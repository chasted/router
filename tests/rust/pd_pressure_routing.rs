#![cfg(all(feature = "cpu", feature = "ready"))]
//! Native HTTP PD shadow routing against explicit, identity-bearing metrics.
use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, Method, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures_util::{stream, StreamExt};
use metrics_exporter_prometheus::PrometheusHandle;
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};
use vllm_router_rs::{
    config::{PolicyConfig, RouterConfig, RoutingMode},
    routers::{RouterFactory, RouterTrait},
    server::AppContext,
};

#[derive(Default)]
struct Seen {
    dispatched: Mutex<HashMap<String, (String, Option<String>)>>,
    scrapes: AtomicUsize,
    scrapes_by_worker: Mutex<HashMap<String, usize>>,
    bad_prefill_signal: AtomicUsize,
    fail_stage: AtomicUsize,
    sequence: AtomicUsize,
}
#[derive(Clone)]
struct Engine {
    name: String,
    url: String,
    seen: Arc<Seen>,
}

async fn capacity(State(engine): State<Engine>) -> Response {
    engine.seen.scrapes.fetch_add(1, Ordering::SeqCst);
    *engine
        .seen
        .scrapes_by_worker
        .lock()
        .unwrap()
        .entry(engine.name.clone())
        .or_default() += 1;
    let fault = if engine.name == "p1" {
        engine.seen.bad_prefill_signal.load(Ordering::SeqCst)
    } else {
        0
    };
    if fault == 3 {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    }
    let stage = if engine.name.starts_with('p') {
        "prefill"
    } else {
        "decode"
    };
    let usage = if fault == 1 {
        "NaN"
    } else if engine.name.ends_with('0') {
        "0.90"
    } else {
        "0.60"
    };
    let epoch = if fault == 2 { "boot-2" } else { "boot-1" };
    format!(
        "vllm:kv_cache_usage_perc{{engine=\"{name}\",model_name=\"pressure-model\"}} {usage}\n\
         router_kv_identity{{stage=\"{stage}\",endpoint=\"{url}\",dp_rank=\"aggregate\",engine_id=\"{name}\",instance_epoch=\"{epoch}\",layout_fingerprint=\"layout-1\",engine=\"{name}\",model_name=\"pressure-model\"}} 1\n",
        name=engine.name, url=engine.url,
    ).into_response()
}

async fn tokenize(Json(body): Json<Value>) -> Json<Value> {
    assert_eq!(body["model"], "pressure-model");
    Json(json!({"tokens": [1,2,3,4,5,6,7,8], "count":8, "max_model_len":4096}))
}

async fn generate(
    State(engine): State<Engine>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let id = headers["x-request-id"].to_str().unwrap().to_owned();
    let fail = engine.seen.fail_stage.load(Ordering::SeqCst);
    if engine.name.starts_with('p') {
        assert_eq!(body["stream"], false);
        assert!(engine
            .seen
            .dispatched
            .lock()
            .unwrap()
            .insert(id.clone(), (engine.name.clone(), None))
            .is_none());
        if fail == 1 {
            return StatusCode::BAD_GATEWAY.into_response();
        }
        return Json(json!({
            "id":id, "model":"pressure-model",
            "choices":[{"index":0,"text":"","finish_reason":"length"}],
            "kv_transfer_params":{"remote_engine_id":engine.name,"remote_block_ids":[[7]],
                "remote_host":"127.0.0.1","remote_port":5600,"tp_size":1,
                "do_remote_prefill":true,"do_remote_decode":false}
        }))
        .into_response();
    }
    {
        let mut records = engine.seen.dispatched.lock().unwrap();
        let (prefill, decode) = records.get_mut(&id).expect("D follows exactly one P");
        assert!(decode.replace(engine.name.clone()).is_none());
        assert_eq!(body["kv_transfer_params"]["remote_engine_id"], *prefill);
        assert_eq!(body["kv_transfer_params"]["remote_block_ids"], json!([[7]]));
    }
    if fail == 2 {
        return StatusCode::BAD_GATEWAY.into_response();
    }
    if body["stream"].as_bool() == Some(true) {
        let text = format!(
            "data: {}\n\ndata: [DONE]\n\n",
            json!({
                "id":id,"object":"text_completion","model":"pressure-model",
                "choices":[{"index":0,"text":"ok","finish_reason":"stop"}],
            })
        );
        (
            [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
            Body::from(text),
        )
            .into_response()
    } else {
        Json(json!({"id":id,"model":"pressure-model",
            "choices":[{"index":0,"text":"ok","finish_reason":"stop"}]}))
        .into_response()
    }
}

struct Fixture {
    router: Box<dyn RouterTrait>,
    seen: Arc<Seen>,
    context: Arc<AppContext>,
    server: tokio::task::JoinHandle<()>,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        self.server.abort();
    }
}
impl Fixture {
    async fn new(p_mode: &str, d_mode: &str) -> Self {
        let seen = Arc::new(Seen::default());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let mut app = Router::new();
        let mappings = |prefix: char| -> Vec<Value> {
            (0..2)
                .map(|index| {
                    let name = format!("{prefix}{index}");
                    json!({
                        "inference_endpoint":format!("{base}/{name}"),"dp_rank":null,
                        "metrics_url":format!("{base}/{name}/pressure-metrics"),
                        "usage":{"name":"vllm:kv_cache_usage_perc",
                            "labels":{"engine":name,"model_name":"pressure-model"}},
                        "identity":{"name":"router_kv_identity","labels":{"engine":name}},
                        "expected_engine_id":name,"expected_instance_epoch":"boot-1",
                        "expected_layout_fingerprint":"layout-1"
                    })
                })
                .collect()
        };
        let pressure = serde_json::from_value(json!({
            "prefill":{"mode":p_mode,"poll_interval_ms":10,"timeout_ms":300,
                "max_age_ms":2000,"max_skew_ms":2000,"mappings":mappings('p')},
            "decode":{"mode":d_mode,"poll_interval_ms":10,"timeout_ms":300,
                "max_age_ms":2000,"max_skew_ms":2000,"mappings":mappings('d')},
        }))
        .unwrap();
        for name in ["p0", "p1", "d0", "d1"] {
            let engine = Engine {
                name: name.to_owned(),
                url: format!("{base}/{name}"),
                seen: seen.clone(),
            };
            let backend = Router::new()
                .route("/health",get(||async{"ok"}))
                .route("/v1/models",get(||async{Json(json!({"data":[{"id":"pressure-model","parent":null}]}))}))
                .route("/get_server_info",get(move||async move{Json(json!({
                    "model_id":"pressure-model","instance_id":name,"block_size":16,
                    "vllm_config":{"model_config":{"model":"pressure-model","max_model_len":4096},
                        "cache_config":{"block_size":16},
                        "parallel_config":{"tensor_parallel_size":1,"data_parallel_size":1}}
                }))}))
                .route("/get_model_info",get(||async{Json(json!({"model_path":"pressure-model","is_generation":true}))}))
                .route("/kv_event_sources",get(||async{Json(json!({}))}))
                .route("/v1/completions/render",post(||async{StatusCode::NOT_FOUND}))
                .route("/tokenize",post(tokenize))
                // Legacy DP startup probing also reads /metrics. Count only
                // the explicitly mapped pressure source, including startup.
                .route("/metrics",get(||async{""}))
                .route("/pressure-metrics",get(capacity))
                .route("/v1/completions",post(generate))
                .with_state(engine);
            app = app.nest(&format!("/{name}"), backend);
        }
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        // All measured calls retain the warm baseline worker. A low-pressure
        // alternative can only be a proposal and must never receive dispatch.
        let policy = PolicyConfig::CacheAware {
            cache_threshold: 0.3,
            balance_abs_threshold: 100_000,
            balance_rel_threshold: 100.0,
            eviction_interval_secs: 3600,
            max_tree_size: 1 << 22,
        };
        let config = RouterConfig {
            mode: RoutingMode::VllmPrefillDecode {
                prefill_urls: vec![(format!("{base}/p0"), None), (format!("{base}/p1"), None)],
                decode_urls: vec![format!("{base}/d0"), format!("{base}/d1")],
                prefill_policy: Some(policy.clone()),
                decode_policy: Some(policy.clone()),
                discovery_address: None,
            },
            policy,
            pd_pressure: Some(pressure),
            disable_retries: true,
            disable_circuit_breaker: true,
            worker_startup_timeout_secs: 5,
            ..RouterConfig::default()
        };
        let client = reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap();
        let context = Arc::new(AppContext::new(config, client, 64, None, vec![]).unwrap());
        let router = RouterFactory::create_router(&context).await.unwrap();
        // Establish the hot-prefix baseline explicitly. Current upstream chooses
        // random cold ties, so registration order cannot define the hot worker.
        for (stage, suffix, policy) in [
            (
                "prefill",
                "/p0",
                context.policy_registry.get_prefill_policy(),
            ),
            ("decode", "/d0", context.policy_registry.get_decode_policy()),
        ] {
            let worker = context
                .worker_registry
                .get_all()
                .into_iter()
                .find(|worker| worker.url().ends_with(suffix))
                .unwrap();
            let cache = policy
                .as_any()
                .downcast_ref::<vllm_router_rs::policies::CacheAwarePolicy>()
                .unwrap();
            assert_eq!(
                cache.select_worker_with_inputs(
                    &[worker],
                    vllm_router_rs::policies::KvRequestTokens::Shared(Some(&[
                        1, 2, 3, 4, 5, 6, 7, 8
                    ])),
                    Some("pressure-model"),
                    None,
                    None,
                    stage,
                ),
                Some(0)
            );
        }
        Self {
            router,
            seen,
            context,
            server,
        }
    }

    async fn request(&self, streaming: bool, expected: StatusCode) {
        let sequence = self.seen.sequence.fetch_add(1, Ordering::SeqCst);
        let response = self
            .router
            .route_transparent(
                None,
                "/v1/completions",
                &Method::POST,
                json!({"model":"pressure-model","prompt":"shared pressure prefix",
                "user":format!("request-{sequence}"),"stream":streaming,"max_tokens":8}),
            )
            .await;
        assert_eq!(response.status(), expected);
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .unwrap();
        if expected.is_success() {
            let text = std::str::from_utf8(&bytes).unwrap();
            assert!(text.contains("ok"), "{text}");
            if streaming {
                assert!(text.contains("[DONE]"));
            }
        }
    }

    fn assert_baseline_and_cleanup(&self, expected: usize) {
        let records = self.seen.dispatched.lock().unwrap();
        assert_eq!(records.len(), expected);
        assert!(records
            .values()
            .all(|(p, d)| p == "p0" && d.as_deref() == Some("d0")));
        assert!(self
            .context
            .worker_registry
            .get_all()
            .iter()
            .all(|w| w.load() == 0));
    }
}

fn counter(handle: &PrometheusHandle, stage: &str, reason: Option<&str>) -> f64 {
    handle
        .render()
        .lines()
        .filter(|line| {
            line.starts_with("vllm_router_pd_pressure_decisions_total{")
                && line.contains(&format!("stage=\"{stage}\""))
                && reason.is_none_or(|r| line.contains(&format!("reason=\"{r}\"")))
        })
        .map(|line| {
            line.split_whitespace()
                .last()
                .unwrap()
                .parse::<f64>()
                .unwrap()
        })
        .sum()
}

fn assert_no_pd_occupancy(handle: &PrometheusHandle) {
    let rendered = handle.render();
    for name in [
        "vllm_router_pd_prefill_occupancy",
        "vllm_router_pd_pending_decode_requests",
        "vllm_router_pd_active_decode_requests",
    ] {
        let values: Vec<f64> = rendered
            .lines()
            .filter(|line| line.starts_with(&format!("{name}{{")))
            .map(|line| line.split_whitespace().last().unwrap().parse().unwrap())
            .collect();
        assert!(!values.is_empty(), "missing {name}");
        assert!(
            values.iter().all(|value| *value == 0.0),
            "{name}: {values:?}"
        );
    }
}

fn actual_counters(fixture: &Fixture, handle: &PrometheusHandle) -> Vec<(String, f64, bool)> {
    let rendered = handle.render();
    let value = |name: &str, labels: &[(&str, &str)]| -> f64 {
        rendered
            .lines()
            .filter(|line| {
                line.starts_with(&format!("{name}{{"))
                    && labels
                        .iter()
                        .all(|(key, val)| line.contains(&format!("{key}=\"{val}\"")))
            })
            .map(|line| {
                line.split_whitespace()
                    .last()
                    .unwrap()
                    .parse::<f64>()
                    .unwrap()
            })
            .sum()
    };
    let mut counters = Vec::new();
    for stage in ["prefill", "decode"] {
        counters.push((
            format!("prefix/{stage}"),
            value(
                "vllm_router_prefix_routing_decisions_total",
                &[("policy", "cache_aware"), ("stage", stage)],
            ),
            true,
        ));
    }
    for worker in fixture.context.worker_registry.get_all() {
        let url = worker.url();
        for name in [
            "vllm_router_processed_requests_total",
            "vllm_router_policy_decisions_total",
        ] {
            counters.push((
                format!("{name}/{url}"),
                value(name, &[("worker", url)]),
                url.ends_with('0'),
            ));
        }
    }
    counters.sort_by(|a, b| a.0.cmp(&b.0));
    counters
}

async fn ready(fixture: &Fixture, handle: &PrometheusHandle, p: bool, d: bool) {
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let before = [
                counter(handle, "prefill", Some("would_override")),
                counter(handle, "decode", Some("would_override")),
            ];
            fixture.request(false, StatusCode::OK).await;
            let after = [
                counter(handle, "prefill", Some("would_override")),
                counter(handle, "decode", Some("would_override")),
            ];
            if (after[0] > before[0]) == p && (after[1] > before[1]) == d {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("pressure collector never reached the requested signal state");
}

async fn invalid_prefill(fixture: &Fixture, handle: &PrometheusHandle, reason: &str) {
    // This must finish before max_age_ms=2000: retaining last-good until TTL
    // expiry must not satisfy an immediate invalidation regression.
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let before = counter(handle, "prefill", Some(reason));
            let proposed_before = [
                counter(handle, "prefill", Some("would_override")),
                counter(handle, "decode", Some("would_override")),
            ];
            fixture.request(false, StatusCode::OK).await;
            if counter(handle, "prefill", Some(reason)) > before {
                assert_eq!(
                    counter(handle, "prefill", Some("would_override")),
                    proposed_before[0]
                );
                assert_eq!(
                    counter(handle, "decode", Some("would_override")),
                    proposed_before[1] + 1.0
                );
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("failed scrape did not invalidate the P signal before its TTL");
}

async fn exercise(rounds: usize, concurrency: usize) -> Value {
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let _scope = metrics::set_default_local_recorder(&recorder);
    let started = Instant::now();
    let mut results = Vec::new();
    let requests = rounds * concurrency;
    for (p, d) in [
        ("off", "off"),
        ("shadow", "off"),
        ("off", "shadow"),
        ("shadow", "shadow"),
    ] {
        let fixture = Fixture::new(p, d).await;
        ready(&fixture, &handle, p == "shadow", d == "shadow").await;
        fixture.seen.dispatched.lock().unwrap().clear();
        let before = [
            counter(&handle, "prefill", None),
            counter(&handle, "decode", None),
        ];
        let proposed_before = [
            counter(&handle, "prefill", Some("would_override")),
            counter(&handle, "decode", Some("would_override")),
        ];
        let actual_before = actual_counters(&fixture, &handle);
        for _ in 0..rounds {
            stream::iter(0..concurrency)
                .map(|i| fixture.request(i % 2 == 0, StatusCode::OK))
                .buffer_unordered(concurrency)
                .collect::<Vec<_>>()
                .await;
        }
        fixture.assert_baseline_and_cleanup(requests);
        assert_no_pd_occupancy(&handle);
        for ((name, after, selected), (before_name, before, before_selected)) in
            actual_counters(&fixture, &handle).iter().zip(actual_before)
        {
            assert_eq!(*name, before_name);
            assert_eq!(*selected, before_selected);
            assert_eq!(
                *after - before,
                if *selected { requests as f64 } else { 0.0 },
                "{name}"
            );
        }
        for (i, stage, mode) in [(0, "prefill", p), (1, "decode", d)] {
            let expected = if mode == "shadow" {
                requests as f64
            } else {
                0.0
            };
            assert_eq!(counter(&handle, stage, None) - before[i], expected);
            assert_eq!(
                counter(&handle, stage, Some("would_override")) - proposed_before[i],
                expected
            );
        }
        if p == "off" && d == "off" {
            assert_eq!(fixture.seen.scrapes.load(Ordering::SeqCst), 0);
        }
        for (prefix, mode) in [('p', p), ('d', d)] {
            if mode == "off" {
                assert!(fixture
                    .seen
                    .scrapes_by_worker
                    .lock()
                    .unwrap()
                    .keys()
                    .all(|name| !name.starts_with(prefix)));
            }
        }
        results.push(
            json!({"prefill":p,"decode":d,"requests":requests,"completed":requests,
            "handoffs_verified":requests,"final_occupancy":0}),
        );
        if p == "shadow" && d == "shadow" {
            for (fault, reason) in [
                (1, "invalid_usage"),
                (2, "identity_mismatch"),
                (3, "scrape_failed"),
            ] {
                fixture
                    .seen
                    .bad_prefill_signal
                    .store(fault, Ordering::SeqCst);
                invalid_prefill(&fixture, &handle, reason).await;
                fixture.seen.bad_prefill_signal.store(0, Ordering::SeqCst);
                ready(&fixture, &handle, true, true).await;
            }
            for failure in [1, 2] {
                fixture.seen.fail_stage.store(failure, Ordering::SeqCst);
                // Existing HTTP PD behavior wraps a P-side 5xx as a network
                // error (500), while forwarding the D-side response status.
                let expected = if failure == 1 {
                    StatusCode::INTERNAL_SERVER_ERROR
                } else {
                    StatusCode::BAD_GATEWAY
                };
                fixture.request(false, expected).await;
                assert!(fixture
                    .context
                    .worker_registry
                    .get_all()
                    .iter()
                    .all(|w| w.load() == 0));
                assert_no_pd_occupancy(&handle);
                fixture.seen.fail_stage.store(0, Ordering::SeqCst);
            }
        }
    }
    json!({"kind":"http_pd_shadow","rounds":rounds,"concurrency":concurrency,
        "scenario_count":results.len(),"requests":requests*results.len(),
        "completed":requests*results.len(),"handoffs_verified":requests*results.len(),
        "cases":results,"final_occupancy":0,"wall_seconds":started.elapsed().as_secs_f64(),
        "evidence":"HTTP PD mocks with explicit identity series; no GPU or TTFT measurement"})
}

#[tokio::test]
async fn http_pd_shadow_modes_and_signal_failures() {
    exercise(2, 4).await;
}

#[tokio::test]
#[ignore = "run with the PR2 pressure validation runner"]
async fn http_pd_pressure_pr2_stress() {
    let parameter = |key, default| max_setting(key, default);
    let result = exercise(
        parameter("ROUTER_PD_STRESS_ROUNDS", 24),
        parameter("ROUTER_PD_STRESS_CONCURRENCY", 16),
    )
    .await;
    println!("PD_PRESSURE_STRESS_RESULT {result}");
}
fn max_setting(key: &str, default: usize) -> usize {
    let value = std::env::var(key).map_or(default, |s| s.parse().unwrap());
    assert!((1..=256).contains(&value));
    value
}
