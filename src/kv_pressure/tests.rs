#![cfg(all(feature = "cpu", feature = "ready"))]

use super::*;
use crate::core::{BasicWorker, Worker, WorkerRegistry, WorkerType};
use crate::engine_pressure::{
    parser::parse_usage,
    store::{EngineKey as BindingKey, Store},
};
use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

fn mapping(endpoint: &str, metrics: &str) -> PressureMapping {
    PressureMapping {
        inference_endpoint: endpoint.into(),
        dp_rank: None,
        metrics_url: metrics.into(),
        auth_token_env: None,
        usage: MetricSelector {
            name: "vllm:kv_cache_usage_perc".into(),
            labels: BTreeMap::from([
                ("engine".into(), "0".into()),
                ("model_name".into(), "m".into()),
            ]),
        },
        identity: MetricSelector {
            name: "router_kv_identity".into(),
            labels: BTreeMap::from([("engine".into(), "0".into())]),
        },
        expected_engine_id: "engine-a".into(),
        expected_instance_epoch: "boot-a".into(),
        expected_layout_fingerprint: "layout-a".into(),
    }
}

fn text(mapping: &PressureMapping, usage: &str) -> String {
    format!("# TYPE vllm:kv_cache_usage_perc gauge\nvllm:kv_cache_usage_perc{{engine=\"0\",model_name=\"m\"}} {usage}\n\
        router_kv_identity{{stage=\"prefill\",endpoint=\"{}\",dp_rank=\"aggregate\",engine_id=\"engine-a\",instance_epoch=\"boot-a\",layout_fingerprint=\"layout-a\",engine=\"0\",model_name=\"m\"}} 1\n",
        mapping.inference_endpoint)
}

fn worker(url: &str) -> Arc<dyn Worker> {
    Arc::new(BasicWorker::new(
        url.into(),
        WorkerType::Prefill {
            bootstrap_port: None,
        },
    ))
}

#[test]
fn defaults_are_off_and_enabled_modes_without_mappings_fall_back() {
    let config: PdPressureConfig = serde_json::from_str("{}").unwrap();
    assert!(!config.enabled());
    config.validate().unwrap();
    let shadow: PdPressureConfig =
        serde_json::from_str(r#"{"prefill":{"mode":"shadow"}}"#).unwrap();
    assert!(shadow.prefill.enabled());
    assert!(!shadow.decode.enabled());
    shadow.validate().unwrap();
    let mut on = shadow.clone();
    on.prefill.mode = PressureMode::On;
    on.validate().unwrap();
    let runtime = super::KvPressureRuntime::new(on, reqwest::Client::new()).unwrap();
    assert!(runtime
        .snapshot(PressureStage::Prefill, &[], &[], Instant::now())
        .is_err());
}

#[test]
fn config_rejects_bad_bounds_ambiguous_sources_and_implicit_identity() {
    let mut config = PdPressureConfig::default();
    config.prefill.mode = PressureMode::Shadow;
    config
        .prefill
        .mappings
        .push(mapping("http://p0", "http://metrics/metrics"));
    config.validate().unwrap();
    let mut bad = config.clone();
    bad.prefill.low = f64::NAN;
    assert!(bad.validate().is_err());
    let mut bad = config.clone();
    bad.prefill.high = bad.prefill.low;
    assert!(bad.validate().is_err());
    let mut bad = config.clone();
    bad.prefill.max_concurrent_scrapes = 0;
    assert!(bad.validate().is_err());
    let mut bad = config.clone();
    bad.prefill.max_response_bytes = 16_777_217;
    assert!(bad.validate().is_err());
    let mut bad = config.clone();
    bad.prefill.mappings = vec![bad.prefill.mappings[0].clone(); 65];
    assert!(bad.validate().is_err());
    let mut bad = config.clone();
    bad.decode.mappings = bad.prefill.mappings.clone();
    assert!(bad.validate().is_err());
    for value in [
        "grpc://metrics",
        "http://secret@metrics",
        "http://metrics?token=secret",
    ] {
        let mut bad = config.clone();
        bad.prefill.mappings[0].metrics_url = value.into();
        assert!(bad.validate().is_err());
    }
    let mut bad = config.clone();
    bad.prefill.mappings[0].usage.labels.remove("engine");
    assert!(bad.validate().is_err());
    let mut ranked = config.clone();
    ranked.prefill.mappings[0].dp_rank = Some(1);
    assert!(ranked.validate().is_err());
    ranked.prefill.mappings[0]
        .usage
        .labels
        .insert("dp_rank".into(), "1".into());
    ranked.validate().unwrap();
    let mut bad = config;
    bad.prefill.mappings[0].expected_instance_epoch.clear();
    assert!(bad.validate().is_err());
}

#[test]
fn config_rejects_overlapping_selectors_across_targets_and_stages() {
    let first = mapping("http://p0", "http://METRICS:80/metrics");
    let mut refined = mapping("http://p1", "http://metrics/metrics");
    refined.usage.labels.insert("source".into(), "x".into());
    let mut config = PdPressureConfig::default();
    config.prefill.mappings = vec![first.clone(), refined.clone()];
    assert!(config.validate().unwrap_err().contains("overlap"));
    config.prefill.mappings = vec![first];
    config.decode.mappings = vec![refined];
    assert!(config.validate().unwrap_err().contains("overlap"));
    // An explicit conflicting label makes the selected series disjoint.
    config.decode.mappings[0]
        .usage
        .labels
        .insert("engine".into(), "1".into());
    config.validate().unwrap();
}

#[test]
fn parser_requires_live_exact_identity_and_unique_series() {
    let mapping = mapping("http://p0", "http://metrics/metrics");
    let body = text(&mapping, "0.91");
    assert_eq!(
        parse_usage(&body, PressureStage::Prefill, &mapping),
        Ok(0.91)
    );
    assert_eq!(
        parse_usage(&body, PressureStage::Decode, &mapping),
        Err(InvalidReason::IdentityMismatch)
    );
    for (from, to) in [
        ("boot-a", "boot-b"),
        ("engine-a", "engine-b"),
        ("layout-a", "layout-b"),
        ("aggregate", "1"),
        ("http://p0", "http://p1"),
    ] {
        assert_eq!(
            parse_usage(&body.replace(from, to), PressureStage::Prefill, &mapping),
            Err(InvalidReason::IdentityMismatch)
        );
    }
    assert_eq!(
        parse_usage(
            body.lines().nth(1).unwrap(),
            PressureStage::Prefill,
            &mapping
        ),
        Err(InvalidReason::MissingSeries)
    );
    assert_eq!(
        parse_usage(
            &format!("{body}\n{}", body.lines().nth(1).unwrap()),
            PressureStage::Prefill,
            &mapping
        ),
        Err(InvalidReason::AmbiguousSeries)
    );
    assert_eq!(
        parse_usage(
            &format!("{body}\n{}", body.lines().nth(2).unwrap()),
            PressureStage::Prefill,
            &mapping
        ),
        Err(InvalidReason::AmbiguousSeries)
    );
    for value in ["NaN", "+Inf", "-Inf", "-0.1", "1.01"] {
        assert_eq!(
            parse_usage(&text(&mapping, value), PressureStage::Prefill, &mapping),
            Err(InvalidReason::InvalidUsage)
        );
    }
    let mismatched_join = body.replacen("model_name=\"m\"} 1", "model_name=\"other\"} 1", 1);
    assert!(parse_usage(&mismatched_join, PressureStage::Prefill, &mapping).is_err());
}

#[test]
fn parser_rejects_duplicate_labels_and_handles_escaped_values() {
    let mut mapping = mapping("http://p0", "http://metrics/metrics");
    let body = text(&mapping, "0.2");
    let duplicate = body.replacen("engine=\"0\",", "engine=\"0\",engine=\"0\",", 1);
    assert_eq!(
        parse_usage(&duplicate, PressureStage::Prefill, &mapping),
        Err(InvalidReason::MalformedResponse)
    );
    let escaped = body.replace("model_name=\"m\"", "model_name=\"a\\\"b\\\\c\\nd\"");
    mapping
        .usage
        .labels
        .insert("model_name".into(), "a\"b\\c\nd".into());
    assert_eq!(
        parse_usage(&escaped, PressureStage::Prefill, &mapping),
        Ok(0.2)
    );
}

#[test]
fn ranked_series_never_borrows_an_aggregate_or_another_rank() {
    let mut mapping = mapping("http://p0", "http://metrics/metrics");
    mapping.dp_rank = Some(1);
    mapping.usage.labels.insert("dp_rank".into(), "1".into());
    let aggregate = text(&mapping, "0.2");
    assert_eq!(
        parse_usage(&aggregate, PressureStage::Prefill, &mapping),
        Err(InvalidReason::MissingSeries)
    );
    let ranked = aggregate
        .replacen("model_name=\"m\"}", "model_name=\"m\",dp_rank=\"1\"}", 1)
        .replace("dp_rank=\"aggregate\"", "dp_rank=\"1\"");
    assert_eq!(
        parse_usage(&ranked, PressureStage::Prefill, &mapping),
        Ok(0.2)
    );
    assert_eq!(
        parse_usage(
            &ranked.replace("dp_rank=\"1\"", "dp_rank=\"0\""),
            PressureStage::Prefill,
            &mapping
        ),
        Err(InvalidReason::MissingSeries)
    );
}

#[test]
fn aggregate_identity_rejects_ranked_or_conflicting_usage_labels() {
    let mapping = mapping("http://p0", "http://metrics/metrics");
    let body = text(&mapping, "0.2");
    for label in [
        "dp_rank=\"0\"",
        "instance_epoch=\"different-boot\"",
        "engine_id=\"different-engine\"",
    ] {
        let conflicting = body.replacen(
            "model_name=\"m\"}",
            &format!("model_name=\"m\",{label}}}"),
            1,
        );
        assert_eq!(
            parse_usage(&conflicting, PressureStage::Prefill, &mapping),
            Err(InvalidReason::IdentityMismatch)
        );
    }
}

#[test]
fn ambiguous_live_worker_objects_are_not_bound_and_future_samples_fail() {
    let first = worker("http://p0");
    let duplicate = worker("http://p0");
    let stage = PressureStage::Prefill;
    let now = Instant::now();
    let config = StagePressureConfig {
        mode: PressureMode::Shadow,
        ..Default::default()
    };
    let mut store = Store::default();
    store.inject(stage, &first, 0.2, now);
    store.reconcile(stage, &[first.clone(), duplicate]);
    assert_eq!(
        store
            .snapshot(
                stage,
                &config.observation_config(),
                std::slice::from_ref(&first),
                &[0],
                now
            )
            .unwrap_err(),
        InvalidReason::UnboundWorker
    );
    store.inject(stage, &first, 0.2, now + Duration::from_millis(1));
    assert_eq!(
        store
            .snapshot(
                stage,
                &config.observation_config(),
                std::slice::from_ref(&first),
                &[0],
                now
            )
            .unwrap_err(),
        InvalidReason::Stale
    );
    store.inject(stage, &first, f64::NAN, now);
    assert_eq!(
        store
            .snapshot(stage, &config.observation_config(), &[first], &[0], now)
            .unwrap_err(),
        InvalidReason::InvalidUsage
    );
}

#[test]
fn store_rebind_fences_old_responses_and_failed_scrapes_invalidate() {
    let first = worker("http://p0");
    let replacement = worker("http://p0");
    let now = Instant::now();
    let mut store = Store::default();
    let key = BindingKey::for_worker(PressureStage::Prefill, first.as_ref());
    store.reconcile(PressureStage::Prefill, std::slice::from_ref(&first));
    let old = store.begin(&key, now).unwrap();
    store.reconcile(PressureStage::Prefill, std::slice::from_ref(&replacement));
    let newer = store.begin(&key, now).unwrap();
    assert!(newer.generation > old.generation);
    assert!(!store.publish(old, now, "layout", Ok(0.1)));
    assert!(store.publish(newer, now, "layout", Ok(0.9)));
    let config = StagePressureConfig {
        mode: PressureMode::Shadow,
        ..Default::default()
    };
    assert_eq!(
        store
            .snapshot(
                PressureStage::Prefill,
                &config.observation_config(),
                std::slice::from_ref(&first),
                &[0],
                now
            )
            .unwrap_err(),
        InvalidReason::UnboundWorker
    );
    assert_eq!(
        store
            .snapshot(
                PressureStage::Prefill,
                &config.observation_config(),
                std::slice::from_ref(&replacement),
                &[0],
                now
            )
            .unwrap()
            .usage(0),
        Some(0.9)
    );
    let earlier = store.begin(&key, now).unwrap();
    let later = store.begin(&key, now).unwrap();
    store.publish(later, now, "layout", Err(InvalidReason::ScrapeFailed));
    assert!(!store.publish(earlier, now, "layout", Ok(0.1)));
    assert_eq!(
        store
            .snapshot(
                PressureStage::Prefill,
                &config.observation_config(),
                &[replacement],
                &[0],
                now
            )
            .unwrap_err(),
        InvalidReason::ScrapeFailed
    );
}

#[test]
fn store_requires_every_candidate_and_stage_layout_age_skew_match() {
    let workers = vec![worker("http://p0"), worker("http://p1")];
    let stage = PressureStage::Prefill;
    let now = Instant::now();
    let mut store = Store::default();
    let config = StagePressureConfig {
        mode: PressureMode::Shadow,
        max_age_ms: 100,
        max_skew_ms: 20,
        timeout_ms: 50,
        ..Default::default()
    };
    store.inject(stage, &workers[0], 0.9, now);
    assert!(store
        .snapshot(stage, &config.observation_config(), &workers, &[0, 1], now)
        .is_err());
    store.inject(stage, &workers[1], 0.2, now);
    let snapshot = store
        .snapshot(stage, &config.observation_config(), &workers, &[1, 0], now)
        .unwrap();
    assert_eq!(snapshot.usage(1), Some(0.2));
    assert!(snapshot
        .samples
        .iter()
        .all(|sample| sample.source_sample_time.is_none()));
    assert!(store
        .snapshot(
            PressureStage::Decode,
            &config.observation_config(),
            &workers,
            &[0, 1],
            now
        )
        .is_err());
    assert_eq!(
        store
            .snapshot(
                stage,
                &config.observation_config(),
                &workers,
                &[0, 1],
                now + Duration::from_millis(101)
            )
            .unwrap_err(),
        InvalidReason::Stale
    );
    store.inject(stage, &workers[1], 0.2, now + Duration::from_millis(21));
    assert_eq!(
        store
            .snapshot(
                stage,
                &config.observation_config(),
                &workers,
                &[0, 1],
                now + Duration::from_millis(21)
            )
            .unwrap_err(),
        InvalidReason::Skew
    );
    let key = BindingKey::for_worker(stage, workers[1].as_ref());
    let ticket = store.begin(&key, now).unwrap();
    store.publish(ticket, now, "different-layout", Ok(0.2));
    assert_eq!(
        store
            .snapshot(stage, &config.observation_config(), &workers, &[0, 1], now)
            .unwrap_err(),
        InvalidReason::LayoutMismatch
    );
    store.reconcile(stage, &[]);
    assert!(store
        .snapshot(stage, &config.observation_config(), &workers, &[0], now)
        .is_err());
}

#[tokio::test]
async fn collector_polls_only_enabled_stage_and_invalidates_failed_response() {
    use axum::{extract::State, routing::get, Router};
    let responses = Arc::new(AtomicUsize::new(0));
    let current = Arc::new(parking_lot::Mutex::new(String::new()));
    let state = (responses.clone(), current.clone());
    let app = Router::new()
        .route(
            "/metrics",
            get(
                |State((calls, body)): State<(
                    Arc<AtomicUsize>,
                    Arc<parking_lot::Mutex<String>>,
                )>| async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    body.lock().clone()
                },
            ),
        )
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let metrics_url = format!("http://{}/metrics", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let p = worker("http://p0");
    let binding = mapping(p.base_url(), &metrics_url);
    *current.lock() = text(&binding, "0.93");
    let mut config = PdPressureConfig::default();
    config.prefill.mode = PressureMode::Shadow;
    config.prefill.poll_interval_ms = 20;
    config.prefill.mappings.push(binding);
    let registry = Arc::new(WorkerRegistry::new());
    registry.register(p.clone());
    let runtime = KvPressureRuntime::new(
        config,
        reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap(),
    )
    .unwrap();
    runtime.start(registry);
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if runtime
                .snapshot(
                    PressureStage::Prefill,
                    std::slice::from_ref(&p),
                    &[0],
                    Instant::now(),
                )
                .is_ok()
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(
        runtime
            .snapshot(
                PressureStage::Decode,
                std::slice::from_ref(&p),
                &[0],
                Instant::now()
            )
            .unwrap_err(),
        InvalidReason::Off
    );
    *current.lock() = "vllm:kv_cache_usage_perc{engine=\"0\",model_name=\"m\"} 0.1".into();
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if matches!(
                runtime.snapshot(
                    PressureStage::Prefill,
                    std::slice::from_ref(&p),
                    &[0],
                    Instant::now()
                ),
                Err(InvalidReason::MissingSeries)
            ) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    runtime.shutdown();
    tokio::time::sleep(Duration::from_millis(10)).await;
    let stopped = responses.load(Ordering::SeqCst);
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(responses.load(Ordering::SeqCst), stopped);
    server.abort();
}

#[tokio::test]
async fn collector_shares_one_scrape_between_distinct_workers_at_one_source() {
    use axum::{extract::State, routing::get, Router};
    let calls = Arc::new(AtomicUsize::new(0));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let metrics_url = format!("http://{}/metrics", listener.local_addr().unwrap());
    let workers = vec![worker("http://shared-p0"), worker("http://shared-p1")];
    let first = mapping(workers[0].base_url(), &metrics_url);
    let mut second = mapping(workers[1].base_url(), &metrics_url);
    second.usage.labels.insert("engine".into(), "1".into());
    second.identity.labels.insert("engine".into(), "1".into());
    let body = format!(
        "{}{}",
        text(&first, "0.9"),
        text(&second, "0.2").replace("engine=\"0\"", "engine=\"1\"")
    );
    let app = Router::new()
        .route(
            "/metrics",
            get(
                |State((calls, body)): State<(Arc<AtomicUsize>, String)>| async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    body
                },
            ),
        )
        .with_state((calls.clone(), body));
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let mut config = PdPressureConfig::default();
    config.prefill.mode = PressureMode::Shadow;
    // A second poll cannot explain both samples becoming available together.
    config.prefill.poll_interval_ms = 10_000;
    config.prefill.mappings = vec![first, second];
    let registry = Arc::new(WorkerRegistry::new());
    for worker in &workers {
        registry.register(worker.clone());
    }
    let runtime = KvPressureRuntime::new(
        config,
        reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap(),
    )
    .unwrap();
    runtime.start(registry);
    let snapshot = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if let Ok(snapshot) =
                runtime.snapshot(PressureStage::Prefill, &workers, &[0, 1], Instant::now())
            {
                break snapshot;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("both workers must be sampled in the first shared scrape");
    assert_eq!(snapshot.usage(0), Some(0.9));
    assert_eq!(snapshot.usage(1), Some(0.2));
    assert_eq!(
        snapshot.samples[0].scrape_started_at,
        snapshot.samples[1].scrape_started_at
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    runtime.shutdown();
    println!(
        "PD_PRESSURE_SHARED_SOURCE={}",
        serde_json::json!({
            "workers": workers.len(), "http_scrapes": calls.load(Ordering::SeqCst),
            "usage": [snapshot.usage(0), snapshot.usage(1)],
        })
    );
    server.abort();
}

#[tokio::test]
async fn collector_enforces_response_size_timeout_and_off_without_health_changes() {
    use axum::{extract::State, http::StatusCode, response::IntoResponse, routing::get, Router};
    let mode = Arc::new(AtomicUsize::new(0));
    let calls = Arc::new(AtomicUsize::new(0));
    let app = Router::new()
        .route(
            "/metrics",
            get(
                |State((mode, calls)): State<(Arc<AtomicUsize>, Arc<AtomicUsize>)>| async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    match mode.load(Ordering::SeqCst) {
                        0 => "x".repeat(2048).into_response(),
                        1 => {
                            tokio::time::sleep(Duration::from_millis(100)).await;
                            "slow".into_response()
                        }
                        _ => StatusCode::SERVICE_UNAVAILABLE.into_response(),
                    }
                },
            ),
        )
        .with_state((mode.clone(), calls.clone()));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let metrics_url = format!("http://{}/metrics", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let p = worker("http://p0");
    let registry = Arc::new(WorkerRegistry::new());
    registry.register(p.clone());
    let mut config = PdPressureConfig::default();
    config
        .prefill
        .mappings
        .push(mapping(p.base_url(), &metrics_url));
    let client = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();
    let off = KvPressureRuntime::new(config.clone(), client.clone()).unwrap();
    off.start(registry.clone());
    tokio::time::sleep(Duration::from_millis(30)).await;
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    off.shutdown();
    config.prefill.mode = PressureMode::Shadow;
    config.prefill.max_response_bytes = 128;
    config.prefill.poll_interval_ms = 10;
    config.prefill.timeout_ms = 20;
    let runtime = KvPressureRuntime::new(config, client).unwrap();
    let healthy = p.is_healthy();
    runtime.start(registry);
    for (setting, expected) in [
        (0, InvalidReason::ResponseTooLarge),
        (1, InvalidReason::Timeout),
        (2, InvalidReason::ScrapeFailed),
    ] {
        mode.store(setting, Ordering::SeqCst);
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if matches!(runtime.snapshot(PressureStage::Prefill, std::slice::from_ref(&p), &[0], Instant::now()), Err(reason) if reason == expected) { break; }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        }).await.unwrap();
        assert_eq!(p.is_healthy(), healthy);
    }
    runtime.shutdown();
    server.abort();
}

#[tokio::test]
async fn collector_bounds_concurrent_sources() {
    use axum::{
        extract::{Path, State},
        routing::get,
        Router,
    };
    let active = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let app = Router::new().route("/{id}", get(|Path(id): Path<usize>, State((active, peak)): State<(Arc<AtomicUsize>, Arc<AtomicUsize>)>| async move {
        let current = active.fetch_add(1, Ordering::SeqCst) + 1;
        peak.fetch_max(current, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(30)).await;
        active.fetch_sub(1, Ordering::SeqCst);
        text(&mapping(&format!("http://p{id}"), "http://unused"), "0.5")
    })).with_state((active, peak.clone()));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let workers: Vec<_> = (0..6).map(|i| worker(&format!("http://p{i}"))).collect();
    let registry = Arc::new(WorkerRegistry::new());
    let mut config = PdPressureConfig::default();
    config.prefill.mode = PressureMode::Shadow;
    config.prefill.poll_interval_ms = 1;
    config.prefill.max_concurrent_scrapes = 2;
    for (i, worker) in workers.iter().enumerate() {
        registry.register(worker.clone());
        config
            .prefill
            .mappings
            .push(mapping(worker.base_url(), &format!("http://{address}/{i}")));
    }
    let runtime = KvPressureRuntime::new(
        config,
        reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap(),
    )
    .unwrap();
    runtime.start(registry);
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if runtime
                .snapshot(
                    PressureStage::Prefill,
                    &workers,
                    &[0, 1, 2, 3, 4, 5],
                    Instant::now(),
                )
                .is_ok()
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    runtime.shutdown();
    assert_eq!(
        peak.load(Ordering::SeqCst),
        2,
        "source requests must obey the configured bound"
    );
    server.abort();
}
