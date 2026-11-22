use super::{
    parser::parse_usage,
    store::{EngineKey, Store},
    EnginePressureConfig, InvalidReason, PressureMapping, PressureStage, StageObservationConfig,
    StageSnapshot,
};
use crate::core::{Worker, WorkerRegistry};
use futures_util::{stream, StreamExt};
use parking_lot::Mutex;
use std::{
    collections::BTreeMap,
    fmt,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::{sync::watch, task::JoinHandle};

pub struct EnginePressureRuntime {
    config: EnginePressureConfig,
    client: reqwest::Client,
    store: Mutex<Store>,
    started: AtomicBool,
    cancelled: watch::Sender<bool>,
    tasks: Mutex<Vec<JoinHandle<()>>>,
}

struct ScrapeGroup {
    url: String,
    auth: Option<String>,
    mappings: Vec<PressureMapping>,
    scheduled_start: tokio::time::Instant,
}

impl fmt::Debug for EnginePressureRuntime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EnginePressureRuntime")
            .field("started", &self.started.load(Ordering::Acquire))
            .field("prefill_enabled", &self.config.prefill.enabled)
            .field("decode_enabled", &self.config.decode.enabled)
            .finish_non_exhaustive()
    }
}

impl EnginePressureRuntime {
    /// Pure construction; callers supply a dedicated, no-redirect HTTP client.
    pub fn new(config: EnginePressureConfig, client: reqwest::Client) -> Result<Arc<Self>, String> {
        config.validate()?;
        let (cancelled, _) = watch::channel(false);
        Ok(Arc::new(Self {
            config,
            client,
            store: Mutex::new(Store::default()),
            started: AtomicBool::new(false),
            cancelled,
            tasks: Mutex::new(Vec::new()),
        }))
    }

    pub fn config(&self, stage: PressureStage) -> &StageObservationConfig {
        self.config.stage(stage)
    }

    pub fn start(self: &Arc<Self>, registry: Arc<WorkerRegistry>) {
        if self.started.swap(true, Ordering::AcqRel) || *self.cancelled.borrow() {
            return;
        }
        for stage in [PressureStage::Prefill, PressureStage::Decode] {
            let config = self.config(stage);
            if !config.enabled() || config.mappings.is_empty() {
                continue;
            }
            let period = Duration::from_millis(config.poll_interval_ms);
            let weak = Arc::downgrade(self);
            let registry = registry.clone();
            let mut cancelled = self.cancelled.subscribe();
            let handle = tokio::spawn(async move {
                let delay = if stage == PressureStage::Decode {
                    period / 2
                } else {
                    Duration::ZERO
                };
                let mut interval =
                    tokio::time::interval_at(tokio::time::Instant::now() + delay, period);
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                loop {
                    if *cancelled.borrow() {
                        break;
                    }
                    tokio::select! {
                        biased;
                        _ = cancelled.changed() => break,
                        _ = interval.tick() => {},
                    }
                    let Some(runtime) = weak.upgrade() else {
                        break;
                    };
                    tokio::select! {
                        biased;
                        _ = cancelled.changed() => break,
                        _ = runtime.collect_round(stage, registry.clone()) => {},
                    }
                }
            });
            self.tasks.lock().push(handle);
        }
    }

    pub fn shutdown(&self) {
        self.cancelled.send_replace(true);
        for task in self.tasks.lock().drain(..) {
            task.abort();
        }
    }

    /// Reads only bound observations; never fetches metrics or changes worker health.
    pub fn snapshot(
        &self,
        stage: PressureStage,
        workers: &[Arc<dyn Worker>],
        eligible: &[usize],
        now: Instant,
    ) -> Result<StageSnapshot, InvalidReason> {
        self.store
            .lock()
            .snapshot(stage, self.config(stage), workers, eligible, now)
    }

    /// Read one engine without choosing a worker or applying policy thresholds.
    /// The same cached sample and freshness rules back stage snapshots.
    pub fn observe(
        &self,
        key: &super::EngineKey,
        worker: &Arc<dyn Worker>,
        now: Instant,
    ) -> super::PressureObservation {
        let config = self.config(key.stage);
        if !config.enabled {
            return super::PressureObservation::Unknown(InvalidReason::Off);
        }
        self.store
            .lock()
            .observe(key, worker, now, Duration::from_millis(config.max_age_ms))
    }

    async fn collect_round(self: &Arc<Self>, stage: PressureStage, registry: Arc<WorkerRegistry>) {
        let workers = self.bound_workers(stage, &registry);
        self.store.lock().reconcile(stage, &workers);
        let config = self.config(stage);
        let mut groups: BTreeMap<(String, Option<String>), Vec<PressureMapping>> = BTreeMap::new();
        for mapping in &config.mappings {
            groups
                .entry((mapping.metrics_url.clone(), mapping.auth_token_env.clone()))
                .or_default()
                .push(mapping.clone());
        }
        let count = groups.len();
        let round_start = tokio::time::Instant::now();
        let jobs: Vec<_> = groups
            .into_iter()
            .enumerate()
            .map(|(index, ((url, auth), mappings))| {
                let offset =
                    Duration::from_millis(config.poll_interval_ms * index as u64 / count as u64);
                self.clone().collect_group(
                    stage,
                    registry.clone(),
                    ScrapeGroup {
                        url,
                        auth,
                        mappings,
                        scheduled_start: round_start + offset,
                    },
                )
            })
            .collect();
        stream::iter(jobs)
            .buffer_unordered(config.max_concurrent_scrapes)
            .collect::<Vec<_>>()
            .await;
    }

    async fn collect_group(
        self: Arc<Self>,
        stage: PressureStage,
        registry: Arc<WorkerRegistry>,
        group: ScrapeGroup,
    ) {
        // Spread source starts within a poll period; queued jobs never add another full delay.
        tokio::time::sleep_until(group.scheduled_start).await;
        let config = self.config(stage);
        let started = Instant::now();
        let tickets: Vec<_> = {
            let mut store = self.store.lock();
            group
                .mappings
                .into_iter()
                .filter_map(|mapping| {
                    let key = EngineKey {
                        stage,
                        endpoint: mapping.inference_endpoint.clone(),
                        rank: mapping.dp_rank,
                    };
                    store.begin(&key, started).map(|ticket| (mapping, ticket))
                })
                .collect()
        };
        if tickets.is_empty() {
            return;
        }
        let response = tokio::time::timeout(
            Duration::from_millis(config.timeout_ms),
            self.fetch(&group.url, group.auth.as_deref(), config.max_response_bytes),
        )
        .await
        .unwrap_or(Err(InvalidReason::Timeout));
        let completed = Instant::now();
        // A delayed executor can observe an already-ready response after the
        // deadline. Classify that observation as a timeout in both storage and
        // scrape metrics, even if the async timeout selected the response.
        let response =
            if completed.duration_since(started) > Duration::from_millis(config.timeout_ms) {
                Err(InvalidReason::Timeout)
            } else {
                response
            };
        // Reconcile again before publishing: a response for a replaced Arc must not land.
        self.store
            .lock()
            .reconcile(stage, &self.bound_workers(stage, &registry));
        for (mapping, ticket) in tickets {
            let result = response
                .as_ref()
                .map_err(|reason| *reason)
                .and_then(|body| parse_usage(body, stage, &mapping));
            let reason = result
                .as_ref()
                .err()
                .map_or("valid", |reason| reason.as_str());
            let accepted = self.store.lock().publish(
                ticket,
                completed,
                &mapping.expected_layout_fingerprint,
                result,
            );
            metrics::counter!("vllm_router_engine_pressure_scrapes_total",
                "stage" => stage.as_str(), "result" => if accepted { reason } else { "superseded" })
            .increment(1);
        }
        metrics::histogram!("vllm_router_engine_pressure_scrape_duration_seconds",
            "stage" => stage.as_str())
        .record(completed.duration_since(started).as_secs_f64());
    }

    fn bound_workers(
        &self,
        stage: PressureStage,
        registry: &WorkerRegistry,
    ) -> Vec<Arc<dyn Worker>> {
        registry
            .get_all()
            .into_iter()
            .filter(|worker| {
                super::store::stage_matches(stage, worker.as_ref())
                    && self.config(stage).mappings.iter().any(|mapping| {
                        mapping.inference_endpoint == worker.base_url()
                            && mapping.dp_rank == worker.dp_rank()
                    })
            })
            .collect()
    }

    async fn fetch(
        &self,
        url: &str,
        auth: Option<&str>,
        limit: usize,
    ) -> Result<String, InvalidReason> {
        let mut request = self.client.get(url);
        if let Some(name) = auth {
            let token = std::env::var(name).map_err(|_| InvalidReason::ScrapeFailed)?;
            if token.is_empty() {
                return Err(InvalidReason::ScrapeFailed);
            }
            request = request.bearer_auth(token);
        }
        let response = request
            .send()
            .await
            .map_err(|_| InvalidReason::ScrapeFailed)?;
        if !response.status().is_success() {
            return Err(InvalidReason::ScrapeFailed);
        }
        if response
            .content_length()
            .is_some_and(|size| size > limit as u64)
        {
            return Err(InvalidReason::ResponseTooLarge);
        }
        let mut body = Vec::new();
        let mut chunks = response.bytes_stream();
        while let Some(chunk) = chunks.next().await {
            let chunk = chunk.map_err(|_| InvalidReason::ScrapeFailed)?;
            if chunk.len() > limit.saturating_sub(body.len()) {
                return Err(InvalidReason::ResponseTooLarge);
            }
            body.extend_from_slice(&chunk);
        }
        String::from_utf8(body).map_err(|_| InvalidReason::MalformedResponse)
    }

    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub(crate) fn inject_for_test(
        &self,
        stage: PressureStage,
        worker: &Arc<dyn Worker>,
        usage: f64,
        started: Instant,
    ) {
        debug_assert!(super::store::stage_matches(stage, worker.as_ref()));
        self.store.lock().inject(stage, worker, usage, started);
    }
}

impl Drop for EnginePressureRuntime {
    fn drop(&mut self) {
        self.shutdown();
    }
}
