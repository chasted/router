use super::{PressureStage, StageObservationConfig};
use crate::core::{Worker, WorkerType};
use std::{
    collections::HashMap,
    sync::{Arc, Weak},
    time::{Duration, Instant, SystemTime},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InvalidReason {
    Off,
    Unsupported,
    UnboundWorker,
    AmbiguousWorker,
    MissingSample,
    ScrapeFailed,
    Timeout,
    ResponseTooLarge,
    MalformedResponse,
    MissingSeries,
    AmbiguousSeries,
    IdentityMismatch,
    InvalidUsage,
    Stale,
    Skew,
    LayoutMismatch,
    Superseded,
}

impl InvalidReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Unsupported => "unsupported",
            Self::UnboundWorker => "unbound_worker",
            Self::AmbiguousWorker => "ambiguous_worker",
            Self::MissingSample => "missing_sample",
            Self::ScrapeFailed => "scrape_failed",
            Self::Timeout => "timeout",
            Self::ResponseTooLarge => "response_too_large",
            Self::MalformedResponse => "malformed_response",
            Self::MissingSeries => "missing_series",
            Self::AmbiguousSeries => "ambiguous_series",
            Self::IdentityMismatch => "identity_mismatch",
            Self::InvalidUsage => "invalid_usage",
            Self::Stale => "stale",
            Self::Skew => "skew",
            Self::LayoutMismatch => "layout_mismatch",
            Self::Superseded => "superseded",
        }
    }
}

#[derive(Clone, Debug)]
pub struct EngineSample {
    pub usage: f64,
    pub scrape_started_at: Instant,
    pub scrape_completed_at: Instant,
    pub source_sample_time: Option<SystemTime>,
    pub binding_generation: u64,
    pub layout_fingerprint: String,
}

/// A validated engine observation projected into one candidate view.
#[derive(Clone, Debug)]
pub struct PressureSample {
    pub worker_index: usize,
    pub observation: EngineSample,
}

impl std::ops::Deref for PressureSample {
    type Target = EngineSample;
    fn deref(&self) -> &Self::Target {
        &self.observation
    }
}

/// Failure is explicit; unavailable data must never be interpreted as zero use.
#[derive(Clone, Debug)]
pub enum PressureObservation {
    Known(EngineSample),
    Unknown(InvalidReason),
}

impl EngineSample {
    fn at(&self, now: Instant, max_age: Duration) -> Result<Self, InvalidReason> {
        let age = now
            .checked_duration_since(self.scrape_started_at)
            .ok_or(InvalidReason::Stale)?;
        if age > max_age.min(super::MAX_OBSERVATION_AGE)
            || self.scrape_completed_at < self.scrape_started_at
            || self.scrape_completed_at > now
        {
            return Err(InvalidReason::Stale);
        }
        Ok(self.clone())
    }
}

#[derive(Clone, Debug)]
pub struct StageSnapshot {
    pub samples: Vec<PressureSample>,
}

impl StageSnapshot {
    pub fn usage(&self, worker_index: usize) -> Option<f64> {
        self.samples
            .iter()
            .find(|sample| sample.worker_index == worker_index)
            .map(|sample| sample.usage)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EngineKey {
    pub stage: PressureStage,
    pub endpoint: String,
    pub rank: Option<usize>,
}

impl EngineKey {
    pub fn for_worker(stage: PressureStage, worker: &dyn Worker) -> Self {
        Self {
            stage,
            endpoint: worker.base_url().to_owned(),
            rank: worker.dp_rank(),
        }
    }
}

pub(crate) fn stage_matches(stage: PressureStage, worker: &dyn Worker) -> bool {
    matches!(
        (stage, worker.worker_type()),
        (PressureStage::Prefill, WorkerType::Prefill { .. })
            | (PressureStage::Decode, WorkerType::Decode)
    )
}

struct Binding {
    worker: Weak<dyn Worker>,
    generation: u64,
    sequence: u64,
    value: Result<EngineSample, InvalidReason>,
}

#[derive(Clone, Debug)]
pub(crate) struct ScrapeTicket {
    pub key: EngineKey,
    pub generation: u64,
    pub sequence: u64,
    pub started: Instant,
}

#[derive(Default)]
pub(crate) struct Store {
    bindings: HashMap<EngineKey, Binding>,
    next_generation: u64,
}

impl Store {
    /// Reconcile live Arc identity, never URL alone. A registry ID can survive replacement.
    pub fn reconcile(&mut self, stage: PressureStage, workers: &[Arc<dyn Worker>]) {
        let mut live: HashMap<EngineKey, Vec<&Arc<dyn Worker>>> = HashMap::new();
        for worker in workers
            .iter()
            .filter(|worker| stage_matches(stage, worker.as_ref()))
        {
            live.entry(EngineKey::for_worker(stage, worker.as_ref()))
                .or_default()
                .push(worker);
        }
        self.bindings
            .retain(|key, _| key.stage != stage || live.contains_key(key));
        for (key, workers) in live {
            if workers.len() != 1 {
                self.bindings.remove(&key);
                continue;
            }
            self.bind(key, workers[0]);
        }
    }

    fn bind(&mut self, key: EngineKey, worker: &Arc<dyn Worker>) {
        if self
            .bindings
            .get(&key)
            .and_then(|binding| binding.worker.upgrade())
            .is_some_and(|old| Arc::ptr_eq(&old, worker))
        {
            return;
        }
        self.next_generation = self
            .next_generation
            .checked_add(1)
            .expect("binding generation exhausted");
        self.bindings.insert(
            key,
            Binding {
                worker: Arc::downgrade(worker),
                generation: self.next_generation,
                sequence: 0,
                value: Err(InvalidReason::MissingSample),
            },
        );
    }

    pub fn begin(&mut self, key: &EngineKey, started: Instant) -> Option<ScrapeTicket> {
        let binding = self.bindings.get_mut(key)?;
        binding.worker.upgrade()?;
        binding.sequence = binding.sequence.checked_add(1)?;
        Some(ScrapeTicket {
            key: key.clone(),
            generation: binding.generation,
            sequence: binding.sequence,
            started,
        })
    }

    pub fn publish(
        &mut self,
        ticket: ScrapeTicket,
        completed: Instant,
        layout: &str,
        value: Result<f64, InvalidReason>,
    ) -> bool {
        let Some(binding) = self.bindings.get_mut(&ticket.key) else {
            return false;
        };
        if binding.generation != ticket.generation
            || binding.sequence != ticket.sequence
            || binding.worker.upgrade().is_none()
        {
            return false;
        }
        binding.value = value.and_then(|usage| {
            if !usage.is_finite() || !(0.0..=1.0).contains(&usage) {
                return Err(InvalidReason::InvalidUsage);
            }
            if completed
                .checked_duration_since(ticket.started)
                .is_none_or(|duration| duration > super::MAX_SCRAPE_DURATION)
            {
                return Err(InvalidReason::Timeout);
            }
            Ok(EngineSample {
                usage,
                scrape_started_at: ticket.started,
                scrape_completed_at: completed,
                source_sample_time: None,
                binding_generation: ticket.generation,
                layout_fingerprint: layout.to_owned(),
            })
        });
        true
    }

    fn sample(
        &self,
        key: &EngineKey,
        worker: &Arc<dyn Worker>,
        now: Instant,
        max_age: Duration,
    ) -> Result<EngineSample, InvalidReason> {
        if *key != EngineKey::for_worker(key.stage, worker.as_ref())
            || !stage_matches(key.stage, worker.as_ref())
        {
            return Err(InvalidReason::UnboundWorker);
        }
        if worker.dp_rank().is_none() && worker.dp_size().is_some_and(|size| size > 1) {
            return Err(InvalidReason::Unsupported);
        }
        let binding = self.bindings.get(key).ok_or(InvalidReason::UnboundWorker)?;
        if !binding
            .worker
            .upgrade()
            .is_some_and(|old| Arc::ptr_eq(&old, worker))
        {
            return Err(InvalidReason::UnboundWorker);
        }
        binding
            .value
            .as_ref()
            .map_err(|reason| *reason)?
            .at(now, max_age)
    }

    pub fn observe(
        &self,
        key: &EngineKey,
        worker: &Arc<dyn Worker>,
        now: Instant,
        max_age: Duration,
    ) -> PressureObservation {
        match self.sample(key, worker, now, max_age) {
            Ok(sample) => PressureObservation::Known(sample),
            Err(reason) => PressureObservation::Unknown(reason),
        }
    }

    pub fn snapshot(
        &self,
        stage: PressureStage,
        config: &StageObservationConfig,
        workers: &[Arc<dyn Worker>],
        eligible: &[usize],
        now: Instant,
    ) -> Result<StageSnapshot, InvalidReason> {
        if !config.enabled() {
            return Err(InvalidReason::Off);
        }
        if eligible.is_empty() {
            return Err(InvalidReason::Unsupported);
        }
        let mut samples = Vec::with_capacity(eligible.len());
        for &index in eligible {
            let worker = workers.get(index).ok_or(InvalidReason::UnboundWorker)?;
            let key = EngineKey::for_worker(stage, worker.as_ref());
            let observation =
                self.sample(&key, worker, now, Duration::from_millis(config.max_age_ms))?;
            samples.push(PressureSample {
                worker_index: index,
                observation,
            });
        }
        let earliest = samples
            .iter()
            .map(|sample| sample.scrape_started_at)
            .min()
            .unwrap();
        let latest = samples
            .iter()
            .map(|sample| sample.scrape_started_at)
            .max()
            .unwrap();
        if latest.duration_since(earliest) > Duration::from_millis(config.max_skew_ms) {
            return Err(InvalidReason::Skew);
        }
        if samples
            .iter()
            .any(|sample| sample.layout_fingerprint != samples[0].layout_fingerprint)
        {
            return Err(InvalidReason::LayoutMismatch);
        }
        Ok(StageSnapshot { samples })
    }

    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn inject(
        &mut self,
        stage: PressureStage,
        worker: &Arc<dyn Worker>,
        usage: f64,
        started: Instant,
    ) {
        let key = EngineKey::for_worker(stage, worker.as_ref());
        self.bind(key.clone(), worker);
        let ticket = self.begin(&key, started).unwrap();
        self.publish(ticket, started, "test-fixture-layout", Ok(usage));
    }
}
