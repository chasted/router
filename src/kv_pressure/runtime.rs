use super::{InvalidReason, PdPressureConfig, PressureStage, StagePressureConfig, StageSnapshot};
use crate::{
    core::{Worker, WorkerRegistry},
    engine_pressure::EnginePressureRuntime,
};
use std::{sync::Arc, time::Instant};

/// Native PD consumer configuration over the shared engine observation service.
#[derive(Debug)]
pub struct KvPressureRuntime {
    config: PdPressureConfig,
    observations: Arc<EnginePressureRuntime>,
}

impl KvPressureRuntime {
    pub fn new(config: PdPressureConfig, client: reqwest::Client) -> Result<Arc<Self>, String> {
        config.validate()?;
        let observations = EnginePressureRuntime::new(config.observation_config(), client)?;
        Ok(Arc::new(Self {
            config,
            observations,
        }))
    }

    pub fn config(&self, stage: PressureStage) -> &StagePressureConfig {
        self.config.stage(stage)
    }
    pub fn observations(&self) -> &Arc<EnginePressureRuntime> {
        &self.observations
    }
    pub fn start(self: &Arc<Self>, registry: Arc<WorkerRegistry>) {
        for stage in [PressureStage::Prefill, PressureStage::Decode] {
            if self.config(stage).mode == super::PressureMode::On {
                // A measured zero distinguishes no overrides from a missing metric.
                metrics::counter!("vllm_router_pd_pressure_overrides_total", "stage" => stage.as_str())
                    .increment(0);
            }
        }
        self.observations.start(registry);
    }
    pub fn shutdown(&self) {
        self.observations.shutdown();
    }
    pub fn snapshot(
        &self,
        stage: PressureStage,
        workers: &[Arc<dyn Worker>],
        eligible: &[usize],
        now: Instant,
    ) -> Result<StageSnapshot, InvalidReason> {
        self.observations.snapshot(stage, workers, eligible, now)
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
        self.observations
            .inject_for_test(stage, worker, usage, started);
    }
}

impl Drop for KvPressureRuntime {
    fn drop(&mut self) {
        self.shutdown();
    }
}
