use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PressureMode {
    #[default]
    Off,
    Shadow,
    On,
}

impl PressureMode {
    pub fn enabled(self) -> bool {
        self != Self::Off
    }
}

use crate::engine_pressure::{EnginePressureConfig, StageObservationConfig};
use crate::engine_pressure::{PressureMapping, PressureStage};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PdPressureConfig {
    pub prefill: StagePressureConfig,
    pub decode: StagePressureConfig,
}

impl PdPressureConfig {
    pub fn enabled(&self) -> bool {
        self.prefill.enabled() || self.decode.enabled()
    }

    pub fn stage(&self, stage: PressureStage) -> &StagePressureConfig {
        match stage {
            PressureStage::Prefill => &self.prefill,
            PressureStage::Decode => &self.decode,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        self.prefill.validate()?;
        self.decode.validate()?;
        self.observation_config().validate()
    }

    pub(crate) fn observation_config(&self) -> EnginePressureConfig {
        EnginePressureConfig {
            prefill: self.prefill.observation_config(),
            decode: self.decode.observation_config(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StagePressureConfig {
    pub mode: PressureMode,
    pub high: f64,
    pub low: f64,
    pub poll_interval_ms: u64,
    pub timeout_ms: u64,
    pub max_age_ms: u64,
    pub max_skew_ms: u64,
    pub max_response_bytes: usize,
    pub max_concurrent_scrapes: usize,
    pub mappings: Vec<PressureMapping>,
}

impl Default for StagePressureConfig {
    fn default() -> Self {
        let observation = StageObservationConfig::default();
        Self {
            mode: PressureMode::Off,
            high: 0.85,
            low: 0.70,
            poll_interval_ms: observation.poll_interval_ms,
            timeout_ms: observation.timeout_ms,
            max_age_ms: observation.max_age_ms,
            max_skew_ms: observation.max_skew_ms,
            max_response_bytes: observation.max_response_bytes,
            max_concurrent_scrapes: observation.max_concurrent_scrapes,
            mappings: observation.mappings,
        }
    }
}

impl StagePressureConfig {
    pub fn enabled(&self) -> bool {
        self.mode.enabled()
    }

    fn validate(&self) -> Result<(), String> {
        if self.mode == PressureMode::On {
            return Err("on is not implemented; use off or shadow".into());
        }
        if !self.low.is_finite()
            || !self.high.is_finite()
            || !(0.0..self.high).contains(&self.low)
            || self.high > 1.0
        {
            return Err("require finite 0 <= low < high <= 1".into());
        }
        self.observation_config().validate()
    }

    pub(crate) fn observation_config(&self) -> StageObservationConfig {
        StageObservationConfig {
            enabled: self.enabled(),
            poll_interval_ms: self.poll_interval_ms,
            timeout_ms: self.timeout_ms,
            max_age_ms: self.max_age_ms,
            max_skew_ms: self.max_skew_ms,
            max_response_bytes: self.max_response_bytes,
            max_concurrent_scrapes: self.max_concurrent_scrapes,
            mappings: self.mappings.clone(),
        }
    }
}
