use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PressureStage {
    Prefill,
    Decode,
}

impl PressureStage {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EnginePressureConfig {
    pub prefill: StageObservationConfig,
    pub decode: StageObservationConfig,
}

impl EnginePressureConfig {
    pub fn enabled(&self) -> bool {
        self.prefill.enabled() || self.decode.enabled()
    }

    pub fn stage(&self, stage: PressureStage) -> &StageObservationConfig {
        match stage {
            PressureStage::Prefill => &self.prefill,
            PressureStage::Decode => &self.decode,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        let mut mappings: Vec<&PressureMapping> = Vec::new();
        for stage in [PressureStage::Prefill, PressureStage::Decode] {
            self.stage(stage)
                .validate()
                .map_err(|error| format!("engine_pressure.{}: {error}", stage.as_str()))?;
            for mapping in &self.stage(stage).mappings {
                if mappings
                    .iter()
                    .any(|other| usage_selectors_overlap(other, mapping))
                {
                    return Err(
                        "engine_pressure: usage selectors may overlap within or across stages"
                            .into(),
                    );
                }
                mappings.push(mapping);
            }
        }
        Ok(())
    }
}

fn usage_selectors_overlap(left: &PressureMapping, right: &PressureMapping) -> bool {
    // Equal selectors are not the only overlap: a less specific selector also
    // matches every series selected by its refinement. URL parsing normalizes
    // host case/default ports before comparing the explicit scrape endpoints.
    url::Url::parse(&left.metrics_url).ok() == url::Url::parse(&right.metrics_url).ok()
        && left.usage.name == right.usage.name
        && !left.usage.labels.iter().any(|(key, value)| {
            right
                .usage
                .labels
                .get(key)
                .is_some_and(|other| other != value)
        })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StageObservationConfig {
    pub enabled: bool,
    pub poll_interval_ms: u64,
    pub timeout_ms: u64,
    pub max_age_ms: u64,
    pub max_skew_ms: u64,
    pub max_response_bytes: usize,
    pub max_concurrent_scrapes: usize,
    pub mappings: Vec<PressureMapping>,
}

impl Default for StageObservationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            poll_interval_ms: 1_000,
            timeout_ms: 500,
            max_age_ms: 2_000,
            max_skew_ms: 1_500,
            max_response_bytes: 1_048_576,
            max_concurrent_scrapes: 4,
            mappings: Vec::new(),
        }
    }
}

impl StageObservationConfig {
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("poll_interval_ms", self.poll_interval_ms),
            ("timeout_ms", self.timeout_ms),
            ("max_age_ms", self.max_age_ms),
            ("max_skew_ms", self.max_skew_ms),
        ] {
            if !(1..=3_600_000).contains(&value) {
                return Err(format!("{name} must be in 1..=3600000"));
            }
        }
        if self.timeout_ms > super::MAX_SCRAPE_DURATION.as_millis() as u64
            || self.max_age_ms > super::MAX_OBSERVATION_AGE.as_millis() as u64
        {
            return Err(
                "timeout_ms must not exceed 500 and max_age_ms must not exceed 2000".into(),
            );
        }
        if self.timeout_ms > self.max_age_ms || self.max_skew_ms > self.max_age_ms {
            return Err("timeout_ms and max_skew_ms must not exceed max_age_ms".into());
        }
        if !(1..=16_777_216).contains(&self.max_response_bytes)
            || !(1..=64).contains(&self.max_concurrent_scrapes)
            || self.mappings.len() > 64
        {
            return Err(
                "require response limit 1..=16MiB, concurrency 1..=64, mappings <= 64".into(),
            );
        }
        let mut endpoints = HashSet::new();
        let mut series = HashSet::new();
        for mapping in &self.mappings {
            mapping.validate()?;
            if !endpoints.insert((&mapping.inference_endpoint, mapping.dp_rank))
                || !series.insert((
                    &mapping.metrics_url,
                    &mapping.usage.name,
                    &mapping.usage.labels,
                ))
            {
                return Err(
                    "duplicate endpoint/rank or reused usage selector within a stage".into(),
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricSelector {
    pub name: String,
    pub labels: BTreeMap<String, String>,
}

/// The identity selector must name a deployment-verified exporter series.
/// Configuration declarations alone never validate a live vLLM engine.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PressureMapping {
    pub inference_endpoint: String,
    pub dp_rank: Option<usize>,
    pub metrics_url: String,
    #[serde(default)]
    pub auth_token_env: Option<String>,
    pub usage: MetricSelector,
    pub identity: MetricSelector,
    pub expected_engine_id: String,
    pub expected_instance_epoch: String,
    pub expected_layout_fingerprint: String,
}

impl PressureMapping {
    pub fn validate(&self) -> Result<(), String> {
        for (value, metrics) in [(&self.inference_endpoint, false), (&self.metrics_url, true)] {
            let url = url::Url::parse(value).map_err(|_| "invalid explicit endpoint URL")?;
            let scheme_ok = matches!(url.scheme(), "http" | "https")
                || (!metrics && matches!(url.scheme(), "grpc" | "grpcs"));
            if !scheme_ok
                || url.host_str().is_none()
                || !url.username().is_empty()
                || url.password().is_some()
                || url.query().is_some()
                || url.fragment().is_some()
            {
                return Err("explicit endpoint must have an allowed scheme/host and no credentials/query/fragment".into());
            }
        }
        for selector in [&self.usage, &self.identity] {
            if !metric_name(&selector.name)
                || selector.labels.len() > 32
                || selector
                    .labels
                    .iter()
                    .any(|(key, value)| !label_name(key) || value.len() > 1024)
            {
                return Err("invalid metric selector name, labels, or size".into());
            }
        }
        for label in ["engine", "model_name"] {
            if self
                .usage
                .labels
                .get(label)
                .is_none_or(|value| value.is_empty())
            {
                return Err(format!("usage selector requires explicit nonempty {label}"));
            }
        }
        if let Some(rank) = self.dp_rank {
            if self.usage.labels.get("dp_rank") != Some(&rank.to_string()) {
                return Err("ranked mapping requires an explicit usage dp_rank label matching the target rank".into());
            }
        }
        if [
            &self.expected_engine_id,
            &self.expected_instance_epoch,
            &self.expected_layout_fingerprint,
        ]
        .iter()
        .any(|value| value.trim().is_empty() || value.len() > 1024)
        {
            return Err(
                "expected engine, epoch and layout must be explicit nonempty values".into(),
            );
        }
        if self
            .auth_token_env
            .as_ref()
            .is_some_and(|name| !label_name(name))
        {
            return Err("auth_token_env must be an environment variable name".into());
        }
        Ok(())
    }
}

pub(super) fn label_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .enumerate()
            .all(|(i, c)| c == b'_' || c.is_ascii_alphabetic() || (i > 0 && c.is_ascii_digit()))
}

pub(super) fn metric_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 256
        && name.bytes().enumerate().all(|(i, c)| {
            c == b':' || c == b'_' || c.is_ascii_alphabetic() || (i > 0 && c.is_ascii_digit())
        })
}
