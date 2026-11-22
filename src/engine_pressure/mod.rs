//! Policy-independent, identity-validated engine KV occupancy observations.
//!
//! Collection and invalidation live here. Consumers decide how to use Known
//! samples; Unknown is never a fabricated zero or a routing recommendation.
//! Native HTTP PD is the first adapter; no KV event source is required.
pub(crate) mod config;
pub(crate) mod parser;
mod runtime;
pub(crate) mod store;

pub use config::{
    EnginePressureConfig, MetricSelector, PressureMapping, PressureStage, StageObservationConfig,
};
pub use runtime::EnginePressureRuntime;
pub use store::{
    EngineKey, EngineSample, InvalidReason, PressureObservation, PressureSample, StageSnapshot,
};

/// Common observation limits, coordinated with the proposed #585 boundary.
pub const MAX_SCRAPE_DURATION: std::time::Duration = std::time::Duration::from_millis(500);
pub const MAX_OBSERVATION_AGE: std::time::Duration = std::time::Duration::from_secs(2);

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests;
