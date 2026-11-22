#![cfg(all(feature = "cpu", feature = "ready"))]

use super::*;
use crate::core::{BasicWorker, Worker, WorkerType};
use std::{sync::Arc, time::Instant};

fn prefill() -> Arc<dyn Worker> {
    Arc::new(BasicWorker::new(
        "http://independent-engine:8000".into(),
        WorkerType::Prefill {
            bootstrap_port: None,
        },
    ))
}

#[test]
fn public_observer_is_policy_independent_and_shares_snapshot_freshness() {
    // No cache-aware policy, mode, threshold, request count or KV event source.
    let mut config = EnginePressureConfig::default();
    config.prefill.enabled = true;
    let runtime = EnginePressureRuntime::new(config, reqwest::Client::new()).unwrap();
    let worker = prefill();
    let key = EngineKey::for_worker(PressureStage::Prefill, worker.as_ref());
    let now = Instant::now();
    assert!(matches!(
        runtime.observe(&key, &worker, now),
        PressureObservation::Unknown(InvalidReason::UnboundWorker)
    ));
    runtime.inject_for_test(PressureStage::Prefill, &worker, 0.93, now);
    for age in [std::time::Duration::ZERO, MAX_OBSERVATION_AGE] {
        let PressureObservation::Known(sample) = runtime.observe(&key, &worker, now + age) else {
            panic!("valid sample missing")
        };
        let snapshot = runtime
            .snapshot(
                PressureStage::Prefill,
                std::slice::from_ref(&worker),
                &[0],
                now + age,
            )
            .unwrap();
        assert_eq!(sample.usage, 0.93);
        assert_eq!(snapshot.usage(0), Some(sample.usage));
        assert_eq!(
            snapshot.samples[0].binding_generation,
            sample.binding_generation
        );
    }
    let stale = now + MAX_OBSERVATION_AGE + std::time::Duration::from_nanos(1);
    assert!(matches!(
        runtime.observe(&key, &worker, stale),
        PressureObservation::Unknown(InvalidReason::Stale)
    ));
    assert_eq!(
        runtime
            .snapshot(
                PressureStage::Prefill,
                std::slice::from_ref(&worker),
                &[0],
                stale
            )
            .unwrap_err(),
        InvalidReason::Stale
    );
    let replacement = prefill();
    assert!(matches!(
        runtime.observe(&key, &replacement, now),
        PressureObservation::Unknown(InvalidReason::UnboundWorker)
    ));
    runtime.inject_for_test(PressureStage::Prefill, &worker, f64::NAN, now);
    assert!(matches!(
        runtime.observe(&key, &worker, now),
        PressureObservation::Unknown(InvalidReason::InvalidUsage)
    ));
}

#[test]
fn shared_limits_cannot_be_relaxed_by_a_consumer() {
    let mut config = EnginePressureConfig::default();
    assert_eq!(config.prefill.timeout_ms, 500);
    assert_eq!(config.prefill.max_age_ms, 2000);
    config.prefill.timeout_ms = 501;
    assert!(config.validate().unwrap_err().contains("500"));
    config.prefill.timeout_ms = 500;
    config.prefill.max_age_ms = 2001;
    assert!(config.validate().unwrap_err().contains("2000"));
}

#[test]
fn late_capture_becomes_unknown_and_engine_keys_do_not_cross_stages_or_ranks() {
    let mut store = store::Store::default();
    let worker = prefill();
    let now = Instant::now();
    let stage = PressureStage::Prefill;
    let key = EngineKey::for_worker(stage, worker.as_ref());
    store.reconcile(stage, std::slice::from_ref(&worker));
    let ticket = store.begin(&key, now).unwrap();
    assert!(store.publish(
        ticket,
        now + MAX_SCRAPE_DURATION + std::time::Duration::from_nanos(1),
        "layout",
        Ok(0.1)
    ));
    assert!(matches!(
        store.observe(
            &key,
            &worker,
            now + MAX_SCRAPE_DURATION,
            MAX_OBSERVATION_AGE
        ),
        PressureObservation::Unknown(InvalidReason::Timeout)
    ));
    let mut other = key.clone();
    other.stage = PressureStage::Decode;
    assert!(matches!(
        store.observe(&other, &worker, now, MAX_OBSERVATION_AGE),
        PressureObservation::Unknown(InvalidReason::UnboundWorker)
    ));
    other.stage = stage;
    other.rank = Some(1);
    assert!(matches!(
        store.observe(&other, &worker, now, MAX_OBSERVATION_AGE),
        PressureObservation::Unknown(InvalidReason::UnboundWorker)
    ));
}
