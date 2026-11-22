use super::{InvalidReason, MetricSelector, PressureMapping, PressureStage};
use std::collections::BTreeMap;

struct Point {
    labels: BTreeMap<String, String>,
    value: f64,
}

/// Join one usage series to one live identity series from the SAME response.
/// The identity exporter must derive these labels from the actual engine, not
/// simply repeat router configuration. Native vLLM does not supply this proof.
pub(crate) fn parse_usage(
    body: &str,
    stage: PressureStage,
    mapping: &PressureMapping,
) -> Result<f64, InvalidReason> {
    let usage = select(body, &mapping.usage)?;
    let identity = select(body, &mapping.identity)?;
    if !usage.value.is_finite() || !(0.0..=1.0).contains(&usage.value) {
        return Err(InvalidReason::InvalidUsage);
    }
    if identity.value != 1.0 {
        return Err(InvalidReason::IdentityMismatch);
    }
    let rank = mapping
        .dp_rank
        .map_or_else(|| "aggregate".into(), |rank| rank.to_string());
    for (label, expected) in [
        ("stage", stage.as_str()),
        ("endpoint", mapping.inference_endpoint.as_str()),
        ("dp_rank", rank.as_str()),
        ("engine_id", mapping.expected_engine_id.as_str()),
        ("instance_epoch", mapping.expected_instance_epoch.as_str()),
        (
            "layout_fingerprint",
            mapping.expected_layout_fingerprint.as_str(),
        ),
    ] {
        if identity.labels.get(label).map(String::as_str) != Some(expected) {
            return Err(InvalidReason::IdentityMismatch);
        }
        // Decorated usage may carry identity fields beyond the selector. They
        // must not contradict the bridge (especially rank versus aggregate).
        if usage
            .labels
            .get(label)
            .is_some_and(|actual| actual != expected)
        {
            return Err(InvalidReason::IdentityMismatch);
        }
    }
    // Explicit series join: at least engine and model_name are required by config.
    if mapping
        .usage
        .labels
        .iter()
        .any(|(key, value)| identity.labels.get(key) != Some(value))
    {
        return Err(InvalidReason::IdentityMismatch);
    }
    Ok(usage.value)
}

fn select(body: &str, selector: &MetricSelector) -> Result<Point, InvalidReason> {
    let mut selected = None;
    for line in body
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
    {
        let end = line
            .find(|c: char| c == '{' || c.is_whitespace())
            .unwrap_or(line.len());
        if line[..end] != selector.name {
            continue;
        }
        let point = parse_point(&line[end..])?;
        if selector
            .labels
            .iter()
            .all(|(key, value)| point.labels.get(key) == Some(value))
            && selected.replace(point).is_some()
        {
            return Err(InvalidReason::AmbiguousSeries);
        }
    }
    selected.ok_or(InvalidReason::MissingSeries)
}

fn parse_point(rest: &str) -> Result<Point, InvalidReason> {
    let mut rest = rest.trim_start();
    let mut labels = BTreeMap::new();
    if rest.starts_with('{') {
        let mut quoted = false;
        let mut escaped = false;
        let mut close = None;
        for (offset, c) in rest.char_indices().skip(1) {
            if escaped {
                escaped = false;
                continue;
            }
            if quoted && c == '\\' {
                escaped = true;
                continue;
            }
            if c == '"' {
                quoted = !quoted;
            }
            if !quoted && c == '}' {
                close = Some(offset);
                break;
            }
        }
        let close = close.ok_or(InvalidReason::MalformedResponse)?;
        let mut chars = rest[1..close].chars().peekable();
        loop {
            while chars.peek().is_some_and(|c| c.is_whitespace()) {
                chars.next();
            }
            if chars.peek().is_none() {
                break;
            }
            let mut key = String::new();
            while chars
                .peek()
                .is_some_and(|c| c.is_ascii_alphanumeric() || *c == '_')
            {
                key.push(chars.next().unwrap());
            }
            while chars.peek().is_some_and(|c| c.is_whitespace()) {
                chars.next();
            }
            if !super::config::label_name(&key) || chars.next() != Some('=') {
                return Err(InvalidReason::MalformedResponse);
            }
            while chars.peek().is_some_and(|c| c.is_whitespace()) {
                chars.next();
            }
            if chars.next() != Some('"') {
                return Err(InvalidReason::MalformedResponse);
            }
            let mut value = String::new();
            loop {
                match chars.next() {
                    Some('"') => break,
                    Some('\\') => value.push(match chars.next() {
                        Some('n') => '\n',
                        Some('\\') => '\\',
                        Some('"') => '"',
                        _ => return Err(InvalidReason::MalformedResponse),
                    }),
                    Some(c) => value.push(c),
                    None => return Err(InvalidReason::MalformedResponse),
                }
            }
            if labels.insert(key, value).is_some() {
                return Err(InvalidReason::MalformedResponse);
            }
            while chars.peek().is_some_and(|c| c.is_whitespace()) {
                chars.next();
            }
            match chars.next() {
                Some(',') => {}
                None => break,
                _ => return Err(InvalidReason::MalformedResponse),
            }
        }
        rest = &rest[close + 1..];
    }
    let mut fields = rest.split_whitespace();
    let value = fields
        .next()
        .ok_or(InvalidReason::MalformedResponse)?
        .parse::<f64>()
        .map_err(|_| InvalidReason::MalformedResponse)?;
    if let Some(timestamp) = fields.next() {
        if timestamp.parse::<i64>().is_err() || fields.next().is_some() {
            return Err(InvalidReason::MalformedResponse);
        }
    }
    // Exposition timestamps are not proof of scheduler refresh time. Keep
    // source_sample_time=None until an independently verified clock exists.
    Ok(Point { labels, value })
}
