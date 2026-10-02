// Copyright 2026 Hermes contributors.
// SPDX-License-Identifier: Apache-2.0

//! Hermes: timed phases for the host agent (gpu-plan.md P12).
//!
//! Each [`Span`] prints one line to stderr when it ends:
//! `HERMES-SPAN {"name":..,"start_unix_ns":..,"duration_ns":..,"traceparent":..,"fields":{..}}`.
//! The host agent reads Firecracker's stderr and turns every such line into a
//! log event and an OpenTelemetry span, parented by the `traceparent` of the
//! API request that ran it. stdout carries the guest's serial console, which a
//! guest could use to forge records, so they never go there. The jailed VMM
//! has no network, so it never exports anything itself.

use std::io::Write;
use std::sync::{Mutex, PoisonError};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// The prefix that marks a span record on stderr.
pub const SPAN_PREFIX: &str = "HERMES-SPAN ";

/// The W3C trace context of the API request being served. The API server
/// serves one request at a time and waits for the VMM's reply, so a single
/// slot is enough.
static CONTEXT: Mutex<Option<String>> = Mutex::new(None);

/// Sets (or clears) the trace context of the request being served. A value
/// that is not a plausible `traceparent` is dropped.
pub fn set_context(traceparent: Option<&str>) {
    let traceparent = traceparent
        .filter(|value| {
            value.len() <= 64
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() || byte == b'-')
        })
        .map(str::to_owned);
    *CONTEXT.lock().unwrap_or_else(PoisonError::into_inner) = traceparent;
}

fn context() -> Option<String> {
    CONTEXT
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone()
}

/// A timed phase, printed when dropped.
#[derive(Debug)]
pub struct Span {
    name: &'static str,
    start: Instant,
    start_unix_ns: u64,
    fields: Vec<(&'static str, u64)>,
}

impl Span {
    /// Starts timing `name`.
    pub fn start(name: &'static str) -> Self {
        Self {
            name,
            start: Instant::now(),
            start_unix_ns: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |since| {
                    u64::try_from(since.as_nanos()).unwrap_or(u64::MAX)
                }),
            fields: Vec::new(),
        }
    }

    /// Attaches a numeric field, such as a byte count.
    pub fn record(&mut self, key: &'static str, value: u64) {
        self.fields.push((key, value));
    }

    fn line(&self) -> String {
        let fields: serde_json::Map<String, serde_json::Value> = self
            .fields
            .iter()
            .map(|(key, value)| ((*key).to_owned(), serde_json::Value::from(*value)))
            .collect();
        let record = serde_json::json!({
            "name": self.name,
            "start_unix_ns": self.start_unix_ns,
            "duration_ns": u64::try_from(self.start.elapsed().as_nanos()).unwrap_or(u64::MAX),
            "traceparent": context(),
            "fields": fields,
        });
        format!("{SPAN_PREFIX}{record}")
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        let line = self.line();
        let mut stderr = std::io::stderr().lock();
        // Tracing must never fail the VMM.
        let _ = writeln!(stderr, "{line}");
        let _ = stderr.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_span_record_carries_its_name_fields_and_context() {
        set_context(Some(
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
        ));
        let mut span = Span::start("snapshot.memory_dump");
        span.record("bytes", 4096);
        let line = span.line();
        let record: serde_json::Value =
            serde_json::from_str(line.strip_prefix(SPAN_PREFIX).unwrap()).unwrap();
        assert_eq!(record["name"], "snapshot.memory_dump");
        assert_eq!(record["fields"]["bytes"], 4096);
        assert_eq!(
            record["traceparent"],
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        );
        assert!(record["start_unix_ns"].as_u64().unwrap() > 0);
        set_context(Some("not a trace context"));
        assert!(context().is_none());
        set_context(None);
    }
}
