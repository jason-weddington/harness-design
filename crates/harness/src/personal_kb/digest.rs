//! The pure, I/O-free turn-digest buffer behind
//! [`super::PersonalKbMemory`]'s `/api/kb/turn` posts.
//!
//! It maps observations to contract items, owns the harness-correction
//! lifecycle (open, fold, resolve, carry over), closes digests on the item /
//! byte budget and the digest period, and fits a lone oversized group with
//! [`fit_items`]. A closed digest comes back as [`Closed`] — the exact bytes
//! to POST, or [`Closed::Unfittable`] — and the parent module does the
//! posting. Holds no API key.

use std::collections::{BTreeSet, HashSet};
use std::time::{Duration, Instant, SystemTime};

use serde_json::{Value, json};

use super::{
    MAX_RESOLVED_BY, MAX_TURN_BODY_BYTES, MAX_TURN_EXCERPT_CHARS, MAX_TURN_ITEMS,
    MAX_TURN_PROMPT_CHARS, MAX_TURN_TARGET_CHARS, MAX_TURN_TEXT_CHARS, native_target,
    truncate_chars,
};
use crate::memory::{CorrectionTrigger, Observation, ToolCallView, ToolResultView};
use crate::run_record::Disposition;

/// The static part of every digest body.
#[derive(Debug, Clone, Default)]
pub(super) struct Envelope {
    pub(super) session_id: String,
    pub(super) host: Option<String>,
    pub(super) hook_version: Option<String>,
    pub(super) project: Option<String>,
}

/// A digest the buffer closed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Closed {
    /// The exact serialized body to POST.
    Body(Vec<u8>),
    /// The digest could not be fitted to the budget; it is not posted.
    Unfittable,
}

/// What [`fit_items`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Fit {
    /// The items already fit.
    Untouched,
    /// Items were dropped until they fit.
    Dropped,
    /// Even the undroppable items do not fit.
    Unfittable,
}

/// An open harness correction.
#[derive(Debug, Clone)]
struct OpenCorrection {
    trigger: CorrectionTrigger,
    detail: String,
    /// The `tool_use_id`s posted since it opened, in dispatch order.
    pending: Vec<String>,
}

impl OpenCorrection {
    /// The last [`MAX_RESOLVED_BY`] pending ids.
    fn resolved_by(&self) -> &[String] {
        &self.pending[self.pending.len().saturating_sub(MAX_RESOLVED_BY)..]
    }

    fn item(&self, resolved: bool) -> Value {
        let resolved_by: &[String] = if resolved { self.resolved_by() } else { &[] };
        json!({
            "kind": "harness_correction",
            "trigger": self.trigger.as_str(),
            "detail": self.detail,
            "resolved_by": resolved_by,
            "resolved": resolved,
        })
    }
}

/// The serialized size of `v`.
fn byte_len(v: &Value) -> usize {
    serde_json::to_vec(v).map_or(0, |b| b.len())
}

/// The serialized size of a body whose `items: []` envelope is
/// `envelope_bytes` long and whose items are `item_bytes` long in total.
fn body_len(envelope_bytes: usize, item_bytes: usize, count: usize) -> usize {
    envelope_bytes + item_bytes + count.saturating_sub(1)
}

fn kind(item: &Value) -> &str {
    item["kind"].as_str().unwrap_or_default()
}

fn tool_use_id(item: &Value) -> &str {
    item["tool_use_id"].as_str().unwrap_or_default()
}

/// Drop items until `items` fits the contract budget (sender rule c).
///
/// A candidate's size is `envelope_bytes + sum(item bytes) + (n - 1)`; it
/// fits when that is at most [`MAX_TURN_BODY_BYTES`] and `n` is at most
/// [`MAX_TURN_ITEMS`]. Drops one item at a time, earliest first within a
/// tier: (1) `reasoning` and `assistant_text`; (2) `tool_call` /
/// `tool_result` items whose id is neither in `protected` nor named by a
/// `harness_correction` in `items`; (3) the remaining `tool_result` items.
/// Never drops a `harness_correction` or a protected / named `tool_call`.
pub(super) fn fit_items(
    items: &mut Vec<Value>,
    envelope_bytes: usize,
    protected: &HashSet<String>,
) -> Fit {
    let mut sizes: Vec<usize> = items.iter().map(byte_len).collect();
    let fits = |sizes: &[usize]| {
        sizes.len() <= MAX_TURN_ITEMS
            && body_len(envelope_bytes, sizes.iter().sum(), sizes.len()) <= MAX_TURN_BODY_BYTES
    };
    if fits(&sizes) {
        return Fit::Untouched;
    }
    let named: HashSet<String> = items
        .iter()
        .filter(|i| kind(i) == "harness_correction")
        .filter_map(|i| i["resolved_by"].as_array())
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect();
    let text_items = |i: &Value| matches!(kind(i), "reasoning" | "assistant_text");
    let unnamed_tools = |i: &Value| {
        matches!(kind(i), "tool_call" | "tool_result")
            && !protected.contains(tool_use_id(i))
            && !named.contains(tool_use_id(i))
    };
    let results = |i: &Value| kind(i) == "tool_result";
    let tiers: [&dyn Fn(&Value) -> bool; 3] = [&text_items, &unnamed_tools, &results];
    for tier in tiers {
        while let Some(at) = items.iter().position(tier) {
            items.remove(at);
            sizes.remove(at);
            if fits(&sizes) {
                return Fit::Dropped;
            }
        }
    }
    Fit::Unfittable
}

/// The turn-digest buffer of one run.
#[derive(Debug)]
pub(super) struct DigestBuffer {
    envelope: Envelope,
    /// The native names that are `tool_map` keys: only these are posted.
    tool_names: BTreeSet<String>,
    period: Duration,
    items: Vec<Value>,
    item_bytes: usize,
    truncated: bool,
    opened_at: Instant,
    next_index: u32,
    user_prompt: Option<String>,
    corrections: Vec<OpenCorrection>,
    finish_message: Option<String>,
    last_text: Option<String>,
    finished: bool,
}

impl DigestBuffer {
    pub(super) fn new(envelope: Envelope, tool_names: BTreeSet<String>, period: Duration) -> Self {
        Self {
            envelope,
            tool_names,
            period,
            items: Vec::new(),
            item_bytes: 0,
            truncated: false,
            opened_at: Instant::now(),
            next_index: 0,
            user_prompt: None,
            corrections: Vec::new(),
            finish_message: None,
            last_text: None,
            finished: false,
        }
    }

    pub(super) fn set_context(&mut self, host: Option<String>, hook_version: Option<String>) {
        self.envelope.host = host;
        self.envelope.hook_version = hook_version;
    }

    #[cfg(test)]
    pub(super) fn set_period(&mut self, period: Duration) {
        self.period = period;
    }

    /// The body of digest `index` with `items`.
    fn body(&self, items: &[Value], final_message: Option<&str>, truncated: bool) -> Value {
        let index = self.next_index;
        json!({
            "event_id": format!("{}:{index}", self.envelope.session_id),
            "session_id": self.envelope.session_id,
            "harness": "talos",
            "mode": "headless",
            "engine": Value::Null,
            "host": self.envelope.host,
            "hook_version": self.envelope.hook_version,
            "project": self.envelope.project,
            "turn_index": index,
            "ts": crate::time::format_rfc3339(SystemTime::now()),
            "user_prompt": if index == 0 { self.user_prompt.as_deref() } else { None },
            "items": items,
            "final_message": final_message,
            "truncated": truncated,
        })
    }

    fn envelope_bytes(&self, final_message: Option<&str>, truncated: bool) -> usize {
        byte_len(&self.body(&[], final_message, truncated))
    }

    /// Close the open digest (consuming its index) and open the next.
    fn close(&mut self, final_message: Option<&str>) -> Closed {
        let items = std::mem::take(&mut self.items);
        let body = self.body(&items, final_message, self.truncated);
        self.next_index += 1;
        self.item_bytes = 0;
        self.truncated = false;
        self.opened_at = Instant::now();
        match serde_json::to_vec(&body) {
            Ok(bytes) => Closed::Body(bytes),
            Err(_) => Closed::Unfittable,
        }
    }

    /// Discard the open digest as unfittable (consuming its index).
    fn discard(&mut self) -> Closed {
        self.items.clear();
        self.next_index += 1;
        self.item_bytes = 0;
        self.truncated = false;
        self.opened_at = Instant::now();
        Closed::Unfittable
    }

    /// Whether `group` would push the open digest over a limit.
    fn overflows(&self, group: &[Value], final_message: Option<&str>) -> bool {
        let count = self.items.len() + group.len();
        let bytes = self.item_bytes + group.iter().map(byte_len).sum::<usize>();
        count > MAX_TURN_ITEMS
            || body_len(
                self.envelope_bytes(final_message, self.truncated),
                bytes,
                count,
            ) > MAX_TURN_BODY_BYTES
    }

    /// The pending ids of every still-open correction.
    fn protected(&self) -> HashSet<String> {
        self.corrections
            .iter()
            .flat_map(|c| c.resolved_by().iter().cloned())
            .collect()
    }

    /// Append `group`, closing the open digest first when it would overflow,
    /// and fitting the group when it does not fit an empty digest. Returns
    /// `true` when the group now sits in the open digest.
    fn append(
        &mut self,
        group: Vec<Value>,
        final_message: Option<&str>,
        out: &mut Vec<Closed>,
    ) -> bool {
        if group.is_empty() {
            return true;
        }
        if !self.items.is_empty() && self.overflows(&group, final_message) {
            out.push(self.close(None));
        }
        let lone = self.items.is_empty();
        self.item_bytes += group.iter().map(byte_len).sum::<usize>();
        self.items.extend(group);
        if lone && self.overflows(&[], final_message) {
            let protected = self.protected();
            let envelope = self.envelope_bytes(final_message, true);
            match fit_items(&mut self.items, envelope, &protected) {
                Fit::Untouched => {}
                Fit::Dropped => {
                    self.truncated = true;
                    self.item_bytes = self.items.iter().map(byte_len).sum();
                }
                Fit::Unfittable => {
                    out.push(self.discard());
                    return false;
                }
            }
        }
        true
    }

    fn call_items(call: &ToolCallView, id: &str, is_error: bool, content: &str) -> [Value; 2] {
        [
            json!({
                "kind": "tool_call",
                "tool_use_id": id,
                "tool": call.name,
                "target": truncate_chars(&native_target(call), MAX_TURN_TARGET_CHARS),
                "target_class": "",
            }),
            json!({
                "kind": "tool_result",
                "tool_use_id": id,
                "is_error": is_error,
                "excerpt": if is_error {
                    truncate_chars(content, MAX_TURN_EXCERPT_CHARS)
                } else {
                    String::new()
                },
            }),
        ]
    }

    /// One call's group: its items when posted, then — when it resolves —
    /// every open correction; otherwise its id joins every open
    /// correction's pending list.
    fn call(
        &mut self,
        call: &ToolCallView,
        iteration: u32,
        call_id: &str,
        is_error: bool,
        content: &str,
        resolving: bool,
    ) -> Vec<Closed> {
        let mut out = Vec::new();
        if self.finished {
            return out;
        }
        let posted = self.tool_names.contains(call.name);
        let id = format!("{iteration}:{call_id}");
        let mut group = Vec::new();
        if posted {
            group.extend(Self::call_items(call, &id, is_error, content));
        }
        if resolving {
            group.extend(self.corrections.drain(..).map(|c| c.item(true)));
        } else if posted {
            for c in &mut self.corrections {
                c.pending.push(id.clone());
            }
        }
        self.append(group, None, &mut out);
        out
    }

    /// One invoked non-finish call.
    pub(super) fn after_tool(
        &mut self,
        call: &ToolCallView,
        result: &ToolResultView,
    ) -> Vec<Closed> {
        let resolving = call.name == "run_checks" && !result.is_error;
        self.call(
            call,
            result.iteration,
            result.call_id,
            result.is_error,
            result.content,
            resolving,
        )
    }

    /// One engine observation; `now` drives the digest period.
    pub(super) fn observe(&mut self, observation: &Observation<'_>, now: Instant) -> Vec<Closed> {
        let mut out = Vec::new();
        if self.finished {
            return out;
        }
        match *observation {
            Observation::RunStart { task } => {
                self.user_prompt = Some(truncate_chars(task, MAX_TURN_PROMPT_CHARS));
            }
            Observation::Turn {
                reasoning, text, ..
            } => {
                if !self.items.is_empty() && now.duration_since(self.opened_at) >= self.period {
                    out.push(self.close(None));
                }
                let mut group = Vec::new();
                if !reasoning.trim().is_empty() {
                    group.push(json!({
                        "kind": "reasoning",
                        "text": truncate_chars(reasoning, MAX_TURN_TEXT_CHARS),
                        "truncated": reasoning.chars().count() > MAX_TURN_TEXT_CHARS,
                    }));
                }
                if !text.trim().is_empty() {
                    self.last_text = Some(text.to_string());
                    group.push(json!({
                        "kind": "assistant_text",
                        "text": truncate_chars(text, MAX_TURN_TEXT_CHARS),
                    }));
                }
                self.append(group, None, &mut out);
            }
            Observation::Finish {
                call_id,
                iteration,
                args,
                is_error,
                content,
                accepted,
            } => {
                if let Some(d) = accepted {
                    self.finish_message = Some(match d {
                        Disposition::Done { summary, .. } | Disposition::Failed { summary, .. } => {
                            summary.clone()
                        }
                        Disposition::AlreadySatisfied { reason, .. } => reason.clone(),
                        Disposition::Blocked { decision_needed } => decision_needed.clone(),
                        Disposition::Answer { result, .. } => {
                            serde_json::to_string(result).unwrap_or_default()
                        }
                    });
                }
                let resolving = matches!(
                    accepted,
                    Some(
                        Disposition::Done { .. }
                            | Disposition::AlreadySatisfied { .. }
                            | Disposition::Answer { .. }
                    )
                );
                let view = ToolCallView {
                    name: "finish",
                    args,
                };
                out = self.call(&view, iteration, call_id, is_error, content, resolving);
            }
            Observation::Correction {
                trigger, detail, ..
            } => {
                if !self.corrections.iter().any(|c| c.trigger == trigger) {
                    self.corrections.push(OpenCorrection {
                        trigger,
                        detail: truncate_chars(detail, MAX_TURN_EXCERPT_CHARS),
                        pending: Vec::new(),
                    });
                }
            }
        }
        out
    }

    /// Close the run: the final digest(s). Empty when already finished.
    pub(super) fn finish(&mut self) -> Vec<Closed> {
        let mut out = Vec::new();
        if self.finished {
            return out;
        }
        self.finished = true;
        let unresolved: Vec<Value> = self.corrections.drain(..).map(|c| c.item(false)).collect();
        let final_message = self
            .finish_message
            .clone()
            .or_else(|| self.last_text.clone())
            .map(|m| truncate_chars(&m, MAX_TURN_PROMPT_CHARS));
        let final_message = final_message.as_deref();
        if !self.items.is_empty() && self.overflows(&unresolved, final_message) {
            out.push(self.close(None));
        }
        if self.append(unresolved, final_message, &mut out) {
            out.push(self.close(final_message));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reasoning(n: usize) -> Value {
        json!({"kind": "reasoning", "text": "r".repeat(n), "truncated": false})
    }

    fn call(id: &str) -> Value {
        json!({"kind": "tool_call", "tool_use_id": id, "tool": "bash", "target": "", "target_class": ""})
    }

    fn result(id: &str) -> Value {
        json!({"kind": "tool_result", "tool_use_id": id, "is_error": false, "excerpt": ""})
    }

    fn correction(names: &[&str]) -> Value {
        json!({"kind": "harness_correction", "trigger": "gate_red", "detail": "d", "resolved_by": names, "resolved": true})
    }

    fn total(items: &[Value]) -> usize {
        body_len(0, items.iter().map(byte_len).sum(), items.len())
    }

    #[test]
    fn fit_items_table() {
        let none = HashSet::new();

        // Under budget: untouched.
        let mut items = vec![call("a"), result("a")];
        let before = items.clone();
        assert_eq!(fit_items(&mut items, 100, &none), Fit::Untouched);
        assert_eq!(items, before);

        // A reasoning item pushing over the byte budget is dropped alone.
        let mut items = vec![reasoning(2000), call("a"), result("a")];
        let envelope = MAX_TURN_BODY_BYTES - total(&items) + 1;
        assert_eq!(fit_items(&mut items, envelope, &none), Fit::Dropped);
        assert_eq!(items, vec![call("a"), result("a")]);

        // Only the named pair and its correction fit.
        let mut items = vec![
            reasoning(10),
            call("u"),
            result("u"),
            call("n"),
            result("n"),
            correction(&["n"]),
        ];
        let kept = vec![call("n"), result("n"), correction(&["n"])];
        let envelope = MAX_TURN_BODY_BYTES - total(&kept);
        assert_eq!(fit_items(&mut items, envelope, &none), Fit::Dropped);
        assert_eq!(items, kept);

        // 201 unnamed tool items: the earliest is dropped.
        let mut items: Vec<Value> = (0..201).map(|i| call(&format!("c{i}"))).collect();
        assert_eq!(fit_items(&mut items, 100, &none), Fit::Dropped);
        assert_eq!(items.len(), 200);
        assert_eq!(items[0], call("c1"));

        // A correction plus its named call larger than the budget.
        let mut items = vec![call("n"), correction(&["n"])];
        let envelope = MAX_TURN_BODY_BYTES;
        assert_eq!(fit_items(&mut items, envelope, &none), Fit::Unfittable);

        // A protected id survives tier 2; the tier-3 result goes first.
        let protected = HashSet::from(["p".to_string()]);
        let mut items = vec![call("p"), result("p"), call("u"), result("u")];
        let envelope = MAX_TURN_BODY_BYTES - total(&[call("p")]);
        assert_eq!(fit_items(&mut items, envelope, &protected), Fit::Dropped);
        assert_eq!(items, vec![call("p")]);
    }

    fn buffer(names: &[&str]) -> DigestBuffer {
        DigestBuffer::new(
            Envelope {
                session_id: "s".into(),
                ..Envelope::default()
            },
            names.iter().map(|n| (*n).to_string()).collect(),
            Duration::from_mins(10),
        )
    }

    fn bodies(closed: &[Closed]) -> Vec<Value> {
        closed
            .iter()
            .map(|c| match c {
                Closed::Body(b) => serde_json::from_slice(b).unwrap(),
                Closed::Unfittable => Value::Null,
            })
            .collect()
    }

    #[test]
    fn a_lone_oversized_group_is_fitted_and_marked_truncated() {
        let mut b = buffer(&["bash"]);
        // Ten 8 KB reasoning items (2000 four-byte chars each) in one group
        // cannot fit an empty digest; no observation builds such a group, so
        // drive `append` directly.
        let mut out = Vec::new();
        let group: Vec<Value> = (0..10)
            .map(|_| json!({"kind": "reasoning", "text": "🦀".repeat(2000), "truncated": false}))
            .collect();
        assert!(b.append(group, None, &mut out));
        assert!(out.is_empty());
        assert!(b.truncated);
        let closed = b.finish();
        let body = &bodies(&closed)[0];
        assert_eq!(body["truncated"], true);
        let Closed::Body(bytes) = &closed[0] else {
            panic!("body")
        };
        assert!(bytes.len() <= MAX_TURN_BODY_BYTES);
    }

    #[test]
    fn an_unfittable_group_is_discarded_and_consumes_its_index() {
        let mut b = buffer(&["bash"]);
        let mut out = Vec::new();
        let huge: Vec<String> = (0..200)
            .map(|i| format!("{i}:{}", "x".repeat(400)))
            .collect();
        let group = vec![json!({
            "kind": "harness_correction", "trigger": "gate_red", "detail": "d",
            "resolved_by": huge, "resolved": true
        })];
        assert!(!b.append(group, None, &mut out));
        assert_eq!(out, vec![Closed::Unfittable]);
        let closed = b.finish();
        assert_eq!(bodies(&closed)[0]["turn_index"], 1);
        assert!(b.finish().is_empty());
    }

    #[test]
    fn final_corrections_that_overflow_the_open_digest_go_in_a_fresh_one() {
        let mut b = buffer(&["read_file"]);
        let args = json!({"path": "p"});
        let view = ToolCallView {
            name: "read_file",
            args: &args,
        };
        // Fill the open digest to exactly 200 items.
        for i in 0..100 {
            let id = format!("c{i}");
            let closed = b.after_tool(
                &view,
                &ToolResultView {
                    call_id: &id,
                    iteration: 1,
                    is_error: false,
                    content: "",
                    duration_ms: None,
                },
            );
            assert!(closed.is_empty());
        }
        b.observe(
            &Observation::Correction {
                trigger: CorrectionTrigger::Nudge,
                iteration: 1,
                detail: "n",
            },
            Instant::now(),
        );
        let closed = bodies(&b.finish());
        assert_eq!(closed.len(), 2);
        assert_eq!(closed[0]["items"].as_array().unwrap().len(), 200);
        assert_eq!(closed[0]["final_message"], Value::Null);
        assert_eq!(
            closed[1]["items"],
            json!([{"kind": "harness_correction", "trigger": "nudge", "detail": "n", "resolved_by": [], "resolved": false}])
        );
    }

    #[test]
    fn finish_messages_for_every_disposition() {
        use crate::exec::ChangeEvidence;
        use crate::run_record::{FailureMode, Verification};
        let args = json!({});
        let cases = [
            (
                Disposition::AlreadySatisfied {
                    reason: "r".into(),
                    verification: Verification::NoChecksConfigured,
                    change: ChangeEvidence::TreeUnchanged,
                },
                "r",
            ),
            (
                Disposition::Failed {
                    mode: FailureMode::Loop,
                    summary: "f".into(),
                },
                "f",
            ),
            (
                Disposition::Answer {
                    result: json!({"a": 1}),
                    verification: Verification::NoChecksConfigured,
                    change: ChangeEvidence::TreeUnchanged,
                },
                "{\"a\":1}",
            ),
        ];
        for (d, expected) in cases {
            let mut b = buffer(&[]);
            b.observe(
                &Observation::Finish {
                    call_id: "cf",
                    iteration: 1,
                    args: &args,
                    is_error: false,
                    content: "",
                    accepted: Some(&d),
                },
                Instant::now(),
            );
            assert_eq!(bodies(&b.finish())[0]["final_message"], expected);
        }
        // No finish: the last non-blank turn text.
        let mut b = buffer(&[]);
        for text in ["first", "  "] {
            b.observe(
                &Observation::Turn {
                    iteration: 1,
                    reasoning: "",
                    text,
                },
                Instant::now(),
            );
        }
        assert_eq!(bodies(&b.finish())[0]["final_message"], "first");
        // Observations after the final digest are ignored.
        let none = b.observe(&Observation::RunStart { task: "t" }, Instant::now());
        assert!(none.is_empty());
    }
}
