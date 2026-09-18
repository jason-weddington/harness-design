//! Opt-in full run transcript: a JSONL sink recording every model
//! request/turn, tool call, and tool result for one [`crate::engine::run`] /
//! [`crate::engine::run_persisted`] / [`crate::engine::resume`] invocation.
//!
//! **Default OFF everywhere.** This module only defines the interface and
//! writes the file; it is wired into the engine loop opt-in via
//! [`crate::engine::RunConfig::with_transcript`], and from there into `talos
//! run --transcript <path>`, `coding_eval` (`CODING_EVAL_TRANSCRIPTS`), and
//! `mined_eval` (`MINED_EVAL_TRANSCRIPTS`). Writing is **best-effort**: a
//! filesystem failure at any point silently disables the writer for the rest
//! of the run — it can never change a run's outcome, its [`crate::engine::RunStats`],
//! or the messages sent to the model. Timestamps are observational only: they
//! read `std::time::SystemTime`/`std::time::Instant` directly, never the
//! injected [`crate::time::Clock`] (routing them through the clock would
//! change what the loop *does*, not just what it reports — see the
//! clock-discrimination tests in `engine.rs`).
//!
//! ## Event kinds
//!
//! One JSON object per line, tagged by an `"event"` key (see [`EVENT_KINDS`]).
//! Every line also carries `"ts"` (an RFC 3339 UTC second-resolution
//! timestamp, `YYYY-MM-DDTHH:MM:SSZ`, via [`crate::time::format_rfc3339`]) and
//! `"elapsed_ms"` (milliseconds since [`TranscriptWriter::open`], as a u64).
//!
//! A transcript file may hold several **run blocks** back to back (append
//! mode — the same path can be reused across a crash-resume). Each block
//! starts with a `run_start` line; a reader splits blocks on that line. A
//! block is **complete** iff its last line is `run_end` — a block with no
//! trailing `run_end` means the writing process was killed (e.g. a worker
//! hard-timeout) or the writer disabled itself on a failure partway through.
//!
//! - **`run_start`** — emitted once per invocation, before the first
//!   iteration. Fields: `transcript_version` (this module's
//!   [`TRANSCRIPT_VERSION`]), `harness_version` (`CARGO_PKG_VERSION`),
//!   `label` (the [`TranscriptConfig::label`] verbatim), `run_id` (the
//!   persisted run id, or `null` for the no-persistence [`crate::engine::run`]
//!   path), `resume` (`true` for both [`crate::engine::ResumeMode::Crash`] and
//!   [`crate::engine::ResumeMode::FreshContext`]), `tree_baseline` (the
//!   serialized [`crate::exec::TreeObservation`] the leg-3 precondition
//!   compares against, with `porcelain` capped for rendering and its
//!   untruncated `porcelain_chars` count alongside), `system` (the exact
//!   rendered system prompt), `tools` (the exact tool-schema array), `messages`
//!   (the full starting [`crate::model::Message`] history — for `Crash` this
//!   is the reconciled history, for `FreshContext` the fresh task seed), and
//!   `config` (an object with exactly `max_iterations`, `max_tokens`, `checks`
//!   — the check command display string, or `null` — `wall_clock_secs`,
//!   `static_tree_k`, `max_nudges`, `max_retries`).
//!
//!   ```json
//!   {"event":"run_start","ts":"2026-09-15T02:00:00Z","elapsed_ms":0,
//!    "transcript_version":1,"harness_version":"0.10.0","label":"claude-sonnet-5",
//!    "run_id":"task-42:1","resume":false,
//!    "tree_baseline":{"Observed":{"porcelain":"","porcelain_chars":0,"head":"abc123"}},
//!    "system":"You are an autonomous coding agent...","tools":[{"name":"echo","...":"..."}],
//!    "messages":[{"User":{"content":[{"Text":"do the task"}]}}],
//!    "config":{"max_iterations":10,"max_tokens":32768,"checks":"cargo test",
//!              "wall_clock_secs":0,"static_tree_k":3,"max_nudges":2,"max_retries":3}}
//!   ```
//!
//! - **`model_request`** — emitted once per logical iteration, right after
//!   `stats.iterations` is incremented and before the retry loop; NOT
//!   re-emitted per retry. Fields: `iteration` (1-based), `message_count`
//!   (`messages.len()` at send time), `block_count` (total content blocks
//!   summed across those messages). Deliberately lean — it does not re-dump
//!   history; a reader reconstructs it (see "Reconstruction" below).
//!
//!   ```json
//!   {"event":"model_request","ts":"2026-09-15T02:00:00Z","elapsed_ms":1,
//!    "iteration":1,"message_count":1,"block_count":1}
//!   ```
//!
//! - **`backend_error`** — emitted for EVERY failed `backend.turn` inside the
//!   retry loop, transients that will be retried included. Fields: `iteration`,
//!   `attempt` (0-based index of this failed call within the iteration),
//!   `retryable`, `will_retry` (`retryable && attempt < config.max_retries`),
//!   `error` (`Display`), `error_debug` (`Debug` — carries `Protocol`'s `raw`/
//!   `kind`/`retry_after`), `latency_ms` (this failed call only),
//!   `retry_delay_ms` (the computed backoff delay in ms when `will_retry`,
//!   else `null`).
//!
//!   ```json
//!   {"event":"backend_error","ts":"2026-09-15T02:00:01Z","elapsed_ms":1002,
//!    "iteration":1,"attempt":0,"retryable":true,"will_retry":true,
//!    "error":"transient backend failure (Network; retry_after=None)",
//!    "error_debug":"Transient { kind: Network, retry_after: None }",
//!    "latency_ms":5,"retry_delay_ms":500}
//!   ```
//!
//! - **`model_response`** — emitted for a successful turn, BEFORE it is
//!   consumed into `messages`. Fields: `iteration`, `attempts` (total
//!   `backend.turn` calls this iteration, successful one included),
//!   `latency_ms` (the successful call only), `stop_reason`
//!   ([`crate::model::StopReason`], externally tagged), `usage`
//!   ([`crate::model::Usage`]), `content` (the full
//!   `Vec<`[`crate::model::ContentBlock`]`>`, tool-call inputs verbatim).
//!
//!   ```json
//!   {"event":"model_response","ts":"2026-09-15T02:00:02Z","elapsed_ms":2003,
//!    "iteration":1,"attempts":1,"latency_ms":998,"stop_reason":"ToolUse",
//!    "usage":{"input_tokens":100,"output_tokens":20,"cache_read_tokens":null,
//!             "cache_write_tokens":null,"reasoning_tokens":null},
//!    "content":[{"ToolCall":{"id":"c1","name":"echo","input":{"i":1}}}]}
//!   ```
//!
//! - **`tool_result`** — emitted once per executed call, in call order, after
//!   the result is produced. Fields: `iteration`, `call_id`, `tool_name`,
//!   `is_error`, `content` (exactly the string fed back as the
//!   [`crate::model::UserBlock::ToolResult`]'s `content`), `offload_path`
//!   (always present — the offload path display string for calls routed
//!   through [`crate::tool::ToolRegistry::invoke`], `null` for finish-routed
//!   calls), `duration_ms`. ONLY for a call routed through the harness's
//!   finish-acceptance path does it ALSO carry `finish_accepted`,
//!   `finish_verification` (the [`crate::exec::CheckReport`], or `null` when
//!   no checks ran), `finish_change` (the
//!   [`crate::exec::ChangeEvidence`], or `null` when the call returned before
//!   observing the tree) and `tree_current` (the
//!   [`crate::exec::TreeObservation`] that evidence was classified from, or
//!   `null`) — those keys are absent on every other `tool_result`, including a
//!   second `finish` in the same batch (which executes as a plain tool call).
//!
//!   ```json
//!   {"event":"tool_result","ts":"2026-09-15T02:00:02Z","elapsed_ms":2004,
//!    "iteration":1,"call_id":"c1","tool_name":"echo","is_error":false,
//!    "content":"{\"i\":1}","offload_path":null,"duration_ms":0}
//!   {"event":"tool_result","ts":"2026-09-15T02:00:03Z","elapsed_ms":3000,
//!    "iteration":2,"call_id":"c2","tool_name":"finish","is_error":true,
//!    "content":"finish(done) rejected: verification failed","offload_path":null,
//!    "duration_ms":12,"finish_accepted":false,
//!    "finish_verification":{"passed":false,"excerpt":"FAIL_DETAIL","exit_code":3,"offload_path":null},
//!    "finish_change":null,"tree_current":null}
//!   ```
//!
//! - **`harness_message`** — emitted for every finish-recovery nudge
//!   injection. Fields: `iteration`, `kind` (`"nudge"` today — any future
//!   harness-injected user-lane message gets a new `kind`, documented here,
//!   plus its own reconstruction-test script), `placement`
//!   (`"new_user_message"` at the stop-terminal nudge site, where a fresh
//!   `Message::User` is pushed; `"appended_to_tool_results"` at the
//!   green-static nudge site, where a `UserBlock::Text` is appended to the
//!   existing tool-results `Message::User`), `text` (the exact injected
//!   string), `last_gate_green` (always `true` — a runtime tripwire),
//!   `iters_since_tree_change`, `static_tree_k`, `nudge_number` (1-based),
//!   `max_nudges`.
//!
//!   ```json
//!   {"event":"harness_message","ts":"2026-09-15T02:00:05Z","elapsed_ms":5000,
//!    "iteration":4,"kind":"nudge","placement":"appended_to_tool_results",
//!    "text":"You have not called finish...","last_gate_green":true,
//!    "iters_since_tree_change":3,"static_tree_k":3,"nudge_number":1,"max_nudges":2}
//!   ```
//!
//! - **`iteration_end`** — emitted immediately before the wall-clock breach
//!   check, so exactly once for each iteration that reaches end-of-iteration
//!   bookkeeping (NOT emitted for iterations that return earlier — finish,
//!   the recovery terminals, the stop terminal, a backend error — nor for the
//!   `continue`d stop-site nudge iteration). Fields: `iteration`, `mutated`,
//!   `last_gate_green`, `tree_dirty`, `iters_since_tree_change`,
//!   `nudges_fired`. Reads no clock.
//!
//!   ```json
//!   {"event":"iteration_end","ts":"2026-09-15T02:00:03Z","elapsed_ms":3001,
//!    "iteration":1,"mutated":false,"last_gate_green":false,"tree_dirty":false,
//!    "iters_since_tree_change":1,"nudges_fired":0}
//!   ```
//!
//! - **`run_end`** — emitted exactly once per invocation, on every exit path
//!   including a `?`-propagated [`crate::store::StoreError`]. Fields:
//!   `outcome` (one of `"Finished"` | `"StoppedWithoutFinish"` |
//!   `"MaxIterations"` | `"BudgetExhausted"` | `"BackendError"` — matching
//!   talos's `outcome_str` byte for byte — or `"StoreError"`, transcript-only),
//!   `disposition` (the [`crate::run_record::Disposition`] for `Finished`,
//!   `null` otherwise), `detail` (the `BudgetExhausted` summary /
//!   `BackendError`'s `Display` / `StoreError`'s `Display`, `null` otherwise),
//!   `stats` (an object with `iterations`, `input_tokens`, `output_tokens`,
//!   `cache_read_tokens`, `cache_write_tokens`, `gates_green_at_exit`,
//!   `nudges_fired`, `tree_dirty`, `iters_since_tree_change_at_exit`,
//!   `peak_iters_since_tree_change`, `mutating_iters`, `bash_calls_ok`,
//!   `edit_file_calls_ok`, `no_change_rejections`,
//!   `already_satisfied_check_rejections`, `tree_baseline_unobservable`).
//!   `wall_clock` is intentionally omitted — the
//!   caller (`run`/`run_persisted`/`resume`) sets `stats.wall_clock` only
//!   AFTER `run_loop_impl` (and therefore this event) returns.
//!
//!   The finish-recovery terminal appears as `outcome: "Finished"` with
//!   `disposition: {"Failed":{"mode":"FinishDiscipline","summary":"..."}}`.
//!
//!   ```json
//!   {"event":"run_end","ts":"2026-09-15T02:00:10Z","elapsed_ms":10000,
//!    "outcome":"Finished",
//!    "disposition":{"Done":{"summary":"ok","verification":"NoChecksConfigured",
//!                            "change":"TreeChanged"}},
//!    "detail":null,
//!    "stats":{"iterations":1,"input_tokens":0,"output_tokens":0,
//!             "cache_read_tokens":0,"cache_write_tokens":0,"gates_green_at_exit":false,
//!             "nudges_fired":0,"tree_dirty":false,"iters_since_tree_change_at_exit":0,
//!             "peak_iters_since_tree_change":0,"mutating_iters":0,"bash_calls_ok":0,
//!             "edit_file_calls_ok":0,"no_change_rejections":0,
//!             "already_satisfied_check_rejections":0,"tree_baseline_unobservable":false}}
//!   ```
//!
//! - **`contract_violation`** — emitted from the same choke point as
//!   `run_end`, and ONLY when the leg-3 invariant was broken: a
//!   `Disposition::Done` reached the terminal carrying
//!   [`crate::exec::ChangeEvidence::TreeUnchanged`]. Fields: `kind`
//!   (`"done_with_unchanged_tree"` today), `run_id` (or `null`), `change`. A
//!   correct run never emits it.
//!
//! ## Nested shapes (externally tagged serde)
//!
//! - `stop_reason`: `"ToolUse"` (unit variant) or `{"Other":".."}` (newtype).
//! - `content` blocks: `{"Text":".."}`, `{"Reasoning":{"text":"..","opaque":".."}}`,
//!   `{"ToolCall":{"id":"..","name":"..","input":{}}}`.
//! - `messages`: `{"User":{"content":[{"ToolResult":{"call_id":"..","content":"..","is_error":false}}]}}`
//!   or `{"Assistant":{"content":[...]}}`.
//! - `disposition`: `{"Done":{"summary":"..","verification":"NoChecksConfigured",
//!   "change":"TreeChanged"}}` or `{..,"Checks":{"passed":true,"excerpt":"..",
//!   "exit_code":0,"offload_path":null,"duration":{"secs":1,"nanos":0}}}`;
//!   `{"AlreadySatisfied":{"reason":"..","verification":"NoChecksConfigured",
//!   "change":"TreeUnchanged"}}`.
//! - `change`: `"TreeChanged"` / `"TreeUnchanged"` (unit variants) or
//!   `{"Unobservable":{"reason":".."}}`.
//!
//! Object key order within a line is unspecified.
//!
//! ## Reconstruction contract
//!
//! The full message history is reconstructible from one run block using only
//! `run_start.messages` + `model_response` + `tool_result` + `harness_message`
//! (a reader ignores `model_request`, `backend_error`, and `iteration_end`):
//!
//! 1. Start from `run_start.messages`.
//! 2. On `model_response`, push `Message::Assistant { content }`.
//! 3. On `tool_result`, append a `UserBlock::ToolResult { call_id, content,
//!    is_error }` to a pending batch.
//! 4. On `harness_message` with `placement == "appended_to_tool_results"`,
//!    append a `UserBlock::Text(text)` to that same pending batch.
//! 5. On `harness_message` with `placement == "new_user_message"`, first flush
//!    any non-empty pending batch as one `Message::User`, then push a fresh
//!    `Message::User { content: [Text(text)] }`.
//! 6. On the NEXT `model_request`, flush any non-empty pending batch as one
//!    `Message::User` before comparing.
//!
//! At every `model_request`, the rebuilt history's length and total block
//! count must equal that event's `message_count`/`block_count` — a mismatch
//! means an engine injection site (a new `messages.push`/`content.push`) was
//! added without a matching transcript event.
//!
//! `offload_path` values (and `"[full check output: ..]"` pointers embedded in
//! rejection content) are valid only while the caller's offload directory
//! still exists: persistent for `talos run`, deleted at trial end by both eval
//! runners. Events are harness-level (pre-adapter) — the provider's raw wire
//! payload is never captured. `talos ralph` does not write transcripts (scoped
//! out of this module). A resume that errors before reaching the loop body
//! (`UnknownRunId`, a reconcile store error) writes nothing at all — there is
//! no partial block.
//!
//! Callers should point the transcript path OUTSIDE the agent's workspace, so
//! the file itself never becomes something the agent reads, edits, or
//! otherwise contaminates its own context with.

use std::fs::{File, OpenOptions};
use std::io::Write as _;
use std::path::PathBuf;
use std::time::Instant;

use serde_json::{Map, Value};

/// Schema version stamped on every `run_start` event as
/// `transcript_version`. Bump when the event shapes documented on this
/// module change incompatibly.
pub const TRANSCRIPT_VERSION: u32 = 1;

/// The complete, closed set of `"event"` tag values a transcript line can
/// carry — see the module docs for each event's fields.
pub const EVENT_KINDS: [&str; 8] = [
    "run_start",
    "model_request",
    "backend_error",
    "model_response",
    "tool_result",
    "harness_message",
    "iteration_end",
    "run_end",
];

/// Opt-in configuration for a run's transcript sink.
///
/// `path` is where the JSONL file lives (append mode; a caller may reuse the
/// same path across a crash-resume — see the module docs on multi-block
/// files). `label` identifies the run in the `run_start` event (e.g. a model
/// label) and is carried verbatim, with no interpretation by this module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptConfig {
    /// Where the JSONL transcript is written. A relative path resolves
    /// against the caller's current working directory.
    pub path: PathBuf,
    /// Free-form label carried verbatim into `run_start.label`.
    pub label: String,
}

/// Parse an opt-in transcript boolean flag read from the environment
/// variable named `var_name` (the name is only used to build the `Err`
/// message).
///
/// The input is trimmed first. Empty (including `None`) or `"0"` means off;
/// `"1"` means on. Anything else is an error naming `var_name` and the
/// offending (untrimmed-echoed-as-trimmed) value.
///
/// # Errors
/// Returns `Err` when the trimmed value is non-empty and neither `"0"` nor
/// `"1"`.
pub fn parse_transcripts_flag(var_name: &str, v: Option<&str>) -> Result<bool, String> {
    let raw = v.map(str::trim).unwrap_or_default();
    match raw {
        "" | "0" => Ok(false),
        "1" => Ok(true),
        other => Err(format!(
            "{var_name}: expected `0`, `1`, or empty, got `{other}`"
        )),
    }
}

/// Best-effort JSONL writer for one transcript sink.
///
/// Disabled (`file: None`) either because no [`TranscriptConfig`] was
/// supplied to [`Self::open`], or because opening, serializing, or writing
/// failed at some point — the first failure disables the writer for the rest
/// of its life, printing exactly one `warning: transcript disabled (..): ..`
/// line to stderr. Every method is infallible from the caller's
/// perspective: a transcript failure never surfaces as an `Err` the engine
/// loop has to handle.
pub(crate) struct TranscriptWriter {
    file: Option<File>,
    path: PathBuf,
    start: Instant,
}

impl TranscriptWriter {
    /// Open the writer. `None` (transcripts off) does NO filesystem I/O at
    /// all and returns a disabled writer. `Some(cfg)` best-effort creates
    /// `cfg.path`'s parent directory (errors ignored — the subsequent open
    /// call surfaces any real problem) and opens the file in
    /// create-if-missing, APPEND mode — never truncate, so a reused path
    /// (crash-resume) accumulates run blocks rather than clobbering earlier
    /// ones. An open failure disables the writer (after printing the one
    /// warning line) rather than propagating.
    pub(crate) fn open(cfg: Option<&TranscriptConfig>) -> Self {
        let Some(cfg) = cfg else {
            return Self {
                file: None,
                path: PathBuf::new(),
                start: Instant::now(),
            };
        };
        if let Some(parent) = cfg.path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let mut writer = Self {
            file: None,
            path: cfg.path.clone(),
            start: Instant::now(),
        };
        match OpenOptions::new().create(true).append(true).open(&cfg.path) {
            Ok(file) => writer.file = Some(file),
            Err(err) => writer.disable(&err.to_string()),
        }
        writer
    }

    /// Whether the writer currently holds an open file — `false` either
    /// because transcripts are off, or because a prior open/serialize/write
    /// failure disabled it.
    pub(crate) fn is_enabled(&self) -> bool {
        self.file.is_some()
    }

    /// Emit one transcript line tagged `event`, with `fields` (a JSON
    /// object) as the event-specific payload. A no-op when disabled — call
    /// sites should guard any expensive payload construction with
    /// [`Self::is_enabled`] first, since this method still has to be called
    /// with an already-built `Value`.
    ///
    /// Inserts `event`, `ts` (RFC 3339 UTC, second resolution, via
    /// [`crate::time::format_rfc3339`] over `SystemTime::now()`), and
    /// `elapsed_ms` (milliseconds since [`Self::open`], via
    /// `Instant::now()`) into `fields`, serializes the result, and writes the
    /// line plus a trailing `\n` in a single unbuffered `write_all`. Any
    /// serialize or write failure disables the writer (see the struct docs).
    pub(crate) fn emit(&mut self, event: &str, fields: Value) {
        if self.file.is_none() {
            return;
        }
        let ts = crate::time::format_rfc3339(std::time::SystemTime::now());
        let elapsed_ms = u64::try_from(self.start.elapsed().as_millis()).unwrap_or(u64::MAX);

        let mut obj = match fields {
            Value::Object(map) => map,
            other => {
                // Contract violation by the caller (every call site in this
                // crate passes an object) — fall back to wrapping rather than
                // dropping the payload, so a mistake is visible in the file
                // instead of silently losing data.
                let mut m = Map::new();
                m.insert("value".to_string(), other);
                m
            }
        };
        obj.insert("event".to_string(), Value::String(event.to_string()));
        obj.insert("ts".to_string(), Value::String(ts));
        obj.insert("elapsed_ms".to_string(), Value::from(elapsed_ms));

        let line = match serde_json::to_string(&Value::Object(obj)) {
            Ok(line) => line,
            Err(err) => {
                self.disable(&err.to_string());
                return;
            }
        };

        // `file` is Some — checked at the top of this method, and nothing
        // above can have cleared it (self.disable does, but every path that
        // calls it returns immediately).
        let file = self.file.as_mut().expect("file is Some (checked above)");
        let mut line = line;
        line.push('\n');
        if let Err(err) = file.write_all(line.as_bytes()) {
            self.disable(&err.to_string());
        }
    }

    /// Disable the writer after a failure, printing exactly one warning line
    /// to stderr. Idempotent in effect (subsequent calls are unreachable —
    /// every call site returns right after calling this), but not asserted
    /// as such since `is_enabled()` already prevents a caller from reaching a
    /// second warning through the normal `emit`/`open` paths.
    fn disable(&mut self, error: &str) {
        eprintln!(
            "warning: transcript disabled ({}): {error}",
            self.path.display()
        );
        self.file = None;
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::{TranscriptConfig, TranscriptWriter, parse_transcripts_flag};
    use serde_json::json;
    use std::io::Read as _;

    // ---- parse_transcripts_flag ------------------------------------------

    #[test]
    fn parse_transcripts_flag_covers_the_pinned_tuples() {
        assert_eq!(parse_transcripts_flag("X_VAR", None), Ok(false));
        assert_eq!(parse_transcripts_flag("X_VAR", Some("")), Ok(false));
        assert_eq!(parse_transcripts_flag("X_VAR", Some("   ")), Ok(false));
        assert_eq!(parse_transcripts_flag("X_VAR", Some("0")), Ok(false));
        assert_eq!(parse_transcripts_flag("X_VAR", Some(" 0 ")), Ok(false));
        assert_eq!(parse_transcripts_flag("X_VAR", Some("1")), Ok(true));
        assert_eq!(parse_transcripts_flag("X_VAR", Some(" 1 ")), Ok(true));
    }

    #[test]
    fn parse_transcripts_flag_error_is_pinned_verbatim() {
        assert_eq!(
            parse_transcripts_flag("X_VAR", Some("yes")),
            Err("X_VAR: expected `0`, `1`, or empty, got `yes`".to_string())
        );
    }

    // ---- TranscriptWriter --------------------------------------------------

    #[test]
    fn open_none_is_disabled_with_no_filesystem_io() {
        let writer = TranscriptWriter::open(None);
        assert!(!writer.is_enabled());
    }

    #[test]
    fn parent_is_a_regular_file_disables_the_writer() {
        let parent_file = tempfile::NamedTempFile::new().expect("create temp file");
        let path = parent_file.path().join("t.jsonl");
        let cfg = TranscriptConfig {
            path,
            label: "t".to_string(),
        };
        let writer = TranscriptWriter::open(Some(&cfg));
        assert!(!writer.is_enabled());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn dev_full_disables_after_first_emit_and_does_not_panic_on_second() {
        let cfg = TranscriptConfig {
            path: std::path::PathBuf::from("/dev/full"),
            label: "t".to_string(),
        };
        let mut writer = TranscriptWriter::open(Some(&cfg));
        assert!(writer.is_enabled(), "opening /dev/full for append succeeds");

        writer.emit("run_start", json!({}));
        assert!(
            !writer.is_enabled(),
            "a write to /dev/full always fails (ENOSPC)"
        );

        // Must not panic.
        writer.emit("run_end", json!({}));
        assert!(!writer.is_enabled());
    }

    #[test]
    fn happy_path_two_emits_produce_two_valid_lines() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("t.jsonl");
        let cfg = TranscriptConfig {
            path: path.clone(),
            label: "t".to_string(),
        };
        let mut writer = TranscriptWriter::open(Some(&cfg));
        assert!(writer.is_enabled());

        writer.emit("run_start", json!({"label": "t"}));
        writer.emit("run_end", json!({"outcome": "Finished"}));

        let contents = std::fs::read_to_string(&path).expect("read transcript");
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 2, "two emits produce two lines");

        for line in lines {
            let value: serde_json::Value = serde_json::from_str(line).expect("valid JSON line");
            let obj = value.as_object().expect("each line is a JSON object");
            assert!(obj.contains_key("event"));
            let ts = obj
                .get("ts")
                .expect("ts present")
                .as_str()
                .expect("ts is a string");
            assert_eq!(ts.len(), 20, "ts is YYYY-MM-DDTHH:MM:SSZ (20 chars)");
            assert!(ts.ends_with('Z'));
            assert!(obj.contains_key("elapsed_ms"));
        }
    }

    #[test]
    fn two_writers_on_the_same_path_append_rather_than_truncate() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("t.jsonl");
        let cfg = TranscriptConfig {
            path: path.clone(),
            label: "t".to_string(),
        };

        let mut first = TranscriptWriter::open(Some(&cfg));
        first.emit("run_start", json!({}));
        first.emit("run_end", json!({}));
        drop(first);

        let first_contents = {
            let mut s = String::new();
            std::fs::File::open(&path)
                .expect("open after first writer")
                .read_to_string(&mut s)
                .expect("read after first writer");
            s
        };
        let first_lines: Vec<String> = first_contents.lines().map(str::to_string).collect();
        assert_eq!(first_lines.len(), 2);

        let mut second = TranscriptWriter::open(Some(&cfg));
        second.emit("run_start", json!({}));
        second.emit("run_end", json!({}));
        drop(second);

        let final_contents = std::fs::read_to_string(&path).expect("read after second writer");
        let final_lines: Vec<&str> = final_contents.lines().collect();
        assert_eq!(final_lines.len(), 4, "append mode: 2 + 2 lines");
        assert_eq!(
            final_lines[0], first_lines[0],
            "first block's lines are unchanged"
        );
        assert_eq!(
            final_lines[1], first_lines[1],
            "first block's lines are unchanged"
        );
    }
}
