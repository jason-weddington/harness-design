//! The `somnus` library target: everything the nightly map-maintenance loop
//! is made of, minus the (not-yet-wired) run body.
//!
//! Every library construct lives HERE rather than in `src/main.rs` so that
//! `pub` items are exempt from `dead_code`: the bin's run path is a not-wired
//! stub this cut (`pub` does NOT exempt items from `dead_code` in a binary,
//! so a bin-only shape would fail `cargo clippy --all-targets -- -D warnings`
//! on first build). `main.rs` stays in place so the publisher's
//! `crates/*/src/main.rs` discovery rule and the `[[bin]]` path are unchanged.
//!
//! Modules:
//!
//! - [`gate`] — the map-lint gate and the six-tool registry.
//! - [`observer`] — the leg-3 pointer-count change observer.
//! - [`ops`] — the closed op vocabulary, the op seam, and the disposition
//!   mapping.
//! - [`loop_input`] — the rung-1 payload types, the loop-input transport, and
//!   the pure payload consumers.
//! - [`ledger`] — the cluster/decline ledger seam and the decline filter.
//! - [`rungs`] — the single-shot rung-1 and rung-2 inferences.
//! - [`materialize`] — rung 3: code composes the map bodies from ops.
//! - [`map_op`] — the `POST /api/kb/map-op` transport (the single
//!   machine-principal write endpoint; the contract of record is the
//!   vendored spec's write-path section).
//! - [`worklist`] — the `GET /api/kb/map-worklist` client, the take-first-
//!   three-in-server-order selection, and the nightly invocation record.
//! - [`unit`] — the per-project pipeline, the map-op application step, and
//!   its run report.
//!
//! Retired with this cut: `build_run_config`, `SOMNUS_MAX_NUDGES`, and
//! `SOMNUS_COMPACT_THRESHOLD_PCT`. The rung discipline — one single-shot
//! inference per project and per cluster, the read path injected as
//! synthetic tool-call/result events, and the read tools deleted from the
//! output union — cannot be expressed by `engine::run`, which builds its
//! own `initial_messages` from `prompt::render_task_prompt` internally (its
//! `RunConfig` exposes no history seam and `run_loop_impl` is private), so
//! there is no engine loop here and therefore no nudge or compaction
//! machinery to configure. The per-unit inference budget survived and was
//! repurposed as the per-unit inference budget instead.

/// The binary's own version token, stamped by `build.rs` from
/// `git describe --tags` — the same string `somnus --version` prints and the
/// same one the artifact host publishes under.
///
/// Every record somnus writes carries it. Without it a report is a set of
/// numbers with no way to say which build produced them, which forces an
/// operator to choose between installing a fix and being able to attribute
/// the run they are in the middle of. That is a false choice and it costs a
/// night: a report that cannot be attributed is a measurement that cannot be
/// compared, and a fix withheld to protect attribution is a fix not running.
pub const SOMNUS_VERSION: &str = env!("SOMNUS_VERSION");

pub mod admission;
pub mod gate;
pub mod ledger;
pub mod loop_input;
pub mod map_op;
pub mod materialize;
pub mod observer;
pub mod ops;
pub mod rungs;
pub mod unit;
pub mod worklist;

use harness::exec::TreeObservation;

/// The seed task for the nightly loop.
pub const SOMNUS_TASK: &str = "somnus nightly map-maintenance loop";

// ===== Grounding notes ====================================================
//
// - LEAD DECISION 1 (token ceiling): somnus enforces the ceiling itself.
//   The vendored spec's crate-obligation paragraph is WRONG on this point —
//   its premise (the harness budget would apply) is false: somnus's pipeline
//   drives `ModelBackend::turn` directly and never calls `engine::run`, so
//   `RunConfig::token_budget` is unreachable from it. The enforcement lives
//   where the turns are ([`crate::unit::run_unit`]), against the breach
//   predicate extracted PUBLIC from harness (`engine::token_budget_breached`)
//   and the shared billed-token core (`model::billed_token_sum`, also used by
//   `Usage::billed_tokens` and `engine::budget_consumed_now`) — ONE
//   definition of a billed token in the workspace, not two.
// - LEAD DECISION 2 (the worklist payload): the
//   `GET /api/kb/map-worklist` field spellings are LEAD-SUPPLIED GROUND
//   TRUTH, unverifiable from this clone (the vendored spec's Rung 0 still
//   names the admin-gated `/api/kb/map-eligibility`, which somnus must NOT
//   call). Recorded in [`crate::worklist`]'s module doc.
//
// ===== The binary-level configuration surface (pure, env-read at the edge) ==
//
// The operator env spellings are FROZEN: they are already provisioned in
// /etc/somnus/env on three hosts. somnus's own code reads exactly these six
// vars plus `XDG_STATE_HOME` and `HOME` (the latter two only inside the
// state-dir default), so every `env::var` spelling in this crate is one of
// those eight. reqwest's transport-layer proxy handling is not somnus code
// and is unchanged.
//
// The PURE functions below take the raw `Option<String>` values; the binary
// (`main.rs`) is the only place that touches `std::env`, so the fault lines
// here are exercisable without any environment mutation (which is forbidden
// in this workspace: `unsafe_code = "forbid"` + edition 2024).

/// The operator env spelling for the KB base URL (required).
pub const KB_BASE_URL_VAR: &str = "SOMNUS_KB_BASE_URL";
/// The operator env spelling for the KB bearer token (required).
pub const KB_API_KEY_VAR: &str = "SOMNUS_KB_API_KEY";
/// The operator env spelling for the Anthropic key (required).
pub const ANTHROPIC_API_KEY_VAR: &str = "ANTHROPIC_API_KEY";
/// The operator env spelling for the nightly SPEND ceiling, in
/// micro-dollars (optional).
pub const COST_BUDGET_VAR: &str = "SOMNUS_COST_BUDGET_MICROS";
/// The operator env spelling for the state-dir override (optional).
pub const STATE_DIR_VAR: &str = "SOMNUS_STATE_DIR";
/// The operator env spelling for the kill switch (optional; exactly `1`).
pub const DISABLED_VAR: &str = "SOMNUS_DISABLED";

/// The pinned clap-usage fault line: EVERY clap usage error (a bare
/// `somnus`, a bad flag, a missing `--project`) exits 1 with this ONE line —
/// clap's default usage-error exit 2 is overridden, and clap's default
/// usage/`error:` rendering is replaced by this byte-pinned shape.
pub const CLI_USAGE_ERROR_MSG: &str = "somnus: exactly one subcommand required: nightly | run --project <ref> | backfill --project <ref>";

/// The pinned kill-switch stderr line. A silent exit 0 is the one outcome an
/// unattended 3am binary must never produce, so the switch announces itself.
pub const DISABLED_MSG: &str =
    "somnus: disabled via SOMNUS_DISABLED=1; exiting before any other env read, fetch, or write";

/// The pinned cost-budget fault line: any non-integer, negative, or empty
/// value is a configuration fault (one shape for all three).
pub const COST_BUDGET_MSG: &str = "somnus: SOMNUS_COST_BUDGET_MICROS is not a non-negative integer";

/// The pinned base-URL fault line. A value ending in `/` is a configuration
/// fault, never a silently-trimmed value.
pub const BASE_URL_TRAILING_SLASH_MSG: &str = "somnus: SOMNUS_KB_BASE_URL must not end in '/'";

/// The pinned fault line for a missing, empty, or whitespace-only required
/// env var: ONE byte-pinned shape covering all three cases, per var.
pub fn render_required_env_missing_line(var: &str) -> String {
    format!("somnus: {var} is not set")
}

/// The subcommand spend ceilings, in micro-dollars.
///
/// **Denominated in money because tokens are the wrong unit for this
/// question, and measuring in the wrong one punished the fix that made the
/// loop cheaper.** The first ceiling was 550,000 billed tokens, intended as
/// roughly $2. A prompt-cache fix then landed that cut cache WRITES from
/// 504k to 51k and moved 434k tokens into cache READS — which bill at a
/// fiftieth of a completion token. The run got sharply cheaper, $1.13, and
/// tripped the ceiling anyway at 575,017 tokens. A token ceiling answers a
/// context-window question; a guard against overspend has to count money.
pub const NIGHTLY_COST_BUDGET_MICROS_DEFAULT: u64 = 2_000_000;
/// The `run` default (one project, the same shape as one night's project).
pub const RUN_COST_BUDGET_MICROS_DEFAULT: u64 = 2_000_000;
/// The `backfill` default (loose enough to admit the backfill is not a
/// ceiling; see the vendored spec's cap section).
pub const BACKFILL_COST_BUDGET_MICROS_DEFAULT: u64 = 7_000_000;

/// The kill switch: EXACTLY the string `1` arms it. Any other value —
/// `true`, `0`, empty — is ignored, so a mistyped `SOMNUS_DISABLED=true`
/// never silently disables the loop.
#[must_use]
pub fn is_disabled(value: Option<&str>) -> bool {
    value == Some("1")
}

/// The required operator env, validated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MachineEnv {
    /// The KB base URL. Guaranteed not to end in `/`.
    pub kb_base_url: String,
    /// The KB bearer token.
    pub kb_api_key: String,
    /// The Anthropic API key.
    pub anthropic_api_key: String,
}

/// Validate the three REQUIRED operator env vars, in pinned order
/// (base URL, KB key, Anthropic key).
///
/// A missing, empty, or whitespace-only var is a configuration fault with
/// the pinned line `somnus: {VAR} is not set` (one shape for all three
/// cases); a base URL ending in `/` is the pinned trailing-slash fault —
/// never a silently-trimmed value.
///
/// # Errors
/// The pinned fault line for the FIRST offending var, in pinned order.
pub fn parse_required_env(
    kb_base_url: Option<&str>,
    kb_api_key: Option<&str>,
    anthropic_api_key: Option<&str>,
) -> Result<MachineEnv, String> {
    let base = required(KB_BASE_URL_VAR, kb_base_url)?;
    let kb_api_key = required(KB_API_KEY_VAR, kb_api_key)?;
    let anthropic_api_key = required(ANTHROPIC_API_KEY_VAR, anthropic_api_key)?;
    if base.ends_with('/') {
        return Err(BASE_URL_TRAILING_SLASH_MSG.to_string());
    }
    Ok(MachineEnv {
        kb_base_url: base,
        kb_api_key,
        anthropic_api_key,
    })
}

/// One required var: missing, empty, and whitespace-only are the same fault.
fn required(var: &str, raw: Option<&str>) -> Result<String, String> {
    match raw {
        Some(value) if !value.trim().is_empty() => Ok(value.to_string()),
        _ => Err(render_required_env_missing_line(var)),
    }
}

/// Parse [`COST_BUDGET_VAR`]: absent → the subcommand default; `0` → 0
/// (disabled/unbounded); any non-integer, negative, or empty value → the
/// pinned fault line.
///
/// # Errors
/// [`COST_BUDGET_MSG`] when `raw` is `Some` but not a non-negative integer.
pub fn parse_cost_budget(default: u64, raw: Option<&str>) -> Result<u64, String> {
    match raw {
        None => Ok(default),
        Some(raw) => raw.parse::<u64>().map_err(|_| COST_BUDGET_MSG.to_string()),
    }
}

/// The default state dir: `$XDG_STATE_HOME/somnus` when `xdg` is set, else
/// `$HOME/.local/state/somnus`, else the process tempdir under `somnus`
/// when both are absent (never panics on a headless box).
#[must_use]
pub fn default_state_dir(xdg: Option<&str>, home: Option<&str>) -> std::path::PathBuf {
    if let Some(xdg) = xdg {
        return std::path::PathBuf::from(xdg).join("somnus");
    }
    if let Some(home) = home {
        return std::path::PathBuf::from(home).join(".local/state/somnus");
    }
    std::env::temp_dir().join("somnus")
}

/// Resolve the state dir with the [`STATE_DIR_VAR`] override: a set and
/// non-empty value overrides ENTIRELY; anything else falls through to
/// [`default_state_dir`].
#[must_use]
pub fn resolve_state_dir(
    state_dir_var: Option<&str>,
    xdg: Option<&str>,
    home: Option<&str>,
) -> std::path::PathBuf {
    match state_dir_var {
        Some(dir) if !dir.trim().is_empty() => std::path::PathBuf::from(dir),
        _ => default_state_dir(xdg, home),
    }
}

/// How many clusters ONE `run_unit` will pay a rung-2 inference for.
///
/// **Raised from an effective 23 on 2026-09-21, and re-expressed in the unit
/// that actually governs.** Rung 2 is exactly one call per cluster, so a cap
/// on CALLS was a cap on clusters with an off-by-one hiding in it — and the
/// number was a compiled guess made before anyone had seen a real project's
/// cluster count. Photoqueue produced 23, which is to say the guess was
/// wrong by about one cluster on the first project that tested it.
///
/// The ceiling that should stop a runaway is MONEY, not a call count: a call
/// count stops legitimate work at a boundary that has nothing to do with
/// what the work is worth, and run 5 spent $1.62 against a $2.00 ceiling
/// while being cut short by this. So this is now a genuine runaway guard —
/// a rung 1 returning 64 clusters for one project has misunderstood the
/// project — rather than a working limit anything normal reaches.
///
/// When it does bind, the clusters beyond it are the ones rung 1 ranked
/// LAST, because the list arrives in merit order. That is the job merit
/// order was always for; it just never had a truncation to govern before.
pub const SOMNUS_MAX_CLUSTERS_PER_UNIT: usize = 64;

/// The nightly wall-clock budget, in seconds: 4 hours. Consumed by
/// [`crate::unit::run_unit`]'s `tokio::time::timeout` wrapper, so an expiry
/// is a named abort and never a hang.
///
/// Token-budget arming is somnus's OWN job (see
/// [`NIGHTLY_COST_BUDGET_MICROS_DEFAULT`]): somnus's pipeline drives
/// `ModelBackend::turn` directly and never calls `engine::run`, so
/// `RunConfig::token_budget` (which exists in `crates/harness/src/engine.rs`)
/// is unreachable from it — the ceiling must live where the turns are, and
/// [`crate::unit::run_unit`] enforces it with
/// [`harness::engine::token_budget_breached`].
pub const NIGHTLY_WALL_CLOCK_SECS: u64 = 14_400;

/// The run-start guard: refuse to start when the run-start baseline could not
/// be observed.
///
/// The baseline is armed by the observer's FIRST successful observation; a
/// first-call failure yields `Unobservable`, and a loop that cannot prove
/// work happened does not get to claim it did — so it does not get to run.
/// The guard's call site is [`crate::unit::run_unit`], which aborts the unit
/// with this exact message when the baseline could not be observed.
///
/// # Errors
///
/// Returns `Err` with the exact refusal message when `baseline` is
/// [`TreeObservation::Unobservable`]; `Ok(())` for any `Observed` baseline.
pub fn run_start_refusal(baseline: &TreeObservation) -> Result<(), String> {
    match baseline {
        TreeObservation::Observed { .. } => Ok(()),
        TreeObservation::Unobservable { reason } => Err(format!(
            "somnus: refusing to start: run-start baseline could not be observed ({reason})"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- the run-start guard ---

    #[test]
    fn run_start_refusal_accepts_any_observed_baseline() {
        let baseline = TreeObservation::Observed {
            porcelain: "17".to_string(),
            head: None,
        };
        assert_eq!(run_start_refusal(&baseline), Ok(()));
    }

    #[test]
    fn run_start_refusal_refuses_an_unobservable_baseline_with_the_exact_message() {
        let baseline = TreeObservation::Unobservable {
            reason: "pointer count timed out after 10s".to_string(),
        };
        assert_eq!(
            run_start_refusal(&baseline),
            Err("somnus: refusing to start: run-start baseline could not be observed (pointer count timed out after 10s)".to_string())
        );
    }

    // --- the constants the vendored spec pins ---

    #[test]
    fn named_constants_match_the_pinned_values() {
        assert_eq!(SOMNUS_MAX_CLUSTERS_PER_UNIT, 64);
        assert_eq!(NIGHTLY_WALL_CLOCK_SECS, 14_400);
        assert_eq!(NIGHTLY_COST_BUDGET_MICROS_DEFAULT, 2_000_000);
        assert_eq!(RUN_COST_BUDGET_MICROS_DEFAULT, 2_000_000);
        assert_eq!(BACKFILL_COST_BUDGET_MICROS_DEFAULT, 7_000_000);
        assert_eq!(
            crate::ops::SOMNUS_DONE_SUMMARY,
            "ops applied, map-lint gate green, pointer count moved"
        );
    }

    // --- the kill switch --------------------------------------------------

    #[test]
    fn only_the_exact_string_one_arms_the_kill_switch() {
        assert!(is_disabled(Some("1")));
        for ignored in [
            None,
            Some(""),
            Some("true"),
            Some("0"),
            Some(" 1"),
            Some("1 "),
        ] {
            assert!(!is_disabled(ignored), "{ignored:?} is ignored");
        }
    }

    // --- the required env --------------------------------------------------

    #[test]
    fn required_env_missing_empty_and_whitespace_are_one_fault_shape_per_var() {
        // For each var in turn, unset / empty / whitespace-only is the same
        // fault, byte-pinned per var; the other two vars carry dummies.
        for (index, var) in [KB_BASE_URL_VAR, KB_API_KEY_VAR, ANTHROPIC_API_KEY_VAR]
            .into_iter()
            .enumerate()
        {
            let raws: [Option<&str>; 3] = [None, Some(""), Some("   ")];
            for raw in raws {
                let base = if index == 0 { raw } else { Some("http://kb") };
                let kb_key = if index == 1 { raw } else { Some("kb-key") };
                let an_key = if index == 2 { raw } else { Some("an-key") };
                assert_eq!(
                    parse_required_env(base, kb_key, an_key),
                    Err(render_required_env_missing_line(var)),
                    "{var} with {raw:?}"
                );
            }
            assert_eq!(
                render_required_env_missing_line(var),
                format!("somnus: {var} is not set")
            );
        }
    }

    #[test]
    fn a_trailing_slash_base_url_is_a_fault_never_a_trim() {
        let parsed = parse_required_env(Some("http://kb/"), Some("k"), Some("k"));
        assert_eq!(parsed, Err(BASE_URL_TRAILING_SLASH_MSG.to_string()));
        assert_eq!(
            BASE_URL_TRAILING_SLASH_MSG,
            "somnus: SOMNUS_KB_BASE_URL must not end in '/'"
        );
    }

    #[test]
    fn a_clean_required_env_parses_in_pinned_order() {
        assert_eq!(
            parse_required_env(Some("http://kb"), Some("kb-key"), Some("an-key")),
            Ok(MachineEnv {
                kb_base_url: "http://kb".to_string(),
                kb_api_key: "kb-key".to_string(),
                anthropic_api_key: "an-key".to_string(),
            })
        );
    }

    // --- the token budget ---------------------------------------------------

    #[test]
    fn the_cost_budget_parses_the_pinned_table() {
        // (raw, armed) for the nightly default.
        let table = [
            (None, Ok(2_000_000u64)),
            (Some("0"), Ok(0)),
            (Some("2000000"), Ok(2_000_000)),
            (Some("7000000"), Ok(7_000_000)),
            (Some(""), Err(COST_BUDGET_MSG.to_string())),
            (Some("abc"), Err(COST_BUDGET_MSG.to_string())),
            (Some("-1"), Err(COST_BUDGET_MSG.to_string())),
            (Some("1e6"), Err(COST_BUDGET_MSG.to_string())),
        ];
        for (raw, expected) in table {
            assert_eq!(parse_cost_budget(2_000_000, raw), expected, "{raw:?}");
        }
        // The fault line is byte-pinned.
        assert_eq!(
            COST_BUDGET_MSG,
            "somnus: SOMNUS_COST_BUDGET_MICROS is not a non-negative integer"
        );
        // The subcommand defaults flow through.
        assert_eq!(
            parse_cost_budget(BACKFILL_COST_BUDGET_MICROS_DEFAULT, None),
            Ok(7_000_000)
        );
    }

    // --- the state dir ------------------------------------------------------

    #[test]
    fn the_state_dir_resolves_the_pinned_arms_plus_the_override() {
        use std::path::PathBuf;
        assert_eq!(
            default_state_dir(Some("/xdg"), Some("/home")),
            PathBuf::from("/xdg/somnus")
        );
        assert_eq!(
            default_state_dir(None, Some("/home")),
            PathBuf::from("/home/.local/state/somnus")
        );
        let headless = default_state_dir(None, None);
        assert_eq!(
            headless,
            std::env::temp_dir().join("somnus"),
            "headless boxes fall back to the process tempdir"
        );
        // The override wins entirely; empty and whitespace fall through.
        assert_eq!(
            resolve_state_dir(Some("/opt/state"), Some("/xdg"), Some("/home")),
            PathBuf::from("/opt/state")
        );
        assert_eq!(
            resolve_state_dir(Some(""), Some("/xdg"), Some("/home")),
            PathBuf::from("/xdg/somnus")
        );
        assert_eq!(
            resolve_state_dir(None, None, None),
            default_state_dir(None, None)
        );
    }
}
