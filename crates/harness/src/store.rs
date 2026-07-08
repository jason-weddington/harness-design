//! Persistence: the [`RunStore`] trait + [`SqliteRunStore`] implementation.
//!
//! Per `docs/design/02-run-record-schema.md` ("Two stores, one source of
//! truth" and "Persistence interface"), the run record needs durable storage
//! so the harness can:
//!
//! 1. **Crash-resume** — reload the snapshot and continue from the last
//!    checkpoint; interrupted steps are handled by synthesizing an
//!    `is_error=true` result for each unpaired `ToolCallStarted` — the harness
//!    **never re-executes** an interrupted call (side effects may have already
//!    happened). The `seq` key makes each step idempotent on resume.
//! 2. **Fresh-context restart** — drop `messages`, keep `durable_facts` /
//!    `phase` / `budgets`, re-orient from git + filesystem.
//!
//! Two tables, two purposes:
//!
//! - `events(run_id, seq, ts, kind, payload)` — append-only audit trail and
//!   source of truth for the trajectory. The `seq` is a per-run monotonic
//!   counter; tool side effects key off it for idempotent replay. An
//!   unpaired `ToolCallStarted` on resume means the call was interrupted:
//!   the harness synthesizes an `is_error=true` `ToolCallResult` rather
//!   than re-executing the call.
//! - `runs(run_id, schema_version, state_blob, updated_at)` — materialized
//!   snapshot. Derivable from the log in principle; materialized so resume is
//!   a single read, not a full replay.
//!
//! ## `SQLite` crate choice — why `rusqlite` (vs. `sqlx`)
//!
//! `rusqlite` 0.31 with the `bundled` feature. Reasons:
//!
//! 1. **License compliance.** Every transitive crate satisfies the
//!    `MIT / Apache-2.0 / Unicode-3.0` policy in `deny.toml` at this pin.
//!    Newer `rusqlite` (0.33+) pulls `hashbrown` 0.15 → `foldhash` 0.1
//!    (Zlib), which is outside the allow list; staying on 0.31 keeps
//!    `cargo deny` green without widening the policy. `sqlx` would pull a
//!    much larger tree (`sqlx-core`, `sqlx-macros`, `sqlx-sqlite`, async-rt
//!    shims, …) with more license surface and more risk of repeating this
//!    fight on every bump.
//! 2. **Right-sized for v1.** v1 is "one process, one `SQLite` file, one
//!    connection." `sqlx`'s pool / migration / multi-backend machinery is
//!    out of scope. `rusqlite` is a focused, single-purpose binding.
//! 3. **Async via `spawn_blocking`.** Each store call wraps the blocking
//!    `SQLite` work in [`tokio::task::spawn_blocking`], satisfying the
//!    trait's async contract. Throughput here is "~1 checkpoint per
//!    inner-loop iteration" — the synchronous-driver + `spawn_blocking`
//!    pattern is more than fast enough, and it keeps the dependency
//!    footprint small.
//!
//! The `bundled` feature statically links a vendored `SQLite` C source so the
//! build is self-contained (no host `SQLite` headers required).
//!
//! ## Determinism
//!
//! Snapshots are serialized with `serde_json`; the run-record types use
//! `BTreeMap` throughout (see [`crate::run_record`]), so the on-disk JSON
//! is byte-stable across runs — the same discipline that keeps the prompt
//! cache hitting.

use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rusqlite::{Connection, OptionalExtension, params};

use crate::run_record::{Event, RunRecord};

// ===== Errors =========================================================

/// Errors a [`RunStore`] implementation can return.
///
/// Persistence is a *steering* surface for the loop: every variant carries
/// enough context for the harness to decide whether to retry, escalate, or
/// surface the failure to the outer review. There is no `unwrap()` in the hot
/// path — a panic here would silently lose run state, which is the failure
/// class this module exists to prevent.
#[derive(Debug)]
pub enum StoreError {
    /// Underlying `SQLite` error (locked DB, schema mismatch, I/O, …).
    Sql(rusqlite::Error),
    /// `serde_json` (de)serialization error — typically a corrupted snapshot.
    Serialization(serde_json::Error),
    /// The background blocking task panicked or was cancelled. Carries the
    /// stringified cause because [`tokio::task::JoinError`] is not `Clone`
    /// and we want the error type itself to stay simple.
    Join(String),
    /// The internal connection mutex was poisoned (a previous holder
    /// panicked). This is unrecoverable for the live store — the connection
    /// state is unknown — and should bubble up.
    LockPoisoned,
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sql(e) => write!(f, "sql error: {e}"),
            Self::Serialization(e) => write!(f, "serialization error: {e}"),
            Self::Join(e) => write!(f, "background task error: {e}"),
            Self::LockPoisoned => f.write_str("internal connection mutex was poisoned"),
        }
    }
}

impl std::error::Error for StoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Sql(e) => Some(e),
            Self::Serialization(e) => Some(e),
            Self::Join(_) | Self::LockPoisoned => None,
        }
    }
}

impl From<rusqlite::Error> for StoreError {
    fn from(e: rusqlite::Error) -> Self {
        Self::Sql(e)
    }
}

impl From<serde_json::Error> for StoreError {
    fn from(e: serde_json::Error) -> Self {
        Self::Serialization(e)
    }
}

impl From<tokio::task::JoinError> for StoreError {
    fn from(e: tokio::task::JoinError) -> Self {
        Self::Join(e.to_string())
    }
}

// ===== Trait ==========================================================

/// The persistence interface — the seam that keeps the harness
/// deployment-agnostic.
///
/// v1 ships [`SqliteRunStore`] (zero-ops, single-file). Other backends
/// (Postgres, an object store, the task tracker itself) can slot in behind
/// this trait without the loop knowing.
///
/// ## Contract
///
/// - [`Self::load`] returns the latest checkpointed snapshot for `run_id`,
///   or `None` if nothing has been checkpointed yet.
/// - [`Self::append_event`] assigns a **monotonic per-run `seq`** to `event`
///   and appends it to the log, returning the assigned `seq`. Sequences are
///   independent across distinct `run_id`s.
/// - [`Self::checkpoint`] upserts the latest snapshot — the snapshot is a
///   cache; the log is the source of truth. The caller writes the event log
///   first, then checkpoints, so a crash between the two reduces to "an
///   interrupted step" rather than lost state.
#[async_trait]
pub trait RunStore: Send + Sync {
    /// Load the latest snapshot for `run_id`, or `None` if no checkpoint
    /// exists yet.
    async fn load(&self, run_id: &str) -> Result<Option<RunRecord>, StoreError>;

    /// Append `event` to `run_id`'s log. The store assigns a monotonic per-run
    /// `seq` (overwriting whatever `seq` field the passed-in event carried)
    /// and returns it; callers should use the returned value as the
    /// authoritative ordering key.
    async fn append_event(&self, run_id: &str, event: Event) -> Result<u64, StoreError>;

    /// Upsert the latest snapshot for `run_id`. Successive calls overwrite
    /// the previous snapshot — the event log carries history.
    async fn checkpoint(&self, run_id: &str, record: &RunRecord) -> Result<(), StoreError>;
}

// ===== SQLite implementation =========================================

/// `SQLite`-backed [`RunStore`]. Holds a single connection behind a
/// [`std::sync::Mutex`] (`SQLite`'s own write lock is per-connection, and we
/// run blocking work via [`tokio::task::spawn_blocking`], so a sync mutex is
/// the right pairing — an async mutex would deadlock against the blocking
/// pool).
#[derive(Debug)]
pub struct SqliteRunStore {
    conn: Arc<Mutex<Connection>>,
}

impl SqliteRunStore {
    /// Open (or create) the `SQLite` DB at `path`, initializing the schema if
    /// the tables don't yet exist.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let conn = Connection::open(path)?;
        Self::init_schema(&conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Open an in-memory `SQLite` DB. Convenient for tests and ephemeral
    /// scratch runs; the data does not survive the process.
    pub fn open_in_memory() -> Result<Self, StoreError> {
        let conn = Connection::open_in_memory()?;
        Self::init_schema(&conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Create the two tables if missing. Idempotent: safe to call on a
    /// re-opened DB.
    fn init_schema(conn: &Connection) -> Result<(), rusqlite::Error> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS events (
                 run_id  TEXT    NOT NULL,
                 seq     INTEGER NOT NULL,
                 ts      TEXT    NOT NULL,
                 kind    TEXT    NOT NULL,
                 payload TEXT    NOT NULL,
                 PRIMARY KEY (run_id, seq)
             );
             CREATE TABLE IF NOT EXISTS runs (
                 run_id         TEXT    PRIMARY KEY,
                 schema_version INTEGER NOT NULL,
                 state_blob     TEXT    NOT NULL,
                 updated_at     TEXT    NOT NULL
             );",
        )
    }

    /// Read all events for `run_id`, ordered by `seq`. Decodes the JSON
    /// payload back into [`Event`]. Useful for replay, the eval flywheel, and
    /// tests that verify ordering.
    pub async fn list_events(&self, run_id: &str) -> Result<Vec<Event>, StoreError> {
        let conn = self.conn.clone();
        let run_id = run_id.to_owned();
        tokio::task::spawn_blocking(move || -> Result<Vec<Event>, StoreError> {
            let guard = conn.lock().map_err(|_| StoreError::LockPoisoned)?;
            let mut stmt =
                guard.prepare("SELECT payload FROM events WHERE run_id = ?1 ORDER BY seq ASC")?;
            let rows = stmt.query_map(params![run_id], |r| r.get::<_, String>(0))?;
            let mut out = Vec::new();
            for row in rows {
                let payload = row?;
                let event: Event = serde_json::from_str(&payload)?;
                out.push(event);
            }
            Ok(out)
        })
        .await?
    }
}

/// Return the discriminant name for an [`Event`] — the value persisted in the
/// `events.kind` column. Kept here (not on the type) because it's a
/// storage-layer concern, not part of the run-record's public API.
fn event_kind(event: &Event) -> &'static str {
    match event {
        Event::ModelCall { .. } => "ModelCall",
        Event::ToolCallStarted { .. } => "ToolCallStarted",
        Event::ToolCallResult { .. } => "ToolCallResult",
        Event::PhaseTransition { .. } => "PhaseTransition",
        Event::BudgetTick { .. } => "BudgetTick",
        Event::DispositionSet { .. } => "DispositionSet",
    }
}

/// Return a copy of `event` with its `seq` field overwritten. The store is the
/// authority on `seq`, so callers shouldn't have to guess; this is what makes
/// the trait's "store assigns the seq" contract enforceable.
fn with_seq(event: Event, seq: u64) -> Event {
    match event {
        Event::ModelCall {
            model,
            prompt_tokens,
            completion_tokens,
            ..
        } => Event::ModelCall {
            seq,
            model,
            prompt_tokens,
            completion_tokens,
        },
        Event::ToolCallStarted {
            name,
            args,
            call_id,
            ..
        } => Event::ToolCallStarted {
            seq,
            name,
            args,
            call_id,
        },
        Event::ToolCallResult {
            name,
            is_error,
            summary,
            offload_path,
            ..
        } => Event::ToolCallResult {
            seq,
            name,
            is_error,
            summary,
            offload_path,
        },
        Event::PhaseTransition { from, to, .. } => Event::PhaseTransition { seq, from, to },
        Event::BudgetTick { consumed, .. } => Event::BudgetTick { seq, consumed },
        Event::DispositionSet { disposition, .. } => Event::DispositionSet { seq, disposition },
    }
}

#[async_trait]
impl RunStore for SqliteRunStore {
    async fn load(&self, run_id: &str) -> Result<Option<RunRecord>, StoreError> {
        let conn = self.conn.clone();
        let run_id = run_id.to_owned();
        tokio::task::spawn_blocking(move || -> Result<Option<RunRecord>, StoreError> {
            let guard = conn.lock().map_err(|_| StoreError::LockPoisoned)?;
            let blob: Option<String> = guard
                .query_row(
                    "SELECT state_blob FROM runs WHERE run_id = ?1",
                    params![run_id],
                    |r| r.get::<_, String>(0),
                )
                .optional()?;
            match blob {
                Some(b) => Ok(Some(serde_json::from_str(&b)?)),
                None => Ok(None),
            }
        })
        .await?
    }

    async fn append_event(&self, run_id: &str, event: Event) -> Result<u64, StoreError> {
        let conn = self.conn.clone();
        let run_id = run_id.to_owned();
        tokio::task::spawn_blocking(move || -> Result<u64, StoreError> {
            let guard = conn.lock().map_err(|_| StoreError::LockPoisoned)?;
            // COALESCE(...) + 1 gives 0 for the empty-log case and one
            // greater than the current max otherwise. The PRIMARY KEY
            // (run_id, seq) backstops this — concurrent appenders on the
            // same connection would serialize through the Mutex, and the
            // table constraint catches any hypothetical race.
            let next_seq: i64 = guard.query_row(
                "SELECT COALESCE(MAX(seq), -1) + 1 FROM events WHERE run_id = ?1",
                params![run_id],
                |r| r.get::<_, i64>(0),
            )?;
            // next_seq is non-negative by construction (COALESCE default -1,
            // plus 1 == 0 minimum). The conversion can only fail on a
            // corrupted/manually-tampered schema, which is a hard invariant
            // violation; expect with a pointed message.
            let seq: u64 = u64::try_from(next_seq)
                .expect("events.seq must be non-negative (schema invariant)");
            let stamped = with_seq(event, seq);
            let kind = event_kind(&stamped);
            let payload = serde_json::to_string(&stamped)?;
            guard.execute(
                "INSERT INTO events (run_id, seq, ts, kind, payload)
                 VALUES (?1, ?2, datetime('now'), ?3, ?4)",
                params![run_id, next_seq, kind, payload],
            )?;
            Ok(seq)
        })
        .await?
    }

    async fn checkpoint(&self, run_id: &str, record: &RunRecord) -> Result<(), StoreError> {
        // Serialize on the calling task to avoid cloning the (potentially
        // large) RunRecord across the blocking-pool boundary.
        let blob = serde_json::to_string(record)?;
        let schema_version = record.schema_version;
        let conn = self.conn.clone();
        let run_id = run_id.to_owned();
        tokio::task::spawn_blocking(move || -> Result<(), StoreError> {
            let guard = conn.lock().map_err(|_| StoreError::LockPoisoned)?;
            guard.execute(
                "INSERT INTO runs (run_id, schema_version, state_blob, updated_at)
                 VALUES (?1, ?2, ?3, datetime('now'))
                 ON CONFLICT(run_id) DO UPDATE SET
                     schema_version = excluded.schema_version,
                     state_blob     = excluded.state_blob,
                     updated_at     = excluded.updated_at",
                params![run_id, schema_version, blob],
            )?;
            Ok(())
        })
        .await?
    }
}

// =======================================================================
// Tests
// =======================================================================

#[cfg(test)]
mod tests {
    use super::{RunStore, SqliteRunStore, StoreError, event_kind, with_seq};
    use crate::model::{ContentBlock, Message, ToolCallRequest, UserBlock};
    use crate::run_record::{
        AcceptanceCriterion, BudgetConsumed, BudgetLimits, Budgets, ChecklistItem, CriterionStatus,
        Disposition, DurableFacts, Event, Evidence, FailureMode, GateOutcome, GateResult, Phase,
        ProjectConfig, RunRecord, SCHEMA_VERSION, Task, Verification,
    };
    use std::collections::BTreeMap;

    // ---- fixtures ----

    fn sample_record(run_id: &str) -> RunRecord {
        let mut run_checks = BTreeMap::new();
        run_checks.insert("fmt".to_string(), "cargo fmt --check".to_string());
        run_checks.insert("test".to_string(), "cargo nextest run".to_string());

        RunRecord {
            run_id: run_id.to_string(),
            schema_version: SCHEMA_VERSION,
            attempt_n: 1,
            task: Task {
                task_id: "task-1".to_string(),
                title: "store smoke".to_string(),
                description: "Verify the SQLite store round-trips.".to_string(),
                acceptance_criteria: vec![AcceptanceCriterion {
                    id: "ac1".to_string(),
                    criterion: "round-trips".to_string(),
                    check: Some("cargo nextest run".to_string()),
                }],
                files_in_scope: vec!["crates/harness/src/store.rs".to_string()],
                scope_out: vec![],
            },
            project_config: ProjectConfig {
                run_checks,
                model_routing_hint: Some("sonnet".to_string()),
            },
            phase: Phase::InnerLoop,
            durable_facts: DurableFacts {
                checklist: vec![ChecklistItem {
                    id: "ac1".to_string(),
                    criterion: "round-trips".to_string(),
                    status: CriterionStatus::Verified(Evidence::Test {
                        name: "checkpoint_load_round_trip".to_string(),
                        command: "cargo nextest run".to_string(),
                    }),
                }],
                findings: vec!["chose rusqlite 0.31 for license compliance".to_string()],
            },
            budgets: Budgets {
                consumed: BudgetConsumed {
                    iterations: 3,
                    tokens: 1000,
                    cost_micros: 50,
                },
                limits: BudgetLimits {
                    iterations: 100,
                    tokens: 1_000_000,
                    cost_micros: 10_000_000,
                },
                wall_clock_start: "2026-06-22T00:00:00Z".to_string(),
            },
            last_gate_result: Some(GateResult {
                passed: true,
                gates: {
                    let mut g = BTreeMap::new();
                    g.insert(
                        "fmt".to_string(),
                        GateOutcome {
                            passed: true,
                            summary: "ok".to_string(),
                            failure_extract: None,
                        },
                    );
                    g
                },
            }),
            disposition: None,
            messages: vec![
                Message::User {
                    content: vec![UserBlock::Text("do the task".to_string())],
                },
                Message::Assistant {
                    content: vec![ContentBlock::ToolCall(ToolCallRequest {
                        id: "c1".to_string(),
                        name: "read_file".to_string(),
                        input: serde_json::json!({ "path": "src/lib.rs" }),
                    })],
                },
            ],
        }
    }

    fn modify(mut r: RunRecord) -> RunRecord {
        r.phase = Phase::Checks;
        r.durable_facts
            .findings
            .push("second checkpoint".to_string());
        r.budgets.consumed.iterations = 99;
        r.disposition = Some(Disposition::Failed {
            mode: FailureMode::Loop,
            summary: "spinning".to_string(),
        });
        r
    }

    // ---- load / checkpoint ----

    #[tokio::test]
    async fn load_returns_none_for_unknown_run() {
        let store = SqliteRunStore::open_in_memory().expect("open");
        let loaded = store.load("never-checkpointed").await.expect("load");
        assert!(loaded.is_none());
    }

    #[tokio::test]
    async fn checkpoint_then_load_round_trips() {
        let store = SqliteRunStore::open_in_memory().expect("open");
        let r = sample_record("run-a");
        store.checkpoint("run-a", &r).await.expect("checkpoint");
        let loaded = store.load("run-a").await.expect("load").expect("present");
        assert_eq!(loaded, r);
    }

    #[tokio::test]
    async fn checkpoint_upserts_latest_snapshot() {
        let store = SqliteRunStore::open_in_memory().expect("open");
        let r1 = sample_record("run-a");
        store
            .checkpoint("run-a", &r1)
            .await
            .expect("first checkpoint");

        let r2 = modify(r1);
        store
            .checkpoint("run-a", &r2)
            .await
            .expect("second checkpoint");

        let loaded = store.load("run-a").await.expect("load").expect("present");
        assert_eq!(loaded, r2, "load should see the most recent snapshot");
    }

    // ---- append_event seq behavior ----

    #[tokio::test]
    async fn append_event_assigns_monotonic_seq_per_run() {
        let store = SqliteRunStore::open_in_memory().expect("open");

        let s0 = store
            .append_event(
                "run-a",
                Event::ModelCall {
                    seq: 999,
                    model: "haiku".to_string(),
                    prompt_tokens: 10,
                    completion_tokens: 5,
                },
            )
            .await
            .expect("append");
        let s1 = store
            .append_event(
                "run-a",
                Event::PhaseTransition {
                    seq: 999,
                    from: Phase::Init,
                    to: Phase::Orient,
                },
            )
            .await
            .expect("append");
        let s2 = store
            .append_event(
                "run-a",
                Event::BudgetTick {
                    seq: 999,
                    consumed: BudgetConsumed {
                        iterations: 1,
                        tokens: 100,
                        cost_micros: 10,
                    },
                },
            )
            .await
            .expect("append");

        assert_eq!((s0, s1, s2), (0, 1, 2), "per-run seq starts at 0 and ticks");
    }

    #[tokio::test]
    async fn append_event_seq_is_isolated_across_runs() {
        let store = SqliteRunStore::open_in_memory().expect("open");

        let a0 = store
            .append_event(
                "run-a",
                Event::ModelCall {
                    seq: 0,
                    model: "haiku".to_string(),
                    prompt_tokens: 1,
                    completion_tokens: 1,
                },
            )
            .await
            .expect("append");
        let a1 = store
            .append_event(
                "run-a",
                Event::ModelCall {
                    seq: 0,
                    model: "haiku".to_string(),
                    prompt_tokens: 2,
                    completion_tokens: 2,
                },
            )
            .await
            .expect("append");

        let b0 = store
            .append_event(
                "run-b",
                Event::ModelCall {
                    seq: 0,
                    model: "haiku".to_string(),
                    prompt_tokens: 3,
                    completion_tokens: 3,
                },
            )
            .await
            .expect("append");
        let b1 = store
            .append_event(
                "run-b",
                Event::ModelCall {
                    seq: 0,
                    model: "haiku".to_string(),
                    prompt_tokens: 4,
                    completion_tokens: 4,
                },
            )
            .await
            .expect("append");

        assert_eq!((a0, a1, b0, b1), (0, 1, 0, 1));
    }

    #[tokio::test]
    async fn events_read_back_in_seq_order_with_assigned_seq() {
        let store = SqliteRunStore::open_in_memory().expect("open");

        // Insert in a non-trivial order; the store's monotonic seq should
        // dictate read-back order regardless of the seq field on the input.
        store
            .append_event(
                "run-a",
                Event::ModelCall {
                    seq: 42, // deliberately wrong — store overwrites it
                    model: "sonnet".to_string(),
                    prompt_tokens: 1,
                    completion_tokens: 1,
                },
            )
            .await
            .expect("append");
        store
            .append_event(
                "run-a",
                Event::ToolCallStarted {
                    seq: 42,
                    name: "edit_file".to_string(),
                    args: serde_json::json!({"path": "src/lib.rs"}),
                    call_id: "c-edit".to_string(),
                },
            )
            .await
            .expect("append");
        store
            .append_event(
                "run-a",
                Event::ToolCallResult {
                    seq: 42,
                    name: "edit_file".to_string(),
                    is_error: false,
                    summary: "ok".to_string(),
                    offload_path: None,
                },
            )
            .await
            .expect("append");
        store
            .append_event(
                "run-a",
                Event::DispositionSet {
                    seq: 42,
                    disposition: Disposition::Done {
                        summary: "done".to_string(),
                        verification: Verification::NoChecksConfigured,
                    },
                },
            )
            .await
            .expect("append");

        let events = store.list_events("run-a").await.expect("list");
        assert_eq!(events.len(), 4, "all four events should be present");

        // The store rewrites seq to 0..N; payloads echo that.
        let seqs: Vec<u64> = events
            .iter()
            .map(|e| match e {
                Event::ModelCall { seq, .. }
                | Event::ToolCallStarted { seq, .. }
                | Event::ToolCallResult { seq, .. }
                | Event::PhaseTransition { seq, .. }
                | Event::BudgetTick { seq, .. }
                | Event::DispositionSet { seq, .. } => *seq,
            })
            .collect();
        assert_eq!(seqs, vec![0, 1, 2, 3], "payloads carry the assigned seq");

        // And the kinds come back in the order they were appended.
        let kinds: Vec<&'static str> = events.iter().map(event_kind).collect();
        assert_eq!(
            kinds,
            vec![
                "ModelCall",
                "ToolCallStarted",
                "ToolCallResult",
                "DispositionSet",
            ]
        );
    }

    #[tokio::test]
    async fn list_events_isolated_by_run_id() {
        let store = SqliteRunStore::open_in_memory().expect("open");
        store
            .append_event(
                "run-a",
                Event::PhaseTransition {
                    seq: 0,
                    from: Phase::Init,
                    to: Phase::Orient,
                },
            )
            .await
            .expect("append");
        store
            .append_event(
                "run-b",
                Event::PhaseTransition {
                    seq: 0,
                    from: Phase::Init,
                    to: Phase::InnerLoop,
                },
            )
            .await
            .expect("append");

        let a = store.list_events("run-a").await.expect("list a");
        let b = store.list_events("run-b").await.expect("list b");

        assert_eq!(a.len(), 1);
        assert_eq!(b.len(), 1);
        assert_ne!(a, b, "different runs see different events");

        let empty = store.list_events("run-c").await.expect("list c");
        assert!(empty.is_empty());
    }

    // ---- resume after store drop ----

    #[tokio::test]
    async fn resume_after_drop_reloads_state() {
        let tmpdir = tempfile::tempdir().expect("tmpdir");
        let path = tmpdir.path().join("runs.sqlite");

        let r = sample_record("run-a");

        // First "process": open, checkpoint, append a couple events, drop.
        {
            let store = SqliteRunStore::open(&path).expect("open");
            store.checkpoint("run-a", &r).await.expect("checkpoint");
            store
                .append_event(
                    "run-a",
                    Event::ModelCall {
                        seq: 0,
                        model: "haiku".to_string(),
                        prompt_tokens: 10,
                        completion_tokens: 5,
                    },
                )
                .await
                .expect("append");
            store
                .append_event(
                    "run-a",
                    Event::PhaseTransition {
                        seq: 0,
                        from: Phase::Init,
                        to: Phase::Orient,
                    },
                )
                .await
                .expect("append");
        }

        // Second "process": reopen, reload, and assert the snapshot survived
        // intact + the per-run seq counter continues from where it left off.
        {
            let store = SqliteRunStore::open(&path).expect("reopen");
            let loaded = store.load("run-a").await.expect("load").expect("present");
            assert_eq!(loaded, r, "snapshot survives a fresh open of the file");

            let events = store.list_events("run-a").await.expect("list");
            assert_eq!(events.len(), 2, "event log survives the drop");

            let next = store
                .append_event(
                    "run-a",
                    Event::PhaseTransition {
                        seq: 0,
                        from: Phase::Orient,
                        to: Phase::InnerLoop,
                    },
                )
                .await
                .expect("append");
            assert_eq!(next, 2, "seq continues from the persisted max+1");
        }
    }

    #[tokio::test]
    async fn reopen_is_idempotent_for_schema_init() {
        let tmpdir = tempfile::tempdir().expect("tmpdir");
        let path = tmpdir.path().join("runs.sqlite");

        // Open and close twice — `CREATE TABLE IF NOT EXISTS` keeps init
        // safe across re-opens; this asserts we don't error on the second
        // open.
        let _ = SqliteRunStore::open(&path).expect("first open");
        let _ = SqliteRunStore::open(&path).expect("second open");
    }

    // ---- error surfaces ----

    #[tokio::test]
    async fn load_surfaces_serialization_error_on_corrupt_blob() {
        // Manually corrupt the snapshot blob to drive the deserialization
        // branch of `load` — proves the error path is wired through.
        let store = SqliteRunStore::open_in_memory().expect("open");
        {
            let conn = store.conn.lock().expect("lock");
            conn.execute(
                "INSERT INTO runs (run_id, schema_version, state_blob, updated_at)
                 VALUES (?1, 1, ?2, datetime('now'))",
                rusqlite::params!["run-bad", "{not valid json"],
            )
            .expect("insert");
        }
        let err = store.load("run-bad").await.expect_err("should error");
        assert!(
            matches!(err, StoreError::Serialization(_)),
            "got {err:?}, expected Serialization"
        );
        // Display + source impls are exercised here too.
        let display = format!("{err}");
        assert!(
            display.contains("serialization error"),
            "Display: {display}"
        );
        assert!(
            std::error::Error::source(&err).is_some(),
            "Serialization error should carry a source"
        );
    }

    #[tokio::test]
    async fn list_events_surfaces_serialization_error_on_corrupt_payload() {
        let store = SqliteRunStore::open_in_memory().expect("open");
        {
            let conn = store.conn.lock().expect("lock");
            conn.execute(
                "INSERT INTO events (run_id, seq, ts, kind, payload)
                 VALUES ('run-x', 0, datetime('now'), 'ModelCall', '{not json')",
                [],
            )
            .expect("insert");
        }
        let err = store.list_events("run-x").await.expect_err("should error");
        assert!(matches!(err, StoreError::Serialization(_)));
    }

    #[tokio::test]
    async fn open_surfaces_sql_error_for_bad_path() {
        // A path inside a non-existent parent directory is a deterministic
        // way to drive the `Sql` branch of `open`.
        let err =
            SqliteRunStore::open("/this/path/does/not/exist/run.sqlite").expect_err("should fail");
        assert!(matches!(err, StoreError::Sql(_)), "got {err:?}");
        let msg = format!("{err}");
        assert!(msg.contains("sql error"), "Display: {msg}");
        // Source link present.
        assert!(std::error::Error::source(&err).is_some());
    }

    #[test]
    fn store_error_display_covers_all_variants() {
        // Build one of each via the From impls / direct construction.
        let sql: StoreError = rusqlite::Error::QueryReturnedNoRows.into();
        assert!(format!("{sql}").contains("sql error"));

        let json_err = serde_json::from_str::<i32>("not json").unwrap_err();
        let ser: StoreError = json_err.into();
        assert!(format!("{ser}").contains("serialization error"));

        let join = StoreError::Join("oops".to_string());
        assert!(format!("{join}").contains("background task error"));
        assert!(std::error::Error::source(&join).is_none());

        let poisoned = StoreError::LockPoisoned;
        assert!(format!("{poisoned}").contains("poisoned"));
        assert!(std::error::Error::source(&poisoned).is_none());

        // Debug impl is derived — touch it so coverage sees the format call.
        let _ = format!("{poisoned:?}");
    }

    // ---- helpers ----

    #[test]
    fn event_kind_returns_variant_name() {
        let cases: &[(Event, &str)] = &[
            (
                Event::ModelCall {
                    seq: 0,
                    model: "m".to_string(),
                    prompt_tokens: 0,
                    completion_tokens: 0,
                },
                "ModelCall",
            ),
            (
                Event::ToolCallStarted {
                    seq: 0,
                    name: "t".to_string(),
                    args: serde_json::Value::Null,
                    call_id: "c-t".to_string(),
                },
                "ToolCallStarted",
            ),
            (
                Event::ToolCallResult {
                    seq: 0,
                    name: "t".to_string(),
                    is_error: false,
                    summary: String::new(),
                    offload_path: None,
                },
                "ToolCallResult",
            ),
            (
                Event::PhaseTransition {
                    seq: 0,
                    from: Phase::Init,
                    to: Phase::Orient,
                },
                "PhaseTransition",
            ),
            (
                Event::BudgetTick {
                    seq: 0,
                    consumed: BudgetConsumed::default(),
                },
                "BudgetTick",
            ),
            (
                Event::DispositionSet {
                    seq: 0,
                    disposition: Disposition::Done {
                        summary: "done".to_string(),
                        verification: Verification::NoChecksConfigured,
                    },
                },
                "DispositionSet",
            ),
        ];
        for (ev, expected) in cases {
            assert_eq!(event_kind(ev), *expected);
        }
    }

    #[test]
    fn with_seq_overwrites_seq_for_every_variant() {
        let cases = [
            Event::ModelCall {
                seq: 99,
                model: "m".to_string(),
                prompt_tokens: 1,
                completion_tokens: 2,
            },
            Event::ToolCallStarted {
                seq: 99,
                name: "t".to_string(),
                args: serde_json::json!({"x": 1}),
                call_id: "c-with-seq".to_string(),
            },
            Event::ToolCallResult {
                seq: 99,
                name: "t".to_string(),
                is_error: true,
                summary: "boom".to_string(),
                offload_path: Some("/tmp/x".to_string()),
            },
            Event::PhaseTransition {
                seq: 99,
                from: Phase::Init,
                to: Phase::Orient,
            },
            Event::BudgetTick {
                seq: 99,
                consumed: BudgetConsumed {
                    iterations: 1,
                    tokens: 1,
                    cost_micros: 1,
                },
            },
            Event::DispositionSet {
                seq: 99,
                disposition: Disposition::Blocked {
                    decision_needed: "?".to_string(),
                },
            },
        ];
        for ev in cases {
            let stamped = with_seq(ev, 7);
            let new_seq = match &stamped {
                Event::ModelCall { seq, .. }
                | Event::ToolCallStarted { seq, .. }
                | Event::ToolCallResult { seq, .. }
                | Event::PhaseTransition { seq, .. }
                | Event::BudgetTick { seq, .. }
                | Event::DispositionSet { seq, .. } => *seq,
            };
            assert_eq!(new_seq, 7);
        }
    }
}
