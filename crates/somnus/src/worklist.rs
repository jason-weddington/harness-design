//! The `GET /api/kb/map-worklist` client, the nightly selection policy, and
//! the nightly invocation record.
//!
//! This endpoint exists BECAUSE the binary needs it: the machine principal
//! is deliberately non-admin, so the admin-gated `GET /api/kb/map-eligibility`
//! is unreachable and there is no non-admin enumeration of eligible projects
//! without it. The payload is LEAD-SUPPLIED GROUND TRUTH from the
//! endpoint's author — somnus CANNOT verify it against anything in this
//! clone (recorded in the item's grounding notes); it is treated as
//! authoritative and the field spellings are never weakened to a guess.
//!
//! The server answers ALREADY RANKED (never-mapped first, then oldest
//! `latest_map_written_at`, then `project_ref` ascending) and returns only
//! ELIGIBLE projects with overrides respected. Selection here is VERBATIM
//! server order — no sorting, no re-ranking, no most-unpointed-first
//! heuristic, no row filter. Ranking lives server-side because it needs
//! map-write history somnus does not hold; a copy of the rule here is one
//! more thing to drift.
//!
//! WHY take the first three and not re-sort: the obvious heuristic,
//! most-unpointed-first, STARVES — a project whose unpointed entries all
//! sit in declined clusters tops the list every night forever and blocks
//! every other project, and declining a cluster does not make its members
//! pointed. Staleness-first rotates whether or not a given night
//! accomplished anything.
//!
//! Layering (pinned): this module references [`crate::unit`] (for the
//! outcome type the invocation record carries verbatim) and nothing else
//! above it.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::materialize::SOMNUS_MAX_PROJECTS_PER_NIGHT;

/// Wall-clock bound on the worklist GET, mirroring
/// [`crate::ledger::LEDGER_HTTP_BOUND`].
pub const WORKLIST_HTTP_BOUND: Duration = Duration::from_secs(10);

/// The worklist endpoint's pinned path.
pub const MAP_WORKLIST_PATH: &str = "/api/kb/map-worklist";

/// The pinned 401 line: a configuration fault (exit 1) — the token is
/// wrong, and burning three metered units against a rejecting KB would be
/// the defect.
pub const WORKLIST_UNAUTHORIZED_MSG: &str =
    "somnus: /api/kb/map-worklist rejected the bearer token (401); check SOMNUS_KB_API_KEY";

/// One worklist row. Field spellings are LEAD-SUPPLIED GROUND TRUTH
/// (unverifiable from this clone; see the module doc) and deserialized
/// verbatim — including a JSON `null` `latest_map_written_at`, which a
/// never-mapped project carries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorklistProject {
    /// The project ref.
    pub project_ref: String,
    /// How many of the project's entries are mappable — a COUNT, not a
    /// flag.
    ///
    /// It was typed `bool` here under the name `mappable`, which reads as a
    /// predicate: *is this project mappable?* The spec said non-negative
    /// integer in as many words, and the type was never ambiguous in
    /// writing — the name fought the spec and the name won, so the first
    /// supervised nightly aborted on `invalid type: integer 26, expected a
    /// boolean`. The producer renamed the field rather than making the
    /// documentation firmer, which is the same move as putting the pointers
    /// in `create_map`'s signature: make the wrong reading unrepresentable
    /// instead of arguing against it. A field name is part of the shape a
    /// consumer expects, and shape outranks prose.
    pub mappable_entries: u64,
    /// How many maps the project already has.
    pub map_count: u64,
    /// When the last map was written to the project, if ever. A
    /// never-mapped project carries JSON null, tolerated here so one null
    /// never fails the whole response.
    pub latest_map_written_at: Option<String>,
}

/// The response envelope: `{"projects": [...]}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct WorklistResponse {
    projects: Vec<WorklistProject>,
}

/// One worklist fetch outcome.
#[derive(Debug, Clone, PartialEq)]
pub enum WorklistOutcome {
    /// `200` with a parseable body: the projects, in server order.
    Ready(Vec<WorklistProject>),
    /// `401` — a configuration fault, not a run fault.
    Unauthorized,
    /// Any other status, a transport failure, or an unparsable body.
    Fault {
        /// Why the fetch failed.
        reason: String,
    },
}

/// Fetch the worklist: exactly ONE `GET {base_url}{MAP_WORKLIST_PATH}` with
/// `Authorization: Bearer <token>` and no other headers, no retry.
pub async fn fetch_worklist(base_url: &str, token: &str) -> WorklistOutcome {
    let url = format!("{}{MAP_WORKLIST_PATH}", base_url.trim_end_matches('/'));
    let exchange = async {
        let response = match reqwest::Client::new()
            .get(&url)
            .header(reqwest::header::AUTHORIZATION, format!("Bearer {token}"))
            .send()
            .await
        {
            Ok(response) => response,
            Err(err) => {
                return WorklistOutcome::Fault {
                    reason: format!("transport failed: {err}"),
                };
            }
        };
        let status = response.status().as_u16();
        let body = match response.text().await {
            Ok(body) => body,
            Err(err) => {
                return WorklistOutcome::Fault {
                    reason: format!("body read failed: {err}"),
                };
            }
        };
        match status {
            200 => match serde_json::from_str::<WorklistResponse>(&body) {
                Ok(response) => WorklistOutcome::Ready(response.projects),
                Err(err) => WorklistOutcome::Fault {
                    reason: format!("malformed body ({err})"),
                },
            },
            401 => WorklistOutcome::Unauthorized,
            status => WorklistOutcome::Fault {
                reason: format!("unexpected HTTP status {status}"),
            },
        }
    };
    match tokio::time::timeout(WORKLIST_HTTP_BOUND, exchange).await {
        Ok(outcome) => outcome,
        Err(_elapsed) => WorklistOutcome::Fault {
            reason: format!("timed out after {}s", WORKLIST_HTTP_BOUND.as_secs()),
        },
    }
}

/// The stderr line for any non-401 worklist failure (a run fault, exit 2).
/// Pure so the shape is byte-pinned; the binary only `eprintln!`s it.
#[must_use]
pub fn render_worklist_fault_line(reason: &str) -> String {
    format!("somnus: map-worklist fetch failed: {reason}")
}

/// Select the nightly projects: the FIRST THREE rows in verbatim server
/// order — no sorting, no re-ranking, no most-unpointed-first heuristic,
/// and no row filter (a row with zero mappable entries, if the server ever
/// sends one, is still taken and consumes a slot; filtering is the server's job
/// because `/api/kb/map-worklist` IS the eligibility surface). See the
/// module doc for why the heuristic is refused.
#[must_use]
pub fn select_first_projects(projects: &[WorklistProject]) -> Vec<String> {
    projects
        .iter()
        .take(SOMNUS_MAX_PROJECTS_PER_NIGHT as usize)
        .map(|project| project.project_ref.clone())
        .collect()
}

// ===== The nightly invocation record ======================================

/// How one unit of a nightly invocation ended, as the record carries it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NightlyUnitRecord {
    /// The unit's project.
    pub project_ref: String,
    /// The [`UnitOutcome`](crate::unit::UnitOutcome) verbatim — it serializes
    /// verbatim (no serde renames).
    pub outcome: crate::unit::UnitOutcome,
    /// The unit's billed-token sum (rung 1 + rung 2).
    pub billed_tokens: u64,
    /// How many maps the unit actually created.
    ///
    /// **`Ready` and `all_done` say the loop got through its work, NOT that
    /// any work happened.** A project whose only cluster sits below the size
    /// floor is worked correctly, mints nothing, and ends `Ready` — which is
    /// right, since a quiet night is a success by contract. But an operator
    /// reading the invocation record could not tell that project from one
    /// that minted six maps without opening each per-project report, and the
    /// invocation record is the artifact a human actually reads at 3am.
    ///
    /// Recorded alongside the outcome rather than folded into it: the
    /// outcome answers "did anything fail", this answers "did anything
    /// happen", and collapsing the two would lose one of them.
    pub maps_created: usize,
    /// Where the unit's run report was written.
    pub report_path: PathBuf,
}

/// How many maps a unit created, from its applied-op record.
#[must_use]
pub fn maps_created(report: &crate::unit::UnitReport) -> usize {
    report
        .applied
        .iter()
        .filter(|applied| applied.op_kind == "create_map")
        .count()
}

/// The nightly invocation record: written to
/// `<state_dir>/nightly-invocation.json` before EVERY nightly exit,
/// whatever the exit code, so a wrong selection, a budget stop, an empty
/// night, and a crash are distinguishable post-hoc from the state dir
/// alone. A pure `Serialize` struct — no store, no database.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NightlyInvocation {
    /// The binary that produced this record ([`crate::SOMNUS_VERSION`]).
    pub somnus_version: String,
    /// When the invocation started, as a u64 UNIX epoch in seconds (passed
    /// in by the binary from `SystemTime::now`).
    pub invoked_at_utc: u64,
    /// The subcommand that ran (always `nightly` today).
    pub subcommand: String,
    /// The armed token ceiling (the sentinel 0 = unbounded).
    pub armed_budget: u64,
    /// The verbatim deserialized worklist array, in server order.
    pub worklist_projects: Vec<WorklistProject>,
    /// The selected refs, in server order (the take-three slice).
    pub selected_refs: Vec<String>,
    /// One record per unit that RAN (a budget-stopped project gets no
    /// entry: no unit, no report).
    pub units: Vec<NightlyUnitRecord>,
    /// Why the invocation stopped: [`STOP_REASON_ALL_DONE`],
    /// [`STOP_REASON_NOTHING_ELIGIBLE`], `budget_stopped_before:<ref>`, or
    /// `fault:<exit_code>`.
    pub stop_reason: String,
    /// The process exit code.
    pub exit_code: i32,
}

/// Stop reason: every selected unit ran to an ordinary end.
pub const STOP_REASON_ALL_DONE: &str = "all_done";

/// Stop reason: the worklist was empty.
pub const STOP_REASON_NOTHING_ELIGIBLE: &str = "nothing_eligible";

/// The `budget_stopped_before:<ref>` stop reason for `skipped_ref`.
#[must_use]
pub fn budget_stopped_before(skipped_ref: &str) -> String {
    format!("budget_stopped_before:{skipped_ref}")
}

/// The `fault:<exit_code>` stop reason for `exit_code`.
#[must_use]
pub fn fault_stop_reason(exit_code: i32) -> String {
    format!("fault:{exit_code}")
}

/// Build the invocation record (a pure builder so the shape is assembled in
/// one tested place). The pinned record shape carries all eight fields, so
/// the wide parameter list is the shape, not an accident.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn build_invocation(
    invoked_at_utc: u64,
    subcommand: &str,
    armed_budget: u64,
    worklist_projects: Vec<WorklistProject>,
    selected_refs: Vec<String>,
    units: Vec<NightlyUnitRecord>,
    stop_reason: String,
    exit_code: i32,
) -> NightlyInvocation {
    NightlyInvocation {
        somnus_version: crate::SOMNUS_VERSION.to_string(),
        invoked_at_utc,
        subcommand: subcommand.to_string(),
        armed_budget,
        worklist_projects,
        selected_refs,
        units,
        stop_reason,
        exit_code,
    }
}

/// Write the invocation record to `<state_dir>/nightly-invocation.json`.
///
/// # Errors
/// Any filesystem failure (unwritable state dir).
///
/// # Panics
/// Panics if the record fails to serialize — impossible by construction
/// (every field is `Serialize` with no renames).
pub fn write_invocation(state_dir: &Path, record: &NightlyInvocation) -> std::io::Result<()> {
    let text = serde_json::to_string_pretty(record).expect("the record serializes by construction");
    std::fs::write(state_dir.join("nightly-invocation.json"), text)
}

// ===== The pinned nightly stderr lines ====================================

/// The pinned per-project nightly line. Pure so the shape is byte-pinned;
/// the binary only `eprintln!`s it.
#[must_use]
pub fn render_nightly_unit_line(project_ref: &str, outcome: &crate::unit::UnitOutcome) -> String {
    format!(
        "somnus: nightly unit for {project_ref} ended {}",
        outcome.name()
    )
}

/// The pinned empty-night line: nonzero must never mean "nothing was
/// eligible".
pub const NOTHING_ELIGIBLE_MSG: &str = "somnus: nothing eligible tonight";

/// The pinned across-units budget-stop line, carrying BOTH decision inputs
/// (the armed ceiling and the invocation's billed total) and naming the
/// project that will NOT run.
#[must_use]
pub fn render_nightly_budget_stop_line(armed: u64, billed: u64, skipped_ref: &str) -> String {
    format!(
        "somnus: token budget exhausted (armed {armed}, billed {billed}); stopping before {skipped_ref}"
    )
}

// ===== tests ==============================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// A row fixture.
    fn row(
        project_ref: &str,
        mappable_entries: u64,
        map_count: u64,
        latest_map_written_at: Option<&str>,
    ) -> WorklistProject {
        WorklistProject {
            project_ref: project_ref.to_string(),
            mappable_entries,
            map_count,
            latest_map_written_at: latest_map_written_at.map(str::to_string),
        }
    }

    #[test]
    fn the_payload_parses_the_pinned_field_spellings_tolerating_null() {
        let text = r#"{"projects":[
            {"project_ref":"never-mapped","mappable_entries":26,"map_count":0,"latest_map_written_at":null},
            {"project_ref":"stale","mappable_entries":26,"map_count":2,"latest_map_written_at":"2026-09-19T03:14:15Z"}
        ]}"#;
        let parsed: WorklistResponse =
            serde_json::from_str(text).expect("the pinned spellings parse");
        assert_eq!(
            parsed.projects,
            vec![
                row("never-mapped", 26, 0, None),
                row("stale", 26, 2, Some("2026-09-19T03:14:15Z")),
            ]
        );
    }

    #[test]
    fn a_malformed_row_fails_the_whole_response_into_the_fault_arm() {
        let server_ready = r#"{"projects":[{"project_ref":"x"}]}"#;
        assert!(serde_json::from_str::<WorklistResponse>(server_ready).is_err());
    }

    // --- the selection: verbatim server order, take three, no filter -----

    #[test]
    fn the_first_three_are_taken_verbatim_with_no_sort_and_no_filter() {
        // A fixture whose order would change under ANY known heuristic: the
        // never-mapped project is listed AFTER an oldest-mapped one, and a
        // a zero-mappable-entries row sits in the taken slice (the policy is
        // tested, not implied).
        let projects = vec![
            row("oldest-mapped", 26, 5, Some("2020-01-01T00:00:00Z")),
            row("not-mappable", 0, 0, None),
            row("mid", 26, 1, Some("2024-01-01T00:00:00Z")),
            row("never-mapped-first-under-any-heuristic", 26, 0, None),
        ];
        assert_eq!(
            select_first_projects(&projects),
            vec![
                "oldest-mapped".to_string(),
                "not-mappable".to_string(),
                "mid".to_string(),
            ]
        );
    }

    #[test]
    fn fewer_than_three_rows_are_all_taken() {
        let projects = vec![row("only", 26, 0, None)];
        assert_eq!(select_first_projects(&projects), vec!["only".to_string()]);
        assert!(select_first_projects(&[]).is_empty());
    }

    // --- the transport -----------------------------------------------------

    async fn mount(server: &wiremock::MockServer, status: u16, body: &str) {
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path(MAP_WORKLIST_PATH))
            .and(wiremock::matchers::header(
                "Authorization",
                "Bearer kb-token",
            ))
            .respond_with(wiremock::ResponseTemplate::new(status).set_body_string(body.to_string()))
            .mount(server)
            .await;
    }

    const WORKLIST_BODY: &str = r#"{"projects":[{"project_ref":"a","mappable_entries":26,"map_count":0,"latest_map_written_at":null}]}"#;

    #[tokio::test]
    async fn a_200_parses_the_projects_in_server_order() {
        let server = wiremock::MockServer::start().await;
        mount(&server, 200, WORKLIST_BODY).await;
        assert_eq!(
            fetch_worklist(&server.uri(), "kb-token").await,
            WorklistOutcome::Ready(vec![row("a", 26, 0, None)])
        );
        assert_eq!(server.received_requests().await.expect("captured").len(), 1);
    }

    #[tokio::test]
    async fn a_401_maps_to_the_configuration_fault() {
        let server = wiremock::MockServer::start().await;
        mount(&server, 401, "denied").await;
        assert_eq!(
            fetch_worklist(&server.uri(), "kb-token").await,
            WorklistOutcome::Unauthorized
        );
    }

    #[tokio::test]
    async fn any_other_status_or_garbage_maps_to_the_run_fault() {
        for (status, body) in [(500u16, "boom"), (409u16, "x"), (200u16, "garbage")] {
            let server = wiremock::MockServer::start().await;
            mount(&server, status, body).await;
            let WorklistOutcome::Fault { reason } = fetch_worklist(&server.uri(), "kb-token").await
            else {
                panic!("({status}, {body:?}) must fault");
            };
            assert!(!reason.is_empty());
        }
        assert_eq!(
            render_worklist_fault_line("unexpected HTTP status 500"),
            "somnus: map-worklist fetch failed: unexpected HTTP status 500"
        );
    }

    #[tokio::test]
    async fn a_failed_body_read_maps_to_the_run_fault_form() {
        // A one-shot raw TCP server that answers 200 with a
        // `content-length` far larger than the body it writes, then drops
        // the connection — the body read fails into the SAME run-fault form.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind a port");
        let address = listener.local_addr().expect("local address");
        let handle = std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                use std::io::Read;
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf);
                let _ = std::io::Write::write_all(
                    &mut stream,
                    b"HTTP/1.1 200 OK\r\ncontent-length: 128\r\n\r\nshort",
                );
            }
        });
        let WorklistOutcome::Fault { reason } =
            fetch_worklist(&format!("http://{address}"), "kb-token").await
        else {
            panic!("expected the run fault");
        };
        assert!(
            reason.starts_with("body read failed: "),
            "reason was {reason}"
        );
        handle.join().expect("the raw server thread");
    }

    #[tokio::test]
    async fn a_dropped_server_maps_to_the_run_fault() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind a port");
        let address = listener.local_addr().expect("local address");
        let server = wiremock::MockServer::builder()
            .listener(listener)
            .start()
            .await;
        let uri = server.uri();
        drop(server);
        let mut closed = false;
        for _ in 0..200 {
            if std::net::TcpStream::connect_timeout(&address, std::time::Duration::from_millis(25))
                .is_err()
            {
                closed = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert!(closed, "the dropped server's listener must close");
        let WorklistOutcome::Fault { reason } = fetch_worklist(&uri, "kb-token").await else {
            panic!("expected the run fault");
        };
        assert!(
            reason.starts_with("transport failed: "),
            "reason was {reason}"
        );
    }

    // --- the invocation record ----------------------------------------------

    #[test]
    fn the_invocation_record_serializes_the_pinned_shape() {
        let record = build_invocation(
            1_760_000_000,
            "nightly",
            550_000,
            vec![row("a", 26, 0, None)],
            vec!["a".to_string()],
            vec![NightlyUnitRecord {
                project_ref: "a".to_string(),
                outcome: crate::unit::UnitOutcome::Aborted {
                    reason: "boom".to_string(),
                },
                billed_tokens: 12_345,
                maps_created: 0,
                report_path: PathBuf::from("/state/a/run-report.json"),
            }],
            STOP_REASON_ALL_DONE.to_string(),
            2,
        );
        let text = serde_json::to_string_pretty(&record).expect("serializes");
        let value: serde_json::Value = serde_json::from_str(&text).expect("parses");
        assert_eq!(value["invoked_at_utc"], 1_760_000_000);
        assert_eq!(value["subcommand"], "nightly");
        assert_eq!(value["armed_budget"], 550_000);
        assert_eq!(value["worklist_projects"][0]["project_ref"], "a");
        assert_eq!(
            value["worklist_projects"][0]["latest_map_written_at"],
            serde_json::Value::Null
        );
        assert_eq!(value["selected_refs"], serde_json::json!(["a"]));
        assert_eq!(
            value["units"][0]["outcome"],
            serde_json::json!({"Aborted": {"reason": "boom"}}),
            "the UnitOutcome serializes verbatim, PascalCase, no rename_all"
        );
        assert_eq!(value["units"][0]["billed_tokens"], 12_345);
        assert_eq!(value["stop_reason"], "all_done");
        assert_eq!(value["exit_code"], 2);
        for field in [
            "invoked_at_utc",
            "subcommand",
            "armed_budget",
            "worklist_projects",
            "selected_refs",
            "units",
            "stop_reason",
            "exit_code",
        ] {
            assert!(value.get(field).is_some(), "{field} must be present");
        }
    }

    #[test]
    fn the_stop_reasons_render_the_pinned_literals() {
        assert_eq!(STOP_REASON_ALL_DONE, "all_done");
        assert_eq!(STOP_REASON_NOTHING_ELIGIBLE, "nothing_eligible");
        assert_eq!(
            budget_stopped_before("proj-b"),
            "budget_stopped_before:proj-b"
        );
        assert_eq!(fault_stop_reason(2), "fault:2");
    }

    #[test]
    fn the_nightly_lines_are_byte_pinned() {
        assert_eq!(
            render_nightly_unit_line("proj-a", &crate::unit::UnitOutcome::Ready),
            "somnus: nightly unit for proj-a ended Ready"
        );
        assert_eq!(
            render_nightly_unit_line(
                "proj-b",
                &crate::unit::UnitOutcome::Aborted {
                    reason: "r".to_string(),
                }
            ),
            "somnus: nightly unit for proj-b ended Aborted"
        );
        assert_eq!(
            render_nightly_unit_line("proj-c", &crate::unit::UnitOutcome::NotMachinePrincipal),
            "somnus: nightly unit for proj-c ended NotMachinePrincipal"
        );
        assert_eq!(NOTHING_ELIGIBLE_MSG, "somnus: nothing eligible tonight");
        assert_eq!(
            render_nightly_budget_stop_line(550_000, 612_004, "proj-b"),
            "somnus: token budget exhausted (armed 550000, billed 612004); stopping before proj-b"
        );
    }
    /// `Ready` says nothing failed; it does not say anything happened. A
    /// project whose only cluster sits below the floor is worked correctly,
    /// mints nothing, and ends `Ready` — and an operator reading the
    /// invocation record could not tell it from a project that minted six.
    #[test]
    fn a_worked_project_that_minted_nothing_is_distinguishable() {
        let mut report = crate::unit::UnitReport::new("quiet", PathBuf::from("/state/quiet.json"));
        assert_eq!(maps_created(&report), 0);
        report.applied.push(crate::unit::AppliedOpRecord {
            op_kind: "add_pointer",
            chain_index: 0,
            submitted_map_id: "kb-20001".to_string(),
            server_map_id: Some("kb-20001".to_string()),
            version: Some(2),
            pointer_count: Some(4),
            budget: Some(1),
            http_status: Some(200),
            body: String::new(),
            admission_reason: None,
        });
        assert_eq!(
            maps_created(&report),
            0,
            "a pointer added to an existing map is not a new map"
        );
        report.applied.push(crate::unit::AppliedOpRecord {
            op_kind: "create_map",
            chain_index: 0,
            submitted_map_id: "somnus-new-0".to_string(),
            server_map_id: Some("kb-30001".to_string()),
            version: Some(1),
            pointer_count: Some(3),
            budget: Some(0),
            http_status: Some(201),
            body: String::new(),
            admission_reason: None,
        });
        assert_eq!(maps_created(&report), 1);
    }
}
