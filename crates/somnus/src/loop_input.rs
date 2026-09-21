//! Rung-1's input: the loop-input payload types, the transport that fetches
//! them, and every pure consumer of the payload that the later rungs build on.
//!
//! The pinned server contract (`reference/somnus-functional-spec.md`, rung-1
//! paragraph) is `GET /api/kb/map-loop-input?project_ref=<ref>` — authed like
//! every `/api/kb` route, deliberately NOT admin-gated, with admission
//! enforced server-side: `404` on an unknown ref, `409` on an ineligible one,
//! overrides honoured in both directions. somnus therefore never touches the
//! admin-gated eligibility list and never surveys through `/api/kb/get`.
//!
//! Layering (pinned): this module references NO other new somnus module. It
//! owns [`Cluster`] — the rung-1 output shape
//! `{label, member_entry_ids[], owning_map_id | null}` — because BOTH
//! [`crate::rungs`] (which parses clusters) and [`crate::materialize`] (which
//! composes bodies from them) consume it, and the layer order pins
//! `materialize` below `rungs` with `materialize` referencing only
//! `crate::loop_input` + `crate::ops`.
//!
//! The two response divergences the spec calls part of the contract (not
//! bugs) are covered by the consumer-trap tests here: `maps[].pointers` may
//! name ids absent from `entries`, and `unpointed == false` does not imply
//! the entry's owning map appears in `maps`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::observer::{CountSourceError, PointerCountSource};

// ===== The payload types =================================================

/// One loop-input response. Field names are byte-identical to the vendored
/// spec's rung-1 paragraph, so the SAME type serializes the rung-1 injected
/// payload and parses the HTTP response (no serde renames anywhere).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoopInput {
    /// The project this payload was assembled for.
    pub project_ref: String,
    /// The excerpt length the server pinned when assembling `entries`.
    pub excerpt_chars: u64,
    /// The project's entries, each with an excerpt sized by `excerpt_chars`.
    pub entries: Vec<LoopInputEntry>,
    /// The project's existing mental maps, with FULL bodies (the
    /// read-modify-write substrate for rung 3 — this is NOT the `MapRef`
    /// shape).
    pub maps: Vec<LoopInputMap>,
    /// The candidate pockets, UNFILTERED by the cluster/decline ledger (this
    /// endpoint does not read the ledger; somnus owns the decline filter and
    /// applies it after fetching).
    pub pockets: Vec<LoopInputPocket>,
    /// `None` means "no pockets found"; any non-null string means "pockets
    /// not computed" (see [`PocketsStatus`]).
    pub pockets_omitted_reason: Option<String>,
}

/// One entry in the loop-input payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoopInputEntry {
    /// The entry's id (`kb-\d{5}`).
    pub id: String,
    /// Short title.
    pub short_title: String,
    /// Long title.
    pub long_title: String,
    /// The entry's type tag.
    pub entry_type: String,
    /// The entry's tags.
    pub tags: Vec<String>,
    /// The excerpt (the first `excerpt_chars` characters of the body).
    pub excerpt: String,
    /// The full body's length, so the model knows how much was elided.
    pub details_length: u64,
    /// True when no map in this project points at the entry. The anti-join
    /// does NOT constrain the owning map's `project_ref`, so `false` does not
    /// imply an owning map appears in `maps`.
    pub unpointed: bool,
    /// Coarse directory tokens the server extracted from this entry's FULL
    /// body, ranked by occurrence, most frequent first.
    ///
    /// The server does the extraction because it holds the full text while
    /// `excerpt` is only `excerpt_chars` long and would miss most mentions.
    /// It is per ENTRY rather than per cluster of necessity: loop-input is
    /// assembled once per project and rung 1 invents the clusters afterwards,
    /// so at assembly time no cluster exists to key anything by.
    ///
    /// Counts, not a single token, so code can tell a real plurality from a
    /// three-way tie and REFUSE the tie — see
    /// [`crate::materialize::lives_in_value`]. Every token already satisfies
    /// the map-lint (at most two segments, no dot in a segment, no leading
    /// slash), so somnus never sanitises one.
    #[serde(default)]
    pub directory_tokens: Vec<DirectoryToken>,
}

/// One coarse directory token from an entry's body, with its occurrence
/// count.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectoryToken {
    /// The directory, at most two path segments.
    pub token: String,
    /// How many times it occurred in the entry's full body.
    pub hits: u64,
}

/// One existing map in the loop-input payload. `body` is the FULL
/// `knowledge_details` (not an excerpt) and `contributor`/`updated_by` are
/// what make the authorship tier enforceable; both are `Option` because the
/// authorship tier must tolerate a server null.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoopInputMap {
    /// The map's id (`kb-\d{5}`).
    pub id: String,
    /// Short title.
    pub short_title: String,
    /// Long title.
    pub long_title: String,
    /// Every `kb-\d{5}` in the map body minus the map's own id — so it MAY
    /// contain ids absent from `entries` (cross-project, `NULL`-`project_ref`,
    /// or unresolvable).
    pub pointers: Vec<String>,
    /// The full map body: the read-modify-write substrate for rung 3 and the
    /// text `strike_gap` edits.
    pub body: String,
    /// Who authored the map (`None` tolerated).
    pub contributor: Option<String>,
    /// Who last updated the map (`None` tolerated).
    pub updated_by: Option<String>,
}

/// One candidate pocket: member ids plus the pairwise geometry that made the
/// pocket. Deliberately NO label/name/title/topic field — naming is the
/// model's job and geometry provably cannot do it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoopInputPocket {
    /// The pocket's member entries.
    pub member_entry_ids: Vec<String>,
    /// Mean pairwise similarity.
    pub mean_similarity: f64,
    /// Minimum pairwise similarity.
    pub min_similarity: f64,
    /// Maximum pairwise similarity.
    pub max_similarity: f64,
    /// The pairwise edges.
    pub edges: Vec<LoopInputEdge>,
}

/// One pairwise similarity edge inside a pocket.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoopInputEdge {
    /// One endpoint's entry id.
    pub a: String,
    /// The other endpoint's entry id.
    pub b: String,
    /// The pairwise similarity of `a` and `b`.
    pub similarity: f64,
}

/// One rung-1 cluster: the model's named grouping of entries.
///
/// `owning_map_id` is the lead-disposition input: a cluster that rung 1
/// reports as already covered by an existing map must CONVERGE (ops add
/// pointers to that map) and may never mint a second map for the remainder —
/// see [`crate::materialize::validate_ops_for_cluster`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Cluster {
    /// The model's name for the subject area.
    pub label: String,
    /// The clustered entries. An entry may appear in no cluster or in
    /// several; overlap is expected and correct, never resolved.
    pub member_entry_ids: Vec<String>,
    /// The existing map that already covers this cluster, if one does.
    pub owning_map_id: Option<String>,
    /// Why rung 1 ranked this cluster where it did — required of the first
    /// cluster, optional elsewhere, and never acted on.
    ///
    /// Recorded rather than used: the admission rules are mechanical and the
    /// merit ORDER is what code consumes. This is the model's account of its
    /// own ranking, kept so that when a night's map turns out to be the wrong
    /// one, the run report says what the model thought made it the best pick.
    #[serde(default)]
    pub merit_reason: Option<String>,
}

// ===== Pure consumers of the payload ======================================

/// The project's map-pointer count: `len(maps[].pointers)` summed. This is
/// the ONLY number the leg-3 observer needs, and it comes from the SAME
/// loop-input call the run is already making — there is no second endpoint,
/// no Postgres read, and no second fetch behind it.
#[must_use]
pub fn pointer_count(input: &LoopInput) -> u64 {
    input.maps.iter().map(|map| map.pointers.len() as u64).sum()
}

/// Look an entry up by id, tolerantly: `maps[].pointers` MAY name ids absent
/// from `entries` (cross-project, `NULL`-`project_ref`, or unresolvable), so a
/// consumer building `entries_by_id[pointer]` must tolerate a missing key
/// rather than panicking or indexing.
#[must_use]
pub fn entry_by_id<'a>(input: &'a LoopInput, id: &str) -> Option<&'a LoopInputEntry> {
    input.entries.iter().find(|entry| entry.id == id)
}

/// Why the pair statement ran with no candidate pockets. `NoneFound` is the
/// ordinary "nothing to do" night; `NotComputed` carries the server's reason
/// VERBATIM — including any future reason string this cut has never seen —
/// so an omitted run is loud and forward-compatible, never folded silently
/// into "no pockets found".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PocketsStatus {
    /// `pockets_omitted_reason` was null: the pair statement found no
    /// pockets, which is an ordinary, unremarkable night.
    NoneFound,
    /// Pockets were not computed, and this is why the server said so.
    NotComputed {
        /// The server's reason string, verbatim.
        reason: String,
    },
}

/// Classify a payload's pockets availability from `pockets_omitted_reason`.
#[must_use]
pub fn pockets_status(input: &LoopInput) -> PocketsStatus {
    match &input.pockets_omitted_reason {
        None => PocketsStatus::NoneFound,
        Some(reason) => PocketsStatus::NotComputed {
            reason: reason.clone(),
        },
    }
}

/// The stderr line for a pockets-not-computed night. Pure so the shape is
/// byte-pinned by a unit test; the pipeline only `eprintln!`s it.
#[must_use]
pub fn render_pockets_not_computed_line(project_ref: &str, reason: &str) -> String {
    format!(
        "somnus: pockets not computed for {project_ref} ({reason}); skipping cluster candidates"
    )
}

/// The stderr line for one night's ledger filter, surfacing the three counts
/// so a zero-match or all-match night is visibly anomalous. Pure so the shape
/// is byte-pinned by a unit test; the pipeline only `eprintln!`s it.
#[must_use]
pub fn render_ledger_filter_line(
    project_ref: &str,
    fetched: usize,
    declined: usize,
    matched: usize,
) -> String {
    format!(
        "somnus: ledger filter for {project_ref}: {fetched} pockets fetched, {declined} declined, {matched} matched"
    )
}

// ===== project_ref charset guard =========================================

/// The named refusal for a `project_ref` outside the single-segment charset
/// (the same `[A-Za-z0-9._-]+` class the version token is pinned to).
pub const INVALID_PROJECT_REF_MSG: &str =
    "somnus: project_ref must be a single URL path segment matching [A-Za-z0-9._-]+";

/// Validate a `project_ref` against the single-segment charset
/// `[A-Za-z0-9._-]+` BEFORE any HTTP call exists — an empty or
/// meta-character-carrying ref is refused here, never on the wire.
///
/// # Errors
///
/// Returns `Err` with [`INVALID_PROJECT_REF_MSG`] for any ref that is empty
/// or contains a character outside the charset.
pub fn validate_project_ref(project_ref: &str) -> Result<(), String> {
    if project_ref.is_empty()
        || !project_ref
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
    {
        return Err(INVALID_PROJECT_REF_MSG.to_string());
    }
    Ok(())
}

// ===== The transport seam ================================================

/// One fetch outcome, mapping the pinned admission contract exactly: `404`
/// and `409` are TERMINAL ORDINARY outcomes (no retry, no escalation — the
/// server enforced eligibility, and overrides are the human's call), `401`
/// is an auth failure, any other status is a named surprise, a body that does
/// not parse is a named surprise, and a transport failure is unreachable.
#[derive(Debug, Clone, PartialEq)]
pub enum FetchOutcome {
    /// `200` with a parseable body.
    Ready(LoopInput),
    /// `404 project_ref not found` — terminal and ordinary.
    UnknownProject,
    /// `409 project_ref not map-eligible` — terminal and ordinary.
    NotEligible,
    /// `401` — the bearer token was rejected.
    Unauthorized,
    /// Any status outside the pinned map.
    BadStatus {
        /// The unexpected HTTP status code.
        status: u16,
    },
    /// `200` whose body did not parse as a [`LoopInput`].
    MalformedBody {
        /// Why the body did not parse.
        reason: String,
    },
    /// No HTTP response at all (transport failure, invalid ref, or a token
    /// that could not be read — no request was ever made).
    Unreachable {
        /// Why no request could be made or answered.
        reason: String,
    },
}

impl FetchOutcome {
    /// Why a non-`Ready` outcome cannot produce a payload, as one line for
    /// the observer's fail-closed error channel. `Ready` has no refusal
    /// reason and answers `None`.
    #[must_use]
    pub fn refusal_reason(&self) -> Option<String> {
        match self {
            Self::Ready(_) => None,
            Self::UnknownProject => Some("project_ref not found".to_string()),
            Self::NotEligible => Some("project_ref not map-eligible".to_string()),
            Self::Unauthorized => Some("401 unauthorized".to_string()),
            Self::BadStatus { status } => Some(format!("unexpected HTTP status {status}")),
            Self::MalformedBody { reason } => Some(format!("malformed body ({reason})")),
            Self::Unreachable { reason } => Some(format!("unreachable ({reason})")),
        }
    }
}

/// The seam between the rungs and whatever fetches a project's loop-input.
/// The ONLY production implementation is [`HttpLoopInputSource`]; tests use
/// scripted fakes so no branch in this crate needs a live KB.
#[async_trait]
pub trait LoopInputSource: std::fmt::Debug + Send + Sync {
    /// Fetch the loop-input payload for `project_ref`.
    async fn fetch(&self, project_ref: &str) -> FetchOutcome;
}

/// The production [`LoopInputSource`]: exactly ONE
/// `GET {base_url}/api/kb/map-loop-input?project_ref=<ref>` with
/// `Authorization: Bearer <token>`, no retry, no second endpoint, and no
/// eligibility-list read anywhere.
///
/// The token is held as the result of reading it from disk, so a source
/// whose token could not be read can never turn into a request: `fetch`
/// answers [`FetchOutcome::Unreachable`] without touching the network. The
/// token is deliberately omitted from the `Debug` rendering.
pub struct HttpLoopInputSource {
    base_url: String,
    token: Result<String, TokenError>,
}

impl HttpLoopInputSource {
    /// Build a source over an already-read token.
    #[must_use]
    pub fn new(base_url: String, token: String) -> Self {
        Self {
            base_url,
            token: Ok(token),
        }
    }

    /// Build a source by reading the token out of `home` (see
    /// [`read_token_from`]). Constructing this from a `Err` token result is
    /// exactly the zero-request posture: `fetch` never dials.
    #[must_use]
    pub fn from_home(base_url: String, home: &Path) -> Self {
        Self {
            base_url,
            token: read_token_from(home),
        }
    }
}

impl std::fmt::Debug for HttpLoopInputSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never render the token, even under Debug.
        f.debug_struct("HttpLoopInputSource")
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl LoopInputSource for HttpLoopInputSource {
    async fn fetch(&self, project_ref: &str) -> FetchOutcome {
        if let Err(reason) = validate_project_ref(project_ref) {
            return FetchOutcome::Unreachable { reason };
        }
        let token = match &self.token {
            Ok(token) => token.as_str(),
            Err(err) => {
                return FetchOutcome::Unreachable {
                    reason: err.to_string(),
                };
            }
        };
        let url = format!(
            "{}/api/kb/map-loop-input?project_ref={project_ref}",
            self.base_url.trim_end_matches('/')
        );
        let response = match reqwest::Client::new()
            .get(url)
            .header(reqwest::header::AUTHORIZATION, format!("Bearer {token}"))
            .send()
            .await
        {
            Ok(response) => response,
            Err(err) => {
                return FetchOutcome::Unreachable {
                    reason: format!("transport failed: {err}"),
                };
            }
        };
        match response.status().as_u16() {
            200 => {
                let body = match response.text().await {
                    Ok(body) => body,
                    Err(err) => {
                        return FetchOutcome::Unreachable {
                            reason: format!("body read failed: {err}"),
                        };
                    }
                };
                match serde_json::from_str::<LoopInput>(&body) {
                    Ok(input) => FetchOutcome::Ready(input),
                    Err(err) => FetchOutcome::MalformedBody {
                        reason: err.to_string(),
                    },
                }
            }
            404 => FetchOutcome::UnknownProject,
            409 => FetchOutcome::NotEligible,
            401 => FetchOutcome::Unauthorized,
            status => FetchOutcome::BadStatus { status },
        }
    }
}

// ===== The pointer-count bridge to the observer ==========================

/// The [`PointerCountSource`] the leg-3 observer consumes: it fetches the
/// project's loop-input through the SAME [`LoopInputSource`] the run already
/// uses and applies [`pointer_count`] to it — one endpoint, no second
/// transport, bounded by the observer's existing [`crate::observer::OBSERVE_BOUND`].
#[derive(Debug)]
pub struct LoopInputCountSource {
    source: Arc<dyn LoopInputSource>,
}

impl LoopInputCountSource {
    /// Wrap `source` as a pointer-count source.
    #[must_use]
    pub fn new(source: Arc<dyn LoopInputSource>) -> Self {
        Self { source }
    }
}

#[async_trait]
impl PointerCountSource for LoopInputCountSource {
    async fn pointer_count(&self, project: &str) -> Result<u64, CountSourceError> {
        let outcome = self.source.fetch(project).await;
        match outcome {
            FetchOutcome::Ready(input) => Ok(pointer_count(&input)),
            FetchOutcome::UnknownProject => Err(CountSourceError::Source {
                reason: "loop-input fetch: project_ref not found".to_string(),
            }),
            FetchOutcome::NotEligible => Err(CountSourceError::Source {
                reason: "loop-input fetch: project_ref not map-eligible".to_string(),
            }),
            FetchOutcome::Unauthorized => Err(CountSourceError::Source {
                reason: "loop-input fetch: 401 unauthorized".to_string(),
            }),
            FetchOutcome::BadStatus { status } => Err(CountSourceError::Source {
                reason: format!("loop-input fetch: unexpected HTTP status {status}"),
            }),
            FetchOutcome::MalformedBody { reason } => Err(CountSourceError::Source {
                reason: format!("loop-input fetch: malformed body ({reason})"),
            }),
            FetchOutcome::Unreachable { reason } => Err(CountSourceError::Source {
                reason: format!("loop-input fetch: unreachable ({reason})"),
            }),
        }
    }
}

// ===== The token source (a pure seam, no env mutation) ===================

/// Why the KB bearer token could not be read. `std::env::set_var` is
/// unusable in this workspace (edition 2024 + `unsafe_code = "forbid"`), so
/// the token path is a pure function of a supplied home directory and the
/// tests use `tempfile` tempdirs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenError {
    /// The token file does not exist (or cannot be read) at
    /// [`token_path`].
    Missing,
    /// The token file exists but is empty (or whitespace-only).
    Empty,
}

impl std::fmt::Display for TokenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing => write!(f, "somnus: kb token file not found"),
            Self::Empty => write!(f, "somnus: kb token file is empty"),
        }
    }
}

impl std::error::Error for TokenError {}

/// Where the KB bearer token lives under a home directory. A pure function
/// of `home` so tests never touch the process environment.
#[must_use]
pub fn token_path(home: &Path) -> PathBuf {
    home.join(".config/somnus/kb-token")
}

/// Read the KB bearer token from `home`'s token file, trimming trailing
/// whitespace.
///
/// # Errors
///
/// [`TokenError::Missing`] when the file cannot be read (the file might as
/// well not be there), [`TokenError::Empty`] when it holds nothing but
/// whitespace.
pub fn read_token_from(home: &Path) -> Result<String, TokenError> {
    let bytes = std::fs::read(token_path(home)).map_err(|_| TokenError::Missing)?;
    let token = String::from_utf8(bytes).map_err(|_| TokenError::Missing)?;
    let trimmed = token.trim().to_string();
    if trimmed.is_empty() {
        return Err(TokenError::Empty);
    }
    Ok(trimmed)
}

/// Read the KB bearer token from the process's real `HOME`. A thin wrapper
/// over [`read_token_from`] for the production call site; the seam below it
/// stays pure and testable.
///
/// # Errors
/// Propagates [`TokenError`]; a missing `HOME` answers [`TokenError::Missing`].
pub fn read_token() -> Result<String, TokenError> {
    let home = std::env::var_os("HOME").ok_or(TokenError::Missing)?;
    read_token_from(&PathBuf::from(home))
}

// ===== The pinned fixture ================================================

/// The loop-input fixture, pinned byte-for-byte in
/// `crates/somnus/fixtures/loop_input.json`: two maps (one loop-owned, one
/// human-authored), seven entries (four unpointed), a five-pointer map whose
/// `kb-99999` pointer resolves to no entry, and two pockets.
pub const FIXTURE: &str = include_str!("../fixtures/loop_input.json");

/// Parse the pinned fixture (the test suite's shared substrate).
///
/// # Panics
/// Panics if the pinned fixture no longer parses — that is a fixture bug,
/// and every test that builds on the fixture should fail loudly.
#[must_use]
pub fn fixture() -> LoopInput {
    serde_json::from_str(FIXTURE).expect("the pinned fixture parses as LoopInput")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    // --- the fixture: byte-for-byte pins ---------------------------------

    #[test]
    fn fixture_parses_with_every_pinned_value() {
        let input = fixture();
        assert_eq!(input.project_ref, "demo-project");
        assert_eq!(input.excerpt_chars, 600);
        assert_eq!(input.entries.len(), 7);
        for entry in &input.entries {
            // `excerpt_chars` is the server's pinned excerpt length; the
            // fixture carries short excerpts well under it.
            assert!(
                entry.excerpt.chars().count() <= 600,
                "fixture excerpts respect the pinned excerpt_chars"
            );
            assert!(entry.details_length > 0);
        }
        let unpointed: Vec<&str> = input
            .entries
            .iter()
            .filter(|entry| entry.unpointed)
            .map(|entry| entry.id.as_str())
            .collect();
        assert_eq!(
            unpointed,
            vec!["kb-10001", "kb-10002", "kb-10003", "kb-10004"]
        );
        let pointed: Vec<&str> = input
            .entries
            .iter()
            .filter(|entry| !entry.unpointed)
            .map(|entry| entry.id.as_str())
            .collect();
        assert_eq!(pointed, vec!["kb-10005", "kb-10006", "kb-10007"]);

        assert_eq!(input.maps.len(), 2);
        let loop_map = &input.maps[0];
        assert_eq!(loop_map.id, "kb-20001");
        assert_eq!(loop_map.contributor.as_deref(), Some("somnus"));
        assert_eq!(loop_map.updated_by.as_deref(), Some("somnus"));
        assert_eq!(
            loop_map.pointers,
            vec![
                "kb-10001".to_string(),
                "kb-10002".to_string(),
                "kb-10003".to_string(),
                "kb-99999".to_string(),
                "kb-10004".to_string(),
            ]
        );
        assert_eq!(
            loop_map.body,
            "Lives in knowledge/linux/network\n\nOrients the switch, DNS, and VPN entries for the home network.\n\nDetail entries:\n- kb-10001 — switch VLAN tagging baseline\n- kb-10002 — recursive DNS resolvers\n- kb-10003 — wireguard client config\n- kb-99999 — cross-project wireguard hub notes\n- kb-10004 — DHCP lease hygiene\n\nNot yet documented:\n- Site-to-site wireguard topology"
        );
        let human_map = &input.maps[1];
        assert_eq!(human_map.id, "kb-20002");
        assert_eq!(human_map.contributor.as_deref(), Some("jason"));
        assert_eq!(human_map.updated_by, None);
        assert_eq!(
            human_map.pointers,
            vec![
                "kb-10005".to_string(),
                "kb-10006".to_string(),
                "kb-10007".to_string(),
            ]
        );
        assert_eq!(
            human_map.body,
            "Lives in knowledge/services\n\nHuman-written orientation for the service entries.\n\nDetail entries:\n- kb-10005 — nginx TLS termination\n- kb-10006 — backup job schedule\n- kb-10007 — log retention policy\n\nNot yet documented:\n- Off-site restore drill"
        );

        assert_eq!(input.pockets.len(), 2);
        let p1 = &input.pockets[0];
        assert_eq!(
            p1.member_entry_ids,
            vec![
                "kb-10001".to_string(),
                "kb-10002".to_string(),
                "kb-10003".to_string(),
            ]
        );
        assert!((p1.mean_similarity - 0.71).abs() < f64::EPSILON);
        assert!((p1.min_similarity - 0.62).abs() < f64::EPSILON);
        assert!((p1.max_similarity - 0.83).abs() < f64::EPSILON);
        assert_eq!(p1.edges.len(), 2);
        assert_eq!(p1.edges[0].a, "kb-10001");
        assert_eq!(p1.edges[0].b, "kb-10002");
        assert!((p1.edges[0].similarity - 0.71).abs() < f64::EPSILON);
        assert_eq!(p1.edges[1].a, "kb-10002");
        assert_eq!(p1.edges[1].b, "kb-10003");
        assert!((p1.edges[1].similarity - 0.83).abs() < f64::EPSILON);
        let p2 = &input.pockets[1];
        assert_eq!(
            p2.member_entry_ids,
            vec!["kb-10004".to_string(), "kb-10005".to_string()]
        );
        assert!((p2.mean_similarity - 0.55).abs() < f64::EPSILON);
        assert!((p2.min_similarity - 0.48).abs() < f64::EPSILON);
        assert!((p2.max_similarity - 0.62).abs() < f64::EPSILON);
        assert_eq!(p2.edges.len(), 1);
        assert_eq!(p2.edges[0].a, "kb-10004");
        assert_eq!(p2.edges[0].b, "kb-10005");
        assert!((p2.edges[0].similarity - 0.62).abs() < f64::EPSILON);

        assert_eq!(input.pockets_omitted_reason, None);
    }

    #[test]
    fn fixture_pointer_count_is_eight() {
        // 5 pointers on kb-20001 + 3 on kb-10002's map = 8, the literal the
        // observer and the e2e run are pinned against.
        assert_eq!(pointer_count(&fixture()), 8);
    }

    // --- consumer trap (a): a pointer with no entry ----------------------

    #[test]
    fn entry_by_id_tolerates_the_absent_pointer_id() {
        let input = fixture();
        // `kb-99999` is in kb-20001's pointers but absent from `entries` —
        // the spec's first pinned divergence. A tolerant lookup answers
        // `None`; nothing panics, and the count is unaffected by the absent
        // id.
        assert!(entry_by_id(&input, "kb-99999").is_none());
        let found = entry_by_id(&input, "kb-10001").expect("a present entry id resolves");
        assert_eq!(found.id, "kb-10001");
        assert_eq!(pointer_count(&input), 8);
    }

    // --- consumer trap (b): unpointed == false with no owning map --------

    #[test]
    fn unpointed_false_entries_assemble_without_an_ownership_lookup() {
        // The anti-join does not constrain the owning map's project_ref, so
        // `unpointed == false` does not imply an owning map appears in
        // `maps`: kb-10005..kb-10007 deserialize and assemble with NO code
        // path consulting `maps` for an owner.
        let input = fixture();
        let assembled: Vec<&LoopInputEntry> = input
            .entries
            .iter()
            .filter(|entry| !entry.unpointed)
            .collect();
        assert_eq!(
            assembled
                .iter()
                .map(|entry| entry.id.as_str())
                .collect::<Vec<_>>(),
            vec!["kb-10005", "kb-10006", "kb-10007"]
        );
        for entry in assembled {
            // No ownership lookup exists to make: the entry is usable as
            // itself, and `maps` is only ever a candidate substrate, never a
            // required back-reference.
            assert!(!entry.unpointed);
            assert!(entry.details_length > 0);
        }
    }

    // --- consumer trap (c): pockets_omitted_reason -----------------------

    #[test]
    fn pockets_status_none_found_on_a_null_reason() {
        let mut input = fixture();
        input.pockets_omitted_reason = None;
        assert_eq!(pockets_status(&input), PocketsStatus::NoneFound);
    }

    #[test]
    fn pockets_status_not_computed_too_few_unpointed_entries() {
        let mut input = fixture();
        input.pockets_omitted_reason = Some("too-few-unpointed-entries".to_string());
        assert_eq!(
            pockets_status(&input),
            PocketsStatus::NotComputed {
                reason: "too-few-unpointed-entries".to_string(),
            }
        );
    }

    #[test]
    fn pockets_status_not_computed_unpointed_set_too_large() {
        let mut input = fixture();
        input.pockets_omitted_reason = Some("unpointed-set-too-large".to_string());
        assert_eq!(
            pockets_status(&input),
            PocketsStatus::NotComputed {
                reason: "unpointed-set-too-large".to_string(),
            }
        );
    }

    #[test]
    fn pockets_status_not_computed_non_postgres_backend() {
        let mut input = fixture();
        input.pockets_omitted_reason = Some("non-postgres-backend".to_string());
        assert_eq!(
            pockets_status(&input),
            PocketsStatus::NotComputed {
                reason: "non-postgres-backend".to_string(),
            }
        );
    }

    #[test]
    fn pockets_status_carries_an_unknown_future_reason_verbatim() {
        // Forward-compatible and loud: a reason this cut has never seen is
        // never folded into NoneFound, it is carried raw.
        let mut input = fixture();
        input.pockets_omitted_reason = Some("some-future-reason".to_string());
        assert_eq!(
            pockets_status(&input),
            PocketsStatus::NotComputed {
                reason: "some-future-reason".to_string(),
            }
        );
    }

    // --- the render fns: byte-pinned surfacings --------------------------

    #[test]
    fn render_pockets_not_computed_line_is_byte_pinned() {
        assert_eq!(
            render_pockets_not_computed_line("demo-project", "too-few-unpointed-entries"),
            "somnus: pockets not computed for demo-project (too-few-unpointed-entries); skipping cluster candidates"
        );
    }

    #[test]
    fn render_ledger_filter_line_is_byte_pinned() {
        assert_eq!(
            render_ledger_filter_line("demo-project", 2, 1, 1),
            "somnus: ledger filter for demo-project: 2 pockets fetched, 1 declined, 1 matched"
        );
        // A zero-match night and an all-match night both render, so either
        // extreme is visibly anomalous in the log rather than silent.
        assert_eq!(
            render_ledger_filter_line("demo-project", 0, 0, 0),
            "somnus: ledger filter for demo-project: 0 pockets fetched, 0 declined, 0 matched"
        );
        assert_eq!(
            render_ledger_filter_line("demo-project", 2, 2, 2),
            "somnus: ledger filter for demo-project: 2 pockets fetched, 2 declined, 2 matched"
        );
    }

    // --- the project_ref charset guard -----------------------------------

    #[test]
    fn validate_project_ref_accepts_the_charset_and_refuses_both_bad_arms() {
        for good in ["demo-project", "a", "A.b_C-9"] {
            assert_eq!(validate_project_ref(good), Ok(()), "{good} must pass");
        }
        // Arm 1: the empty ref.
        assert_eq!(
            validate_project_ref(""),
            Err(INVALID_PROJECT_REF_MSG.to_string())
        );
        // Arm 2: a character outside the single-segment charset.
        assert_eq!(
            validate_project_ref("demo project"),
            Err(INVALID_PROJECT_REF_MSG.to_string())
        );
        assert_eq!(
            validate_project_ref("demo/project"),
            Err(INVALID_PROJECT_REF_MSG.to_string())
        );
        assert_eq!(
            validate_project_ref("demo?id=1"),
            Err(INVALID_PROJECT_REF_MSG.to_string())
        );
    }

    // --- the token seam ---------------------------------------------------

    #[test]
    fn token_path_is_pinned_under_the_home_directory() {
        assert_eq!(
            token_path(Path::new("/home/demo")),
            PathBuf::from("/home/demo/.config/somnus/kb-token")
        );
    }

    #[test]
    fn read_token_from_answers_missing_for_an_absent_file() {
        let home = tempfile::tempdir().expect("tempdir");
        assert_eq!(read_token_from(home.path()), Err(TokenError::Missing));
        assert_eq!(
            read_token_from(home.path()).unwrap_err().to_string(),
            "somnus: kb token file not found"
        );
    }

    #[test]
    fn read_token_from_answers_empty_for_a_whitespace_only_file() {
        let home = tempfile::tempdir().expect("tempdir");
        let path = token_path(home.path());
        std::fs::create_dir_all(path.parent().expect("parent")).expect("create .config");
        let mut file = std::fs::File::create(&path).expect("create token file");
        file.write_all(b"  \n\t").expect("write whitespace");
        assert_eq!(read_token_from(home.path()), Err(TokenError::Empty));
        assert_eq!(
            read_token_from(home.path()).unwrap_err().to_string(),
            "somnus: kb token file is empty"
        );
    }

    #[test]
    fn read_token_from_trims_and_returns_a_real_token() {
        let home = tempfile::tempdir().expect("tempdir");
        let path = token_path(home.path());
        std::fs::create_dir_all(path.parent().expect("parent")).expect("create .config");
        std::fs::write(&path, b"  kb-token-value \n").expect("write token");
        assert_eq!(
            read_token_from(home.path()),
            Ok("kb-token-value".to_string())
        );
    }

    #[test]
    fn read_token_wrapper_delegates_to_the_real_process_home() {
        // The `mined_eval.rs` precedent: HOME exists in every reasonable
        // test environment, so the wrapper is exercised against the real
        // process HOME and pinned to agree with the pure seam below it. (The
        // single missing-HOME line inside the wrapper is the one line this
        // suite cannot cover; it is named here and accepted under the
        // workspace's 98% tolerance.)
        let home = PathBuf::from(std::env::var_os("HOME").expect("HOME is set in test envs"));
        assert_eq!(read_token(), read_token_from(&home));
    }

    // --- the transport: wiremock, one request, the pinned status map ------

    mod transport {
        use super::*;
        use wiremock::matchers::{method, path, query_param};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        async fn mount(server: &MockServer, status: u16, body: &str) {
            Mock::given(method("GET"))
                .and(path("/api/kb/map-loop-input"))
                .and(query_param("project_ref", "demo-project"))
                .respond_with(ResponseTemplate::new(status).set_body_string(body))
                .mount(server)
                .await;
        }

        #[tokio::test]
        async fn happy_path_hits_exactly_one_route_with_the_pinned_query_and_header() {
            let server = MockServer::start().await;
            mount(&server, 200, FIXTURE).await;
            let source = HttpLoopInputSource::new(server.uri(), "kb-token".to_string());

            let outcome = source.fetch("demo-project").await;

            let input = match outcome {
                FetchOutcome::Ready(input) => input,
                other => panic!("expected Ready, got {other:?}"),
            };
            assert_eq!(input.project_ref, "demo-project");
            assert_eq!(pointer_count(&input), 8);

            let received = server.received_requests().await.expect("captured");
            assert_eq!(received.len(), 1, "no other route may be hit");
            let request = &received[0];
            assert_eq!(request.method.as_str(), "GET");
            assert_eq!(request.url.path(), "/api/kb/map-loop-input");
            assert_eq!(
                request.url.query(),
                Some("project_ref=demo-project".to_string()).as_deref()
            );
            assert_eq!(
                request
                    .headers
                    .get("authorization")
                    .and_then(|value| value.to_str().ok()),
                Some("Bearer kb-token")
            );
        }

        #[tokio::test]
        async fn not_found_maps_to_unknown_project_with_one_request() {
            let server = MockServer::start().await;
            mount(&server, 404, "project_ref not found").await;
            let source = HttpLoopInputSource::new(server.uri(), "kb-token".to_string());
            assert_eq!(
                source.fetch("demo-project").await,
                FetchOutcome::UnknownProject
            );
            assert_eq!(
                server.received_requests().await.expect("captured").len(),
                1,
                "404 is terminal: no retry, no escalation"
            );
        }

        #[tokio::test]
        async fn conflict_maps_to_not_eligible_with_one_request() {
            let server = MockServer::start().await;
            mount(&server, 409, "project_ref not map-eligible").await;
            let source = HttpLoopInputSource::new(server.uri(), "kb-token".to_string());
            assert_eq!(
                source.fetch("demo-project").await,
                FetchOutcome::NotEligible
            );
            assert_eq!(
                server.received_requests().await.expect("captured").len(),
                1,
                "409 is terminal: the server enforced eligibility"
            );
        }

        #[tokio::test]
        async fn unauthorized_maps_to_the_auth_failure_with_one_request() {
            let server = MockServer::start().await;
            mount(&server, 401, "unauthorized").await;
            let source = HttpLoopInputSource::new(server.uri(), "kb-token".to_string());
            assert_eq!(
                source.fetch("demo-project").await,
                FetchOutcome::Unauthorized
            );
            assert_eq!(server.received_requests().await.expect("captured").len(), 1);
        }

        #[tokio::test]
        async fn an_unpinned_status_maps_to_bad_status_with_one_request() {
            let server = MockServer::start().await;
            mount(&server, 500, "internal server error").await;
            let source = HttpLoopInputSource::new(server.uri(), "kb-token".to_string());
            assert_eq!(
                source.fetch("demo-project").await,
                FetchOutcome::BadStatus { status: 500 }
            );
            assert_eq!(server.received_requests().await.expect("captured").len(), 1);
        }

        #[tokio::test]
        async fn a_200_with_a_garbage_body_maps_to_malformed_body() {
            let server = MockServer::start().await;
            mount(&server, 200, "not a loop-input payload at all").await;
            let source = HttpLoopInputSource::new(server.uri(), "kb-token".to_string());
            match source.fetch("demo-project").await {
                FetchOutcome::MalformedBody { reason } => {
                    assert!(!reason.is_empty(), "the reason names the parse failure");
                }
                other => panic!("expected MalformedBody, got {other:?}"),
            }
            assert_eq!(server.received_requests().await.expect("captured").len(), 1);
        }

        #[tokio::test]
        async fn a_dropped_server_maps_to_unreachable() {
            // The transport branch, not a status branch. The server is built
            // over an EXPLICIT listener (and is therefore a bare server, not
            // a pooled facade) so that dropping it tears the listener down
            // and the connect is refused — wiremock's default
            // `MockServer::start()` hands its listener back to a process
            // pool on drop, which keeps serving instead of refusing.
            let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind a port");
            let address = listener.local_addr().expect("local address");
            let server = MockServer::builder().listener(listener).start().await;
            mount(&server, 200, FIXTURE).await;
            let source = HttpLoopInputSource::new(server.uri(), "kb-token".to_string());
            drop(server);
            // The shutdown is graceful, so wait for the listener to actually
            // close before asserting on the transport outcome.
            let mut closed = false;
            for _ in 0..200 {
                if std::net::TcpStream::connect_timeout(
                    &address,
                    std::time::Duration::from_millis(25),
                )
                .is_err()
                {
                    closed = true;
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
            assert!(closed, "the dropped server's listener must close");
            match source.fetch("demo-project").await {
                FetchOutcome::Unreachable { reason } => {
                    assert!(!reason.is_empty(), "the reason names the transport failure");
                }
                other => panic!("expected Unreachable, got {other:?}"),
            }
        }

        #[tokio::test]
        async fn an_invalid_ref_never_reaches_the_wire() {
            let server = MockServer::start().await;
            mount(&server, 200, FIXTURE).await;
            let source = HttpLoopInputSource::new(server.uri(), "kb-token".to_string());
            for bad_ref in ["", "de mo", "demo/project"] {
                match source.fetch(bad_ref).await {
                    FetchOutcome::Unreachable { reason } => {
                        assert_eq!(reason, INVALID_PROJECT_REF_MSG);
                    }
                    other => panic!("expected Unreachable for {bad_ref:?}, got {other:?}"),
                }
            }
            assert!(
                server
                    .received_requests()
                    .await
                    .expect("captured")
                    .is_empty(),
                "no request may exist for an invalid ref"
            );
        }

        #[tokio::test]
        async fn an_unreadable_token_never_reaches_the_wire() {
            // The zero-requests pin: a source built from an `Err` token
            // result cannot turn into a request, and the token error
            // propagates in the outcome's reason.
            let server = MockServer::start().await;
            mount(&server, 200, FIXTURE).await;
            let home = tempfile::tempdir().expect("tempdir");
            let source = HttpLoopInputSource::from_home(server.uri(), home.path());
            match source.fetch("demo-project").await {
                FetchOutcome::Unreachable { reason } => {
                    assert_eq!(reason, TokenError::Missing.to_string());
                }
                other => panic!("expected Unreachable, got {other:?}"),
            }
            assert!(
                server
                    .received_requests()
                    .await
                    .expect("captured")
                    .is_empty(),
                "no HTTP call may be made without a readable token"
            );
        }

        #[test]
        fn debug_never_renders_the_token() {
            let source = HttpLoopInputSource::new(
                "http://kb.invalid".to_string(),
                "secret-token".to_string(),
            );
            assert_eq!(
                format!("{source:?}"),
                "HttpLoopInputSource { base_url: \"http://kb.invalid\", .. }"
            );
        }
    }

    // --- the count-source bridge over the same endpoint ------------------

    mod count_source {
        use super::*;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        #[tokio::test]
        async fn the_count_source_reads_the_same_endpoint() {
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/api/kb/map-loop-input"))
                .respond_with(ResponseTemplate::new(200).set_body_string(FIXTURE))
                .mount(&server)
                .await;
            let count_source = LoopInputCountSource::new(Arc::new(HttpLoopInputSource::new(
                server.uri(),
                "kb-token".to_string(),
            )));
            // Drive the bridge end to end: fixture body → count 8.
            assert_eq!(
                count_source.pointer_count("demo-project").await,
                Ok(8),
                "the count comes from the loop-input call, nothing else"
            );
            assert_eq!(server.received_requests().await.expect("captured").len(), 1);
        }

        #[tokio::test]
        async fn an_unknown_project_maps_to_a_count_source_error() {
            // The observer's fail-closed path, unchanged: a fetch that
            // cannot produce a payload is an error, never a zero count.
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/api/kb/map-loop-input"))
                .respond_with(ResponseTemplate::new(404).set_body_string("not found"))
                .mount(&server)
                .await;
            let count_source = LoopInputCountSource::new(Arc::new(HttpLoopInputSource::new(
                server.uri(),
                "kb-token".to_string(),
            )));
            match count_source.pointer_count("demo-project").await {
                Err(CountSourceError::Source { reason }) => {
                    assert!(reason.contains("not found"), "reason was {reason}");
                }
                other => panic!("expected a CountSourceError, got {other:?}"),
            }
        }

        /// A scripted fake source, so every remaining outcome arm of the
        /// bridge is covered with no network at all.
        #[derive(Debug)]
        struct ScriptedSource(std::sync::Mutex<Vec<FetchOutcome>>);

        #[async_trait::async_trait]
        impl LoopInputSource for ScriptedSource {
            async fn fetch(&self, _project_ref: &str) -> FetchOutcome {
                let mut script = self.0.lock().expect("script lock");
                if script.len() > 1 {
                    script.remove(0)
                } else {
                    script[0].clone()
                }
            }
        }

        #[tokio::test]
        async fn every_non_ready_outcome_is_a_count_source_error() {
            let outcomes = vec![
                FetchOutcome::NotEligible,
                FetchOutcome::Unauthorized,
                FetchOutcome::BadStatus { status: 500 },
                FetchOutcome::MalformedBody {
                    reason: "garbage".to_string(),
                },
                FetchOutcome::Unreachable {
                    reason: "connection refused".to_string(),
                },
            ];
            for outcome in outcomes {
                let count_source = LoopInputCountSource::new(Arc::new(ScriptedSource(
                    std::sync::Mutex::new(vec![outcome.clone()]),
                )));
                match count_source.pointer_count("demo-project").await {
                    Err(CountSourceError::Source { reason }) => {
                        assert!(
                            reason.starts_with("loop-input fetch:"),
                            "{outcome:?} must surface a named reason, got {reason}"
                        );
                    }
                    other => panic!("{outcome:?} must be an error, got {other:?}"),
                }
            }
        }

        #[test]
        fn refusal_reason_is_none_only_for_ready() {
            assert_eq!(FetchOutcome::Ready(fixture()).refusal_reason(), None);
            for (outcome, expected) in [
                (FetchOutcome::UnknownProject, "project_ref not found"),
                (FetchOutcome::NotEligible, "project_ref not map-eligible"),
                (FetchOutcome::Unauthorized, "401 unauthorized"),
                (
                    FetchOutcome::BadStatus { status: 500 },
                    "unexpected HTTP status 500",
                ),
                (
                    FetchOutcome::MalformedBody {
                        reason: "garbage".to_string(),
                    },
                    "malformed body (garbage)",
                ),
                (
                    FetchOutcome::Unreachable {
                        reason: "refused".to_string(),
                    },
                    "unreachable (refused)",
                ),
            ] {
                assert_eq!(outcome.refusal_reason().as_deref(), Some(expected));
            }
        }
    }
}
