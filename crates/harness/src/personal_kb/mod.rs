//! The **personal-kb adapter** for the [`crate::memory`] seam: phase 1 of
//! `docs/design/11-memory.md` (the read path) and phase 3's native capture
//! (failed-tool events and turn digests).
//!
//! [`PersonalKbMemory::connect`] makes ONE `GET /api/kb/prevention` request
//! per run, before the run starts, and keeps the result: the KB's rendered
//! session slice (`slice_text`) and its Bash cue index. Every [`Memory`]
//! method then answers from that stored state — none does I/O, blocks or
//! mutates. [`Memory::before_tool`] runs the hook's lexical matcher
//! ([`cues`]) over talos `bash` commands and advises `DenyOnce` on a match;
//! the engine owns the mode ceiling, the per-entry deny state and the deny
//! cap ([`crate::memory::MAX_MEMORY_DENIES`]).
//!
//! **KB tools come over MCP, not from here.** The agent-facing KB tools
//! (`kb_search`, `kb_get`, …) are NOT hand-ported: design 11 has a generic
//! MCP client ([`crate::mcp`]) supply them. [`PersonalKbMemory::attach_kb_tools`]
//! connects to the KB's MCP endpoint (`<PERSONAL_KB_URL>/mcp`, same bearer
//! token as `/prevention`) and hands the discovered tools back through
//! [`Memory::tools`], so the engine registers them on M1's tools path and
//! counts their calls. The KB server filters tools per key surface and routes
//! writes through its own pipeline, so the client keeps no allowlist. The
//! MCP connect is skipped when the memory is already degraded (the engine
//! never calls `tools()` on a degraded memory). Both the `/prevention` and
//! the MCP requests carry [`KB_HARNESS_HEADERS`].
//!
//! ## Fail-open
//!
//! `connect` never errors and never panics. Every failure leaves the memory
//! [`Memory::degraded`] — no slice, no cues, no tools — and the engine falls
//! back to Off behaviour with one warning. The degraded reasons:
//!
//! | Reason | Cause |
//! |---|---|
//! | `PERSONAL_KB_URL is not a valid http(s) URL` | unparseable URL, non-http(s) scheme, or no host |
//! | `PERSONAL_KB_URL must not contain credentials; use PERSONAL_KB_API_KEY` | userinfo in the URL |
//! | `PERSONAL_KB_API_KEY is unset and PERSONAL_KB_URL is not a loopback URL` | a remote KB with no key |
//! | `no kb project (pass --kb-project or add a .kb_project file at or above the workspace)` | no project resolved; no request is made |
//! | `prevention fetch failed: timeout` | no response within the prevention timeout |
//! | `prevention fetch failed: unreachable (<cause>)` | connect/send/body error; `<cause>` is the innermost error |
//! | `prevention fetch failed: http <status>` | a non-2xx status |
//! | `prevention fetch failed: malformed response` | the body failed the hook's own shape validation |
//!
//! A 2xx with a parseable body is never degraded — including the KB's inert
//! response (empty index and slice), which the route serves as a
//! 200 on any internal failure. Three silent-but-visible states are carried
//! in [`Memory::label`] instead, so they never read as "the KB had nothing":
//!
//! - `tool_map=absent` — an older KB that omits `tool_map`: the slice is
//!   delivered but no cues are kept, so there is no advice;
//! - `dropped>0` — index elements failed the observed-once / shape filter:
//!   the expected signature of KB-side schema drift;
//! - `truncated>0` — the KB itself cut the index.
//!
//! A failed MCP connect does not degrade the memory: the run gets no KB
//! tools, the caller records one `mcp_connect_failed` report, and the slice
//! and advice still work.
//!
//! ## Eval isolation
//!
//! The adapter is constructed only when `talos run --memory` is not `off`.
//! With memory off nothing reads `PERSONAL_KB_*`, touches `.kb_project` or
//! opens a socket.
//!
//! ## Request shape
//!
//! `project` and `session_id` only. No `cwd` is sent: given an empty project
//! the KB derives one from the cwd basename, which would contradict "no
//! project → degraded". No `harness` parameter, no retry, no mid-run refresh.
//!
//! **Ignored KB settings.** `gate.shadow` and the KB's rate-limit and re-arm
//! settings (`max_denies*`, `rearm_hours`) are ignored: `--memory` is the
//! explicit operator switch, and honouring shadow would silently collapse a
//! deny arm into annotate.
//!
//! **Deny-policy divergence from the Claude Code hook (documented, not
//! reconciled).** The hook denies at most 1 call per turn and 6 per rolling
//! hour; a lesson re-arms after 24 h or a compact/resume/clear session start,
//! and an identical retry marks it overridden until re-arm. talos under the
//! memory seam denies at most 2 calls per run, annotates every later match,
//! and never re-arms.
//!
//! **Fixture.** `testdata/personal_kb/prevention_response.json` is the
//! interim stand-in for the golden `/prevention` fixture kb-service will
//! publish; replace it when that lands.
//!
//! ## Turn digests
//!
//! Phase 3 native capture: the session is posted to `POST /api/kb/turn`
//! ([`TURN_PATH`]) as periodic digests, contract-checked against the vendored
//! `testdata/personal_kb_contracts/turn.json`.
//!
//! - **Gating.** Digests go to `turn_target`, which is `Some` under exactly
//!   the condition failed-tool events use: not degraded AND the `/prevention`
//!   body carried a `tool_map` key. Otherwise [`Memory::observe`] and
//!   [`Memory::after_tool`] buffer nothing for digests and
//!   [`Memory::flush`] posts none. Only calls whose native name is a
//!   `tool_map` key become `tool_call`/`tool_result` items, with
//!   `tool_use_id` = `<iteration>:<call id>`.
//! - **Headers.** Every event and digest POST sends the bearer token,
//!   [`KB_HARNESS_HEADERS`] and, when the engine id given to
//!   [`PersonalKbMemory::with_event_context`] is a non-blank valid header
//!   value, [`KB_ENGINE_HEADER`]; digests also send
//!   `Content-Type: application/json`.
//! - **Body.** The contract's 14 keys; `event_id` = `<session_id>:<turn_index>`,
//!   `turn_index` 0, 1, 2, ... per closed digest (a failed or unfittable
//!   digest still consumes its index), `user_prompt` only on index 0,
//!   `final_message` only on the final digest. The bytes budgeted are the
//!   bytes posted.
//! - **Items.** A turn contributes `reasoning` then `assistant_text` (each
//!   when non-blank); a call contributes `tool_call` + `tool_result` (the
//!   excerpt only on an error) and then the corrections it resolves. Fields
//!   are cut to the contract's `maxLength`s.
//! - **Corrections.** The adapter owns the lifecycle: open on
//!   `observe(Correction)`, fold a repeat of an open trigger, resolve every
//!   open correction at the next green `run_checks` or accepted
//!   `done`/`already_satisfied`/`answer` with `resolved_by` = the posted calls
//!   in between (last [`MAX_RESOLVED_BY`]), carry still-open ones into the
//!   final digest with `resolved: false`.
//! - **Closing.** A digest closes when the next group would exceed
//!   [`MAX_TURN_ITEMS`] or [`MAX_TURN_BODY_BYTES`], or at a turn once
//!   [`DIGEST_PERIOD`] has passed since it opened; it is never closed empty
//!   mid-run, and a closed digest is spawned, never awaited in the loop.
//!   `flush` always sends a final digest, even an empty one.
//! - **Budget fallback.** A lone group that does not fit an empty digest is
//!   fitted by dropping text items, then tool items no correction names,
//!   then tool results (sender rule c); `truncated` marks a digest that lost
//!   items, and one that still does not fit is not posted and counts as
//!   failed.
//! - **Flush.** Awaits every event and digest POST (each digest bounded by
//!   [`TURN_TIMEOUT`]) and reports `digests_sent` / `digests_failed`; POSTs
//!   still pending when the engine's bound drops the flush are aborted and
//!   count as neither. A second flush posts nothing.

pub mod cues;
mod digest;

use std::collections::BTreeMap;
use std::error::Error as _;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant, SystemTime};

use async_trait::async_trait;
use serde_json::Value;

use crate::mcp::{McpConnectReport, McpServerConfig};
use crate::memory::{Advice, Memory, MemoryFlush, Observation, ToolCallView, ToolResultView};
use crate::tool::{Tool, ToolCtx, ToolRegistry, ToolResult};
use cues::{IndexCue, find_match};
use digest::{Closed, DigestBuffer, Envelope};

/// The KB URL used when `PERSONAL_KB_URL` is unset or blank.
pub const LOCAL_KB_URL: &str = "http://127.0.0.1:8765";

/// The bearer token used for a loopback KB when `PERSONAL_KB_API_KEY` is
/// unset or blank.
pub const LOCAL_KB_API_KEY: &str = "local-no-auth";

/// The per-directory file naming the KB project (first non-comment line).
pub const KB_PROJECT_FILENAME: &str = ".kb_project";

/// The KB's prevention route.
pub const PREVENTION_PATH: &str = "/api/kb/prevention";

/// The bound on the one prevention request. Longer than the hook's 1.5 s: a
/// talos run lasts minutes, and a cold KB would otherwise drop memory for the
/// whole run.
pub const PREVENTION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// The longest fact handed to the engine, in chars; longer facts end in `…`.
pub const MAX_FACT_CHARS: usize = 1000;

/// The KB's MCP endpoint, appended to the base URL.
pub const KB_MCP_PATH: &str = "/mcp";

/// [`McpServerConfig::name`] of the KB's MCP server.
pub const KB_MCP_SERVER_NAME: &str = "kb";

/// The bound on the KB MCP connect (initialize + every `tools/list` page).
pub const KB_MCP_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Headers identifying talos to the KB, sent on the `/prevention` request and
/// on every MCP request. The KB keys its per-surface policy on these.
pub const KB_HARNESS_HEADERS: [(&str, &str); 2] =
    [("X-KB-Harness", "talos"), ("X-KB-Mode", "headless")];

/// The KB's harness-event route (failed tool calls).
pub const EVENT_PATH: &str = "/api/kb/event";

/// The per-request bound on one `/api/kb/event` post.
pub const EVENT_TIMEOUT: Duration = Duration::from_secs(2);

/// The most `/api/kb/event` posts one run sends; later qualifying calls are
/// counted as capped.
pub const MAX_EVENTS_PER_RUN: u32 = 50;

/// The longest `command` / `error` sent in an event, in chars.
pub const MAX_EVENT_FIELD_CHARS: usize = 2000;

/// The KB's turn-digest route.
pub const TURN_PATH: &str = "/api/kb/turn";

/// The per-request bound on one `/api/kb/turn` post — inside the engine's
/// 3 s flush bound.
pub const TURN_TIMEOUT: Duration = Duration::from_secs(2);

/// How long a digest stays open before it closes at the next iteration
/// boundary.
pub const DIGEST_PERIOD: Duration = Duration::from_mins(10);

/// The most items one digest carries (the contract's `items.maxItems`).
pub const MAX_TURN_ITEMS: usize = 200;

/// The largest digest body, in bytes (the contract's `max_body_bytes`).
pub const MAX_TURN_BODY_BYTES: usize = 65_536;

/// The longest `reasoning` / `assistant_text` item text, in chars.
pub const MAX_TURN_TEXT_CHARS: usize = 2000;

/// The longest `tool_result` excerpt and correction `detail`, in chars.
pub const MAX_TURN_EXCERPT_CHARS: usize = 1500;

/// The longest `tool_call` target, in chars.
pub const MAX_TURN_TARGET_CHARS: usize = 500;

/// The longest `user_prompt` / `final_message`, in chars.
pub const MAX_TURN_PROMPT_CHARS: usize = 4000;

/// The most ids one correction's `resolved_by` names (the last ones kept).
pub const MAX_RESOLVED_BY: usize = 200;

/// The header carrying the model id on every event and digest post.
pub const KB_ENGINE_HEADER: &str = "X-KB-Engine";

/// Where the host name is read from for the event `host` field.
const HOSTNAME_PATH: &str = "/proc/sys/kernel/hostname";

/// The degraded reason when no project resolves.
const NO_PROJECT: &str =
    "no kb project (pass --kb-project or add a .kb_project file at or above the workspace)";

/// Where the KB lives and how to authenticate to it. Holds the API key, so it
/// deliberately has no `Debug`.
pub struct KbEndpoint {
    base_url: String,
    api_key: String,
    label: String,
}

impl KbEndpoint {
    /// Resolve the endpoint from `PERSONAL_KB_URL` / `PERSONAL_KB_API_KEY`
    /// read through `env` (both trimmed; blank means unset). The URL defaults
    /// to [`LOCAL_KB_URL`]; the key defaults to [`LOCAL_KB_API_KEY`] only for
    /// a loopback URL (`localhost`, `127.0.0.1`, `[::1]`).
    ///
    /// # Errors
    /// An invalid or non-http(s) URL, a URL carrying credentials, or a remote
    /// URL with no key. The error never contains the URL's userinfo.
    pub fn from_env(env: &impl Fn(&str) -> Option<String>) -> Result<Self, String> {
        let raw = env("PERSONAL_KB_URL")
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| LOCAL_KB_URL.to_string());
        let invalid = || "PERSONAL_KB_URL is not a valid http(s) URL".to_string();
        let url = reqwest::Url::parse(&raw).map_err(|_| invalid())?;
        if !matches!(url.scheme(), "http" | "https") {
            return Err(invalid());
        }
        let host = url
            .host_str()
            .filter(|h| !h.is_empty())
            .ok_or_else(invalid)?
            .to_string();
        if !url.username().is_empty() || url.password().is_some() {
            return Err(
                "PERSONAL_KB_URL must not contain credentials; use PERSONAL_KB_API_KEY".into(),
            );
        }
        let loopback =
            host.eq_ignore_ascii_case("localhost") || host == "127.0.0.1" || host == "[::1]";
        let api_key = match env("PERSONAL_KB_API_KEY")
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
        {
            Some(key) => key,
            None if loopback => LOCAL_KB_API_KEY.to_string(),
            None => {
                return Err(
                    "PERSONAL_KB_API_KEY is unset and PERSONAL_KB_URL is not a loopback URL".into(),
                );
            }
        };
        let port = url.port_or_known_default().unwrap_or_default();
        Ok(Self {
            base_url: raw.trim_end_matches('/').to_string(),
            api_key,
            label: format!("{host}:{port}"),
        })
    }

    /// The base URL, without a trailing `/`.
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// `host:port` — safe to print (no key, no userinfo, no path).
    #[must_use]
    pub fn endpoint_label(&self) -> &str {
        &self.label
    }

    /// The MCP server config for the KB's MCP endpoint: `<base>/mcp`, the
    /// same bearer token as `/prevention`, and [`KB_HARNESS_HEADERS`].
    #[must_use]
    pub fn mcp_server_config(&self) -> McpServerConfig {
        McpServerConfig {
            name: KB_MCP_SERVER_NAME.to_string(),
            url: format!("{}{KB_MCP_PATH}", self.base_url),
            bearer_token: Some(self.api_key.clone()),
            headers: KB_HARNESS_HEADERS
                .iter()
                .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                .collect(),
        }
    }
}

/// Walk up from `start` (itself first) to the nearest `.kb_project` FILE and
/// return its first non-blank, non-`#` line, trimmed. A file with no such
/// line stops the walk with `None` (an outer file belongs to a different
/// project); an unreadable file is skipped. A port of the personal-kb hook's
/// `resolver.py`.
#[must_use]
pub fn resolve_kb_project(start: &Path) -> Option<String> {
    for dir in start.ancestors() {
        let file = dir.join(KB_PROJECT_FILENAME);
        if !file.is_file() {
            continue;
        }
        let Ok(bytes) = std::fs::read(&file) else {
            continue;
        };
        return String::from_utf8_lossy(&bytes)
            .lines()
            .map(str::trim)
            .find(|l| !l.is_empty() && !l.starts_with('#'))
            .map(str::to_string);
    }
    None
}

/// The parts of a `/prevention` response the adapter uses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Prevention {
    pub(crate) slice_text: String,
    pub(crate) cues: Vec<IndexCue>,
    pub(crate) gate_enabled: bool,
    /// Native tool name to the KB's canonical name; `None` when the KB omits it.
    pub(crate) tool_map: Option<BTreeMap<String, String>>,
    pub(crate) received: usize,
    pub(crate) index_truncated: u64,
}

/// A string field, trimmed; empty when absent, null or not a string.
fn str_field(obj: &serde_json::Map<String, Value>, key: &str) -> String {
    obj.get(key)
        .and_then(Value::as_str)
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

fn parse_cue(element: &Value) -> Option<IndexCue> {
    let obj = element.as_object()?;
    if obj.get("observed_once") == Some(&Value::Bool(true)) {
        return None;
    }
    let required = |key: &str| {
        obj.get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    Some(IndexCue {
        resolution_id: required("resolution_id")?,
        tool: required("tool")?,
        target_class: required("target_class")?,
        corrected_fact: required("corrected_fact")?,
        args_prefix: str_field(obj, "args_prefix"),
        wrong_belief: str_field(obj, "wrong_belief"),
        provenance_label: str_field(obj, "provenance_label"),
    })
}

/// Parse a `/prevention` body. The wire shape is personal-kb's
/// `PreventionResponse` (models.py at `3bd9131`); only `slice_text`,
/// `gate.enabled`, `index` and `diagnostics.index_truncated` are read.
pub(crate) fn parse_prevention(body: &[u8]) -> Result<Prevention, String> {
    let malformed = || "malformed response".to_string();
    let value: Value = serde_json::from_slice(body).map_err(|_| malformed())?;
    let obj = value.as_object().ok_or_else(malformed)?;
    let gate = obj
        .get("gate")
        .and_then(Value::as_object)
        .ok_or_else(malformed)?;
    let index = obj
        .get("index")
        .and_then(Value::as_array)
        .ok_or_else(malformed)?;
    let gate_enabled = gate.get("enabled") == Some(&Value::Bool(true));
    let tool_map = match obj.get("tool_map") {
        None => None,
        Some(Value::Object(m)) => Some(
            m.iter()
                .filter_map(|(k, v)| v.as_str().map(|v| (k.clone(), v.to_string())))
                .collect::<BTreeMap<_, _>>(),
        ),
        Some(_) => return Err(malformed()),
    };
    // An older KB (no `tool_map`) gates its index on Claude Code's switch and
    // does not normalize talos digests, so its index is not a talos contract:
    // keep no cues from it. The gate switch itself never filters cues.
    let cues = if tool_map.is_some() {
        index.iter().filter_map(parse_cue).collect()
    } else {
        Vec::new()
    };
    Ok(Prevention {
        slice_text: obj
            .get("slice_text")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        cues,
        gate_enabled,
        tool_map,
        received: index.len(),
        index_truncated: obj
            .get("diagnostics")
            .and_then(|d| d.get("index_truncated"))
            .and_then(Value::as_u64)
            .unwrap_or(0),
    })
}

/// The fact shown for a matched cue: corrected fact, then the earlier wrong
/// belief and the provenance when present, capped at [`MAX_FACT_CHARS`]
/// chars (a cut fact ends in `…`). The entry id is not included — the engine
/// appends `(kb: <id>)` itself.
pub(crate) fn cue_fact(cue: &IndexCue) -> String {
    let mut fact = cue.corrected_fact.clone();
    if !cue.wrong_belief.is_empty() {
        fact.push_str(" Earlier wrong belief: ");
        fact.push_str(&cue.wrong_belief);
    }
    if !cue.provenance_label.is_empty() {
        fact.push_str(" [");
        fact.push_str(&cue.provenance_label);
        fact.push(']');
    }
    if fact.chars().count() > MAX_FACT_CHARS {
        let mut cut: String = fact.chars().take(MAX_FACT_CHARS - 1).collect();
        cut.push('…');
        return cut;
    }
    fact
}

/// The Display of the innermost error in `err`'s source chain, if any.
fn root_cause(err: &reqwest::Error) -> Option<String> {
    let mut cur = err.source()?;
    while let Some(next) = cur.source() {
        cur = next;
    }
    Some(cur.to_string())
}

fn transport_failure(err: &reqwest::Error) -> String {
    if err.is_timeout() {
        "timeout".to_string()
    } else {
        match root_cause(err) {
            Some(cause) => format!("unreachable ({cause})"),
            None => "unreachable".to_string(),
        }
    }
}

async fn fetch_prevention(
    endpoint: &KbEndpoint,
    project: &str,
    session_id: &str,
    timeout: Duration,
) -> Result<Prevention, String> {
    let mut request = reqwest::Client::new()
        .get(format!("{}{PREVENTION_PATH}", endpoint.base_url))
        .query(&[
            ("project", project),
            ("session_id", session_id),
            ("harness", "talos"),
        ])
        .bearer_auth(&endpoint.api_key)
        .timeout(timeout);
    for (k, v) in KB_HARNESS_HEADERS {
        request = request.header(k, v);
    }
    let response = request.send().await.map_err(|e| transport_failure(&e))?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!("http {}", status.as_u16()));
    }
    let body = response.bytes().await.map_err(|e| transport_failure(&e))?;
    parse_prevention(&body)
}

/// A shared handle to a KB tool, so [`Memory::tools`] can hand out fresh
/// boxes from `&self` without interior mutability.
struct SharedTool(Arc<dyn Tool>);

#[async_trait]
impl Tool for SharedTool {
    fn name(&self) -> &str {
        self.0.name()
    }

    fn schema(&self) -> Value {
        self.0.schema()
    }

    async fn run(&self, input: Value, ctx: &ToolCtx) -> ToolResult {
        self.0.run(input, ctx).await
    }

    fn is_mcp(&self) -> bool {
        self.0.is_mcp()
    }
}

/// The personal-kb [`Memory`]: one `/prevention` fetch at connect, then
/// answers from stored state. Holds no key, but has no `Debug` either (the
/// endpoint it keeps does hold one).
pub struct PersonalKbMemory {
    degraded: Option<String>,
    label: String,
    slice_text: String,
    cues: Vec<IndexCue>,
    tool_map: BTreeMap<String, String>,
    endpoint: Option<KbEndpoint>,
    kb_tools: Vec<Arc<dyn Tool>>,
    session_id: String,
    project: Option<String>,
    /// Where failed-tool events go: `(url, api key)`. `Some` only when not
    /// degraded AND the `/prevention` body carried a `tool_map` key — an
    /// older KB without it gets no events (same rule as digests).
    event_target: Option<(String, String)>,
    /// Where turn digests go: `(url, api key)`. `Some` under exactly the
    /// condition `event_target` is.
    turn_target: Option<(String, String)>,
    events: EventContext,
    posts: Mutex<PostState>,
    digest: Mutex<DigestBuffer>,
}

/// The static part of every event body, fixed before the run.
#[derive(Default)]
struct EventContext {
    cwd: Option<String>,
    client_version: Option<String>,
    host: Option<String>,
    /// The `X-KB-Engine` header value, when valid and non-blank.
    engine: Option<String>,
}

/// Spawned event and digest posts and the per-run counters.
#[derive(Default)]
struct PostState {
    /// Posts started this run (≤ [`MAX_EVENTS_PER_RUN`]).
    started: u32,
    capped: u32,
    /// Each pending post resolves to `true` when the KB answered 2xx.
    pending: Vec<tokio::task::JoinHandle<bool>>,
    /// Spawned digest posts, same convention.
    digests: Vec<tokio::task::JoinHandle<bool>>,
    /// Digests closed but not posted because they could not be fitted.
    unfittable: u32,
}

/// Aborts every post still pending when dropped — the engine's flush bound
/// cancels [`Memory::flush`] by dropping it.
struct AbortOnDrop(Vec<tokio::task::JoinHandle<bool>>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        for handle in &self.0 {
            handle.abort();
        }
    }
}

/// The trimmed contents of `path`, or `None` when unreadable or blank.
fn read_host(path: &Path) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// `s` cut to its first `max` chars (on a char boundary).
fn truncate_chars(s: &str, max: usize) -> String {
    match s.char_indices().nth(max) {
        Some((i, _)) => s[..i].to_string(),
        None => s.to_string(),
    }
}

/// The `tool_input` of an event: only the call's target argument, never
/// edit contents. `None` for a tool that is not posted.
fn event_tool_input(call: &ToolCallView) -> Option<Value> {
    let arg = |key: &str| call.args.get(key).cloned().unwrap_or(Value::Null);
    Some(match call.name {
        "bash" => {
            let command = match call.args.get("command") {
                Some(Value::String(c)) => Value::String(truncate_chars(c, MAX_EVENT_FIELD_CHARS)),
                Some(other) => other.clone(),
                None => Value::Null,
            };
            serde_json::json!({ "command": command })
        }
        "edit_file" | "read_file" => serde_json::json!({ "path": arg("path") }),
        "list_files" => {
            let path = call
                .args
                .get("path")
                .cloned()
                .unwrap_or_else(|| Value::String(".".to_string()));
            serde_json::json!({ "path": path })
        }
        _ => return None,
    })
}

/// A POST to the KB carrying the bearer token, [`KB_HARNESS_HEADERS`] and,
/// when kept, [`KB_ENGINE_HEADER`].
fn kb_post(url: String, api_key: &str, engine: Option<&str>) -> reqwest::RequestBuilder {
    let mut request = reqwest::Client::new().post(url).bearer_auth(api_key);
    for (k, v) in KB_HARNESS_HEADERS {
        request = request.header(k, v);
    }
    if let Some(engine) = engine {
        request = request.header(KB_ENGINE_HEADER, engine);
    }
    request
}

/// POST one event; `true` when the KB answered 2xx (whatever its `reason`).
async fn post_event(url: String, api_key: String, engine: Option<String>, body: Value) -> bool {
    let request = kb_post(url, &api_key, engine.as_deref())
        .timeout(EVENT_TIMEOUT)
        .json(&body);
    matches!(request.send().await, Ok(r) if r.status().is_success())
}

/// POST one digest's exact bytes; `true` when the KB answered 2xx.
async fn post_turn(url: String, api_key: String, engine: Option<String>, body: Vec<u8>) -> bool {
    let request = kb_post(url, &api_key, engine.as_deref())
        .timeout(TURN_TIMEOUT)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(body);
    matches!(request.send().await, Ok(r) if r.status().is_success())
}

/// The engine id kept for [`KB_ENGINE_HEADER`]: trimmed, non-blank and a
/// valid header value.
fn header_engine(engine: &str) -> Option<String> {
    let engine = engine.trim();
    (!engine.is_empty() && reqwest::header::HeaderValue::from_str(engine).is_ok())
        .then(|| engine.to_string())
}

impl PersonalKbMemory {
    /// Fetch the prevention slice and index for `project` (see the module
    /// doc). Never fails: every failure is recorded as
    /// [`Memory::degraded`]. No request is made when `endpoint` is an error
    /// or `project` is missing or blank.
    pub async fn connect(
        endpoint: Result<KbEndpoint, String>,
        project: Option<String>,
        session_id: &str,
        prevention_timeout: std::time::Duration,
    ) -> PersonalKbMemory {
        let project = project
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty());
        let endpoint_label = endpoint
            .as_ref()
            .map_or_else(|_| "-".to_string(), |e| e.endpoint_label().to_string());
        let project_label = project.as_deref().unwrap_or("-").to_string();
        let degraded_label = || {
            format!(
                "personal-kb {endpoint_label} project={project_label} gate=- index=0 dropped=0 truncated=0"
            )
        };
        let degraded = |reason: String, endpoint: Option<KbEndpoint>| PersonalKbMemory {
            degraded: Some(reason),
            label: degraded_label(),
            slice_text: String::new(),
            cues: Vec::new(),
            tool_map: BTreeMap::new(),
            endpoint,
            kb_tools: Vec::new(),
            session_id: session_id.to_string(),
            project: None,
            event_target: None,
            turn_target: None,
            events: EventContext::default(),
            posts: Mutex::new(PostState::default()),
            digest: Mutex::new(DigestBuffer::new(
                Envelope::default(),
                std::collections::BTreeSet::new(),
                DIGEST_PERIOD,
            )),
        };
        let endpoint = match endpoint {
            Ok(e) => e,
            Err(reason) => return degraded(reason, None),
        };
        let Some(project) = project else {
            return degraded(NO_PROJECT.to_string(), Some(endpoint));
        };
        match fetch_prevention(&endpoint, &project, session_id, prevention_timeout).await {
            Err(e) => degraded(format!("prevention fetch failed: {e}"), Some(endpoint)),
            Ok(p) => {
                let index = p.cues.len();
                let label = format!(
                    "personal-kb {endpoint_label} project={project_label} gate={} tool_map={} index={index} dropped={} truncated={}",
                    if p.gate_enabled { "on" } else { "off" },
                    p.tool_map
                        .as_ref()
                        .map_or_else(|| "absent".to_string(), |m| m.len().to_string()),
                    p.received.saturating_sub(index),
                    p.index_truncated,
                );
                let target = |route: &str| {
                    p.tool_map.is_some().then(|| {
                        (
                            format!("{}{route}", endpoint.base_url),
                            endpoint.api_key.clone(),
                        )
                    })
                };
                let event_target = target(EVENT_PATH);
                let turn_target = target(TURN_PATH);
                let tool_map = p.tool_map.unwrap_or_default();
                let digest = DigestBuffer::new(
                    Envelope {
                        session_id: session_id.to_string(),
                        host: None,
                        hook_version: None,
                        project: Some(project.clone()),
                    },
                    tool_map.keys().cloned().collect(),
                    DIGEST_PERIOD,
                );
                PersonalKbMemory {
                    degraded: None,
                    label,
                    slice_text: p.slice_text,
                    cues: p.cues,
                    event_target,
                    turn_target,
                    tool_map,
                    endpoint: Some(endpoint),
                    kb_tools: Vec::new(),
                    session_id: session_id.to_string(),
                    project: Some(project),
                    events: EventContext::default(),
                    posts: Mutex::new(PostState::default()),
                    digest: Mutex::new(digest),
                }
            }
        }
    }

    /// Set the run context every `/api/kb/event` and `/api/kb/turn` body
    /// carries: `cwd` (the workspace root, as given — the caller makes it
    /// absolute; events only) and `client_version` (sent as
    /// `hook_version`), plus `engine`, the model id sent as
    /// [`KB_ENGINE_HEADER`] on every event and digest post (trimmed; dropped
    /// when blank or not a valid header value). Also reads the host name
    /// from `/proc/sys/kernel/hostname` once; unreadable or blank → `null`.
    #[must_use]
    pub fn with_event_context(self, cwd: &Path, client_version: &str, engine: &str) -> Self {
        self.with_event_context_from(cwd, client_version, engine, Path::new(HOSTNAME_PATH))
    }

    fn with_event_context_from(
        mut self,
        cwd: &Path,
        client_version: &str,
        engine: &str,
        host: &Path,
    ) -> Self {
        self.events = EventContext {
            cwd: Some(cwd.display().to_string()),
            client_version: Some(client_version.to_string()),
            host: read_host(host),
            engine: header_engine(engine),
        };
        self.digest
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .set_context(self.events.host.clone(), self.events.client_version.clone());
        self
    }

    /// Override [`DIGEST_PERIOD`] (tests only).
    #[cfg(test)]
    pub(crate) fn with_digest_period(mut self, period: Duration) -> Self {
        self.digest
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .set_period(period);
        self
    }

    /// Spawn a POST for every closed digest; count the unfittable ones.
    fn post_digests(&self, closed: Vec<Closed>) {
        let Some((url, api_key)) = self.turn_target.as_ref() else {
            return;
        };
        if closed.is_empty() {
            return;
        }
        let mut posts = self.posts.lock().unwrap_or_else(PoisonError::into_inner);
        for digest in closed {
            match digest {
                Closed::Body(bytes) => posts.digests.push(tokio::spawn(post_turn(
                    url.clone(),
                    api_key.clone(),
                    self.events.engine.clone(),
                    bytes,
                ))),
                Closed::Unfittable => posts.unfittable += 1,
            }
        }
    }

    /// The `/api/kb/event` body for one failed call — exactly the contract's
    /// `fields.client_supplied` keys.
    fn event_body(
        &self,
        call: &ToolCallView,
        result: &ToolResultView,
        tool_input: &Value,
    ) -> Value {
        serde_json::json!({
            "type": "post_tool",
            "event_id": format!("{}:{}:{}", self.session_id, result.iteration, result.call_id),
            "session_id": self.session_id,
            "harness": "talos",
            "mode": "headless",
            "engine": Value::Null,
            "host": self.events.host,
            "hook_version": self.events.client_version,
            "cwd": self.events.cwd,
            "project": self.project,
            "ts": crate::time::format_rfc3339(SystemTime::now()),
            "tool_name": call.name,
            "tool_input": tool_input,
            "tool_use_id": result.call_id,
            "error": truncate_chars(result.content, MAX_EVENT_FIELD_CHARS),
            "is_error": true,
            "is_interrupt": false,
            "duration_ms": result.duration_ms,
        })
    }

    /// Connect to the KB's MCP endpoint ([`KbEndpoint::mcp_server_config`])
    /// under `timeout` and keep its tools for [`Memory::tools`]. Skipped —
    /// `None` report, no request — when the memory is degraded. A failed
    /// connect is fail-open: no tools, and the returned
    /// [`McpConnectReport::Failed`] is the caller's to record.
    pub async fn attach_kb_tools(mut self, timeout: Duration) -> (Self, Option<McpConnectReport>) {
        let Some(endpoint) = self.endpoint.as_ref().filter(|_| self.degraded.is_none()) else {
            return (self, None);
        };
        let cfg = endpoint.mcp_server_config();
        let mut registry = ToolRegistry::new();
        let reports = crate::mcp::connect_and_register(&mut registry, &[cfg], timeout).await;
        let report = reports.into_iter().next();
        if let Some(McpConnectReport::Connected { tools, .. }) = &report {
            self.kb_tools = tools.iter().filter_map(|n| registry.get(n)).collect();
        }
        (self, report)
    }
}

/// The raw target of a native call, per the KB's `harness_tools.py` rule 3:
/// `bash` → `command`; `edit_file` / `read_file` → `path`; `list_files` →
/// `path` defaulting to `.`; anything else `""`. A non-string value is `""`.
fn native_target(call: &ToolCallView) -> String {
    let key = match call.name {
        "bash" => "command",
        "edit_file" | "read_file" | "list_files" => "path",
        _ => return String::new(),
    };
    match call.args.get(key) {
        Some(v) => v.as_str().unwrap_or_default().to_string(),
        None if call.name == "list_files" => ".".to_string(),
        None => String::new(),
    }
}

impl Memory for PersonalKbMemory {
    fn session_slice(&self) -> Option<String> {
        if self.degraded.is_some() || self.slice_text.trim().is_empty() {
            None
        } else {
            Some(self.slice_text.clone())
        }
    }

    fn before_tool(&self, call: &ToolCallView) -> Advice {
        if self.degraded.is_some() {
            return Advice::None;
        }
        let Some(canonical) = self.tool_map.get(call.name) else {
            return Advice::None;
        };
        match find_match(canonical, &native_target(call), &self.cues) {
            Some(cue) => Advice::DenyOnce {
                fact: cue_fact(cue),
                entry_id: cue.resolution_id.clone(),
            },
            None => Advice::None,
        }
    }

    fn tools(&self) -> Vec<Box<dyn Tool>> {
        self.kb_tools
            .iter()
            .map(|t| Box::new(SharedTool(Arc::clone(t))) as Box<dyn Tool>)
            .collect()
    }

    fn degraded(&self) -> Option<String> {
        self.degraded.clone()
    }

    fn label(&self) -> String {
        self.label.clone()
    }

    fn session_id(&self) -> Option<String> {
        Some(self.session_id.clone())
    }

    /// Post a failed `bash` / `edit_file` / `read_file` / `list_files` call
    /// to `/api/kb/event` — spawned, never awaited here. Nothing is posted
    /// when degraded or when the KB sent no `tool_map`; past
    /// [`MAX_EVENTS_PER_RUN`] posts a qualifying call is only counted.
    fn after_tool(&self, call: &ToolCallView, result: &ToolResultView) {
        if self.turn_target.is_some() {
            let closed = self
                .digest
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .after_tool(call, result);
            self.post_digests(closed);
        }
        if !result.is_error {
            return;
        }
        let Some((url, api_key)) = self.event_target.as_ref() else {
            return;
        };
        // `None` for every tool but bash / edit_file / read_file / list_files.
        let Some(tool_input) = event_tool_input(call) else {
            return;
        };
        let mut posts = self.posts.lock().unwrap_or_else(PoisonError::into_inner);
        if posts.started >= MAX_EVENTS_PER_RUN {
            posts.capped += 1;
            return;
        }
        posts.started += 1;
        let body = self.event_body(call, result, &tool_input);
        let handle = tokio::spawn(post_event(
            url.clone(),
            api_key.clone(),
            self.events.engine.clone(),
            body,
        ));
        posts.pending.push(handle);
    }

    /// Feed the turn-digest buffer (see the module doc's `## Turn
    /// digests`); a digest closed here is spawned, never awaited.
    fn observe(&self, observation: &Observation<'_>) {
        if self.turn_target.is_none() {
            return;
        }
        let closed = self
            .digest
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .observe(observation, Instant::now());
        self.post_digests(closed);
    }

    /// Send the final digest, then await every pending event and digest
    /// post. Dropping this future (the engine's bound) aborts the posts
    /// still running; those count as neither sent nor failed. A second call
    /// posts nothing.
    fn flush(&self) -> Pin<Box<dyn Future<Output = MemoryFlush> + Send + '_>> {
        let final_digests = if self.turn_target.is_some() {
            self.digest
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .finish()
        } else {
            Vec::new()
        };
        Box::pin(async move {
            self.post_digests(final_digests);
            let (pending, digests, capped, unfittable) = {
                let mut posts = self.posts.lock().unwrap_or_else(PoisonError::into_inner);
                (
                    std::mem::take(&mut posts.pending),
                    std::mem::take(&mut posts.digests),
                    std::mem::take(&mut posts.capped),
                    std::mem::take(&mut posts.unfittable),
                )
            };
            let mut events = AbortOnDrop(pending);
            let mut digests = AbortOnDrop(digests);
            let mut flushed = MemoryFlush {
                capped,
                digests_failed: unfittable,
                ..MemoryFlush::default()
            };
            while let Some(handle) = events.0.last_mut() {
                if matches!(handle.await, Ok(true)) {
                    flushed.sent += 1;
                } else {
                    flushed.failed += 1;
                }
                events.0.pop();
            }
            while let Some(handle) = digests.0.last_mut() {
                if matches!(handle.await, Ok(true)) {
                    flushed.digests_sent += 1;
                } else {
                    flushed.digests_failed += 1;
                }
                digests.0.pop();
            }
            flushed
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DIGEST_PERIOD, EVENT_PATH, EVENT_TIMEOUT, HOSTNAME_PATH, KB_ENGINE_HEADER,
        MAX_EVENT_FIELD_CHARS, MAX_EVENTS_PER_RUN, MAX_RESOLVED_BY, MAX_TURN_BODY_BYTES,
        MAX_TURN_EXCERPT_CHARS, MAX_TURN_ITEMS, MAX_TURN_PROMPT_CHARS, MAX_TURN_TARGET_CHARS,
        MAX_TURN_TEXT_CHARS, TURN_PATH, TURN_TIMEOUT, event_tool_input, read_host, truncate_chars,
    };
    use super::{
        KB_HARNESS_HEADERS, KB_MCP_CONNECT_TIMEOUT, KB_PROJECT_FILENAME, KbEndpoint,
        LOCAL_KB_API_KEY, LOCAL_KB_URL, MAX_FACT_CHARS, NO_PROJECT, PREVENTION_PATH,
        PREVENTION_TIMEOUT, PersonalKbMemory, Prevention, cue_fact, parse_prevention,
        resolve_kb_project,
    };
    use crate::engine::{FINISH_TOOL_NAME, FinishTool, RunConfig, RunStats};
    use crate::mcp::McpConnectReport;
    use crate::memory::{
        self, Advice, CorrectionTrigger, Memory, MemoryFlush, MemoryMode, Observation,
        ToolCallView, ToolResultView,
    };
    use crate::model::{
        AssistantTurn, ContentBlock, Message, StopReason, ToolCallRequest, Usage, UserBlock,
    };
    use crate::personal_kb::cues::IndexCue;
    use crate::test_support::MockBackend;
    use crate::tool::{Tool, ToolCtx, ToolRegistry, ToolResult};
    use async_trait::async_trait;
    use serde_json::{Value, json};
    use std::path::Path;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Duration;
    use wiremock::matchers::{
        any, body_partial_json, header, method, path, query_param, query_param_is_missing,
    };
    use wiremock::{Mock, MockServer, Request, ResponseTemplate};

    const FIXTURE: &str = include_str!("../../testdata/personal_kb/prevention_response.json");

    const FACT_1: &str = "Push to origin; github is release-only Earlier wrong belief: push to github [deliberate/observed]";

    const REMOTE_NO_KEY: &str =
        "PERSONAL_KB_API_KEY is unset and PERSONAL_KB_URL is not a loopback URL";
    const INVALID_URL: &str = "PERSONAL_KB_URL is not a valid http(s) URL";
    const CREDENTIALS: &str =
        "PERSONAL_KB_URL must not contain credentials; use PERSONAL_KB_API_KEY";

    fn fixture() -> Value {
        serde_json::from_str(FIXTURE).expect("fixture is JSON")
    }

    fn slice_text() -> String {
        fixture()["slice_text"]
            .as_str()
            .expect("slice_text")
            .to_string()
    }

    /// FIXTURE with `edit` applied, re-serialized.
    fn variant(edit: impl FnOnce(&mut Value)) -> String {
        let mut v = fixture();
        edit(&mut v);
        v.to_string()
    }

    fn env_of(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> + use<> {
        let pairs: Vec<(String, String)> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        move |key: &str| pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone())
    }

    fn endpoint(url: &str) -> Result<KbEndpoint, String> {
        KbEndpoint::from_env(&env_of(&[("PERSONAL_KB_URL", url)]))
    }

    fn view<'a>(name: &'a str, args: &'a Value) -> ToolCallView<'a> {
        ToolCallView { name, args }
    }

    async fn prevention_server(status: u16, body: String) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path(PREVENTION_PATH))
            .respond_with(ResponseTemplate::new(status).set_body_string(body))
            .mount(&server)
            .await;
        server
    }

    async fn connect_to(url: &str, project: Option<&str>) -> PersonalKbMemory {
        PersonalKbMemory::connect(
            endpoint(url),
            project.map(str::to_string),
            "t:1",
            PREVENTION_TIMEOUT,
        )
        .await
    }

    /// Connect to a server serving `body` with status 200.
    async fn connected(body: String) -> (MockServer, PersonalKbMemory) {
        let server = prevention_server(200, body).await;
        let memory = connect_to(&server.uri(), Some("fake-project")).await;
        (server, memory)
    }

    fn port(server: &MockServer) -> u16 {
        server.address().port()
    }

    // ---- AC2: constants ---------------------------------------------------

    #[test]
    fn constants_are_pinned() {
        assert_eq!(LOCAL_KB_URL, "http://127.0.0.1:8765");
        assert_eq!(LOCAL_KB_API_KEY, "local-no-auth");
        assert_eq!(KB_PROJECT_FILENAME, ".kb_project");
        assert_eq!(PREVENTION_PATH, "/api/kb/prevention");
        assert_eq!(PREVENTION_TIMEOUT, Duration::from_secs(5));
        assert_eq!(MAX_FACT_CHARS, 1000);
        assert_eq!(KB_MCP_CONNECT_TIMEOUT, Duration::from_secs(10));
        assert_eq!(
            KB_HARNESS_HEADERS,
            [("X-KB-Harness", "talos"), ("X-KB-Mode", "headless")]
        );
    }

    // ---- AC3: fixture -------------------------------------------------------

    #[test]
    fn fixture_slice_text_shape() {
        let s = slice_text();
        assert_eq!(s.chars().count(), 230);
        assert_eq!(memory::slice_item_count(&s), 2);
        assert_eq!(s.trim(), s);
    }

    // ---- AC4: KbEndpoint ----------------------------------------------------

    /// `(PERSONAL_KB_URL, PERSONAL_KB_API_KEY, base, label, key)`.
    type OkRow = (
        Option<&'static str>,
        Option<&'static str>,
        &'static str,
        &'static str,
        &'static str,
    );

    #[test]
    fn endpoint_from_env_table() {
        // (url, key) -> Ok((base, label, key))
        let ok: &[OkRow] = &[
            (
                None,
                None,
                "http://127.0.0.1:8765",
                "127.0.0.1:8765",
                "local-no-auth",
            ),
            (
                Some("http://localhost:9/"),
                None,
                "http://localhost:9",
                "localhost:9",
                "local-no-auth",
            ),
            (
                Some("http://LOCALHOST:9"),
                None,
                "http://LOCALHOST:9",
                "localhost:9",
                "local-no-auth",
            ),
            (
                Some("http://[::1]:9"),
                None,
                "http://[::1]:9",
                "[::1]:9",
                "local-no-auth",
            ),
            (
                Some("https://kb.example.com"),
                Some("sk-1"),
                "https://kb.example.com",
                "kb.example.com:443",
                "sk-1",
            ),
            (
                Some("https://kb.example.com/kb/"),
                Some("sk-1"),
                "https://kb.example.com/kb",
                "kb.example.com:443",
                "sk-1",
            ),
            (
                Some("   "),
                None,
                "http://127.0.0.1:8765",
                "127.0.0.1:8765",
                "local-no-auth",
            ),
        ];
        for (url, key, base, label, want_key) in ok {
            let mut pairs = Vec::new();
            if let Some(u) = url {
                pairs.push(("PERSONAL_KB_URL", *u));
            }
            if let Some(k) = key {
                pairs.push(("PERSONAL_KB_API_KEY", *k));
            }
            let Ok(e) = KbEndpoint::from_env(&env_of(&pairs)) else {
                panic!("{url:?} {key:?} should resolve");
            };
            assert_eq!(e.base_url(), *base, "{url:?}");
            assert_eq!(e.endpoint_label(), *label, "{url:?}");
            assert_eq!(e.api_key, *want_key, "{url:?}");
        }
        let err: &[(&str, Option<&str>, &str)] = &[
            ("https://kb.example.com", None, REMOTE_NO_KEY),
            ("https://kb.example.com", Some("   "), REMOTE_NO_KEY),
            ("http://127.0.0.2:9", None, REMOTE_NO_KEY),
            ("not a url", None, INVALID_URL),
            ("ftp://x", None, INVALID_URL),
            ("http://user:pw@127.0.0.1:9/x", Some("sk-1"), CREDENTIALS),
            ("http://user@127.0.0.1:9", None, CREDENTIALS),
        ];
        for (url, key, want) in err {
            let mut pairs = vec![("PERSONAL_KB_URL", *url)];
            if let Some(k) = key {
                pairs.push(("PERSONAL_KB_API_KEY", *k));
            }
            assert_eq!(
                KbEndpoint::from_env(&env_of(&pairs)).err().as_deref(),
                Some(*want),
                "{url:?} {key:?}"
            );
        }
    }

    #[test]
    fn endpoint_without_host_is_invalid() {
        assert_eq!(endpoint("http:").err().as_deref(), Some(INVALID_URL));
    }

    #[test]
    fn mcp_server_config_reuses_the_token_and_headers() {
        let e = KbEndpoint::from_env(&env_of(&[
            ("PERSONAL_KB_URL", "https://kb.example.com/kb/"),
            ("PERSONAL_KB_API_KEY", "sk-1"),
        ]))
        .expect("endpoint");
        let cfg = e.mcp_server_config();
        assert_eq!(cfg.name, "kb");
        assert_eq!(cfg.url, "https://kb.example.com/kb/mcp");
        assert_eq!(cfg.bearer_token.as_deref(), Some("sk-1"));
        assert_eq!(
            cfg.headers,
            vec![
                ("X-KB-Harness".to_string(), "talos".to_string()),
                ("X-KB-Mode".to_string(), "headless".to_string()),
            ]
        );
        let local = endpoint("http://127.0.0.1:9").expect("endpoint");
        assert_eq!(
            local.mcp_server_config().bearer_token.as_deref(),
            Some("local-no-auth")
        );
    }

    // ---- AC5: resolve_kb_project -------------------------------------------

    #[test]
    fn resolve_kb_project_walks_up() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let root = tmp.path();

        // (a)
        let dir_a = root.join("a");
        std::fs::create_dir_all(dir_a.join("b")).expect("mkdir");
        std::fs::write(
            dir_a.join(".kb_project"),
            "# comment\n\n  my-proj  \nother\n",
        )
        .expect("w");
        assert_eq!(
            resolve_kb_project(&dir_a.join("b")).as_deref(),
            Some("my-proj")
        );

        // (b)
        let dir_b = root.join("bb");
        std::fs::create_dir_all(dir_b.join("inner")).expect("mkdir");
        std::fs::write(dir_b.join(".kb_project"), "outer").expect("w");
        std::fs::write(
            dir_b.join("inner").join(".kb_project"),
            "# only a comment\n",
        )
        .expect("w");
        assert_eq!(resolve_kb_project(&dir_b.join("inner")), None);

        // (c)
        let dir_c = root.join("c");
        std::fs::create_dir_all(&dir_c).expect("mkdir");
        std::fs::write(dir_c.join(".kb_project"), "p").expect("w");
        assert_eq!(resolve_kb_project(&dir_c).as_deref(), Some("p"));

        // (d)
        let dir_d = root.join("d");
        std::fs::create_dir_all(dir_d.join("x").join(".kb_project")).expect("mkdir");
        std::fs::write(dir_d.join(".kb_project"), "outer").expect("w");
        assert_eq!(
            resolve_kb_project(&dir_d.join("x")).as_deref(),
            Some("outer")
        );

        // (e)
        let dir_e = root.join("e");
        std::fs::create_dir_all(&dir_e).expect("mkdir");
        std::fs::write(dir_e.join(".kb_project"), "  # indented\nproj\n").expect("w");
        assert_eq!(resolve_kb_project(&dir_e).as_deref(), Some("proj"));
    }

    // ---- AC8: parse_prevention ---------------------------------------------

    fn fixture_cue() -> IndexCue {
        IndexCue {
            resolution_id: "kb-00001".into(),
            tool: "Bash".into(),
            target_class: "git push".into(),
            args_prefix: "github".into(),
            wrong_belief: "push to github".into(),
            corrected_fact: "Push to origin; github is release-only".into(),
            provenance_label: "deliberate/observed".into(),
        }
    }

    fn parse(body: &str) -> Prevention {
        parse_prevention(body.as_bytes()).expect("parses")
    }

    #[test]
    fn parse_rejects_malformed_bodies() {
        for body in [
            "not json",
            "[]",
            r#"{"index":[]}"#,
            r#"{"gate":true,"index":[]}"#,
            r#"{"gate":{},"index":{}}"#,
        ] {
            assert_eq!(
                parse_prevention(body.as_bytes()),
                Err("malformed response".to_string()),
                "{body}"
            );
        }
    }

    #[test]
    fn parse_minimal_body() {
        assert_eq!(
            parse(r#"{"gate":{},"index":[]}"#),
            Prevention {
                slice_text: String::new(),
                cues: vec![],
                gate_enabled: false,
                tool_map: None,
                received: 0,
                index_truncated: 0,
            }
        );
    }

    #[test]
    fn parse_fixture() {
        assert_eq!(
            parse(FIXTURE),
            Prevention {
                slice_text: slice_text(),
                cues: vec![fixture_cue()],
                gate_enabled: true,
                tool_map: Some(
                    fixture()["tool_map"]
                        .as_object()
                        .unwrap()
                        .iter()
                        .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
                        .collect()
                ),
                received: 1,
                index_truncated: 0,
            }
        );
    }

    #[test]
    fn parse_gate_variants() {
        let off = parse(&variant(|v| v["gate"]["enabled"] = json!(false)));
        assert_eq!(off.cues, vec![fixture_cue()]);
        assert!(!off.gate_enabled);
        assert_eq!(off.received, 1);
        assert_eq!(off.slice_text, slice_text());

        let stringly = parse(&variant(|v| v["gate"]["enabled"] = json!("true")));
        assert!(!stringly.gate_enabled);
        assert_eq!(stringly.cues, vec![fixture_cue()]);
    }

    #[test]
    fn parse_tool_map_variants() {
        let none = parse(&variant(|v| {
            v.as_object_mut().unwrap().remove("tool_map");
        }));
        assert_eq!(none.tool_map, None);
        assert!(none.cues.is_empty());
        assert_eq!(none.received, 1);

        let skipped = parse(&variant(|v| {
            v["tool_map"] = json!({"bash": "Bash", "x": 5});
        }));
        assert_eq!(skipped.tool_map.unwrap().len(), 1);

        for bad in [json!([]), json!("x"), Value::Null] {
            let body = variant(|v| v["tool_map"] = bad.clone());
            assert_eq!(
                parse_prevention(body.as_bytes()),
                Err("malformed response".to_string())
            );
        }
    }

    #[test]
    fn native_targets() {
        let t = |name: &str, args: Value| super::native_target(&view(name, &args));
        assert_eq!(t("bash", json!({"command": "ls"})), "ls");
        assert_eq!(t("bash", json!({"command": 5})), "");
        assert_eq!(t("bash", json!({})), "");
        assert_eq!(t("edit_file", json!({"path": "a.rs"})), "a.rs");
        assert_eq!(t("read_file", json!({"path": "b.rs"})), "b.rs");
        assert_eq!(t("list_files", json!({"path": "src"})), "src");
        assert_eq!(t("list_files", json!({})), ".");
        assert_eq!(t("list_files", json!({"path": 3})), "");
        assert_eq!(t("finish", json!({"path": "x"})), "");
        assert_eq!(t("edit_file", json!({})), "");
    }

    fn contract_example(name: &str) -> Value {
        let c: Value = serde_json::from_str(include_str!(
            "../../testdata/personal_kb_contracts/prevention.json"
        ))
        .unwrap();
        c["response"]["examples"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["name"] == name)
            .unwrap()["body"]
            .clone()
    }

    #[test]
    fn contract_examples_all_parse() {
        let c: Value = serde_json::from_str(include_str!(
            "../../testdata/personal_kb_contracts/prevention.json"
        ))
        .unwrap();
        let examples = c["response"]["examples"].as_array().unwrap();
        assert!(!examples.is_empty());
        for e in examples {
            assert!(
                parse_prevention(e["body"].to_string().as_bytes()).is_ok(),
                "{}",
                e["name"]
            );
        }
    }

    /// A bash command that the cue's class and args prefix match.
    fn command_for(cue: &IndexCue) -> String {
        format!("{} {}", cue.target_class, cue.args_prefix)
    }

    #[tokio::test]
    async fn connect_query_keys_match_the_contract() {
        let c: Value = serde_json::from_str(include_str!(
            "../../testdata/personal_kb_contracts/prevention.json"
        ))
        .unwrap();
        let example = c["request"]["examples"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["name"] == "talos")
            .unwrap();
        let want: std::collections::BTreeSet<String> = example["query"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        let schema = c["request"]["query_schema"]["properties"]
            .as_object()
            .unwrap();
        assert!(want.iter().all(|k| schema.contains_key(k)));
        let (server, _m) = connected(FIXTURE.to_string()).await;
        let reqs = server.received_requests().await.unwrap();
        let got: std::collections::BTreeSet<String> = reqs[0]
            .url
            .query_pairs()
            .map(|(k, _)| k.to_string())
            .collect();
        assert_eq!(got, want);
        assert!(
            reqs[0]
                .url
                .query_pairs()
                .any(|(k, v)| k == "harness" && v == "talos")
        );
    }

    #[tokio::test]
    async fn contract_talos_examples_deny() {
        for name in ["talos_gate_armed", "talos_gate_switch_off"] {
            let body = contract_example(name);
            let p = parse_prevention(body.to_string().as_bytes()).unwrap();
            assert_eq!(p.tool_map.as_ref().unwrap().len(), 6, "{name}");
            let cue = p.cues.iter().find(|c| c.tool == "Bash").expect("bash cue");
            let (_s, memory) = connected(body.to_string()).await;
            let args = json!({"command": command_for(cue)});
            assert_eq!(
                memory.before_tool(&view("bash", &args)),
                Advice::DenyOnce {
                    fact: cue_fact(cue),
                    entry_id: cue.resolution_id.clone()
                },
                "{name}"
            );
        }
    }

    #[tokio::test]
    async fn contract_empty_tool_map_keeps_cues_but_gives_no_advice() {
        let body = contract_example("gate_armed");
        let p = parse_prevention(body.to_string().as_bytes()).unwrap();
        assert!(!p.cues.is_empty());
        let cue = p.cues.iter().find(|c| c.tool == "Bash").expect("bash cue");
        let (_s, memory) = connected(body.to_string()).await;
        let args = json!({"command": command_for(cue)});
        assert_eq!(memory.before_tool(&view("bash", &args)), Advice::None);
        assert!(memory.label().contains("tool_map=0"));
    }

    #[tokio::test]
    async fn contract_absent_tool_map_keeps_no_cues() {
        let mut body = contract_example("talos_gate_armed");
        body.as_object_mut().unwrap().remove("tool_map");
        let p = parse_prevention(body.to_string().as_bytes()).unwrap();
        assert!(p.cues.is_empty());
        let (_s, memory) = connected(body.to_string()).await;
        assert!(memory.label().contains("tool_map=absent"));
        assert!(memory.session_slice().is_some() || p.slice_text.trim().is_empty());
    }

    #[test]
    fn parse_index_truncated() {
        let t = parse(&variant(|v| v["diagnostics"]["index_truncated"] = json!(3)));
        assert_eq!(t.index_truncated, 3);
        let none = parse(&variant(|v| {
            v.as_object_mut().expect("obj").remove("diagnostics");
        }));
        assert_eq!(none.index_truncated, 0);
        let neg = parse(&variant(|v| {
            v["diagnostics"]["index_truncated"] = json!(-1);
        }));
        assert_eq!(neg.index_truncated, 0);
    }

    #[test]
    fn parse_filters_index_elements() {
        let base = fixture()["index"][0].clone();
        let with = |edit: &dyn Fn(&mut Value)| {
            let mut e = base.clone();
            edit(&mut e);
            e
        };
        let rejected = [
            with(&|e| e["observed_once"] = json!(true)),
            with(&|e| e["tool"] = json!("  ")),
            with(&|e| {
                e.as_object_mut().unwrap().remove("tool");
            }),
            with(&|e| e["resolution_id"] = json!("  ")),
            with(&|e| e["corrected_fact"] = json!("")),
            json!(5),
        ];
        for extra in rejected {
            let p = parse(&variant(|v| {
                v["index"]
                    .as_array_mut()
                    .expect("index")
                    .push(extra.clone());
            }));
            assert_eq!(p.cues, vec![fixture_cue()], "{extra}");
            assert_eq!(p.received, 2, "{extra}");
        }

        let null_prefix = parse(&variant(|v| v["index"][0]["args_prefix"] = Value::Null));
        assert_eq!(null_prefix.cues[0].args_prefix, "");
        let padded = parse(&variant(|v| {
            v["index"][0]["target_class"] = json!(" git push ");
        }));
        assert_eq!(padded.cues[0].target_class, "git push");
        let non_string = parse(&variant(|v| v["slice_text"] = json!(5)));
        assert_eq!(non_string.slice_text, "");
    }

    // ---- AC9: cue_fact ------------------------------------------------------

    #[test]
    fn cue_fact_rendering() {
        assert_eq!(cue_fact(&fixture_cue()), FACT_1);
        let bare = |fact: String| IndexCue {
            resolution_id: "kb-x".into(),
            tool: "Bash".into(),
            target_class: "x".into(),
            args_prefix: String::new(),
            wrong_belief: String::new(),
            corrected_fact: fact,
            provenance_label: String::new(),
        };
        assert_eq!(cue_fact(&bare("Use X".into())), "Use X");
        let long = cue_fact(&bare("x".repeat(1200)));
        assert_eq!(long, format!("{}…", "x".repeat(999)));
        let wide = cue_fact(&bare("é".repeat(1200)));
        assert_eq!(wide.chars().count(), 1000);
        assert!(wide.ends_with('…'));
        let exact = cue_fact(&bare("y".repeat(1000)));
        assert_eq!(exact, "y".repeat(1000));
    }

    // ---- AC10: connect ------------------------------------------------------

    #[tokio::test]
    async fn connect_sends_the_pinned_request() {
        let server = MockServer::start().await;
        let mut mock = Mock::given(method("GET"))
            .and(path("/api/kb/prevention"))
            .and(query_param("project", "fake-project"))
            .and(query_param("session_id", "t:1"))
            .and(query_param_is_missing("cwd"))
            .and(header("authorization", "Bearer local-no-auth"));
        for (k, v) in KB_HARNESS_HEADERS {
            mock = mock.and(header(k, v));
        }
        mock.respond_with(ResponseTemplate::new(200).set_body_string(FIXTURE))
            .expect(1)
            .mount(&server)
            .await;
        let memory = connect_to(&server.uri(), Some("fake-project")).await;
        assert_eq!(memory.degraded(), None);
    }

    #[tokio::test]
    async fn connect_sends_the_configured_key() {
        let server = MockServer::start().await;
        Mock::given(header("authorization", "Bearer sk-1"))
            .respond_with(ResponseTemplate::new(200).set_body_string(FIXTURE))
            .expect(1)
            .mount(&server)
            .await;
        let env = env_of(&[
            ("PERSONAL_KB_URL", &server.uri()),
            ("PERSONAL_KB_API_KEY", "sk-1"),
        ]);
        let memory = PersonalKbMemory::connect(
            KbEndpoint::from_env(&env),
            Some("fake-project".into()),
            "t:1",
            PREVENTION_TIMEOUT,
        )
        .await;
        assert_eq!(memory.degraded(), None);
    }

    #[tokio::test]
    async fn connect_refused_is_unreachable() {
        let memory = connect_to("http://127.0.0.1:1", Some("fake-project")).await;
        let d = memory.degraded().expect("degraded");
        assert!(d.starts_with("prevention fetch failed: unreachable"), "{d}");
        assert!(d.contains(" ("), "cause is appended: {d}");
    }

    #[tokio::test]
    async fn connect_timeout() {
        let server = MockServer::start().await;
        Mock::given(any())
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(FIXTURE)
                    .set_delay(Duration::from_secs(2)),
            )
            .mount(&server)
            .await;
        let memory = PersonalKbMemory::connect(
            endpoint(&server.uri()),
            Some("fake-project".into()),
            "t:1",
            Duration::from_millis(200),
        )
        .await;
        assert_eq!(
            memory.degraded().as_deref(),
            Some("prevention fetch failed: timeout")
        );
    }

    #[tokio::test]
    async fn connect_http_and_body_failures() {
        let cases = [
            (
                401,
                FIXTURE.to_string(),
                "prevention fetch failed: http 401",
            ),
            (
                500,
                FIXTURE.to_string(),
                "prevention fetch failed: http 500",
            ),
            (
                200,
                "not json".to_string(),
                "prevention fetch failed: malformed response",
            ),
            (
                200,
                r#"{"gate":{},"index":{}}"#.to_string(),
                "prevention fetch failed: malformed response",
            ),
        ];
        for (status, body, want) in cases {
            let server = prevention_server(status, body).await;
            let memory = connect_to(&server.uri(), Some("fake-project")).await;
            assert_eq!(memory.degraded().as_deref(), Some(want), "{status}");
        }
    }

    async fn silent_server() -> MockServer {
        let server = MockServer::start().await;
        Mock::given(any())
            .respond_with(ResponseTemplate::new(200).set_body_string(FIXTURE))
            .expect(0)
            .mount(&server)
            .await;
        server
    }

    #[tokio::test]
    async fn connect_without_endpoint_or_project_makes_no_request() {
        let server = silent_server().await;
        let remote = PersonalKbMemory::connect(
            Err(REMOTE_NO_KEY.to_string()),
            Some("fake-project".into()),
            "t:1",
            PREVENTION_TIMEOUT,
        )
        .await;
        assert_eq!(remote.degraded().as_deref(), Some(REMOTE_NO_KEY));
        for project in [None, Some("  ")] {
            let memory = connect_to(&server.uri(), project).await;
            assert_eq!(memory.degraded().as_deref(), Some(NO_PROJECT));
        }
        assert_eq!(
            NO_PROJECT,
            "no kb project (pass --kb-project or add a .kb_project file at or above the workspace)"
        );
    }

    #[tokio::test]
    async fn connect_inert_response_is_not_degraded() {
        let inert = r#"{"project":"fake-project","gate":{"enabled":false,"shadow":true,"max_denies":1000,"max_denies_per_turn":1,"max_denies_per_hour":6,"rearm_hours":24},"index":[],"slice":[],"slice_text":"","diagnostics":{},"surprise_capture":"off"}"#;
        let (_server, memory) = connected(inert.to_string()).await;
        assert_eq!(memory.degraded(), None);
        assert_eq!(memory.session_slice(), None);
    }

    // ---- AC11: the Memory impl ---------------------------------------------

    #[tokio::test]
    async fn memory_over_the_fixture() {
        let (server, memory) = connected(FIXTURE.to_string()).await;
        assert_eq!(
            memory.label(),
            format!(
                "personal-kb 127.0.0.1:{} project=fake-project gate=on tool_map=6 index=1 dropped=0 truncated=0",
                port(&server)
            )
        );
        assert_eq!(memory.session_slice(), Some(slice_text()));
        assert!(memory.tools().is_empty());
        assert_eq!(memory.degraded(), None);
        let deny = Advice::DenyOnce {
            fact: FACT_1.to_string(),
            entry_id: "kb-00001".to_string(),
        };
        let cases = [
            (
                "bash",
                json!({"command": "git commit -qam x && git push github main"}),
                deny,
            ),
            (
                "bash",
                json!({"command": "git push origin main"}),
                Advice::None,
            ),
            (
                "Bash",
                json!({"command": "git push github main"}),
                Advice::None,
            ),
            (
                "echo",
                json!({"command": "git push github main"}),
                Advice::None,
            ),
            ("bash", json!({"command": 5}), Advice::None),
            ("bash", json!({}), Advice::None),
        ];
        for (name, args, want) in cases {
            assert_eq!(
                memory.before_tool(&view(name, &args)),
                want,
                "{name} {args}"
            );
        }
    }

    #[tokio::test]
    async fn memory_slice_and_label_variants() {
        let (_s, blank) = connected(variant(|v| v["slice_text"] = json!("  \n"))).await;
        assert_eq!(blank.session_slice(), None);

        let (_s, off) = connected(variant(|v| v["gate"]["enabled"] = json!(false))).await;
        assert!(
            off.label()
                .ends_with("gate=off tool_map=6 index=1 dropped=0 truncated=0"),
            "{}",
            off.label()
        );
        let push = json!({"command": "git push github main"});
        assert!(matches!(
            off.before_tool(&view("bash", &push)),
            Advice::DenyOnce { .. }
        ));
        assert_eq!(off.session_slice(), Some(slice_text()));

        let (_s, dropped) = connected(variant(|v| {
            let mut extra = v["index"][0].clone();
            extra["observed_once"] = json!(true);
            v["index"].as_array_mut().expect("index").push(extra);
        }))
        .await;
        assert!(
            dropped.label().contains("index=1 dropped=1"),
            "{}",
            dropped.label()
        );

        let (_s, truncated) =
            connected(variant(|v| v["diagnostics"]["index_truncated"] = json!(3))).await;
        assert!(
            truncated.label().ends_with("truncated=3"),
            "{}",
            truncated.label()
        );
    }

    #[tokio::test]
    async fn degraded_memory_is_inert() {
        let memory = connect_to("http://127.0.0.1:1", Some("fake-project")).await;
        let push = json!({"command": "git push github main"});
        assert_eq!(memory.session_slice(), None);
        assert!(memory.tools().is_empty());
        assert_eq!(memory.before_tool(&view("bash", &push)), Advice::None);
        assert_eq!(
            memory.label(),
            "personal-kb 127.0.0.1:1 project=fake-project gate=- index=0 dropped=0 truncated=0"
        );

        let no_endpoint =
            PersonalKbMemory::connect(Err("x".into()), None, "t:1", PREVENTION_TIMEOUT).await;
        assert_eq!(
            no_endpoint.label(),
            "personal-kb - project=- gate=- index=0 dropped=0 truncated=0"
        );

        let server = silent_server().await;
        let blank = connect_to(&server.uri(), Some("  ")).await;
        assert!(blank.label().contains(" project=- "), "{}", blank.label());
    }

    // ---- AC17: no secret leakage --------------------------------------------

    #[tokio::test]
    async fn the_key_never_leaks() {
        let env = env_of(&[
            ("PERSONAL_KB_URL", "http://127.0.0.1:1"),
            ("PERSONAL_KB_API_KEY", "sk-SECRET-1"),
        ]);
        let memory = PersonalKbMemory::connect(
            KbEndpoint::from_env(&env),
            Some("p".into()),
            "t:1",
            PREVENTION_TIMEOUT,
        )
        .await;
        assert!(!memory.label().contains("sk-SECRET-1"));
        let d = memory.degraded().expect("degraded");
        assert!(!d.contains("sk-SECRET-1"), "{d}");

        let err = endpoint("http://user:pw@127.0.0.1:1").err().expect("err");
        assert!(!err.contains("pw@") && !err.contains("user:"), "{err}");
    }

    #[test]
    fn no_debug_on_key_holders() {
        let src = include_str!("mod.rs");
        let (body, _) = src.split_once("#[cfg(test)]").expect("test module");
        for ty in ["KbEndpoint", "PersonalKbMemory"] {
            assert!(!body.contains(&format!("Debug for {ty}")), "{ty}");
            let decl = format!("pub struct {ty} {{");
            let at = body.find(&decl).expect("declared");
            let prev = body[..at].trim_end().lines().last().unwrap_or("");
            assert!(!prev.contains("derive"), "{ty}: {prev}");
        }
    }

    // ---- KB MCP tools (AC-ADD-1) --------------------------------------------

    fn rpc_ok(req: &Request, result: &Value) -> ResponseTemplate {
        let id = serde_json::from_slice::<Value>(&req.body).unwrap_or(Value::Null)["id"].clone();
        ResponseTemplate::new(200)
            .set_body_json(json!({"jsonrpc": "2.0", "id": id, "result": result}))
    }

    /// Mount a fake KB MCP server at `/mcp` listing `tools`; `tools/call`
    /// echoes its arguments.
    async fn mount_fake_mcp(server: &MockServer, tools: Vec<Value>) {
        let on = |rpc: &str| {
            Mock::given(method("POST"))
                .and(path("/mcp"))
                .and(body_partial_json(json!({"method": rpc})))
        };
        let init = json!({
            "protocolVersion": "2025-11-25",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake-kb", "version": "1.0"},
        });
        on("initialize")
            .respond_with(move |r: &Request| rpc_ok(r, &init))
            .mount(server)
            .await;
        on("notifications/initialized")
            .respond_with(ResponseTemplate::new(202))
            .mount(server)
            .await;
        let listed = json!({"tools": tools});
        on("tools/list")
            .respond_with(move |r: &Request| rpc_ok(r, &listed))
            .mount(server)
            .await;
        on("tools/call")
            .respond_with(|r: &Request| {
                let body: Value = serde_json::from_slice(&r.body).unwrap_or(Value::Null);
                let text = format!("hits for {}", body["params"]["arguments"]);
                rpc_ok(r, &json!({"content": [{"type": "text", "text": text}]}))
            })
            .mount(server)
            .await;
    }

    fn kb_tool_def(name: &str) -> Value {
        json!({
            "name": name,
            "description": format!("{name} the KB"),
            "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}},
        })
    }

    #[tokio::test]
    async fn attach_kb_tools_registers_the_servers_tools() {
        let server = prevention_server(200, FIXTURE.to_string()).await;
        mount_fake_mcp(
            &server,
            vec![kb_tool_def("kb_search"), kb_tool_def("kb_get")],
        )
        .await;
        let memory = connect_to(&server.uri(), Some("fake-project")).await;
        let (memory, report) = memory.attach_kb_tools(KB_MCP_CONNECT_TIMEOUT).await;
        match report {
            Some(McpConnectReport::Connected {
                name, tools, auth, ..
            }) => {
                assert_eq!(name, "kb");
                assert_eq!(tools, vec!["kb_search", "kb_get"]);
                assert_eq!(auth, "bearer");
            }
            other => panic!("expected Connected, got {other:?}"),
        }
        let tools = memory.tools();
        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert_eq!(names, vec!["kb_search", "kb_get"]);
        assert!(tools.iter().all(|t| t.is_mcp()));
        assert_eq!(tools[0].schema()["name"], "kb_search");
        // A second call hands out the same tools again.
        assert_eq!(memory.tools().len(), 2);
        let out = tools[0].run(json!({"query": "q"}), &ToolCtx::stub()).await;
        assert!(!out.is_error, "{out:?}");

        // Every MCP request carried the KB token and the harness headers.
        let mcp: Vec<Request> = server
            .received_requests()
            .await
            .expect("recorded")
            .into_iter()
            .filter(|r| r.url.path() == "/mcp")
            .collect();
        assert!(mcp.len() >= 3, "{}", mcp.len());
        for r in &mcp {
            let h = |k: &str| {
                r.headers
                    .get(k)
                    .and_then(|v| v.to_str().ok())
                    .map(str::to_string)
            };
            assert_eq!(h("authorization").as_deref(), Some("Bearer local-no-auth"));
            assert_eq!(h("x-kb-harness").as_deref(), Some("talos"));
            assert_eq!(h("x-kb-mode").as_deref(), Some("headless"));
        }
        // The slice and advice are unaffected.
        assert_eq!(memory.session_slice(), Some(slice_text()));
    }

    #[tokio::test]
    async fn attach_kb_tools_failure_is_fail_open() {
        // No /mcp mocks: wiremock answers 404.
        let (_server, memory) = connected(FIXTURE.to_string()).await;
        let (memory, report) = memory.attach_kb_tools(KB_MCP_CONNECT_TIMEOUT).await;
        assert!(
            matches!(report, Some(McpConnectReport::Failed { ref name, .. }) if name == "kb"),
            "{report:?}"
        );
        assert!(memory.tools().is_empty());
        assert_eq!(memory.degraded(), None);
        assert_eq!(memory.session_slice(), Some(slice_text()));
        let push = json!({"command": "git push github main"});
        assert!(matches!(
            memory.before_tool(&view("bash", &push)),
            Advice::DenyOnce { .. }
        ));
    }

    #[tokio::test]
    async fn attach_kb_tools_rejects_a_reserved_name() {
        let server = prevention_server(200, FIXTURE.to_string()).await;
        mount_fake_mcp(&server, vec![kb_tool_def("bash")]).await;
        let memory = connect_to(&server.uri(), Some("fake-project")).await;
        let (memory, report) = memory.attach_kb_tools(KB_MCP_CONNECT_TIMEOUT).await;
        assert!(
            matches!(
                report,
                Some(McpConnectReport::Failed {
                    error_kind: "name_collision",
                    ..
                })
            ),
            "{report:?}"
        );
        assert!(memory.tools().is_empty());
    }

    #[tokio::test]
    async fn attach_kb_tools_is_skipped_when_degraded() {
        let server = silent_server().await;
        let memory = connect_to(&server.uri(), None).await;
        let (memory, report) = memory.attach_kb_tools(KB_MCP_CONNECT_TIMEOUT).await;
        assert!(report.is_none());
        assert!(memory.tools().is_empty());
        let no_endpoint =
            PersonalKbMemory::connect(Err("x".into()), Some("p".into()), "t:1", PREVENTION_TIMEOUT)
                .await;
        let (_, report) = no_endpoint.attach_kb_tools(KB_MCP_CONNECT_TIMEOUT).await;
        assert!(report.is_none());
    }

    // ---- AC12: end to end through the engine -------------------------------

    struct CountingBash {
        runs: Arc<AtomicU32>,
    }

    #[async_trait]
    impl Tool for CountingBash {
        #[allow(clippy::unnecessary_literal_bound)]
        fn name(&self) -> &str {
            "bash"
        }

        fn schema(&self) -> Value {
            json!({"name": "bash", "description": "test bash", "input_schema": {"type": "object"}})
        }

        async fn run(&self, _input: Value, _ctx: &ToolCtx) -> ToolResult {
            self.runs.fetch_add(1, Ordering::SeqCst);
            ToolResult::ok("ran")
        }
    }

    fn usage() -> Usage {
        Usage {
            input_tokens: 1,
            output_tokens: 1,
            cache_read_tokens: None,
            cache_write_tokens: None,
            reasoning_tokens: None,
        }
    }

    fn call_turn(id: &str, name: &str, input: Value) -> AssistantTurn {
        AssistantTurn {
            content: vec![ContentBlock::ToolCall(ToolCallRequest {
                id: id.to_string(),
                name: name.to_string(),
                input,
            })],
            stop_reason: StopReason::ToolUse,
            usage: usage(),
        }
    }

    fn script() -> Vec<AssistantTurn> {
        vec![
            call_turn("c1", "bash", json!({"command": "git push github main"})),
            call_turn("c2", "bash", json!({"command": "git push github main"})),
            call_turn("c3", "bash", json!({"command": "git push origin main"})),
            call_turn(
                "cf",
                FINISH_TOOL_NAME,
                json!({"disposition": "blocked", "decision_needed": "q"}),
            ),
        ]
    }

    fn registry() -> (ToolRegistry, Arc<AtomicU32>) {
        let runs = Arc::new(AtomicU32::new(0));
        let mut registry = ToolRegistry::new();
        registry.register(
            "bash",
            Arc::new(CountingBash {
                runs: Arc::clone(&runs),
            }),
        );
        registry.register(FINISH_TOOL_NAME, Arc::new(FinishTool::default()));
        (registry, runs)
    }

    /// The fed-back `ToolResult` for `call_id`, across every request.
    fn fed_back(backend: &MockBackend, call_id: &str) -> (String, bool) {
        for history in backend.messages_seen().iter().rev() {
            for m in history {
                if let Message::User { content } = m {
                    for b in content {
                        if let UserBlock::ToolResult {
                            call_id: id,
                            content,
                            is_error,
                        } = b
                            && id == call_id
                        {
                            return (content.clone(), *is_error);
                        }
                    }
                }
            }
        }
        panic!("no fed-back result for {call_id}");
    }

    async fn run_with(
        memory: &Arc<dyn Memory>,
        registry: &ToolRegistry,
        mode: MemoryMode,
        turns: Vec<AssistantTurn>,
    ) -> (MockBackend, RunStats) {
        let backend = MockBackend::from_turns(turns);
        let result = crate::engine::run(
            &backend,
            registry,
            &ToolCtx::stub(),
            &RunConfig::new("t", 10).with_memory(Arc::clone(memory), mode),
        )
        .await;
        (backend, result.stats)
    }

    #[tokio::test]
    async fn end_to_end_with_the_engine() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path(PREVENTION_PATH))
            .respond_with(ResponseTemplate::new(200).set_body_string(FIXTURE))
            .expect(1)
            .mount(&server)
            .await;
        let env = env_of(&[("PERSONAL_KB_URL", &server.uri())]);
        let memory: Arc<dyn Memory> = Arc::new(
            PersonalKbMemory::connect(
                KbEndpoint::from_env(&env),
                Some("fake-project".into()),
                "t:1",
                PREVENTION_TIMEOUT,
            )
            .await,
        );
        let (registry, runs) = registry();
        let annotated = format!("ran\n\n{}", memory::annotation_block(FACT_1, "kb-00001"));

        // Run A: deny-once.
        let (backend, stats) = run_with(&memory, &registry, MemoryMode::DenyOnce, script()).await;
        let system = backend.systems_seen()[0].clone().expect("system");
        assert!(system.contains(memory::MEMORY_SLICE_HEADING), "{system}");
        assert!(system.contains("- Use cargo nextest [deliberate/asserted; kb-00002]"));
        assert!(
            backend.tools_seen()[0]
                .iter()
                .all(|s| !s["name"].as_str().unwrap_or("").starts_with("kb_"))
        );
        assert_eq!(
            fed_back(&backend, "c1"),
            (memory::deny_content(FACT_1, "kb-00001"), true)
        );
        assert_eq!(fed_back(&backend, "c2"), (annotated.clone(), false));
        assert_eq!(fed_back(&backend, "c3"), ("ran".to_string(), false));
        assert_eq!(runs.load(Ordering::SeqCst), 2);
        assert_eq!(stats.memory_denies, 1);
        assert_eq!(stats.memory_annotations, 1);
        assert_eq!(stats.memory_deny_overrides, 1);
        assert_eq!(stats.memory_kb_tool_calls, 0);
        assert_eq!(stats.memory_degraded, None);

        // Run B: annotate, same memory (no second fetch).
        let (backend, stats) = run_with(&memory, &registry, MemoryMode::Annotate, script()).await;
        assert_eq!(fed_back(&backend, "c1"), (annotated.clone(), false));
        assert_eq!(fed_back(&backend, "c2"), (annotated, false));
        assert_eq!(fed_back(&backend, "c3"), ("ran".to_string(), false));
        assert_eq!(runs.load(Ordering::SeqCst), 5);
        assert_eq!(stats.memory_denies, 0);
        assert_eq!(stats.memory_annotations, 2);
    }

    #[tokio::test]
    async fn end_to_end_kb_tools_are_registered_and_counted() {
        let server = prevention_server(200, FIXTURE.to_string()).await;
        mount_fake_mcp(&server, vec![kb_tool_def("kb_search")]).await;
        let memory = connect_to(&server.uri(), Some("fake-project")).await;
        let (memory, _) = memory.attach_kb_tools(KB_MCP_CONNECT_TIMEOUT).await;
        let memory: Arc<dyn Memory> = Arc::new(memory);
        let (registry, _) = registry();
        let turns = vec![
            call_turn("k1", "kb_search", json!({"query": "push"})),
            call_turn(
                "cf",
                FINISH_TOOL_NAME,
                json!({"disposition": "blocked", "decision_needed": "q"}),
            ),
        ];
        let (backend, stats) = run_with(&memory, &registry, MemoryMode::Annotate, turns).await;
        assert!(
            backend.tools_seen()[0]
                .iter()
                .any(|s| s["name"] == "kb_search")
        );
        let (content, is_error) = fed_back(&backend, "k1");
        assert!(!is_error, "{content}");
        assert!(content.contains("hits for"), "{content}");
        assert_eq!(stats.memory_kb_tool_calls, 1);
        assert_eq!(stats.mcp_tool_calls, 1);

        // Memory off: the KB tools never reach the model.
        let turns = vec![call_turn(
            "cf",
            FINISH_TOOL_NAME,
            json!({"disposition": "blocked", "decision_needed": "q"}),
        )];
        let (backend, _) = run_with(&memory, &registry, MemoryMode::Off, turns).await;
        assert!(
            backend.tools_seen()[0]
                .iter()
                .all(|s| s["name"] != "kb_search")
        );
    }

    // ---- failed-tool events (POST /api/kb/event) ---------------------------

    const EVENT_CONTRACT: &str = include_str!("../../testdata/personal_kb_contracts/event.json");

    fn event_contract() -> Value {
        serde_json::from_str(EVENT_CONTRACT).expect("event.json is JSON")
    }

    fn event_example(name: &str) -> Value {
        event_contract()["request"]["examples"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["name"] == name)
            .unwrap()["body"]
            .clone()
    }

    fn client_supplied() -> std::collections::BTreeSet<String> {
        event_contract()["fields"]["client_supplied"]
            .as_array()
            .unwrap()
            .iter()
            .map(|k| k.as_str().unwrap().to_string())
            .collect()
    }

    /// A native tool that succeeds iff its input contains `OK`.
    struct FlakyTool(&'static str);

    #[async_trait]
    impl Tool for FlakyTool {
        fn name(&self) -> &str {
            self.0
        }

        fn schema(&self) -> Value {
            json!({"name": self.0, "description": "flaky", "input_schema": {"type": "object"}})
        }

        async fn run(&self, input: Value, _ctx: &ToolCtx) -> ToolResult {
            if input.to_string().contains("OK") {
                ToolResult::ok("fine")
            } else {
                ToolResult::error(format!("{} failed", self.0))
            }
        }
    }

    fn flaky_registry() -> ToolRegistry {
        let mut registry = ToolRegistry::new();
        for name in ["bash", "edit_file", "read_file", "list_files", "run_checks"] {
            registry.register(name, Arc::new(FlakyTool(name)));
        }
        registry.register(FINISH_TOOL_NAME, Arc::new(FinishTool::default()));
        registry
    }

    fn finish_turn() -> AssistantTurn {
        call_turn(
            "cf",
            FINISH_TOOL_NAME,
            json!({"disposition": "blocked", "decision_needed": "q"}),
        )
    }

    /// One turn carrying every `(id, name, input)` call.
    fn calls_turn(calls: Vec<(String, &str, Value)>) -> AssistantTurn {
        AssistantTurn {
            content: calls
                .into_iter()
                .map(|(id, name, input)| {
                    ContentBlock::ToolCall(ToolCallRequest {
                        id,
                        name: name.to_string(),
                        input,
                    })
                })
                .collect(),
            stop_reason: StopReason::ToolUse,
            usage: usage(),
        }
    }

    /// A fake KB serving `prevention` and answering events with `event`.
    async fn event_kb(prevention: Value, event: ResponseTemplate) -> MockServer {
        let server = prevention_server(200, prevention.to_string()).await;
        Mock::given(method("POST"))
            .and(path(EVENT_PATH))
            .respond_with(event)
            .mount(&server)
            .await;
        server
    }

    async fn event_memory(server: &MockServer) -> Arc<dyn Memory> {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let host = dir.path().join("hostname");
        std::fs::write(&host, "  test-host\n").expect("write");
        let memory = connect_to(&server.uri(), Some("fake-project"))
            .await
            .with_event_context_from(
                Path::new("/ws/root"),
                "9.9.9-test",
                "glm-5.3-flash:cloud",
                &host,
            );
        Arc::new(memory)
    }

    async fn event_requests(server: &MockServer) -> Vec<Request> {
        server
            .received_requests()
            .await
            .unwrap()
            .into_iter()
            .filter(|r| r.url.path() == EVENT_PATH)
            .collect()
    }

    async fn event_bodies(server: &MockServer) -> Vec<Value> {
        event_requests(server)
            .await
            .iter()
            .map(|r| serde_json::from_slice(&r.body).unwrap())
            .collect()
    }

    async fn run_events(
        server: &MockServer,
        mode: MemoryMode,
        turns: Vec<AssistantTurn>,
    ) -> crate::engine::RunResult {
        let memory = event_memory(server).await;
        crate::engine::run(
            &MockBackend::from_turns(turns),
            &flaky_registry(),
            &ToolCtx::stub(),
            &RunConfig::new("t", 10).with_memory(memory, mode),
        )
        .await
    }

    fn assert_blocked(result: &crate::engine::RunResult) {
        assert!(
            matches!(
                result.outcome,
                crate::engine::LoopOutcome::Finished(
                    crate::run_record::Disposition::Blocked { .. }
                )
            ),
            "{:?}",
            result.outcome
        );
    }

    fn ok_event() -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_json(json!({
            "recorded": true, "cue_key": "x", "normalizer_version": 1, "reason": "recorded"
        }))
    }

    #[test]
    fn event_constants_are_pinned() {
        assert_eq!(EVENT_PATH, "/api/kb/event");
        assert_eq!(EVENT_PATH, event_contract()["path"]);
        assert_eq!(event_contract()["method"], "POST");
        assert_eq!(EVENT_TIMEOUT, Duration::from_secs(2));
        assert_eq!(MAX_EVENTS_PER_RUN, 50);
        assert_eq!(MAX_EVENT_FIELD_CHARS, 2000);
    }

    #[test]
    fn event_helpers() {
        assert_eq!(truncate_chars("héllo", 2), "hé");
        assert_eq!(truncate_chars("hé", 5), "hé");
        let dir = tempfile::TempDir::new().unwrap();
        assert_eq!(read_host(&dir.path().join("missing")), None);
        std::fs::write(dir.path().join("blank"), " \n").unwrap();
        assert_eq!(read_host(&dir.path().join("blank")), None);
        std::fs::write(dir.path().join("h"), "box-1\n").unwrap();
        assert_eq!(read_host(&dir.path().join("h")).as_deref(), Some("box-1"));

        let input = |name: &str, args: Value| event_tool_input(&view(name, &args));
        let long = "é".repeat(MAX_EVENT_FIELD_CHARS + 5);
        assert_eq!(
            input("bash", json!({"command": long, "timeout": 5})),
            Some(json!({"command": "é".repeat(MAX_EVENT_FIELD_CHARS)}))
        );
        assert_eq!(
            input("bash", json!({"command": 3})),
            Some(json!({"command": 3}))
        );
        assert_eq!(input("bash", json!({})), Some(json!({"command": null})));
        assert_eq!(
            input(
                "edit_file",
                json!({"path": "a", "old_string": "x", "new_string": "y"})
            ),
            Some(json!({"path": "a"}))
        );
        assert_eq!(input("read_file", json!({})), Some(json!({"path": null})));
        assert_eq!(input("list_files", json!({})), Some(json!({"path": "."})));
        assert_eq!(
            input("list_files", json!({"path": "src"})),
            Some(json!({"path": "src"}))
        );
        assert_eq!(input("run_checks", json!({})), None);
        assert_eq!(input("kb_search", json!({})), None);
    }

    #[test]
    fn event_context_uses_the_proc_hostname() {
        let m = futures_free_degraded().with_event_context(Path::new("/w"), "1.0", " m-1 ");
        assert_eq!(m.events.cwd.as_deref(), Some("/w"));
        assert_eq!(m.events.client_version.as_deref(), Some("1.0"));
        assert_eq!(m.events.engine.as_deref(), Some("m-1"));
        assert_eq!(
            m.events.host,
            read_host(Path::new(HOSTNAME_PATH)),
            "host comes from {HOSTNAME_PATH}"
        );
    }

    /// A degraded memory built without a runtime (no request is made).
    fn futures_free_degraded() -> PersonalKbMemory {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        rt.block_on(PersonalKbMemory::connect(
            Err("x".into()),
            None,
            "t:1",
            PREVENTION_TIMEOUT,
        ))
    }

    // AC7 (a)
    #[tokio::test]
    async fn failing_bash_posts_one_contract_shaped_event() {
        let server = event_kb(contract_example("talos_gate_armed"), ok_event()).await;
        let result = run_events(
            &server,
            MemoryMode::Annotate,
            vec![
                call_turn("c1", "bash", json!({"command": "git push mirror feat/x"})),
                finish_turn(),
            ],
        )
        .await;
        assert_blocked(&result);
        let requests = event_requests(&server).await;
        assert_eq!(requests.len(), 1);
        let r = &requests[0];
        assert_eq!(r.method.as_str(), "POST");
        assert_eq!(r.headers["authorization"], "Bearer local-no-auth");
        for (k, v) in KB_HARNESS_HEADERS {
            assert_eq!(r.headers[k], v);
        }
        let body: Value = serde_json::from_slice(&r.body).unwrap();
        let keys: std::collections::BTreeSet<String> =
            body.as_object().unwrap().keys().cloned().collect();
        assert_eq!(keys, client_supplied());
        let example = event_example("talos_bash_failure");
        for k in [
            "type",
            "harness",
            "mode",
            "tool_name",
            "tool_input",
            "is_error",
            "is_interrupt",
        ] {
            assert_eq!(body[k], example[k], "{k}");
        }
        assert_eq!(body["event_id"], "t:1:1:c1");
        assert_eq!(body["session_id"], "t:1");
        assert_eq!(body["tool_use_id"], "c1");
        assert_eq!(body["engine"], Value::Null);
        assert_eq!(body["host"], "test-host");
        assert_eq!(body["hook_version"], "9.9.9-test");
        assert_eq!(body["cwd"], "/ws/root");
        assert_eq!(body["project"], "fake-project");
        assert!(body["duration_ms"].is_u64(), "{body}");
        assert!(body["ts"].as_str().unwrap().ends_with('Z'), "{body}");
        let error = body["error"].as_str().unwrap();
        assert!(error.starts_with("bash failed"), "{error}");
        assert_eq!(result.stats.memory_events_sent, 1);
        assert_eq!(result.stats.memory_events_failed, 0);
        assert_eq!(result.stats.memory_events_capped, 0);
    }

    // AC7 (b)
    #[tokio::test]
    async fn failing_edit_file_sends_only_the_path() {
        let server = event_kb(contract_example("talos_gate_armed"), ok_event()).await;
        run_events(
            &server,
            MemoryMode::DenyOnce,
            vec![
                call_turn(
                    "e1",
                    "edit_file",
                    json!({"path": "src/a.py", "old_string": "secret", "new_string": "body"}),
                ),
                finish_turn(),
            ],
        )
        .await;
        let bodies = event_bodies(&server).await;
        assert_eq!(bodies.len(), 1);
        assert_eq!(bodies[0]["tool_input"], json!({"path": "src/a.py"}));
        assert_eq!(bodies[0]["tool_name"], "edit_file");
        let example = event_example("talos_edit_file_failure");
        assert_eq!(bodies[0]["tool_input"], example["tool_input"]);
    }

    // AC7 (c)
    #[tokio::test]
    async fn success_run_checks_and_denied_calls_post_nothing() {
        let server = event_kb(contract_example("talos_gate_armed"), ok_event()).await;
        let result = run_events(
            &server,
            MemoryMode::DenyOnce,
            vec![
                call_turn("ok", "bash", json!({"command": "echo OK"})),
                call_turn("rc", "run_checks", json!({})),
                call_turn("deny", "bash", json!({"command": "git push mirror feat/x"})),
                finish_turn(),
            ],
        )
        .await;
        assert_blocked(&result);
        assert_eq!(result.stats.memory_denies, 1);
        assert!(event_requests(&server).await.is_empty());
        assert_eq!(result.stats.memory_events_sent, 0);
        assert_eq!(result.stats.memory_events_failed, 0);
    }

    // AC7 (d)
    #[tokio::test]
    async fn no_tool_map_posts_nothing() {
        let mut body = contract_example("talos_gate_armed");
        body.as_object_mut().unwrap().remove("tool_map");
        let server = event_kb(body, ok_event()).await;
        run_events(
            &server,
            MemoryMode::Annotate,
            vec![
                call_turn("c1", "bash", json!({"command": "false"})),
                finish_turn(),
            ],
        )
        .await;
        assert!(event_requests(&server).await.is_empty());
    }

    #[tokio::test]
    async fn degraded_memory_posts_nothing_even_when_called_directly() {
        let memory = connect_to("http://127.0.0.1:1", None).await;
        let args = json!({"command": "false"});
        memory.after_tool(
            &view("bash", &args),
            &ToolResultView {
                call_id: "c1",
                iteration: 1,
                is_error: true,
                content: "boom",
                duration_ms: None,
            },
        );
        assert_eq!(memory.flush().await, MemoryFlush::default());
    }

    // AC7 (e)
    #[tokio::test]
    async fn same_call_id_in_two_iterations_gets_distinct_event_ids() {
        let server = event_kb(contract_example("talos_gate_armed"), ok_event()).await;
        run_events(
            &server,
            MemoryMode::Annotate,
            vec![
                call_turn("ollama-call-0", "read_file", json!({"path": "a"})),
                call_turn("ollama-call-0", "list_files", json!({})),
                finish_turn(),
            ],
        )
        .await;
        let mut ids: Vec<String> = event_bodies(&server)
            .await
            .iter()
            .map(|b| b["event_id"].as_str().unwrap().to_string())
            .collect();
        ids.sort();
        assert_eq!(ids, ["t:1:1:ollama-call-0", "t:1:2:ollama-call-0"]);
    }

    // AC7 (f)
    #[tokio::test]
    async fn a_slow_kb_does_not_hold_the_run() {
        let server = event_kb(
            contract_example("talos_gate_armed"),
            ok_event().set_delay(Duration::from_secs(10)),
        )
        .await;
        let started = std::time::Instant::now();
        let result = run_events(
            &server,
            MemoryMode::Annotate,
            vec![
                call_turn("c1", "bash", json!({"command": "false"})),
                finish_turn(),
            ],
        )
        .await;
        assert!(
            started.elapsed() < Duration::from_secs(4),
            "{:?}",
            started.elapsed()
        );
        assert_blocked(&result);
        assert_eq!(result.stats.memory_events_sent, 0);
    }

    #[tokio::test]
    async fn a_cut_off_flush_aborts_its_pending_posts() {
        let server = event_kb(
            contract_example("talos_gate_armed"),
            ok_event().set_delay(Duration::from_secs(10)),
        )
        .await;
        let memory = event_memory(&server).await;
        let args = json!({"command": "false"});
        memory.after_tool(
            &view("bash", &args),
            &ToolResultView {
                call_id: "c1",
                iteration: 1,
                is_error: true,
                content: "boom",
                duration_ms: Some(1),
            },
        );
        let cut = tokio::time::timeout(Duration::from_millis(200), memory.flush()).await;
        assert!(cut.is_err(), "flush must still be pending");
        // The aborted post is gone: a second flush has nothing to settle.
        assert_eq!(memory.flush().await, MemoryFlush::default());
    }

    // AC7 (g)
    #[tokio::test]
    async fn a_500_counts_as_failed() {
        let server = event_kb(
            contract_example("talos_gate_armed"),
            ResponseTemplate::new(500),
        )
        .await;
        let result = run_events(
            &server,
            MemoryMode::Annotate,
            vec![
                call_turn("c1", "bash", json!({"command": "false"})),
                finish_turn(),
            ],
        )
        .await;
        assert_blocked(&result);
        assert_eq!(event_requests(&server).await.len(), 1);
        assert_eq!(result.stats.memory_events_failed, 1);
        assert_eq!(result.stats.memory_events_sent, 0);
    }

    // AC7 (h)
    #[tokio::test]
    async fn the_51st_failure_is_capped() {
        let server = event_kb(contract_example("talos_gate_armed"), ok_event()).await;
        let calls = (0..=MAX_EVENTS_PER_RUN)
            .map(|i| {
                (
                    format!("c{i}"),
                    "bash",
                    json!({"command": format!("false {i}")}),
                )
            })
            .collect();
        let result = run_events(
            &server,
            MemoryMode::Annotate,
            vec![calls_turn(calls), finish_turn()],
        )
        .await;
        assert_blocked(&result);
        assert_eq!(event_requests(&server).await.len(), 50);
        assert_eq!(result.stats.memory_events_sent, 50);
        assert_eq!(result.stats.memory_events_capped, 1);
        assert_eq!(result.stats.memory_events_failed, 0);
    }

    // ---- turn digests (POST /api/kb/turn) ----------------------------------

    const TURN_CONTRACT: &str = include_str!("../../testdata/personal_kb_contracts/turn.json");

    fn turn_contract() -> Value {
        serde_json::from_str(TURN_CONTRACT).expect("turn.json is JSON")
    }

    fn turn_example(name: &str) -> Value {
        turn_contract()["request"]["examples"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["name"] == name)
            .unwrap()["body"]
            .clone()
    }

    fn keys_of(v: &Value) -> std::collections::BTreeSet<String> {
        v.as_object().unwrap().keys().cloned().collect()
    }

    /// Validate `body` against turn.json's request schema and assert exact
    /// key sets for the body and for every item (unknown keys are dropped
    /// silently by the KB, so the schema alone cannot catch them).
    fn assert_turn_contract(body: &Value) {
        let contract = turn_contract();
        let schema = &contract["request"]["schema"];
        let validator = jsonschema::validator_for(schema).expect("schema compiles");
        let errors: Vec<String> = validator
            .iter_errors(body)
            .map(|e| format!("{e} at {}", e.instance_path()))
            .collect();
        assert!(errors.is_empty(), "schema errors: {errors:#?}\n{body}");
        assert_eq!(keys_of(body), keys_of(&schema["properties"]), "{body}");
        let mapping = &schema["properties"]["items"]["items"]["discriminator"]["mapping"];
        for item in body["items"].as_array().expect("items") {
            let kind = item["kind"].as_str().expect("kind");
            let def = mapping[kind]
                .as_str()
                .and_then(|r| r.strip_prefix("#/$defs/"))
                .unwrap_or_else(|| panic!("unknown kind {kind}"));
            assert_eq!(
                keys_of(item),
                keys_of(&schema["$defs"][def]["properties"]),
                "{item}"
            );
        }
    }

    fn recorded() -> ResponseTemplate {
        let body = turn_contract()["response"]["examples"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["name"] == "recorded")
            .unwrap()["body"]
            .clone();
        ResponseTemplate::new(200).set_body_json(body)
    }

    /// A fake KB serving `prevention`, answering events 200 and digests with
    /// `turn`.
    async fn turn_kb(prevention: Value, turn: ResponseTemplate) -> MockServer {
        let server = event_kb(prevention, ok_event()).await;
        Mock::given(method("POST"))
            .and(path(TURN_PATH))
            .respond_with(turn)
            .mount(&server)
            .await;
        server
    }

    async fn digest_memory(server: &MockServer, engine: &str) -> PersonalKbMemory {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let host = dir.path().join("hostname");
        std::fs::write(&host, "test-host\n").expect("write");
        connect_to(&server.uri(), Some("fake-project"))
            .await
            .with_event_context_from(Path::new("/ws/root"), "9.9.9-test", engine, &host)
    }

    /// Turn requests sorted by `turn_index`, every body contract-checked.
    async fn turn_requests(server: &MockServer) -> Vec<Request> {
        let mut requests: Vec<Request> = server
            .received_requests()
            .await
            .unwrap()
            .into_iter()
            .filter(|r| r.url.path() == TURN_PATH)
            .collect();
        requests.sort_by_key(|r| {
            serde_json::from_slice::<Value>(&r.body).unwrap()["turn_index"]
                .as_u64()
                .unwrap()
        });
        for r in &requests {
            assert_turn_contract(&serde_json::from_slice(&r.body).unwrap());
        }
        requests
    }

    async fn digests(server: &MockServer) -> Vec<Value> {
        turn_requests(server)
            .await
            .iter()
            .map(|r| serde_json::from_slice(&r.body).unwrap())
            .collect()
    }

    async fn run_digests(
        memory: PersonalKbMemory,
        mode: MemoryMode,
        turns: Vec<AssistantTurn>,
        max_iterations: u32,
    ) -> (MockBackend, crate::engine::RunResult) {
        let backend = MockBackend::from_turns(turns);
        let result = crate::engine::run(
            &backend,
            &flaky_registry(),
            &ToolCtx::stub(),
            &RunConfig::new("t", max_iterations).with_memory(Arc::new(memory), mode),
        )
        .await;
        (backend, result)
    }

    fn tc(id: &str, tool: &str, target: &str) -> Value {
        json!({"kind": "tool_call", "tool_use_id": id, "tool": tool, "target": target, "target_class": ""})
    }

    fn tr(id: &str, is_error: bool, excerpt: &str) -> Value {
        json!({"kind": "tool_result", "tool_use_id": id, "is_error": is_error, "excerpt": excerpt})
    }

    fn corr(trigger: &str, detail: &str, resolved_by: &[&str], resolved: bool) -> Value {
        json!({
            "kind": "harness_correction", "trigger": trigger, "detail": detail,
            "resolved_by": resolved_by, "resolved": resolved
        })
    }

    fn items(body: &Value) -> Vec<Value> {
        body["items"].as_array().unwrap().clone()
    }

    fn turn_of(content: Vec<ContentBlock>) -> AssistantTurn {
        AssistantTurn {
            content,
            stop_reason: StopReason::ToolUse,
            usage: usage(),
        }
    }

    fn tool(id: &str, name: &str, input: Value) -> ContentBlock {
        ContentBlock::ToolCall(ToolCallRequest {
            id: id.into(),
            name: name.into(),
            input,
        })
    }

    #[test]
    fn turn_constants_are_pinned() {
        let c = turn_contract();
        let schema = &c["request"]["schema"];
        let defs = &schema["$defs"];
        assert_eq!(TURN_PATH, "/api/kb/turn");
        assert_eq!(TURN_PATH, c["path"]);
        assert_eq!(c["method"], "POST");
        assert_eq!(TURN_TIMEOUT, Duration::from_secs(2));
        assert_eq!(DIGEST_PERIOD.as_secs(), 600);
        assert_eq!(KB_ENGINE_HEADER, "X-KB-Engine");
        assert_eq!(MAX_TURN_BODY_BYTES, 65_536);
        assert_eq!(MAX_TURN_BODY_BYTES as u64, c["request"]["max_body_bytes"]);
        assert_eq!(MAX_TURN_ITEMS, 200);
        assert_eq!(
            MAX_TURN_ITEMS as u64,
            schema["properties"]["items"]["maxItems"]
        );
        assert_eq!(MAX_TURN_TEXT_CHARS, 2000);
        for def in ["TurnReasoningItem", "TurnAssistantTextItem"] {
            assert_eq!(
                MAX_TURN_TEXT_CHARS as u64,
                defs[def]["properties"]["text"]["maxLength"]
            );
        }
        assert_eq!(MAX_TURN_EXCERPT_CHARS, 1500);
        assert_eq!(
            MAX_TURN_EXCERPT_CHARS as u64,
            defs["TurnToolResultItem"]["properties"]["excerpt"]["maxLength"]
        );
        assert_eq!(
            MAX_TURN_EXCERPT_CHARS as u64,
            defs["TurnHarnessCorrectionItem"]["properties"]["detail"]["maxLength"]
        );
        assert_eq!(MAX_RESOLVED_BY, 200);
        assert_eq!(
            MAX_RESOLVED_BY as u64,
            defs["TurnHarnessCorrectionItem"]["properties"]["resolved_by"]["maxItems"]
        );
        assert_eq!(MAX_TURN_TARGET_CHARS, 500);
        assert_eq!(
            MAX_TURN_TARGET_CHARS as u64,
            defs["TurnToolCallItem"]["properties"]["target"]["maxLength"]
        );
        assert_eq!(MAX_TURN_PROMPT_CHARS, 4000);
        for key in ["user_prompt", "final_message"] {
            assert_eq!(
                MAX_TURN_PROMPT_CHARS as u64,
                schema["properties"][key]["anyOf"][0]["maxLength"]
            );
        }
    }

    #[test]
    fn turn_contract_checker_accepts_the_vendored_talos_example() {
        let example = turn_example("talos_turn");
        assert_turn_contract(&example);
        let mut extra = example;
        extra["items"][0]["surprise"] = json!(1);
        let caught = std::panic::catch_unwind(|| assert_turn_contract(&extra));
        assert!(caught.is_err(), "an extra item key must fail the checker");
    }

    // AC15 (a)
    #[tokio::test]
    async fn a_talos_run_posts_one_contract_shaped_digest() {
        let server = turn_kb(contract_example("talos_gate_armed"), recorded()).await;
        let memory = digest_memory(&server, "glm-5.3-flash:cloud").await;
        let turns = vec![
            turn_of(vec![
                ContentBlock::Reasoning {
                    text: "weighing the remote".into(),
                    opaque: None,
                },
                ContentBlock::Text("Pushing now.".into()),
                tool("c1", "bash", json!({"command": "git push mirror feat/x"})),
            ]),
            call_turn(
                "e1",
                "edit_file",
                json!({"path": "src/a.py", "old_string": "x", "new_string": "OK"}),
            ),
            finish_turn(),
        ];
        let (_, result) = run_digests(memory, MemoryMode::Annotate, turns, 10).await;
        assert_blocked(&result);
        let requests = turn_requests(&server).await;
        assert_eq!(requests.len(), 1);
        let r = &requests[0];
        assert_eq!(r.headers["authorization"], "Bearer local-no-auth");
        assert_eq!(r.headers["x-kb-harness"], "talos");
        assert_eq!(r.headers["x-kb-mode"], "headless");
        assert_eq!(r.headers["x-kb-engine"], "glm-5.3-flash:cloud");
        assert_eq!(r.headers["content-type"], "application/json");
        let body: Value = serde_json::from_slice(&r.body).unwrap();
        assert_eq!(body["event_id"], "t:1:0");
        assert_eq!(body["session_id"], "t:1");
        assert_eq!(body["turn_index"], 0);
        assert_eq!(body["harness"], "talos");
        assert_eq!(body["mode"], "headless");
        assert_eq!(body["engine"], Value::Null);
        assert_eq!(body["host"], "test-host");
        assert_eq!(body["hook_version"], "9.9.9-test");
        assert_eq!(body["project"], "fake-project");
        assert_eq!(body["user_prompt"], "t");
        assert_eq!(body["final_message"], "q");
        assert_eq!(body["truncated"], false);
        assert!(body["ts"].as_str().unwrap().ends_with('Z'), "{body}");
        assert_eq!(
            items(&body),
            vec![
                json!({"kind": "reasoning", "text": "weighing the remote", "truncated": false}),
                json!({"kind": "assistant_text", "text": "Pushing now."}),
                tc("1:c1", "bash", "git push mirror feat/x"),
                tr("1:c1", true, "bash failed"),
                tc("2:e1", "edit_file", "src/a.py"),
                tr("2:e1", false, ""),
                tc("3:cf", "finish", ""),
                tr("3:cf", false, ""),
            ]
        );
        assert_eq!(result.stats.memory_digests_sent, 1);
        assert_eq!(result.stats.memory_digests_failed, 0);
        let events = event_requests(&server).await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].headers["x-kb-engine"], "glm-5.3-flash:cloud");
    }

    // AC15 (b)
    #[tokio::test]
    async fn digests_close_on_the_period_and_a_correction_names_an_earlier_digest() {
        let server = turn_kb(contract_example("talos_gate_armed"), recorded()).await;
        let memory = digest_memory(&server, "m")
            .await
            .with_digest_period(Duration::ZERO);
        let turns = vec![
            call_turn("rc1", "run_checks", json!({})),
            call_turn("e1", "edit_file", json!({"path": "OK"})),
            call_turn("rc2", "run_checks", json!({"x": "OK"})),
            finish_turn(),
        ];
        let (_, result) = run_digests(memory, MemoryMode::Annotate, turns, 10).await;
        let bodies = digests(&server).await;
        let ids: Vec<&str> = bodies
            .iter()
            .map(|b| b["event_id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, ["t:1:0", "t:1:1", "t:1:2", "t:1:3"]);
        assert_eq!(
            items(&bodies[0]),
            vec![
                tc("1:rc1", "run_checks", ""),
                tr("1:rc1", true, "run_checks failed")
            ]
        );
        assert_eq!(
            items(&bodies[1]),
            vec![tc("2:e1", "edit_file", "OK"), tr("2:e1", false, "")]
        );
        assert_eq!(
            items(&bodies[2]),
            vec![
                tc("3:rc2", "run_checks", ""),
                tr("3:rc2", false, ""),
                corr("gate_red", "run_checks failed", &["2:e1"], true),
            ]
        );
        assert_eq!(
            items(&bodies[3]),
            vec![tc("4:cf", "finish", ""), tr("4:cf", false, "")]
        );
        assert_eq!(bodies[0]["user_prompt"], "t");
        for b in &bodies[1..] {
            assert_eq!(b["user_prompt"], Value::Null);
        }
        for b in &bodies[..3] {
            assert_eq!(b["final_message"], Value::Null);
        }
        assert_eq!(bodies[3]["final_message"], "q");
        assert_eq!(result.stats.memory_digests_sent, 4);
    }

    // AC15 (c)
    #[tokio::test]
    async fn an_unresolved_correction_lands_in_the_final_digest() {
        let server = turn_kb(contract_example("talos_gate_armed"), recorded()).await;
        let memory = digest_memory(&server, "m").await;
        let turns = vec![call_turn("rc1", "run_checks", json!({})), finish_turn()];
        run_digests(memory, MemoryMode::Annotate, turns, 10).await;
        let bodies = digests(&server).await;
        assert_eq!(bodies.len(), 1);
        assert_eq!(
            items(&bodies[0]).last().unwrap(),
            &corr("gate_red", "run_checks failed", &[], false)
        );
    }

    // AC15 (d)
    #[tokio::test]
    async fn a_repeat_trigger_folds_and_an_accepted_done_resolves() {
        let server = turn_kb(contract_example("talos_gate_armed"), recorded()).await;
        let memory = digest_memory(&server, "m").await;
        let turns = vec![
            call_turn("rc1", "run_checks", json!({})),
            call_turn("rc2", "run_checks", json!({})),
            call_turn("b1", "bash", json!({"command": "echo OK"})),
            call_turn(
                "cf",
                FINISH_TOOL_NAME,
                json!({"disposition": "done", "summary": "fixed"}),
            ),
        ];
        let (_, result) = run_digests(memory, MemoryMode::Annotate, turns, 10).await;
        assert!(
            matches!(
                result.outcome,
                crate::engine::LoopOutcome::Finished(crate::run_record::Disposition::Done { .. })
            ),
            "{:?}",
            result.outcome
        );
        let bodies = digests(&server).await;
        assert_eq!(bodies.len(), 1);
        let all = items(&bodies[0]);
        let corrections: Vec<&Value> = all
            .iter()
            .filter(|i| i["kind"] == "harness_correction")
            .collect();
        assert_eq!(
            corrections,
            [&corr(
                "gate_red",
                "run_checks failed",
                &["2:rc2", "3:b1"],
                true
            )]
        );
        let at = all
            .iter()
            .position(|i| i == &tr("4:cf", false, ""))
            .unwrap();
        assert_eq!(all[at + 1], *corrections[0]);
        assert_eq!(bodies[0]["final_message"], "fixed");
    }

    // AC15 (e)
    #[tokio::test]
    async fn a_digest_closes_at_the_item_budget() {
        let server = turn_kb(contract_example("talos_gate_armed"), recorded()).await;
        let memory = digest_memory(&server, "m").await;
        let calls = (0..=100)
            .map(|i| {
                (
                    format!("r{i}"),
                    "read_file",
                    json!({"path": format!("OK{i}")}),
                )
            })
            .collect();
        run_digests(
            memory,
            MemoryMode::Annotate,
            vec![calls_turn(calls), finish_turn()],
            10,
        )
        .await;
        let bodies = digests(&server).await;
        assert_eq!(bodies.len(), 2);
        let expected: Vec<Value> = (0..100)
            .flat_map(|i| {
                let id = format!("1:r{i}");
                [tc(&id, "read_file", &format!("OK{i}")), tr(&id, false, "")]
            })
            .collect();
        assert_eq!(bodies[0]["turn_index"], 0);
        assert_eq!(items(&bodies[0]), expected);
        assert_eq!(bodies[1]["turn_index"], 1);
        assert_eq!(
            items(&bodies[1]),
            vec![
                tc("1:r100", "read_file", "OK100"),
                tr("1:r100", false, ""),
                tc("2:cf", "finish", ""),
                tr("2:cf", false, ""),
            ]
        );
    }

    // AC15 (f)
    #[tokio::test]
    async fn a_digest_closes_at_the_byte_budget() {
        let server = turn_kb(contract_example("talos_gate_armed"), recorded()).await;
        let memory = digest_memory(&server, "m").await;
        let reasoning = "r".repeat(2000);
        let mut turns: Vec<AssistantTurn> = (1..=40)
            .map(|i| {
                turn_of(vec![
                    ContentBlock::Reasoning {
                        text: reasoning.clone(),
                        opaque: None,
                    },
                    tool(&format!("c{i}"), "read_file", json!({"path": "OK"})),
                ])
            })
            .collect();
        turns.push(finish_turn());
        run_digests(memory, MemoryMode::Annotate, turns, 50).await;
        let requests = turn_requests(&server).await;
        assert!(requests.len() >= 2, "{}", requests.len());
        let mut all = Vec::new();
        for r in &requests {
            assert!(r.body.len() <= MAX_TURN_BODY_BYTES, "{}", r.body.len());
            let body: Value = serde_json::from_slice(&r.body).unwrap();
            assert_eq!(body["truncated"], false);
            let its = items(&body);
            assert!(its.len() <= MAX_TURN_ITEMS);
            all.extend(its);
        }
        let mut expected = Vec::new();
        for i in 1..=40 {
            let id = format!("{i}:c{i}");
            expected.push(json!({"kind": "reasoning", "text": reasoning, "truncated": false}));
            expected.push(tc(&id, "read_file", "OK"));
            expected.push(tr(&id, false, ""));
        }
        expected.push(tc("41:cf", "finish", ""));
        expected.push(tr("41:cf", false, ""));
        assert_eq!(expected.len(), 122);
        assert_eq!(all, expected);
    }

    // AC15 (g)
    #[tokio::test]
    async fn fields_are_truncated_to_the_contract() {
        let server = turn_kb(contract_example("talos_gate_armed"), recorded()).await;
        let memory = digest_memory(&server, "m").await;
        let task = "p".repeat(5000);
        memory.observe(&Observation::RunStart { task: &task });
        let (x, y) = ("x".repeat(2500), "y".repeat(2500));
        memory.observe(&Observation::Turn {
            iteration: 1,
            reasoning: &x,
            text: &y,
        });
        let args = json!({"command": "z".repeat(800)});
        let error = "e".repeat(3000);
        memory.after_tool(
            &view("bash", &args),
            &ToolResultView {
                call_id: "c1",
                iteration: 1,
                is_error: true,
                content: &error,
                duration_ms: None,
            },
        );
        let detail = "d".repeat(2000);
        memory.observe(&Observation::Correction {
            trigger: CorrectionTrigger::GateRed,
            iteration: 1,
            detail: &detail,
        });
        let blocked = crate::run_record::Disposition::Blocked {
            decision_needed: "q".repeat(5000),
        };
        let finish_args = json!({});
        memory.observe(&Observation::Finish {
            call_id: "cf",
            iteration: 2,
            args: &finish_args,
            is_error: false,
            content: "",
            accepted: Some(&blocked),
        });
        let flushed = memory.flush().await;
        assert_eq!(flushed.digests_sent, 1);
        let bodies = digests(&server).await;
        assert_eq!(bodies.len(), 1);
        let b = &bodies[0];
        assert_eq!(b["user_prompt"], "p".repeat(4000));
        assert_eq!(b["final_message"], "q".repeat(4000));
        assert_eq!(
            items(b),
            vec![
                json!({"kind": "reasoning", "text": "x".repeat(2000), "truncated": true}),
                json!({"kind": "assistant_text", "text": "y".repeat(2000)}),
                tc("1:c1", "bash", &"z".repeat(500)),
                tr("1:c1", true, &"e".repeat(1500)),
                tc("2:cf", "finish", ""),
                tr("2:cf", false, ""),
                corr("gate_red", &"d".repeat(1500), &[], false),
            ]
        );
    }

    // AC15 (h)
    #[tokio::test]
    async fn same_call_id_in_two_iterations_gets_distinct_tool_use_ids() {
        let server = turn_kb(contract_example("talos_gate_armed"), recorded()).await;
        let memory = digest_memory(&server, "m").await;
        let turns = vec![
            call_turn("ollama-call-0", "read_file", json!({"path": "OK"})),
            call_turn("ollama-call-0", "read_file", json!({"path": "OK"})),
            finish_turn(),
        ];
        run_digests(memory, MemoryMode::Annotate, turns, 10).await;
        let bodies = digests(&server).await;
        let ids: Vec<String> = items(&bodies[0])
            .iter()
            .filter(|i| i["kind"] == "tool_call" && i["tool"] == "read_file")
            .map(|i| i["tool_use_id"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(ids, ["1:ollama-call-0", "2:ollama-call-0"]);
    }

    // AC15 (i)
    #[tokio::test]
    async fn only_tool_map_names_are_posted() {
        let mut prevention = contract_example("talos_gate_armed");
        prevention["tool_map"] = json!({"bash": "Bash"});
        let server = turn_kb(prevention, recorded()).await;
        let memory = digest_memory(&server, "m").await;
        let turns = vec![
            call_turn("b1", "bash", json!({"command": "echo OK"})),
            call_turn("r1", "read_file", json!({"path": "OK"})),
            finish_turn(),
        ];
        run_digests(memory, MemoryMode::Annotate, turns, 10).await;
        let bodies = digests(&server).await;
        assert_eq!(bodies.len(), 1);
        assert_eq!(
            items(&bodies[0]),
            vec![tc("1:b1", "bash", "echo OK"), tr("1:b1", false, "")]
        );
    }

    // AC15 (j)
    #[tokio::test]
    async fn a_blank_or_invalid_engine_sends_no_engine_header() {
        for engine in ["  ", "bad\nengine"] {
            let server = turn_kb(contract_example("talos_gate_armed"), recorded()).await;
            let memory = digest_memory(&server, engine).await;
            let turns = vec![
                call_turn("c1", "bash", json!({"command": "false"})),
                finish_turn(),
            ];
            run_digests(memory, MemoryMode::Annotate, turns, 10).await;
            let events = event_requests(&server).await;
            let turns = turn_requests(&server).await;
            assert_eq!((events.len(), turns.len()), (1, 1));
            for r in events.iter().chain(&turns) {
                assert!(!r.headers.contains_key("x-kb-engine"), "{engine:?}");
            }
        }
    }

    fn outcome_shape(result: &crate::engine::RunResult) -> String {
        format!("{:?}", result.outcome)
    }

    // AC16 (a)
    #[tokio::test]
    async fn a_500_turn_counts_as_failed_and_changes_nothing() {
        let script = || {
            vec![
                call_turn("rc1", "run_checks", json!({})),
                call_turn("c1", "bash", json!({"command": "false"})),
                finish_turn(),
            ]
        };
        let ok_server = turn_kb(contract_example("talos_gate_armed"), recorded()).await;
        let (ok_backend, ok) = run_digests(
            digest_memory(&ok_server, "m").await,
            MemoryMode::Annotate,
            script(),
            10,
        )
        .await;
        let bad_server = turn_kb(
            contract_example("talos_gate_armed"),
            ResponseTemplate::new(500),
        )
        .await;
        let (bad_backend, bad) = run_digests(
            digest_memory(&bad_server, "m").await,
            MemoryMode::Annotate,
            script(),
            10,
        )
        .await;
        assert_eq!(outcome_shape(&ok), outcome_shape(&bad));
        assert_eq!(ok.stats.iterations, bad.stats.iterations);
        assert_eq!(ok_backend.messages_seen(), bad_backend.messages_seen());
        assert_eq!(
            (ok.stats.memory_digests_sent, ok.stats.memory_digests_failed),
            (1, 0)
        );
        assert_eq!(
            (
                bad.stats.memory_digests_sent,
                bad.stats.memory_digests_failed
            ),
            (0, 1)
        );
    }

    // AC16 (b)
    #[tokio::test]
    async fn a_slow_turn_endpoint_does_not_hold_the_run() {
        let server = turn_kb(
            contract_example("talos_gate_armed"),
            recorded().set_delay(Duration::from_secs(10)),
        )
        .await;
        let started = std::time::Instant::now();
        let (_, result) = run_digests(
            digest_memory(&server, "m").await,
            MemoryMode::Annotate,
            vec![finish_turn()],
            10,
        )
        .await;
        assert!(
            started.elapsed() < Duration::from_secs(4),
            "{:?}",
            started.elapsed()
        );
        assert_blocked(&result);
        assert_eq!(result.stats.memory_digests_sent, 0);
        // The per-POST TURN_TIMEOUT (2 s) fires inside the engine's 3 s
        // flush bound, so the slow digest settles as failed, not aborted;
        // the abort-at-the-bound path is pinned by the next test.
        assert_eq!(result.stats.memory_digests_failed, 1);
    }

    // AC16 (c)
    #[tokio::test]
    async fn a_cut_off_flush_aborts_its_pending_digests() {
        let server = turn_kb(
            contract_example("talos_gate_armed"),
            recorded().set_delay(Duration::from_secs(10)),
        )
        .await;
        let memory = digest_memory(&server, "m").await;
        memory.observe(&Observation::RunStart { task: "t" });
        let cut = tokio::time::timeout(Duration::from_millis(200), memory.flush()).await;
        assert!(cut.is_err(), "flush must still be pending");
        assert_eq!(memory.flush().await, MemoryFlush::default());
    }

    // AC16 (d)
    #[tokio::test]
    async fn no_tool_map_posts_no_digest() {
        let mut prevention = contract_example("talos_gate_armed");
        prevention.as_object_mut().unwrap().remove("tool_map");
        let server = turn_kb(prevention, recorded()).await;
        let (_, result) = run_digests(
            digest_memory(&server, "m").await,
            MemoryMode::Annotate,
            vec![call_turn("rc1", "run_checks", json!({})), finish_turn()],
            10,
        )
        .await;
        assert!(turn_requests(&server).await.is_empty());
        assert_eq!(result.stats.memory_digests_sent, 0);
        assert_eq!(result.stats.memory_digests_failed, 0);
    }

    #[tokio::test]
    async fn degraded_memory_posts_no_digest_even_when_called_directly() {
        let server = turn_kb(contract_example("talos_gate_armed"), recorded()).await;
        // A project-less connect degrades without a request.
        let memory = connect_to(&server.uri(), None).await;
        let args = json!({});
        let blocked = crate::run_record::Disposition::Blocked {
            decision_needed: "q".into(),
        };
        for observation in [
            Observation::RunStart { task: "t" },
            Observation::Turn {
                iteration: 1,
                reasoning: "r",
                text: "t",
            },
            Observation::Finish {
                call_id: "cf",
                iteration: 1,
                args: &args,
                is_error: false,
                content: "",
                accepted: Some(&blocked),
            },
            Observation::Correction {
                trigger: CorrectionTrigger::Nudge,
                iteration: 1,
                detail: "n",
            },
        ] {
            memory.observe(&observation);
        }
        assert_eq!(memory.flush().await, MemoryFlush::default());
        assert!(server.received_requests().await.unwrap().is_empty());
    }

    // AC17 (a)
    #[tokio::test]
    async fn memory_off_posts_no_digest() {
        let server = turn_kb(contract_example("talos_gate_armed"), recorded()).await;
        let memory = digest_memory(&server, "m").await;
        let after_connect = server.received_requests().await.unwrap().len();
        let (_, result) = run_digests(
            memory,
            MemoryMode::Off,
            vec![
                call_turn("c1", "bash", json!({"command": "false"})),
                finish_turn(),
            ],
            10,
        )
        .await;
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), after_connect);
        assert!(
            requests
                .iter()
                .all(|r| r.url.path() != TURN_PATH && r.url.path() != EVENT_PATH)
        );
        assert_eq!(result.stats.memory_digests_sent, 0);
        assert_eq!(result.stats.memory_digests_failed, 0);
    }
}
