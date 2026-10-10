//! The **personal-kb adapter** for the [`crate::memory`] seam: phase 1 of
//! `docs/design/11-memory.md` (the read path).
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

pub mod cues;

use std::collections::BTreeMap;
use std::error::Error as _;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;

use crate::mcp::{McpConnectReport, McpServerConfig};
use crate::memory::{Advice, Memory, ToolCallView};
use crate::tool::{Tool, ToolCtx, ToolRegistry, ToolResult};
use cues::{IndexCue, find_match};

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
                PersonalKbMemory {
                    degraded: None,
                    label,
                    slice_text: p.slice_text,
                    cues: p.cues,
                    tool_map: p.tool_map.unwrap_or_default(),
                    endpoint: Some(endpoint),
                    kb_tools: Vec::new(),
                }
            }
        }
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
}

#[cfg(test)]
mod tests {
    use super::{
        KB_HARNESS_HEADERS, KB_MCP_CONNECT_TIMEOUT, KB_PROJECT_FILENAME, KbEndpoint,
        LOCAL_KB_API_KEY, LOCAL_KB_URL, MAX_FACT_CHARS, NO_PROJECT, PREVENTION_PATH,
        PREVENTION_TIMEOUT, PersonalKbMemory, Prevention, cue_fact, parse_prevention,
        resolve_kb_project,
    };
    use crate::engine::{FINISH_TOOL_NAME, FinishTool, RunConfig, RunStats};
    use crate::mcp::McpConnectReport;
    use crate::memory::{self, Advice, Memory, MemoryMode, ToolCallView};
    use crate::model::{
        AssistantTurn, ContentBlock, Message, StopReason, ToolCallRequest, Usage, UserBlock,
    };
    use crate::personal_kb::cues::IndexCue;
    use crate::test_support::MockBackend;
    use crate::tool::{Tool, ToolCtx, ToolRegistry, ToolResult};
    use async_trait::async_trait;
    use serde_json::{Value, json};
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
}
