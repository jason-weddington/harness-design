//! A generic **MCP client** over Streamable HTTP: connect to a server at run
//! start, discover its tools with `tools/list`, and register each one as an
//! ordinary [`Tool`] whose `run` proxies to `tools/call`.
//!
//! ## Protocol era
//!
//! This implements the **initialize-based** Streamable HTTP client of MCP
//! revisions 2025-03-26 through 2025-11-25: an `initialize` handshake, an
//! optional `MCP-Session-Id` the client echoes on every later request, and the
//! `MCP-Protocol-Version` header after initialization. It does NOT implement
//! the stateless 2026-07-28 revision (per-request `_meta`, `server/discover`).
//! Legacy-only servers and dual-era servers (the Python `mcp` 2.x SDK keeps the
//! legacy streamable-HTTP path) both accept an initialize-based client; a
//! modern-only server does not, and in that case the connect fails — which a
//! caller such as `talos run --mcp` treats as fail-open (the run proceeds
//! without that server's tools).
//!
//! ## Why hand-rolled, not `rmcp`
//!
//! The transport is a few hundred lines on the workspace's existing reqwest
//! 0.12. `rmcp` 3.5.1's streamable-HTTP client transport requires reqwest
//! ^0.13.2, which would put a second HTTP stack beside the workspace's reqwest
//! 0.12; with rmcp's default `reqwest` TLS feature it would also bring
//! `aws-lc-rs` into the shared rustls, against the workspace's ring-only
//! posture. The client needs only `initialize`, `notifications/initialized`,
//! `tools/list` and `tools/call`.
//!
//! ## Deliberately unsupported
//!
//! - the stateless 2026-07-28 revision and dual-era detection;
//! - the deprecated 2024-11-05 HTTP+SSE transport, stdio, and spawning servers;
//! - the standalone GET SSE stream and server-initiated requests (sampling,
//!   elicitation, roots, ping) — they are ignored on a response stream;
//! - resumable streams (`Last-Event-ID`, `retry`): a stream that closes before
//!   the response is a [`McpError::Protocol`];
//! - HTTP DELETE session termination — the session is simply abandoned;
//! - `notifications/tools/list_changed`: the tool set is fixed at connect;
//! - server `instructions`, resources, prompts, completions, and any auth
//!   other than a static bearer token;
//! - prefixing or namespacing tool names: names stay the server's verbatim,
//!   and [`MCP_RESERVED_TOOL_NAMES`] plus the registry check fail closed on a
//!   collision;
//! - CR-only SSE line endings; an unterminated trailing SSE event is
//!   discarded.
//!
//! ## Server-side policy must not key on `clientInfo`
//!
//! The client sends `clientInfo.name` = [`MCP_CLIENT_NAME`] on `initialize`,
//! but a server's write policy must never key on it: a stateless server never
//! sees `clientInfo` on `tools/call`. Use a request header (see
//! [`McpServerConfig::headers`]) or the bearer token instead.

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use reqwest::header::{ACCEPT, CONTENT_TYPE, HeaderValue};
use serde_json::{Map, Value, json};

use crate::tool::{Tool, ToolCtx, ToolRegistry, ToolResult};

/// One MCP server to connect to.
#[derive(Clone)]
pub struct McpServerConfig {
    /// The caller's name for the server (used in reports and error messages,
    /// never sent on the wire).
    pub name: String,
    /// The Streamable HTTP endpoint every message is sent to with `POST`.
    pub url: String,
    /// Optional static bearer token, sent as `Authorization: Bearer <token>`
    /// (marked sensitive). Never printed by `Debug` or any report.
    pub bearer_token: Option<String>,
    /// Extra headers sent verbatim on EVERY request to this server, after the
    /// standard headers. An `Authorization` header (any case) is rejected at
    /// connect — use [`McpServerConfig::bearer_token`]. `Debug` prints the
    /// header names only, never the values.
    pub headers: Vec<(String, String)>,
}

impl fmt::Debug for McpServerConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let header_names: Vec<&str> = self.headers.iter().map(|(k, _)| k.as_str()).collect();
        f.debug_struct("McpServerConfig")
            .field("name", &self.name)
            .field("url", &self.url)
            .field(
                "bearer_token",
                &self.bearer_token.as_ref().map(|_| "<redacted>"),
            )
            .field("headers", &header_names)
            .finish()
    }
}

/// The protocol revision requested on `initialize`.
pub const MCP_PROTOCOL_VERSION: &str = "2025-11-25";

/// The negotiated `protocolVersion` values the client accepts.
pub const MCP_SUPPORTED_PROTOCOL_VERSIONS: [&str; 3] = ["2025-11-25", "2025-06-18", "2025-03-26"];

/// The bound on one `tools/call` (including a session re-handshake and its
/// single retry) when connecting through [`connect_mcp_tools`].
#[allow(clippy::duration_suboptimal_units)] // pinned spelling: 120 s
pub const MCP_TOOL_CALL_TIMEOUT: Duration = Duration::from_secs(120);

/// The most `tools/list` requests one connect will send.
pub const MCP_MAX_TOOLS_LIST_PAGES: usize = 64;

/// `clientInfo.name` sent on `initialize`.
pub const MCP_CLIENT_NAME: &str = "harness";

/// Tool names an MCP server may never register, whether or not the registry
/// currently holds them. The engine keys behaviour on these literal names —
/// [`crate::engine::FINISH_TOOL_NAME`],
/// [`crate::tools::read_file::READ_FILE_TOOL_NAME`], and the `run_checks` /
/// `edit_file` / `bash` comparisons in the engine loop (`run_loop_body`) — so
/// a proxied tool under one of them would drive the done-oracle, the
/// tree-dirty latch, compaction rereads or finish routing.
pub const MCP_RESERVED_TOOL_NAMES: [&str; 6] = [
    "bash",
    "edit_file",
    "finish",
    "list_files",
    "read_file",
    "run_checks",
];

/// Everything that can go wrong talking to an MCP server.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum McpError {
    /// The bounded operation did not finish in time.
    #[error("timed out after {0:?}")]
    Timeout(Duration),
    /// The HTTP request could not be sent or its body read.
    #[error("transport error: {0}")]
    Transport(String),
    /// A non-2xx HTTP status (body excerpt redacted and capped at 200 chars).
    #[error("HTTP {status}: {body}")]
    HttpStatus {
        /// The HTTP status code.
        status: u16,
        /// The redacted, truncated response body.
        body: String,
    },
    /// A JSON-RPC error object answering the request.
    #[error("JSON-RPC error {code}: {message}")]
    JsonRpc {
        /// The JSON-RPC error code.
        code: i64,
        /// The JSON-RPC error message.
        message: String,
    },
    /// The server broke the protocol.
    #[error("protocol error: {0}")]
    Protocol(String),
    /// A tool name is reserved, already registered, or listed twice.
    #[error("tool name collision: `{0}` is already registered")]
    NameCollision(String),
    /// A listed tool definition is unusable.
    #[error("invalid tool `{tool}`: {reason}")]
    InvalidTool {
        /// The offending tool name.
        tool: String,
        /// Why it was rejected.
        reason: String,
    },
}

impl McpError {
    /// A stable machine-readable tag for the variant.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Timeout(_) => "timeout",
            Self::Transport(_) => "transport",
            Self::HttpStatus { .. } => "http_status",
            Self::JsonRpc { .. } => "json_rpc",
            Self::Protocol(_) => "protocol",
            Self::NameCollision(_) => "name_collision",
            Self::InvalidTool { .. } => "invalid_tool",
        }
    }
}

fn protocol(msg: impl Into<String>) -> McpError {
    McpError::Protocol(msg.into())
}

/// The outcome of connecting one server in [`connect_and_register`]. `auth`
/// is `"bearer"` when a bearer token was configured, else `"none"`; the token
/// value never appears.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McpConnectReport {
    /// The server connected and every tool registered.
    Connected {
        /// [`McpServerConfig::name`].
        name: String,
        /// The registered tool names, in server listing order.
        tools: Vec<String>,
        /// The negotiated protocol version.
        protocol_version: String,
        /// `serverInfo.name`, when a string.
        server_name: Option<String>,
        /// `serverInfo.version`, when a string.
        server_version: Option<String>,
        /// `"bearer"` or `"none"`.
        auth: &'static str,
        /// Wall time of the connect, in milliseconds.
        connect_ms: u64,
    },
    /// The connect or the registration failed; none of the server's tools
    /// are registered.
    Failed {
        /// [`McpServerConfig::name`].
        name: String,
        /// The [`McpError`] Display.
        error: String,
        /// [`McpError::kind`].
        error_kind: &'static str,
        /// The status, only for [`McpError::HttpStatus`].
        http_status: Option<u16>,
        /// `"bearer"` or `"none"`.
        auth: &'static str,
        /// Wall time of the connect, in milliseconds.
        connect_ms: u64,
    },
}

// ---------------------------------------------------------------------------
// SSE parsing
// ---------------------------------------------------------------------------

/// Incremental `text/event-stream` parser. Buffers raw bytes, splits lines on
/// `\n` (stripping one trailing `\r`), and decodes a line only once complete.
#[derive(Debug, Default)]
struct SseParser {
    buf: Vec<u8>,
    data: Vec<String>,
}

impl SseParser {
    /// Feed `bytes`, returning the data payloads of the events they complete.
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<String>, McpError> {
        self.buf.extend_from_slice(bytes);
        let mut out = Vec::new();
        while let Some(pos) = self.buf.iter().position(|b| *b == b'\n') {
            let mut raw: Vec<u8> = self.buf.drain(..=pos).collect();
            raw.pop();
            if raw.last() == Some(&b'\r') {
                raw.pop();
            }
            let line =
                String::from_utf8(raw).map_err(|_| protocol("SSE line is not valid UTF-8"))?;
            if line.is_empty() {
                let joined = self.data.join("\n");
                self.data.clear();
                if !joined.is_empty() {
                    out.push(joined);
                }
                continue;
            }
            if line.starts_with(':') {
                continue;
            }
            let (field, value) = match line.split_once(':') {
                Some((f, v)) => (f, v.strip_prefix(' ').unwrap_or(v)),
                None => (line.as_str(), ""),
            };
            if field == "data" {
                self.data.push(value.to_string());
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// JSON-RPC classification
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    JsonBody,
    Sse,
}

/// Classify one candidate message against request `want`. `None` means
/// "ignore and keep reading" (SSE only).
fn classify(text: &str, want: u64, source: Source) -> Option<Result<Value, McpError>> {
    let v: Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(e) => {
            return Some(Err(protocol(match source {
                Source::JsonBody => format!("response body is not JSON: {e}"),
                Source::Sse => format!("SSE event data is not JSON: {e}"),
            })));
        }
    };
    let obj = v.as_object();
    let id = obj.and_then(|o| o.get("id"));
    let id_matches = id == Some(&json!(want));
    let has_method = obj.is_some_and(|o| o.contains_key("method"));
    if has_method && source == Source::Sse {
        return None;
    }
    if !has_method
        && let Some(err) = obj.and_then(|o| o.get("error"))
        && (id_matches || id.is_none_or(Value::is_null))
    {
        let code = err.get("code").and_then(Value::as_i64);
        let message = err.get("message").and_then(Value::as_str);
        return Some(Err(match (code, message) {
            (Some(code), Some(message)) => McpError::JsonRpc {
                code,
                message: message.to_string(),
            },
            _ => protocol("malformed JSON-RPC error object"),
        }));
    }
    if has_method || !id_matches {
        return match source {
            Source::Sse => None,
            Source::JsonBody => {
                let got = id.map_or_else(|| "null".to_string(), Value::to_string);
                Some(Err(protocol(format!(
                    "response id mismatch: expected {want}, got {got}"
                ))))
            }
        };
    }
    Some(match obj.and_then(|o| o.get("result")) {
        Some(result) => Ok(result.clone()),
        None => Err(protocol("response has neither result nor error")),
    })
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
struct SessionState {
    session_id: Option<String>,
    protocol_version: Option<String>,
}

/// What `initialize` told us.
struct InitInfo {
    protocol_version: String,
    server_name: Option<String>,
    server_version: Option<String>,
}

/// One server connection, shared by all of its tools.
struct Session {
    client: reqwest::Client,
    server: String,
    url: String,
    bearer: Option<String>,
    headers: Vec<(String, String)>,
    next_id: AtomicU64,
    state: tokio::sync::Mutex<SessionState>,
}

fn transport(e: reqwest::Error) -> McpError {
    let e = e.without_url();
    let mut s = e.to_string();
    let mut cur = std::error::Error::source(&e);
    while let Some(c) = cur {
        s.push_str(": ");
        s.push_str(&c.to_string());
        cur = c.source();
    }
    McpError::Transport(s)
}

impl Session {
    fn post(&self, body: &Value, state: &SessionState) -> reqwest::RequestBuilder {
        let mut rb = self
            .client
            .post(&self.url)
            .header(CONTENT_TYPE, "application/json")
            .header(ACCEPT, "application/json, text/event-stream");
        if let Some(token) = &self.bearer {
            rb = rb.bearer_auth(token);
        }
        if let Some(sid) = &state.session_id {
            rb = rb.header("MCP-Session-Id", sid);
        }
        if let Some(ver) = &state.protocol_version {
            rb = rb.header("MCP-Protocol-Version", ver);
        }
        for (k, v) in &self.headers {
            rb = rb.header(k, v);
        }
        rb.body(body.to_string())
    }

    /// Turn a non-2xx response into [`McpError::HttpStatus`].
    async fn status_error(&self, resp: reqwest::Response) -> McpError {
        let status = resp.status().as_u16();
        let mut body = resp.text().await.unwrap_or_default();
        if let Some(token) = self.bearer.as_deref().filter(|t| !t.is_empty()) {
            body = body.replace(token, "<redacted>");
        }
        McpError::HttpStatus {
            status,
            body: body.chars().take(200).collect(),
        }
    }

    /// Send one JSON-RPC request; return its `result` plus the raw
    /// `MCP-Session-Id` response header.
    async fn request(
        &self,
        method: &str,
        params: Value,
        state: &SessionState,
    ) -> Result<(Value, Option<HeaderValue>), McpError> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let body = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        let mut resp = self.post(&body, state).send().await.map_err(transport)?;
        if !resp.status().is_success() {
            return Err(self.status_error(resp).await);
        }
        let sid = resp.headers().get("mcp-session-id").cloned();
        let ct = resp
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|v| {
                v.split(';')
                    .next()
                    .unwrap_or_default()
                    .trim()
                    .to_ascii_lowercase()
            });
        match ct.as_deref() {
            Some("application/json") => {
                let bytes = resp.bytes().await.map_err(transport)?;
                let text = String::from_utf8_lossy(&bytes);
                let result = classify(&text, id, Source::JsonBody)
                    .unwrap_or_else(|| Err(protocol("response has neither result nor error")))?;
                Ok((result, sid))
            }
            Some("text/event-stream") => {
                let mut parser = SseParser::default();
                while let Some(chunk) = resp.chunk().await.map_err(transport)? {
                    for payload in parser.push(&chunk)? {
                        if let Some(r) = classify(&payload, id, Source::Sse) {
                            return r.map(|v| (v, sid));
                        }
                    }
                }
                Err(protocol(format!(
                    "SSE stream ended without a response to request {id}"
                )))
            }
            other => Err(protocol(format!(
                "unexpected content type `{}`",
                other.unwrap_or("<none>")
            ))),
        }
    }

    async fn snapshot(&self) -> SessionState {
        self.state.lock().await.clone()
    }

    /// `initialize` + `notifications/initialized`, replacing the session state.
    async fn handshake(&self) -> Result<InitInfo, McpError> {
        let params = json!({
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": MCP_CLIENT_NAME, "version": env!("CARGO_PKG_VERSION")},
        });
        let (result, sid) = self
            .request("initialize", params, &SessionState::default())
            .await?;
        let version = result
            .get("protocolVersion")
            .and_then(Value::as_str)
            .ok_or_else(|| protocol("initialize result has no string protocolVersion"))?;
        if !MCP_SUPPORTED_PROTOCOL_VERSIONS.contains(&version) {
            return Err(protocol(format!("unsupported protocolVersion `{version}`")));
        }
        let session_id = match sid {
            None => None,
            Some(v) => {
                let bytes = v.as_bytes();
                if bytes.is_empty() || !bytes.iter().all(|b| (0x21..=0x7E).contains(b)) {
                    return Err(protocol("invalid MCP-Session-Id header"));
                }
                Some(String::from_utf8_lossy(bytes).into_owned())
            }
        };
        let info = result.get("serverInfo");
        let field = |k: &str| {
            info.and_then(|i| i.get(k))
                .and_then(Value::as_str)
                .map(str::to_string)
        };
        let init = InitInfo {
            protocol_version: version.to_string(),
            server_name: field("name"),
            server_version: field("version"),
        };
        let state = SessionState {
            session_id,
            protocol_version: Some(init.protocol_version.clone()),
        };
        let note = json!({"jsonrpc": "2.0", "method": "notifications/initialized"});
        let resp = self.post(&note, &state).send().await.map_err(transport)?;
        if !resp.status().is_success() {
            return Err(self.status_error(resp).await);
        }
        *self.state.lock().await = state;
        Ok(init)
    }

    /// All `tools/list` pages, validated, in listing order.
    async fn list_tools(&self) -> Result<Vec<ToolDef>, McpError> {
        let mut defs = Vec::new();
        let mut seen = HashSet::new();
        let mut cursor: Option<String> = None;
        let mut pages = 0usize;
        loop {
            if pages == MCP_MAX_TOOLS_LIST_PAGES {
                return Err(protocol(format!(
                    "tools/list exceeded {MCP_MAX_TOOLS_LIST_PAGES} pages"
                )));
            }
            pages += 1;
            let params = cursor
                .take()
                .map_or_else(|| json!({}), |c| json!({"cursor": c}));
            let state = self.snapshot().await;
            let (result, _) = self.request("tools/list", params, &state).await?;
            let entries = result
                .get("tools")
                .and_then(Value::as_array)
                .ok_or_else(|| protocol("tools/list result has no tools array"))?;
            for (i, entry) in entries.iter().enumerate() {
                let def = validate_tool(i, entry)?;
                if !seen.insert(def.name.clone()) {
                    return Err(McpError::NameCollision(def.name));
                }
                defs.push(def);
            }
            match result.get("nextCursor") {
                None | Some(Value::Null) => return Ok(defs),
                Some(Value::String(c)) => cursor = Some(c.clone()),
                Some(_) => return Err(protocol("tools/list nextCursor is not a string")),
            }
        }
    }

    /// `tools/call`, with the one-shot 404 re-handshake on a held session.
    async fn call_tool(&self, params: Value) -> Result<Value, McpError> {
        let state = self.snapshot().await;
        match self.request("tools/call", params.clone(), &state).await {
            Err(McpError::HttpStatus { status: 404, .. }) if state.session_id.is_some() => {
                self.handshake().await?;
                let state = self.snapshot().await;
                self.request("tools/call", params, &state)
                    .await
                    .map(|(v, _)| v)
            }
            r => r.map(|(v, _)| v),
        }
    }
}

/// A validated `tools/list` entry.
struct ToolDef {
    name: String,
    description: Option<String>,
    input_schema: Value,
}

fn valid_tool_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

fn validate_tool(i: usize, entry: &Value) -> Result<ToolDef, McpError> {
    let name = entry
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| protocol(format!("tools/list entry {i} has no string name")))?;
    if !valid_tool_name(name) {
        return Err(McpError::InvalidTool {
            tool: name.to_string(),
            reason: "name must match ^[a-zA-Z0-9_-]{1,64}$".to_string(),
        });
    }
    let input_schema = entry
        .get("inputSchema")
        .filter(|s| s.get("type").and_then(Value::as_str) == Some("object"))
        .ok_or_else(|| McpError::InvalidTool {
            tool: name.to_string(),
            reason: "inputSchema must be a JSON object with type \"object\"".to_string(),
        })?;
    Ok(ToolDef {
        name: name.to_string(),
        description: entry
            .get("description")
            .and_then(Value::as_str)
            .filter(|d| !d.is_empty())
            .map(str::to_string),
        input_schema: input_schema.clone(),
    })
}

// ---------------------------------------------------------------------------
// The proxied tool
// ---------------------------------------------------------------------------

/// A tool proxied to an MCP server's `tools/call`.
struct McpTool {
    session: Arc<Session>,
    name: String,
    description: Option<String>,
    input_schema: Value,
    call_timeout: Duration,
}

/// Render a `tools/call` result into the tool's text and `isError` flag.
fn render_call_result(result: &Value) -> Result<(String, bool), McpError> {
    let malformed = || protocol("malformed tools/call result");
    let obj = result.as_object().ok_or_else(malformed)?;
    let content = match obj.get("content") {
        None => &[][..],
        Some(Value::Array(a)) => a.as_slice(),
        Some(_) => return Err(malformed()),
    };
    let text = if content.is_empty() {
        obj.get("structuredContent")
            .map(Value::to_string)
            .unwrap_or_default()
    } else {
        content
            .iter()
            .map(|block| {
                let ty = block.get("type").and_then(Value::as_str);
                match (ty, block.get("text").and_then(Value::as_str)) {
                    (Some("text"), Some(t)) => t.to_string(),
                    (Some(ty), _) => format!("[{ty} content omitted]"),
                    (None, _) => "[unknown content omitted]".to_string(),
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    let is_error = obj.get("isError").and_then(Value::as_bool).unwrap_or(false);
    Ok((text, is_error))
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn schema(&self) -> Value {
        let mut m = Map::new();
        m.insert("name".into(), Value::String(self.name.clone()));
        if let Some(d) = &self.description {
            m.insert("description".into(), Value::String(d.clone()));
        }
        m.insert("input_schema".into(), self.input_schema.clone());
        Value::Object(m)
    }

    fn is_mcp(&self) -> bool {
        true
    }

    async fn run(&self, input: Value, ctx: &ToolCtx) -> ToolResult {
        let tool = &self.name;
        if !input.is_object() {
            return ToolResult::error(format!("{tool}: arguments must be a JSON object"));
        }
        let params = json!({"name": tool, "arguments": input});
        let outcome = tokio::time::timeout(self.call_timeout, self.session.call_tool(params))
            .await
            .unwrap_or(Err(McpError::Timeout(self.call_timeout)))
            .and_then(|result| render_call_result(&result));
        match outcome {
            Err(err) => ToolResult::error(format!(
                "mcp server `{}`: tools/call `{tool}` failed: {err}",
                self.session.server
            )),
            Ok((text, false)) if text.is_empty() => ToolResult::ok(format!("{tool} ok")),
            Ok((text, false)) => ToolResult::with_detail(format!("{tool} ok"), text, ctx),
            Ok((text, true)) if text.is_empty() => {
                ToolResult::error(format!("{tool} returned an error"))
            }
            Ok((text, true)) => {
                let mut r = ToolResult::with_detail(format!("{tool} returned an error"), text, ctx);
                r.is_error = true;
                r
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Connect + register
// ---------------------------------------------------------------------------

/// A successful connect: the tools plus what `initialize` negotiated.
struct ConnectedServer {
    tools: Vec<Box<dyn Tool>>,
    protocol_version: String,
    server_name: Option<String>,
    server_version: Option<String>,
}

async fn connect_server(
    cfg: &McpServerConfig,
    timeout: Duration,
    call_timeout: Duration,
) -> Result<ConnectedServer, McpError> {
    if cfg
        .headers
        .iter()
        .any(|(k, _)| k.eq_ignore_ascii_case("authorization"))
    {
        return Err(protocol("use bearer_token, not an Authorization header"));
    }
    let client = reqwest::Client::builder().build().map_err(transport)?;
    let session = Arc::new(Session {
        client,
        server: cfg.name.clone(),
        url: cfg.url.clone(),
        bearer: cfg.bearer_token.clone(),
        headers: cfg.headers.clone(),
        next_id: AtomicU64::new(1),
        state: tokio::sync::Mutex::new(SessionState::default()),
    });
    let sequence = async {
        let init = session.handshake().await?;
        let defs = session.list_tools().await?;
        Ok::<_, McpError>((init, defs))
    };
    let (init, defs) = tokio::time::timeout(timeout, sequence)
        .await
        .map_err(|_| McpError::Timeout(timeout))??;
    let tools = defs
        .into_iter()
        .map(|d| {
            Box::new(McpTool {
                session: Arc::clone(&session),
                name: d.name,
                description: d.description,
                input_schema: d.input_schema,
                call_timeout,
            }) as Box<dyn Tool>
        })
        .collect();
    Ok(ConnectedServer {
        tools,
        protocol_version: init.protocol_version,
        server_name: init.server_name,
        server_version: init.server_version,
    })
}

/// Connect to `cfg` and return its tools, each proxying `tools/call` under
/// [`MCP_TOOL_CALL_TIMEOUT`]. The whole connect sequence (`initialize`,
/// `notifications/initialized`, every `tools/list` page) runs under one
/// `timeout`.
///
/// # Errors
/// Any [`McpError`] from the connect; an invalid or duplicate tool definition
/// fails the whole connect.
pub async fn connect_mcp_tools(
    cfg: &McpServerConfig,
    timeout: Duration,
) -> Result<Vec<Box<dyn Tool>>, McpError> {
    connect_mcp_tools_with_call_timeout(cfg, timeout, MCP_TOOL_CALL_TIMEOUT).await
}

/// [`connect_mcp_tools`] with an injected per-call timeout.
pub(crate) async fn connect_mcp_tools_with_call_timeout(
    cfg: &McpServerConfig,
    timeout: Duration,
    call_timeout: Duration,
) -> Result<Vec<Box<dyn Tool>>, McpError> {
    connect_server(cfg, timeout, call_timeout)
        .await
        .map(|c| c.tools)
}

/// Register `tools` into `registry`, failing closed: a name in
/// [`MCP_RESERVED_TOOL_NAMES`], a name the registry already holds, or a name
/// repeated in the batch returns [`McpError::NameCollision`] and registers
/// NOTHING. On success returns the registered names in order.
///
/// # Errors
/// [`McpError::NameCollision`] on the first colliding name.
pub fn register_mcp_tools(
    registry: &mut ToolRegistry,
    tools: Vec<Box<dyn Tool>>,
) -> Result<Vec<String>, McpError> {
    let mut names: Vec<String> = Vec::with_capacity(tools.len());
    for tool in &tools {
        let name = tool.name();
        if MCP_RESERVED_TOOL_NAMES.contains(&name)
            || registry.get(name).is_some()
            || names.iter().any(|n| n == name)
        {
            return Err(McpError::NameCollision(name.to_string()));
        }
        names.push(name.to_string());
    }
    for (name, tool) in names.iter().zip(tools) {
        registry.register(name.clone(), Arc::from(tool));
    }
    Ok(names)
}

/// Connect each server in order and register its tools, reporting per
/// server. A failed server (connect or registration) registers nothing and
/// never affects the others; a later server colliding with an earlier one's
/// tools fails, and the earlier stays registered.
pub async fn connect_and_register(
    registry: &mut ToolRegistry,
    servers: &[McpServerConfig],
    timeout: Duration,
) -> Vec<McpConnectReport> {
    let mut reports = Vec::with_capacity(servers.len());
    for cfg in servers {
        let auth = if cfg.bearer_token.is_some() {
            "bearer"
        } else {
            "none"
        };
        let start = Instant::now();
        let connected = connect_server(cfg, timeout, MCP_TOOL_CALL_TIMEOUT).await;
        let connect_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
        let outcome = connected.and_then(|c| {
            let ConnectedServer {
                tools,
                protocol_version,
                server_name,
                server_version,
            } = c;
            register_mcp_tools(registry, tools)
                .map(|names| (names, protocol_version, server_name, server_version))
        });
        reports.push(match outcome {
            Ok((tools, protocol_version, server_name, server_version)) => {
                McpConnectReport::Connected {
                    name: cfg.name.clone(),
                    tools,
                    protocol_version,
                    server_name,
                    server_version,
                    auth,
                    connect_ms,
                }
            }
            Err(err) => McpConnectReport::Failed {
                name: cfg.name.clone(),
                error: err.to_string(),
                error_kind: err.kind(),
                http_status: match err {
                    McpError::HttpStatus { status, .. } => Some(status),
                    _ => None,
                },
                auth,
                connect_ms,
            },
        });
    }
    reports
}

#[cfg(test)]
mod tests {
    use super::{
        MCP_CLIENT_NAME, MCP_MAX_TOOLS_LIST_PAGES, MCP_PROTOCOL_VERSION, MCP_RESERVED_TOOL_NAMES,
        MCP_SUPPORTED_PROTOCOL_VERSIONS, MCP_TOOL_CALL_TIMEOUT, McpConnectReport, McpError,
        McpServerConfig, Source, SseParser, classify, connect_and_register, connect_mcp_tools,
        connect_mcp_tools_with_call_timeout, register_mcp_tools,
    };
    use crate::exec::{CheckCommand, ChecksRunner};
    use crate::prompt::{ToolLine, tool_lines};
    use crate::tool::{EchoTool, Tool, ToolCtx, ToolRegistry, ToolResult};
    use crate::tools::{answer_registry, standard_registry};
    use serde_json::{Value, json};
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use wiremock::matchers::{body_partial_json, method};
    use wiremock::{Mock, MockServer, Request, ResponseTemplate};

    const T: Duration = Duration::from_secs(10);

    fn body(req: &Request) -> Value {
        serde_json::from_slice(&req.body).unwrap_or(Value::Null)
    }

    #[allow(clippy::needless_pass_by_value)]
    fn ok(req: &Request, result: Value) -> ResponseTemplate {
        ResponseTemplate::new(200)
            .set_body_json(json!({"jsonrpc": "2.0", "id": body(req)["id"], "result": result}))
    }

    fn init_result() -> Value {
        json!({
            "protocolVersion": "2025-11-25",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake", "version": "1.0"},
        })
    }

    fn tool_def(name: &str) -> Value {
        json!({
            "name": name,
            "description": format!("{name} tool"),
            "inputSchema": {"type": "object", "properties": {"x": {"type": "string"}}},
        })
    }

    async fn on<F>(server: &MockServer, rpc: &str, priority: u8, f: F)
    where
        F: Fn(&Request) -> ResponseTemplate + Send + Sync + 'static,
    {
        Mock::given(method("POST"))
            .and(body_partial_json(json!({"method": rpc})))
            .respond_with(f)
            .with_priority(priority)
            .mount(server)
            .await;
    }

    /// Override `rpc` on `server` (wins over the standard fake).
    async fn over<F>(server: &MockServer, rpc: &str, f: F)
    where
        F: Fn(&Request) -> ResponseTemplate + Send + Sync + 'static,
    {
        on(server, rpc, 1, f).await;
    }

    /// A standard fake: initialize, initialized (202), one tools/list page,
    /// and a tools/call that echoes its arguments as text.
    async fn fake(tools: Vec<Value>) -> MockServer {
        let server = MockServer::start().await;
        on(&server, "initialize", 5, |r| ok(r, init_result())).await;
        on(&server, "notifications/initialized", 5, |_| {
            ResponseTemplate::new(202)
        })
        .await;
        on(&server, "tools/list", 5, move |r| {
            ok(r, json!({"tools": tools.clone()}))
        })
        .await;
        on(&server, "tools/call", 5, |r| {
            let args = body(r)["params"]["arguments"].to_string();
            ok(r, json!({"content": [{"type": "text", "text": args}]}))
        })
        .await;
        server
    }

    fn cfg(server: &MockServer) -> McpServerConfig {
        McpServerConfig {
            name: "fake".into(),
            url: format!("{}/mcp", server.uri()),
            bearer_token: None,
            headers: vec![],
        }
    }

    async fn requests(server: &MockServer, rpc: &str) -> Vec<Request> {
        server
            .received_requests()
            .await
            .expect("captured")
            .into_iter()
            .filter(|r| body(r)["method"] == rpc)
            .collect()
    }

    fn header<'a>(req: &'a Request, name: &str) -> Option<&'a str> {
        req.headers.get(name).and_then(|v| v.to_str().ok())
    }

    async fn connect_err(cfg: &McpServerConfig) -> McpError {
        match connect_mcp_tools(cfg, T).await {
            Ok(_) => panic!("connect unexpectedly succeeded"),
            Err(e) => e,
        }
    }

    async fn one_tool(server: &MockServer) -> Box<dyn Tool> {
        connect_mcp_tools(&cfg(server), T)
            .await
            .expect("connect")
            .remove(0)
    }

    async fn call_with(result: Value) -> ToolResult {
        let server = fake(vec![tool_def("alpha")]).await;
        over(&server, "tools/call", move |r| ok(r, result.clone())).await;
        let tool = one_tool(&server).await;
        tool.run(json!({}), &ToolCtx::stub()).await
    }

    fn names(registry: &ToolRegistry) -> Vec<Value> {
        registry
            .list()
            .into_iter()
            .map(|s| s["name"].clone())
            .collect()
    }

    // ---- public surface ----------------------------------------------------

    #[test]
    fn constants_are_pinned() {
        assert_eq!(MCP_PROTOCOL_VERSION, "2025-11-25");
        assert_eq!(
            MCP_SUPPORTED_PROTOCOL_VERSIONS,
            ["2025-11-25", "2025-06-18", "2025-03-26"]
        );
        assert_eq!(MCP_TOOL_CALL_TIMEOUT.as_secs(), 120);
        assert_eq!(MCP_MAX_TOOLS_LIST_PAGES, 64);
        assert_eq!(MCP_CLIENT_NAME, "harness");
    }

    #[test]
    fn debug_redacts_the_token_and_header_values() {
        let mut c = McpServerConfig {
            name: "fake".into(),
            url: "http://x/mcp".into(),
            bearer_token: Some("secret-tok".into()),
            headers: vec![("X-Mode".into(), "hidden-value".into())],
        };
        let d = format!("{c:?}");
        assert!(!d.contains("secret-tok"), "{d}");
        assert!(!d.contains("hidden-value"), "{d}");
        assert!(d.contains("Some(\"<redacted>\")"), "{d}");
        assert!(d.contains("X-Mode"), "{d}");
        c.bearer_token = None;
        assert!(format!("{c:?}").contains("bearer_token: None"));
    }

    #[test]
    fn reserved_names_cover_every_native_tool() {
        assert!(MCP_RESERVED_TOOL_NAMES.contains(&crate::engine::FINISH_TOOL_NAME));
        assert!(MCP_RESERVED_TOOL_NAMES.contains(&crate::tools::read_file::READ_FILE_TOOL_NAME));
        let runner = ChecksRunner::new(
            CheckCommand {
                program: "/bin/sh".to_string(),
                args: vec!["-c".to_string(), "exit 0".to_string()],
            },
            PathBuf::from("/"),
            Duration::from_secs(10),
        );
        for registry in [standard_registry(Some(runner)), answer_registry(None)] {
            for schema in registry.list() {
                let name = schema["name"].as_str().expect("name");
                assert!(MCP_RESERVED_TOOL_NAMES.contains(&name), "{name}");
            }
        }
    }

    #[test]
    fn error_display_and_kind() {
        let cases = [
            (
                McpError::Timeout(Duration::from_millis(200)),
                "timed out after 200ms",
                "timeout",
            ),
            (
                McpError::Transport("x".into()),
                "transport error: x",
                "transport",
            ),
            (
                McpError::HttpStatus {
                    status: 401,
                    body: "no".into(),
                },
                "HTTP 401: no",
                "http_status",
            ),
            (
                McpError::JsonRpc {
                    code: -1,
                    message: "m".into(),
                },
                "JSON-RPC error -1: m",
                "json_rpc",
            ),
            (
                McpError::Protocol("p".into()),
                "protocol error: p",
                "protocol",
            ),
            (
                McpError::NameCollision("a".into()),
                "tool name collision: `a` is already registered",
                "name_collision",
            ),
            (
                McpError::InvalidTool {
                    tool: "a".into(),
                    reason: "r".into(),
                },
                "invalid tool `a`: r",
                "invalid_tool",
            ),
        ];
        for (err, display, kind) in cases {
            assert_eq!(err.to_string(), display);
            assert_eq!(err.kind(), kind);
        }
    }

    #[tokio::test]
    async fn is_mcp_marks_only_proxied_tools() {
        assert!(!EchoTool.is_mcp());
        let server = fake(vec![tool_def("alpha")]).await;
        assert!(one_tool(&server).await.is_mcp());
    }

    // ---- (1) JSON connect --------------------------------------------------

    #[tokio::test]
    async fn json_connect_returns_tools_and_report() {
        let server = fake(vec![tool_def("alpha"), tool_def("beta")]).await;
        let tools = connect_mcp_tools(&cfg(&server), T).await.expect("connect");
        let got: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert_eq!(got, ["alpha", "beta"]);
        assert_eq!(tools[0].schema()["description"], "alpha tool");
        assert_eq!(
            tools[0].schema()["input_schema"],
            tool_def("alpha")["inputSchema"]
        );

        let mut reg = ToolRegistry::new();
        let reports = connect_and_register(&mut reg, &[cfg(&server)], T).await;
        match &reports[..] {
            [
                McpConnectReport::Connected {
                    name,
                    tools,
                    protocol_version,
                    server_name,
                    server_version,
                    auth,
                    ..
                },
            ] => {
                assert_eq!(name, "fake");
                assert_eq!(tools, &["alpha", "beta"]);
                assert_eq!(protocol_version, "2025-11-25");
                assert_eq!(server_name.as_deref(), Some("fake"));
                assert_eq!(server_version.as_deref(), Some("1.0"));
                assert_eq!(*auth, "none");
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(names(&reg), [json!("alpha"), json!("beta")]);

        let init = &requests(&server, "initialize").await[0];
        let b = body(init);
        assert_eq!(b["id"], 1);
        assert_eq!(b["params"]["protocolVersion"], "2025-11-25");
        assert_eq!(b["params"]["capabilities"], json!({}));
        assert_eq!(b["params"]["clientInfo"]["name"], "harness");
        assert_eq!(header(init, "content-type"), Some("application/json"));
        assert_eq!(
            header(init, "accept"),
            Some("application/json, text/event-stream")
        );
        let note = &requests(&server, "notifications/initialized").await[0];
        assert!(body(note).get("id").is_none());
        // Ids are never reused: initialize = 1, tools/list = 2.
        assert_eq!(body(&requests(&server, "tools/list").await[0])["id"], 2);
    }

    // ---- (2) SSE -----------------------------------------------------------

    fn sse_fixture(id: &Value, result: &Value, eol: &str) -> String {
        let response = json!({"jsonrpc": "2.0", "id": id, "result": result});
        [
            "id: e0".to_string(),
            "data:".to_string(),
            String::new(),
            ": a comment".to_string(),
            "event: message".to_string(),
            r#"data: {"jsonrpc":"2.0","method":"notifications/progress","params":{}}"#.to_string(),
            String::new(),
            r#"data: {"jsonrpc":"2.0","id":1,"method":"ping"}"#.to_string(),
            String::new(),
            format!("data: {response}"),
            String::new(),
            "data: not-json".to_string(),
            String::new(),
            String::new(),
        ]
        .join(eol)
    }

    async fn sse_server(eol: &'static str) -> MockServer {
        let server = fake(vec![]).await;
        over(&server, "initialize", move |r| {
            ResponseTemplate::new(200).set_body_raw(
                sse_fixture(&body(r)["id"], &init_result(), eol),
                "text/event-stream",
            )
        })
        .await;
        over(&server, "tools/list", move |r| {
            ResponseTemplate::new(200).set_body_raw(
                sse_fixture(&body(r)["id"], &json!({"tools": [tool_def("alpha")]}), eol),
                "text/event-stream; charset=utf-8",
            )
        })
        .await;
        server
    }

    #[tokio::test]
    async fn sse_response_skips_priming_notifications_and_requests() {
        let server = sse_server("\n").await;
        let tools = connect_mcp_tools(&cfg(&server), T).await.expect("connect");
        assert_eq!(tools[0].name(), "alpha");
    }

    #[tokio::test]
    async fn sse_response_with_crlf_line_endings() {
        let server = sse_server("\r\n").await;
        let tools = connect_mcp_tools(&cfg(&server), T).await.expect("connect");
        assert_eq!(tools[0].name(), "alpha");
    }

    #[tokio::test]
    async fn sse_stream_without_a_response_is_a_protocol_error() {
        let server = fake(vec![]).await;
        over(&server, "initialize", |_| {
            ResponseTemplate::new(200).set_body_raw(
                "data: {\"jsonrpc\":\"2.0\",\"id\":99,\"result\":{}}\n\ndata: x",
                "text/event-stream",
            )
        })
        .await;
        assert_eq!(
            connect_err(&cfg(&server)).await.to_string(),
            "protocol error: SSE stream ended without a response to request 1"
        );
    }

    #[tokio::test]
    async fn sse_error_and_bad_data_are_classified() {
        let server = fake(vec![]).await;
        over(&server, "initialize", |r| {
            let id = &body(r)["id"];
            ResponseTemplate::new(200).set_body_raw(
                format!(
                    "data: {}\n\n",
                    json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32000, "message": "bad"}})
                ),
                "text/event-stream",
            )
        })
        .await;
        assert_eq!(
            connect_err(&cfg(&server)).await,
            McpError::JsonRpc {
                code: -32000,
                message: "bad".into()
            }
        );
        let server = fake(vec![]).await;
        over(&server, "initialize", |_| {
            ResponseTemplate::new(200).set_body_raw("data: {nope\n\n", "text/event-stream")
        })
        .await;
        let e = connect_err(&cfg(&server)).await.to_string();
        assert!(
            e.starts_with("protocol error: SSE event data is not JSON: "),
            "{e}"
        );
    }

    // ---- (2b) SseParser ----------------------------------------------------

    #[test]
    fn sse_parser_is_chunking_independent() {
        let mut fixture = sse_fixture(&json!(1), &json!({}), "\r\n");
        fixture.push_str("data: caf\u{e9}\r\ndata: two\r\n\r\n");
        let bytes = fixture.as_bytes();

        let mut whole = SseParser::default();
        let a = whole.push(bytes).expect("whole");

        let mut bytewise = SseParser::default();
        let mut b = Vec::new();
        for byte in bytes {
            b.extend(bytewise.push(std::slice::from_ref(byte)).expect("byte"));
        }

        let mut split = SseParser::default();
        let mut c = Vec::new();
        let mut start = 0;
        for (i, byte) in bytes.iter().enumerate() {
            if *byte == b'\r' {
                c.extend(split.push(&bytes[start..=i]).expect("split"));
                start = i + 1;
            }
        }
        c.extend(split.push(&bytes[start..]).expect("tail"));

        assert_eq!(a, b);
        assert_eq!(a, c);
        assert_eq!(a.len(), 5);
        assert_eq!(a[3], "not-json");
        assert_eq!(a[4], "caf\u{e9}\ntwo");
    }

    #[test]
    fn sse_parser_rejects_invalid_utf8_and_drops_unterminated_events() {
        let mut p = SseParser::default();
        assert_eq!(
            p.push(b"data: \xff\n"),
            Err(McpError::Protocol("SSE line is not valid UTF-8".into()))
        );
        let mut p = SseParser::default();
        assert_eq!(
            p.push(b"data: x\ndata\n").expect("ok"),
            Vec::<String>::new()
        );
        // A field with no colon has an empty value.
        assert_eq!(p.push(b"\n").expect("ok"), vec!["x\n".to_string()]);
        assert_eq!(p.push(b"data: tail").expect("ok"), Vec::<String>::new());
    }

    #[test]
    fn classify_json_body_rules() {
        let c = |t: &str| classify(t, 3, Source::JsonBody).expect("json body always classifies");
        assert_eq!(
            c(r#"{"id":3,"method":"ping"}"#),
            Err(McpError::Protocol(
                "response id mismatch: expected 3, got 3".into()
            ))
        );
        assert_eq!(
            c("[1]"),
            Err(McpError::Protocol(
                "response id mismatch: expected 3, got null".into()
            ))
        );
        assert_eq!(
            c(r#"{"error":{"code":1,"message":"m"}}"#),
            Err(McpError::JsonRpc {
                code: 1,
                message: "m".into()
            })
        );
        assert_eq!(c(r#"{"id":3,"result":7}"#), Ok(json!(7)));
        assert!(classify(r#"{"id":4,"result":7}"#, 3, Source::Sse).is_none());
    }

    // ---- (3) session id ----------------------------------------------------

    #[tokio::test]
    async fn session_id_and_protocol_version_are_echoed() {
        let server = fake(vec![tool_def("alpha")]).await;
        over(&server, "initialize", |r| {
            ok(r, init_result()).insert_header("Mcp-Session-Id", "sess-1")
        })
        .await;
        let tool = one_tool(&server).await;
        let r = tool.run(json!({"x": "y"}), &ToolCtx::stub()).await;
        assert!(!r.is_error, "{r:?}");
        let init = &requests(&server, "initialize").await[0];
        assert!(header(init, "mcp-session-id").is_none());
        assert!(header(init, "mcp-protocol-version").is_none());
        for rpc in ["notifications/initialized", "tools/list", "tools/call"] {
            let reqs = requests(&server, rpc).await;
            assert!(!reqs.is_empty(), "{rpc}");
            for req in reqs {
                assert_eq!(header(&req, "mcp-session-id"), Some("sess-1"), "{rpc}");
                assert_eq!(
                    header(&req, "mcp-protocol-version"),
                    Some("2025-11-25"),
                    "{rpc}"
                );
            }
        }
    }

    #[tokio::test]
    async fn invalid_session_id_header_is_a_protocol_error() {
        let server = fake(vec![]).await;
        over(&server, "initialize", |r| {
            ok(r, init_result()).insert_header("Mcp-Session-Id", "bad id")
        })
        .await;
        assert_eq!(
            connect_err(&cfg(&server)).await,
            McpError::Protocol("invalid MCP-Session-Id header".into())
        );
    }

    // ---- (4) pagination, (5) page cap --------------------------------------

    #[tokio::test]
    async fn tools_list_follows_next_cursor() {
        let server = fake(vec![]).await;
        Mock::given(method("POST"))
            .and(body_partial_json(
                json!({"method": "tools/list", "params": {"cursor": "p2"}}),
            ))
            .respond_with(|r: &Request| {
                ok(r, json!({"tools": [tool_def("beta")], "nextCursor": null}))
            })
            .with_priority(1)
            .mount(&server)
            .await;
        on(&server, "tools/list", 2, |r| {
            ok(r, json!({"tools": [tool_def("alpha")], "nextCursor": "p2"}))
        })
        .await;
        let tools = connect_mcp_tools(&cfg(&server), T).await.expect("connect");
        let got: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert_eq!(got, ["alpha", "beta"]);
        let lists = requests(&server, "tools/list").await;
        assert_eq!(body(&lists[0])["params"], json!({}));
    }

    #[tokio::test]
    async fn tools_list_page_cap() {
        let server = fake(vec![]).await;
        over(&server, "tools/list", |r| {
            ok(r, json!({"tools": [], "nextCursor": "c"}))
        })
        .await;
        assert_eq!(
            connect_err(&cfg(&server)).await.to_string(),
            "protocol error: tools/list exceeded 64 pages"
        );
        assert_eq!(requests(&server, "tools/list").await.len(), 64);
    }

    #[tokio::test]
    async fn duplicate_across_pages_is_a_collision() {
        let server = fake(vec![]).await;
        Mock::given(method("POST"))
            .and(body_partial_json(
                json!({"method": "tools/list", "params": {"cursor": "p2"}}),
            ))
            .respond_with(|r: &Request| ok(r, json!({"tools": [tool_def("alpha")]})))
            .with_priority(1)
            .mount(&server)
            .await;
        on(&server, "tools/list", 2, |r| {
            ok(r, json!({"tools": [tool_def("alpha")], "nextCursor": "p2"}))
        })
        .await;
        assert_eq!(
            connect_err(&cfg(&server)).await,
            McpError::NameCollision("alpha".into())
        );
    }

    // ---- (6)/(7) tools/call mapping ----------------------------------------

    #[tokio::test]
    async fn is_error_result_is_an_error() {
        let r = call_with(json!({"content": [{"type": "text", "text": "boom"}], "isError": true}))
            .await;
        assert!(r.is_error);
        assert_eq!(r.summary, "alpha returned an error");
        assert!(r.detail.as_deref().is_some_and(|d| d.contains("boom")));
    }

    #[tokio::test]
    async fn large_is_error_result_is_offloaded() {
        let big = "e".repeat(25_001);
        let r =
            call_with(json!({"content": [{"type": "text", "text": big}], "isError": true})).await;
        assert!(r.is_error);
        assert!(r.offload_path.is_some());
    }

    #[tokio::test]
    async fn non_text_content_is_summarised() {
        let r = call_with(json!({"content": [
            {"type": "text", "text": "a"},
            {"type": "image", "data": "..", "mimeType": "image/png"},
            {"kind": "odd"},
        ]}))
        .await;
        assert!(!r.is_error);
        assert_eq!(r.summary, "alpha ok");
        assert_eq!(
            r.detail.as_deref(),
            Some("a\n[image content omitted]\n[unknown content omitted]")
        );
    }

    #[tokio::test]
    async fn empty_and_structured_results() {
        let r = call_with(json!({"content": []})).await;
        assert_eq!(r, ToolResult::ok("alpha ok"));
        let r = call_with(json!({"isError": true})).await;
        assert_eq!(r, ToolResult::error("alpha returned an error"));
        let r = call_with(json!({"content": [], "structuredContent": {"k": 1}})).await;
        assert_eq!(r.detail.as_deref(), Some(r#"{"k":1}"#));
        assert!(!r.is_error);
    }

    #[tokio::test]
    async fn call_sends_name_and_arguments() {
        let server = fake(vec![tool_def("alpha")]).await;
        let tool = one_tool(&server).await;
        let r = tool.run(json!({"x": "y"}), &ToolCtx::stub()).await;
        assert_eq!(r.detail.as_deref(), Some(r#"{"x":"y"}"#));
        let call = &requests(&server, "tools/call").await[0];
        assert_eq!(
            body(call)["params"],
            json!({"name": "alpha", "arguments": {"x": "y"}})
        );
    }

    // ---- (8)/(9)/(10) status and timeouts ----------------------------------

    #[tokio::test]
    async fn http_401_on_initialize() {
        let server = fake(vec![]).await;
        over(&server, "initialize", |_| {
            ResponseTemplate::new(401).set_body_string("nope")
        })
        .await;
        let e = connect_err(&cfg(&server)).await;
        assert!(
            matches!(e, McpError::HttpStatus { status: 401, .. }),
            "{e:?}"
        );
        assert_eq!(e.kind(), "http_status");
        assert!(e.to_string().contains("401"));
    }

    #[tokio::test]
    async fn http_error_on_initialized_notification() {
        let server = fake(vec![]).await;
        over(&server, "notifications/initialized", |_| {
            ResponseTemplate::new(500).set_body_string("x".repeat(300))
        })
        .await;
        match connect_err(&cfg(&server)).await {
            McpError::HttpStatus { status, body } => {
                assert_eq!(status, 500);
                assert_eq!(body.len(), 200);
            }
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn connect_timeout() {
        let server = fake(vec![]).await;
        over(&server, "initialize", |r| {
            ok(r, init_result()).set_delay(Duration::from_secs(5))
        })
        .await;
        let t = Duration::from_millis(200);
        assert!(matches!(
            connect_mcp_tools(&cfg(&server), t).await,
            Err(McpError::Timeout(d)) if d == t
        ));
    }

    #[tokio::test]
    async fn call_timeout() {
        let server = fake(vec![tool_def("alpha")]).await;
        over(&server, "tools/call", |r| {
            ok(r, json!({"content": []})).set_delay(Duration::from_secs(5))
        })
        .await;
        let tool =
            connect_mcp_tools_with_call_timeout(&cfg(&server), T, Duration::from_millis(200))
                .await
                .expect("connect")
                .remove(0);
        let r = tool.run(json!({}), &ToolCtx::stub()).await;
        assert!(r.is_error);
        assert!(r.summary.contains("mcp server `fake`"), "{}", r.summary);
        assert!(r.summary.contains("timed out"), "{}", r.summary);
    }

    // ---- (11)/(12) malformed responses -------------------------------------

    type Responder = fn(&Request) -> ResponseTemplate;

    fn malformed_cases() -> Vec<(Responder, &'static str)> {
        vec![
            (
                |_| ResponseTemplate::new(200).set_body_raw("nope", "application/json"),
                "protocol error: response body is not JSON: ",
            ),
            (
                |_| {
                    ResponseTemplate::new(200)
                        .set_body_json(json!({"jsonrpc": "2.0", "id": 999, "result": {}}))
                },
                "protocol error: response id mismatch: expected ",
            ),
            (
                |r| {
                    ResponseTemplate::new(200)
                        .set_body_json(json!({"jsonrpc": "2.0", "id": body(r)["id"]}))
                },
                "protocol error: response has neither result nor error",
            ),
            (
                |r| {
                    ResponseTemplate::new(200).set_body_json(json!({"jsonrpc": "2.0", "id": body(r)["id"], "error": {"code": "x", "message": "m"}}))
                },
                "protocol error: malformed JSON-RPC error object",
            ),
            (
                |r| {
                    ResponseTemplate::new(200).set_body_json(json!({"jsonrpc": "2.0", "id": body(r)["id"], "error": {"code": -32601, "message": "no such method"}}))
                },
                "JSON-RPC error -32601: no such method",
            ),
            (
                |r| {
                    ResponseTemplate::new(200).set_body_raw(body(r)["id"].to_string(), "text/plain")
                },
                "protocol error: unexpected content type `text/plain`",
            ),
            (
                |_| ResponseTemplate::new(200),
                "protocol error: unexpected content type `<none>`",
            ),
        ]
    }

    #[tokio::test]
    async fn malformed_connect_responses() {
        for (responder, want) in malformed_cases() {
            let server = fake(vec![]).await;
            over(&server, "initialize", responder).await;
            let e = connect_err(&cfg(&server)).await.to_string();
            assert!(e.starts_with(want), "{e} vs {want}");
        }
        let id_mismatch = {
            let server = fake(vec![]).await;
            over(&server, "initialize", malformed_cases()[1].0).await;
            connect_err(&cfg(&server)).await
        };
        assert_eq!(
            id_mismatch,
            McpError::Protocol("response id mismatch: expected 1, got 999".into())
        );
        let list_cases: [(Value, &str); 4] = [
            (
                json!({}),
                "protocol error: tools/list result has no tools array",
            ),
            (
                json!({"tools": [], "nextCursor": 7}),
                "protocol error: tools/list nextCursor is not a string",
            ),
            (
                json!({"tools": [tool_def("alpha"), {"description": "x"}]}),
                "protocol error: tools/list entry 1 has no string name",
            ),
            (
                json!({"tools": ["alpha"]}),
                "protocol error: tools/list entry 0 has no string name",
            ),
        ];
        for (result, want) in list_cases {
            let server = fake(vec![]).await;
            over(&server, "tools/list", move |r| ok(r, result.clone())).await;
            assert_eq!(connect_err(&cfg(&server)).await.to_string(), want);
        }
        let server = fake(vec![]).await;
        over(&server, "initialize", |r| ok(r, json!({"serverInfo": {}}))).await;
        assert_eq!(
            connect_err(&cfg(&server)).await.to_string(),
            "protocol error: initialize result has no string protocolVersion"
        );
        let server = fake(vec![]).await;
        over(&server, "initialize", |r| {
            ResponseTemplate::new(200).set_body_json(json!({"jsonrpc": "2.0", "id": body(r)["id"], "error": {"code": -1, "message": "no"}}))
        })
        .await;
        assert_eq!(
            connect_err(&cfg(&server)).await,
            McpError::JsonRpc {
                code: -1,
                message: "no".into()
            }
        );
    }

    #[tokio::test]
    async fn malformed_call_responses_are_error_results() {
        let mut cases = malformed_cases();
        cases.push((
            |r| ok(r, json!("x")),
            "protocol error: malformed tools/call result",
        ));
        cases.push((
            |r| ok(r, json!({"content": "x"})),
            "protocol error: malformed tools/call result",
        ));
        for (responder, want) in cases {
            let server = fake(vec![tool_def("alpha")]).await;
            over(&server, "tools/call", responder).await;
            let tool = one_tool(&server).await;
            let r = tool.run(json!({}), &ToolCtx::stub()).await;
            assert!(r.is_error, "{want}");
            assert!(
                r.summary.starts_with(&format!(
                    "mcp server `fake`: tools/call `alpha` failed: {want}"
                )),
                "{} vs {want}",
                r.summary
            );
        }
    }

    // ---- (13)/(14) registration --------------------------------------------

    fn failed_kind(report: &McpConnectReport) -> &'static str {
        match report {
            McpConnectReport::Failed { error_kind, .. } => error_kind,
            McpConnectReport::Connected { .. } => panic!("expected Failed: {report:?}"),
        }
    }

    #[tokio::test]
    async fn collision_with_a_registered_tool_fails_closed() {
        let server = fake(vec![tool_def("alpha"), tool_def("echo")]).await;
        let mut reg = ToolRegistry::new();
        reg.register("echo", Arc::new(EchoTool));
        let before = reg.list();
        let reports = connect_and_register(&mut reg, &[cfg(&server)], T).await;
        assert_eq!(failed_kind(&reports[0]), "name_collision");
        assert_eq!(reg.list(), before);
    }

    #[tokio::test]
    async fn intra_server_duplicate_fails() {
        let server = fake(vec![tool_def("alpha"), tool_def("alpha")]).await;
        let mut reg = ToolRegistry::new();
        let reports = connect_and_register(&mut reg, &[cfg(&server)], T).await;
        assert_eq!(failed_kind(&reports[0]), "name_collision");
        assert!(reg.list().is_empty());
    }

    #[tokio::test]
    async fn later_server_colliding_with_earlier_fails_alone() {
        let first = fake(vec![tool_def("alpha")]).await;
        let second = fake(vec![tool_def("beta"), tool_def("alpha")]).await;
        let mut c2 = cfg(&second);
        c2.name = "second".into();
        let mut reg = ToolRegistry::new();
        let reports = connect_and_register(&mut reg, &[cfg(&first), c2], T).await;
        assert!(matches!(&reports[0], McpConnectReport::Connected { name, .. } if name == "fake"));
        assert!(
            matches!(&reports[1], McpConnectReport::Failed { name, error_kind: "name_collision", http_status: None, .. } if name == "second")
        );
        assert_eq!(names(&reg), [json!("alpha")]);
    }

    #[tokio::test]
    async fn reserved_name_in_answer_registry_fails() {
        let server = fake(vec![tool_def("edit_file")]).await;
        let mut reg = answer_registry(None);
        assert!(reg.get("edit_file").is_none());
        let before = reg.list();
        let reports = connect_and_register(&mut reg, &[cfg(&server)], T).await;
        assert_eq!(failed_kind(&reports[0]), "name_collision");
        assert_eq!(reg.list(), before);
    }

    #[tokio::test]
    async fn reserved_run_checks_in_gateless_registry_fails() {
        let server = fake(vec![tool_def("run_checks")]).await;
        let mut reg = standard_registry(None);
        assert!(reg.get("run_checks").is_none());
        let before = reg.list();
        let reports = connect_and_register(&mut reg, &[cfg(&server)], T).await;
        assert_eq!(failed_kind(&reports[0]), "name_collision");
        assert_eq!(reg.list(), before);
    }

    #[tokio::test]
    async fn invalid_tool_definitions_fail() {
        let bad_name = json!({"name": "alpha.beta", "inputSchema": {"type": "object"}});
        let bad_schema = json!({"name": "alpha", "inputSchema": {"type": "string"}});
        let no_schema = json!({"name": "alpha"});
        let long = json!({"name": "a".repeat(65), "inputSchema": {"type": "object"}});
        for def in [bad_name, bad_schema, no_schema, long] {
            let server = fake(vec![def]).await;
            let mut reg = ToolRegistry::new();
            let reports = connect_and_register(&mut reg, &[cfg(&server)], T).await;
            assert_eq!(failed_kind(&reports[0]), "invalid_tool");
        }
        let server = fake(vec![
            json!({"name": "alpha.beta", "inputSchema": {"type": "object"}}),
        ])
        .await;
        assert_eq!(
            connect_err(&cfg(&server)).await.to_string(),
            "invalid tool `alpha.beta`: name must match ^[a-zA-Z0-9_-]{1,64}$"
        );
        let server = fake(vec![json!({"name": "alpha", "inputSchema": []})]).await;
        assert_eq!(
            connect_err(&cfg(&server)).await.to_string(),
            "invalid tool `alpha`: inputSchema must be a JSON object with type \"object\""
        );
    }

    #[test]
    fn register_mcp_tools_registers_in_order_or_nothing() {
        let mut reg = ToolRegistry::new();
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(EchoTool)];
        assert_eq!(register_mcp_tools(&mut reg, tools).expect("ok"), ["echo"]);
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(EchoTool)];
        assert_eq!(
            register_mcp_tools(&mut reg, tools),
            Err(McpError::NameCollision("echo".into()))
        );
        let mut reg = ToolRegistry::new();
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(EchoTool), Box::new(EchoTool)];
        assert!(register_mcp_tools(&mut reg, tools).is_err());
        assert!(reg.list().is_empty());
    }

    // ---- (5) schema shape --------------------------------------------------

    #[tokio::test]
    async fn schema_matches_a_native_tool() {
        let input_schema =
            json!({"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]});
        let def = json!({
            "name": "alpha",
            "title": "Alpha",
            "description": "Alpha does things",
            "inputSchema": input_schema,
            "annotations": {"readOnlyHint": true},
            "outputSchema": {"type": "object"},
            "_meta": {},
        });
        let server = fake(vec![
            def,
            json!({"name": "beta", "description": "", "inputSchema": {"type": "object"}}),
        ])
        .await;
        let mut reg = ToolRegistry::new();
        connect_and_register(&mut reg, &[cfg(&server)], T).await;
        let alpha = reg.get("alpha").expect("alpha").schema();
        let mut keys: Vec<&String> = alpha.as_object().expect("obj").keys().collect();
        keys.sort();
        assert_eq!(keys, ["description", "input_schema", "name"]);
        assert_eq!(alpha["input_schema"], input_schema);
        let beta = reg.get("beta").expect("beta").schema();
        assert!(beta.get("description").is_none());
        assert_eq!(
            tool_lines(&reg)[0],
            ToolLine {
                name: "alpha".into(),
                description: "Alpha does things".into()
            }
        );
    }

    // ---- (15)/(22) auth ----------------------------------------------------

    #[tokio::test]
    async fn bearer_token_is_sent_on_every_request() {
        let server = fake(vec![tool_def("alpha")]).await;
        let mut c = cfg(&server);
        c.bearer_token = Some("tok".into());
        let tool = connect_mcp_tools(&c, T).await.expect("connect").remove(0);
        tool.run(json!({}), &ToolCtx::stub()).await;
        let all = server.received_requests().await.expect("captured");
        assert_eq!(all.len(), 4);
        for req in &all {
            assert_eq!(header(req, "authorization"), Some("Bearer tok"));
        }

        let server = fake(vec![tool_def("alpha")]).await;
        connect_mcp_tools(&cfg(&server), T).await.expect("connect");
        for req in &server.received_requests().await.expect("captured") {
            assert!(req.headers.get("authorization").is_none());
        }
    }

    #[tokio::test]
    async fn token_never_leaks_into_errors() {
        let server = fake(vec![]).await;
        over(&server, "initialize", |_| {
            ResponseTemplate::new(401).set_body_string("bad credentials: Bearer secret-tok")
        })
        .await;
        let mut c = cfg(&server);
        c.bearer_token = Some("secret-tok".into());
        let e = connect_err(&c).await;
        assert!(!e.to_string().contains("secret-tok"), "{e}");
        assert!(e.to_string().contains("<redacted>"), "{e}");
        let mut reg = ToolRegistry::new();
        let reports = connect_and_register(&mut reg, &[c], T).await;
        match &reports[0] {
            McpConnectReport::Failed {
                error,
                http_status,
                auth,
                ..
            } => {
                assert!(!error.contains("secret-tok"));
                assert_eq!(*http_status, Some(401));
                assert_eq!(*auth, "bearer");
            }
            other @ McpConnectReport::Connected { .. } => panic!("{other:?}"),
        }
    }

    // ---- extra headers ------------------------------------------------------

    #[tokio::test]
    async fn extra_headers_are_sent_on_every_request() {
        let server = fake(vec![tool_def("alpha")]).await;
        let mut c = cfg(&server);
        c.headers = vec![
            ("X-Harness".into(), "talos".into()),
            ("X-Mode".into(), "headless".into()),
        ];
        let tool = connect_mcp_tools(&c, T).await.expect("connect").remove(0);
        tool.run(json!({}), &ToolCtx::stub()).await;
        for rpc in ["initialize", "tools/list", "tools/call"] {
            for req in requests(&server, rpc).await {
                assert_eq!(header(&req, "x-harness"), Some("talos"), "{rpc}");
                assert_eq!(header(&req, "x-mode"), Some("headless"), "{rpc}");
            }
        }
    }

    #[tokio::test]
    async fn authorization_header_is_rejected() {
        let server = fake(vec![]).await;
        let mut c = cfg(&server);
        c.headers = vec![("authorization".into(), "Bearer x".into())];
        assert_eq!(
            connect_err(&c).await,
            McpError::Protocol("use bearer_token, not an Authorization header".into())
        );
        assert!(
            server
                .received_requests()
                .await
                .expect("captured")
                .is_empty()
        );
    }

    // ---- (16) version ------------------------------------------------------

    #[tokio::test]
    async fn unsupported_protocol_version() {
        let server = fake(vec![]).await;
        over(&server, "initialize", |r| {
            ok(r, json!({"protocolVersion": "1999-01-01"}))
        })
        .await;
        assert_eq!(
            connect_err(&cfg(&server)).await.to_string(),
            "protocol error: unsupported protocolVersion `1999-01-01`"
        );
    }

    #[tokio::test]
    async fn older_supported_version_is_negotiated() {
        let server = fake(vec![tool_def("alpha")]).await;
        over(&server, "initialize", |r| {
            ok(
                r,
                json!({"protocolVersion": "2025-03-26", "serverInfo": {"name": 7}}),
            )
        })
        .await;
        let mut reg = ToolRegistry::new();
        let reports = connect_and_register(&mut reg, &[cfg(&server)], T).await;
        assert!(matches!(
            &reports[0],
            McpConnectReport::Connected { protocol_version, server_name: None, server_version: None, .. }
                if protocol_version == "2025-03-26"
        ));
        let list = &requests(&server, "tools/list").await[0];
        assert_eq!(header(list, "mcp-protocol-version"), Some("2025-03-26"));
    }

    // ---- (17) session expiry -----------------------------------------------

    async fn expiring_server(second_init: fn(&Request) -> ResponseTemplate) -> MockServer {
        let server = fake(vec![tool_def("alpha")]).await;
        let n = Arc::new(AtomicUsize::new(0));
        over(&server, "initialize", move |r| {
            if n.fetch_add(1, Ordering::SeqCst) == 0 {
                ok(r, init_result()).insert_header("Mcp-Session-Id", "sess-1")
            } else {
                second_init(r)
            }
        })
        .await;
        over(&server, "tools/call", |r| {
            if header(r, "mcp-session-id") == Some("sess-2") {
                ok(r, json!({"content": [{"type": "text", "text": "again"}]}))
            } else {
                ResponseTemplate::new(404)
            }
        })
        .await;
        server
    }

    #[tokio::test]
    async fn expired_session_is_rehandshaken_once() {
        let server =
            expiring_server(|r| ok(r, init_result()).insert_header("Mcp-Session-Id", "sess-2"))
                .await;
        let tool = one_tool(&server).await;
        let r = tool.run(json!({}), &ToolCtx::stub()).await;
        assert!(!r.is_error, "{r:?}");
        assert_eq!(r.detail.as_deref(), Some("again"));
        let inits = requests(&server, "initialize").await;
        assert_eq!(inits.len(), 2);
        assert!(header(&inits[1], "mcp-session-id").is_none());
        // The id counter is never reset: initialize, list, call, initialize, call.
        assert_eq!(body(&inits[1])["id"], 4);
        let calls = requests(&server, "tools/call").await;
        assert_eq!(calls.len(), 2);
        assert_eq!(header(&calls[1], "mcp-session-id"), Some("sess-2"));
    }

    #[tokio::test]
    async fn failed_rehandshake_is_an_error_result() {
        let server = expiring_server(|_| ResponseTemplate::new(500)).await;
        let tool = one_tool(&server).await;
        let r = tool.run(json!({}), &ToolCtx::stub()).await;
        assert!(r.is_error);
        assert!(r.summary.contains("mcp server `fake`"), "{}", r.summary);
        assert!(r.summary.contains("HTTP 500"), "{}", r.summary);
        assert_eq!(requests(&server, "initialize").await.len(), 2);
    }

    #[tokio::test]
    async fn rehandshake_without_session_id_still_retries_once() {
        let server = expiring_server(|r| ok(r, init_result())).await;
        let tool = one_tool(&server).await;
        let r = tool.run(json!({}), &ToolCtx::stub()).await;
        assert!(r.summary.contains("HTTP 404"), "{}", r.summary);
        assert_eq!(requests(&server, "tools/call").await.len(), 2);
        assert!(header(&requests(&server, "tools/call").await[1], "mcp-session-id").is_none());
    }

    #[tokio::test]
    async fn not_found_without_a_session_is_not_retried() {
        let server = fake(vec![tool_def("alpha")]).await;
        over(&server, "tools/call", |_| ResponseTemplate::new(404)).await;
        let tool = one_tool(&server).await;
        let r = tool.run(json!({}), &ToolCtx::stub()).await;
        assert!(r.is_error);
        assert!(r.summary.contains("HTTP 404"), "{}", r.summary);
        assert_eq!(requests(&server, "initialize").await.len(), 1);
        assert_eq!(requests(&server, "tools/call").await.len(), 1);
    }

    // ---- (18)–(21) ---------------------------------------------------------

    #[tokio::test]
    async fn refused_port_is_a_transport_error() {
        let c = McpServerConfig {
            name: "fake".into(),
            url: "http://127.0.0.1:1/mcp".into(),
            bearer_token: None,
            headers: vec![],
        };
        let e = connect_err(&c).await;
        assert_eq!(e.kind(), "transport");
        assert!(matches!(e, McpError::Transport(_)));
        assert!(!e.to_string().contains("127.0.0.1:1/mcp"), "{e}");
    }

    #[tokio::test]
    async fn text_plain_content_type() {
        let server = fake(vec![]).await;
        over(&server, "initialize", |_| {
            ResponseTemplate::new(200).set_body_raw("{}", "text/plain")
        })
        .await;
        assert_eq!(
            connect_err(&cfg(&server)).await.to_string(),
            "protocol error: unexpected content type `text/plain`"
        );
    }

    #[tokio::test]
    async fn non_object_arguments_send_nothing() {
        let server = fake(vec![tool_def("alpha")]).await;
        let tool = one_tool(&server).await;
        let r = tool.run(json!([1]), &ToolCtx::stub()).await;
        assert_eq!(
            r,
            ToolResult::error("alpha: arguments must be a JSON object")
        );
        assert!(requests(&server, "tools/call").await.is_empty());
    }

    #[tokio::test]
    async fn no_servers_is_a_no_op() {
        let mut reg = ToolRegistry::new();
        reg.register("echo", Arc::new(EchoTool));
        let before = reg.list();
        assert!(connect_and_register(&mut reg, &[], T).await.is_empty());
        assert_eq!(reg.list(), before);
    }
}
