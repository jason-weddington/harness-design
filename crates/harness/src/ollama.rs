//! Ollama implementation of [`crate::model::ModelBackend`] — the second
//! backend behind the anti-corruption boundary, and the one that validates the
//! [`crate::model::ModelBackend`] trait against a genuinely different wire
//! format from Anthropic's.
//!
//! ## Scope
//!
//! Non-streaming only (this slice). A single [`ModelBackend::turn`] call POSTs
//! `{base_url}/api/chat` with `"stream": false`, awaits the full JSON
//! response, and translates Ollama's native chat shape into the normalized
//! types in [`crate::model`]. Streaming/SSE, structured outputs (`format`),
//! the OpenAI-compat `/v1` path, and capability discovery
//! (`/api/show|tags|ps`) are all out of scope here.
//!
//! ## One adapter, local + cloud
//!
//! [`OllamaBackend`] serves **both** a local Ollama daemon
//! (`http://localhost:11434`, no auth) and Ollama's hosted cloud
//! (`https://ollama.com`, `Authorization: Bearer <key>`) — they speak the same
//! `/api/chat` wire. The only difference is the base URL and whether an api
//! key was supplied; nothing else in this file branches on "cloud vs local".
//!
//! ## Design pins (mirrors [`crate::anthropic`]; don't re-derive)
//!
//! - **No community Ollama SDK.** `reqwest` + `serde` only — the same
//!   anti-corruption argument as the Anthropic adapter: a third-party SDK just
//!   adds another shape to translate and another supply-chain surface.
//! - **`"stream": false` is always explicit.** Ollama defaults `/api/chat` to
//!   *streaming*; a missing `stream` field would hand us an NDJSON stream this
//!   slice can't parse. The field is emitted on every request.
//! - **Model id is constructor config, never hardcoded.** [`OllamaBackend::new`]
//!   takes the model id (e.g. `glm-5.2:cloud` on cloud, a `qwen3.6`-family tag
//!   locally); the wire never carries a literal.
//! - **No api-key leakage in `Debug`.** [`OllamaBackend`] deliberately does
//!   **not** derive `Debug`; the api key only travels into the
//!   `Authorization` header.
//! - **The backend only classifies; the loop reacts.** Failures map to
//!   [`BackendError`] variants; nothing here retries or backs off.
//!
//! ## Testing
//!
//! Tests use `wiremock` — a local HTTP mock server — so the suite never
//! touches a live daemon, a cloud key, or the external network.

use async_trait::async_trait;
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize, Serializer};
use serde_json::{Map, Value};

use crate::model::{
    AssistantTurn, BackendError, ContentBlock, Message, ModelBackend, StopReason, TerminalKind,
    ToolCallRequest, TransientKind, TurnRequest, Usage, UserBlock,
};

// ============================================================================
// Public adapter
// ============================================================================

/// How hard Ollama should "think" for a turn — the value of the `think`
/// request field.
///
/// The variants serialize to the exact wire forms Ollama accepts:
/// [`Self::Off`] → `false`, [`Self::On`] → `true`, and the graded levels to
/// the strings `"low"`, `"medium"`, `"high"`, `"max"`. The field is **omitted
/// entirely** when [`OllamaBackend::with_think`] was never called — see
/// [`RequestBody::think`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkLevel {
    /// Reasoning disabled — serializes to `false`.
    Off,
    /// Reasoning enabled at the model's default depth — serializes to `true`.
    On,
    /// Serializes to `"low"`.
    Low,
    /// Serializes to `"medium"`.
    Medium,
    /// Serializes to `"high"`.
    High,
    /// Serializes to `"max"`.
    Max,
}

impl Serialize for ThinkLevel {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Off => serializer.serialize_bool(false),
            Self::On => serializer.serialize_bool(true),
            Self::Low => serializer.serialize_str("low"),
            Self::Medium => serializer.serialize_str("medium"),
            Self::High => serializer.serialize_str("high"),
            Self::Max => serializer.serialize_str("max"),
        }
    }
}

/// Ollama-backed [`ModelBackend`], serving both a local daemon and Ollama
/// cloud over the same `/api/chat` wire.
///
/// Construct with [`Self::new`] (model id + base URL) and layer optional
/// configuration via the builder methods: [`Self::with_api_key`] (cloud auth),
/// [`Self::with_num_ctx`] (context window + the pre-flight guard), and
/// [`Self::with_think`] (reasoning depth).
///
/// Does not derive [`Debug`] on purpose: the api key must not show up in a
/// formatter chain (panic messages, `dbg!`, structured logs).
pub struct OllamaBackend {
    client: Client,
    model: String,
    base_url: String,
    api_key: Option<String>,
    num_ctx: Option<u32>,
    think: Option<ThinkLevel>,
}

impl OllamaBackend {
    /// Build a backend pinned to `model` and pointed at `base_url`.
    ///
    /// No default origin: local vs cloud is a deployment choice the caller
    /// makes explicitly (`http://localhost:11434` or `https://ollama.com`). A
    /// trailing slash on `base_url` is tolerated — it is stripped at request
    /// time so `{base}/api/chat` is always a single-slash URL.
    pub fn new(model: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            model: model.into(),
            base_url: base_url.into(),
            api_key: None,
            num_ctx: None,
            think: None,
        }
    }

    /// Attach a bearer token — enables `Authorization: Bearer <key>` on every
    /// request. Required for Ollama cloud; omit entirely for a local daemon.
    #[must_use]
    pub fn with_api_key(mut self, api_key: String) -> Self {
        self.api_key = Some(api_key);
        self
    }

    /// Pin the model's context window (`options.num_ctx`) **and** arm the
    /// client-side pre-flight context guard (see [`estimate_prompt_tokens`]).
    #[must_use]
    pub fn with_num_ctx(mut self, num_ctx: u32) -> Self {
        self.num_ctx = Some(num_ctx);
        self
    }

    /// Set the reasoning depth (the `think` request field). When never called,
    /// the field is omitted from the wire entirely.
    #[must_use]
    pub fn with_think(mut self, think: ThinkLevel) -> Self {
        self.think = Some(think);
        self
    }
}

#[async_trait]
impl ModelBackend for OllamaBackend {
    async fn turn(&self, req: &TurnRequest<'_>) -> Result<AssistantTurn, BackendError> {
        // Build first: request assembly is fallible (an unresolvable
        // tool-result `call_id` is a Protocol error we must catch *before*
        // touching the network).
        let body = build_request_body(&self.model, self.num_ctx, self.think, req)?;

        // Pre-flight context guard. Ollama SILENTLY drops the oldest messages
        // on context overflow with no response signal (ollama/ollama#11885),
        // so the only way to make that invisible failure loud is to refuse the
        // request client-side before it is sent.
        //
        // A *post-hoc* prompt_eval_count-vs-estimate check is deliberately NOT
        // an error: Ollama's KV-cache prefix reuse makes `prompt_eval_count`
        // report only the *newly* evaluated tokens on a multi-turn
        // conversation, so a post-hoc comparison would false-positive on
        // perfectly healthy runs. The guard only runs when `num_ctx` is set;
        // with it unset (typical for cloud, which defaults to the model max)
        // there is nothing to compare against.
        if let Some(num_ctx) = self.num_ctx
            && estimate_prompt_tokens(&body) >= num_ctx as usize
        {
            return Err(BackendError::ContextLengthExceeded);
        }

        let url = format!("{}/api/chat", self.base_url.trim_end_matches('/'));
        let mut builder = self.client.post(&url).json(&body);
        if let Some(key) = &self.api_key {
            builder = builder.header("authorization", format!("Bearer {key}"));
        }

        let response = builder.send().await.map_err(|e| map_reqwest_error(&e))?;
        let status = response.status();
        let body_text = response.text().await.map_err(|e| map_reqwest_error(&e))?;

        if !status.is_success() {
            return Err(map_error_status(status, &body_text));
        }

        let parsed: ResponseBody =
            serde_json::from_str(&body_text).map_err(|e| BackendError::Protocol {
                message: format!("response body not parseable: {e}"),
                raw: Some(body_text.clone()),
            })?;

        Ok(map_response(parsed))
    }
}

// ============================================================================
// Request side — wire shapes that mirror Ollama's `/api/chat` body
// ============================================================================

/// Outgoing request body for `POST /api/chat`.
///
/// Field declaration order is fixed (serde emits in declaration order) so the
/// serialized bytes are deterministic across turns — the same prompt-cache
/// discipline the Anthropic adapter keeps.
#[derive(Serialize)]
struct RequestBody<'a> {
    model: &'a str,
    messages: Vec<WireMessage<'a>>,
    /// Always `false` this slice — Ollama defaults `/api/chat` to streaming,
    /// so the field must be present and explicit.
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Value>>,
    /// Omitted entirely when [`OllamaBackend::with_think`] was never called.
    #[serde(skip_serializing_if = "Option::is_none")]
    think: Option<ThinkLevel>,
    options: Options<'a>,
}

/// The `options` sub-object — sampling knobs plus the context-window pin.
#[derive(Serialize)]
struct Options<'a> {
    num_predict: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<&'a [String]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_ctx: Option<u32>,
}

/// One wire message. Ollama models `system`/`user`/`assistant`/`tool` roles
/// with `content` as a **string** (not Anthropic's block array), plus a couple
/// of role-specific optional fields (`thinking`, `tool_name`, `tool_calls`).
#[derive(Serialize)]
struct WireMessage<'a> {
    role: &'static str,
    content: String,
    /// Assistant-only echo-back of the reasoning trace. Omitted when empty.
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<String>,
    /// `tool`-role only: the name of the tool whose result this message
    /// carries (resolved from the matching prior assistant tool call).
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_name: Option<&'a str>,
    /// Assistant-only: the tool calls the model requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<WireToolCall<'a>>>,
}

impl WireMessage<'_> {
    /// A plain `role`/`content` message with every optional field absent.
    fn simple(role: &'static str, content: String) -> Self {
        Self {
            role,
            content,
            thinking: None,
            tool_name: None,
            tool_calls: None,
        }
    }
}

#[derive(Serialize)]
struct WireToolCall<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    function: WireToolCallFunction<'a>,
}

#[derive(Serialize)]
struct WireToolCallFunction<'a> {
    index: usize,
    name: &'a str,
    arguments: &'a Value,
}

#[allow(clippy::too_many_lines)]
fn build_request_body<'a>(
    model: &'a str,
    num_ctx: Option<u32>,
    think: Option<ThinkLevel>,
    req: &'a TurnRequest<'a>,
) -> Result<RequestBody<'a>, BackendError> {
    let mut messages: Vec<WireMessage<'a>> = Vec::new();

    // The system prompt rides on TurnRequest, not as a Message — Ollama has no
    // top-level `system` field, so it becomes the FIRST `system`-role message.
    if let Some(system) = req.system {
        messages.push(WireMessage::simple("system", system.to_string()));
    }

    for (idx, message) in req.messages.iter().enumerate() {
        match message {
            Message::User { content } => {
                let mut text_parts: Vec<&str> = Vec::new();
                let mut tool_messages: Vec<WireMessage<'a>> = Vec::new();
                for block in content {
                    match block {
                        UserBlock::Text(text) => text_parts.push(text),
                        UserBlock::ToolResult {
                            call_id,
                            content,
                            is_error,
                        } => {
                            // tool_name is RESOLVED by scanning backwards
                            // through this request's own history for the
                            // assistant ToolCall whose id == call_id. Result
                            // order is preserved = call order, the only
                            // disambiguator for duplicate-name parallel calls.
                            let name = resolve_tool_name(req.messages, idx, call_id).ok_or_else(
                                || BackendError::Protocol {
                                    message: format!(
                                        "tool result call_id {call_id:?} does not resolve to any prior assistant tool call"
                                    ),
                                    raw: None,
                                },
                            )?;
                            let body = if *is_error {
                                format!("ERROR: {content}")
                            } else {
                                content.clone()
                            };
                            tool_messages.push(WireMessage {
                                role: "tool",
                                content: body,
                                thinking: None,
                                tool_name: Some(name),
                                tool_calls: None,
                            });
                        }
                    }
                }
                if !text_parts.is_empty() {
                    messages.push(WireMessage::simple("user", text_parts.join("\n\n")));
                }
                messages.extend(tool_messages);
            }
            Message::Assistant { content } => {
                let mut text_parts: Vec<&str> = Vec::new();
                let mut think_parts: Vec<&str> = Vec::new();
                let mut tool_calls: Vec<WireToolCall<'a>> = Vec::new();
                for block in content {
                    match block {
                        ContentBlock::Text(text) => text_parts.push(text),
                        // The reasoning ECHO-BACK the docs require: a prior
                        // Reasoning block goes back out on the assistant
                        // message's `thinking` field.
                        ContentBlock::Reasoning { text, .. } => think_parts.push(text),
                        ContentBlock::ToolCall(call) => {
                            let index = tool_calls.len();
                            tool_calls.push(WireToolCall {
                                kind: "function",
                                function: WireToolCallFunction {
                                    index,
                                    name: &call.name,
                                    arguments: &call.input,
                                },
                            });
                        }
                    }
                }
                messages.push(WireMessage {
                    role: "assistant",
                    content: text_parts.join("\n\n"),
                    thinking: if think_parts.is_empty() {
                        None
                    } else {
                        Some(think_parts.join("\n\n"))
                    },
                    tool_name: None,
                    tool_calls: if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    },
                });
            }
        }
    }

    let tools = if req.tools.is_empty() {
        None
    } else {
        Some(req.tools.iter().map(map_tool).collect())
    };

    Ok(RequestBody {
        model,
        messages,
        stream: false,
        tools,
        think,
        options: Options {
            num_predict: req.params.max_tokens,
            temperature: req.params.temperature,
            stop: if req.params.stop_sequences.is_empty() {
                None
            } else {
                Some(req.params.stop_sequences.as_slice())
            },
            num_ctx,
        },
    })
}

/// Scan backwards through the messages *before* `upto` for the assistant
/// `ToolCall` whose id equals `call_id`, returning its tool name.
///
/// Stateless by design: resolution reads only the request's own history, so
/// the adapter carries no cross-call state. Backwards + most-recent-wins keeps
/// the newest matching call authoritative if an id were ever reused.
fn resolve_tool_name<'a>(messages: &'a [Message], upto: usize, call_id: &str) -> Option<&'a str> {
    for message in messages[..upto].iter().rev() {
        if let Message::Assistant { content } = message {
            for block in content.iter().rev() {
                if let ContentBlock::ToolCall(call) = block
                    && call.id == call_id
                {
                    return Some(call.name.as_str());
                }
            }
        }
    }
    None
}

/// Translate one of our tool schemas (`{name, description, input_schema}`)
/// into Ollama's function-tool shape. The load-bearing detail is the
/// `input_schema` → `parameters` **rename**; everything else is a passthrough.
fn map_tool(tool: &Value) -> Value {
    let mut function = Map::new();
    if let Some(name) = tool.get("name") {
        function.insert("name".to_string(), name.clone());
    }
    if let Some(description) = tool.get("description") {
        function.insert("description".to_string(), description.clone());
    }
    if let Some(schema) = tool.get("input_schema") {
        function.insert("parameters".to_string(), schema.clone());
    }
    let mut root = Map::new();
    root.insert("type".to_string(), Value::String("function".to_string()));
    root.insert("function".to_string(), Value::Object(function));
    Value::Object(root)
}

/// Rough client-side token estimate for the pre-flight context guard:
/// the char length of every message/system/tool text field, divided by 4.
/// Deliberately crude — it only has to be a conservative tripwire against the
/// silent-drop overflow, not an accurate tokenizer.
fn estimate_prompt_tokens(body: &RequestBody) -> usize {
    let mut chars = 0usize;
    for message in &body.messages {
        chars += message.content.len();
        if let Some(thinking) = &message.thinking {
            chars += thinking.len();
        }
        if let Some(tool_name) = message.tool_name {
            chars += tool_name.len();
        }
        if let Some(calls) = &message.tool_calls {
            for call in calls {
                chars += call.function.name.len();
                chars += call.function.arguments.to_string().len();
            }
        }
    }
    if let Some(tools) = &body.tools {
        for tool in tools {
            chars += tool.to_string().len();
        }
    }
    chars / 4
}

// ============================================================================
// Response side — wire shapes that Ollama's `/api/chat` returns
// ============================================================================

#[derive(Deserialize)]
struct ResponseBody {
    message: ResponseMessage,
    #[serde(default)]
    done_reason: Option<String>,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
    #[serde(default)]
    eval_count: Option<u32>,
}

#[derive(Deserialize)]
struct ResponseMessage {
    #[serde(default)]
    content: String,
    #[serde(default)]
    thinking: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ResponseToolCall>,
}

#[derive(Deserialize)]
struct ResponseToolCall {
    function: ResponseToolCallFunction,
}

#[derive(Deserialize)]
struct ResponseToolCallFunction {
    name: String,
    // `arguments` is a JSON OBJECT on the wire (not a string) — no double
    // parse. Default to `null` if a model ever omits it.
    #[serde(default)]
    arguments: Value,
}

fn map_response(body: ResponseBody) -> AssistantTurn {
    let ResponseMessage {
        content,
        thinking,
        tool_calls,
    } = body.message;

    let mut blocks: Vec<ContentBlock> = Vec::new();

    // Order: thinking FIRST, then visible content, then tool calls.
    if let Some(thinking) = thinking
        && !thinking.is_empty()
    {
        blocks.push(ContentBlock::Reasoning {
            text: thinking,
            opaque: None,
        });
    }
    if !content.is_empty() {
        blocks.push(ContentBlock::Text(content));
    }
    let has_tool_calls = !tool_calls.is_empty();
    for (i, call) in tool_calls.into_iter().enumerate() {
        blocks.push(ContentBlock::ToolCall(ToolCallRequest {
            id: format!("ollama-call-{i}"),
            name: call.function.name,
            input: call.function.arguments,
        }));
    }

    AssistantTurn {
        content: blocks,
        stop_reason: map_stop_reason(has_tool_calls, body.done_reason.as_deref()),
        usage: Usage {
            // Absent ≠ zero, but the trait's Usage requires a concrete u32 for
            // the two mandatory counters; Ollama always reports these on a
            // successful non-streaming turn, so `unwrap_or(0)` is a
            // belt-and-braces fallback rather than an expected path.
            input_tokens: body.prompt_eval_count.unwrap_or(0),
            output_tokens: body.eval_count.unwrap_or(0),
            // Ollama reports none of these — leave them None (absent ≠ zero).
            cache_read_tokens: None,
            cache_write_tokens: None,
            reasoning_tokens: None,
        },
    }
}

/// Map Ollama's stop signal to a normalized [`StopReason`].
///
/// **Tool-call presence beats `done_reason`.** Ollama has NO `tool_use`
/// `done_reason`, so a non-empty `tool_calls` array is the only signal that the
/// model wants to call a tool — we infer [`StopReason::ToolUse`] from presence
/// and ignore whatever `done_reason` says.
///
/// `done_reason` is otherwise **advisory**: known GLM bugs report `"stop"` on
/// a truncated turn, and — because Ollama gives no distinct stop-sequence
/// signal — a stop-sequence hit is **indistinguishable from a natural
/// [`StopReason::EndTurn`]** on this backend.
fn map_stop_reason(has_tool_calls: bool, done_reason: Option<&str>) -> StopReason {
    if has_tool_calls {
        return StopReason::ToolUse;
    }
    match done_reason {
        Some("stop") => StopReason::EndTurn,
        Some("length") => StopReason::MaxTokens,
        Some(other) => StopReason::Other(other.to_string()),
        None => StopReason::Other(String::new()),
    }
}

// ============================================================================
// Error mapping
// ============================================================================

/// The `{"error": "..."}` body shape. This convention is undocumented folklore
/// — hence the [`estimate_prompt_tokens`]-style skepticism: we parse it to
/// extract a human message for classification, but always fall back to (and,
/// in the Protocol case, preserve) the raw body.
#[derive(Deserialize)]
struct ErrorBody {
    #[serde(default)]
    error: String,
}

fn extract_error_message(body_text: &str) -> String {
    serde_json::from_str::<ErrorBody>(body_text)
        .ok()
        .map(|e| e.error)
        .filter(|e| !e.is_empty())
        .unwrap_or_else(|| body_text.to_string())
}

/// Reqwest's pre-response errors (connect refused, DNS, reset, timeout) → the
/// transient bucket the loop knows how to retry.
///
/// Note: [`BackendError::Transient`] carries no message field, so the
/// diagnostic detail a connect failure would ideally surface (the target
/// `base_url`, and an "is ollama running?" hint for a refused localhost
/// daemon) has nowhere to ride today. When the loop gains structured logging
/// that hint belongs there; the *classification* is all the trait can carry.
fn map_reqwest_error(e: &reqwest::Error) -> BackendError {
    classify_transport_error(e.is_timeout())
}

fn classify_transport_error(is_timeout: bool) -> BackendError {
    let kind = if is_timeout {
        TransientKind::Timeout
    } else {
        // connect / DNS / reset / body / decode all collapse to a
        // network-class transient: the request never produced a usable
        // response, and retrying is the right shape.
        TransientKind::Network
    };
    BackendError::Transient {
        kind,
        retry_after: None,
    }
}

/// Translate a non-2xx response into the appropriate [`BackendError`].
fn map_error_status(status: StatusCode, body_text: &str) -> BackendError {
    let message = extract_error_message(body_text);
    let lower = message.to_lowercase();

    // Model-not-found surfaces as a 404 OR as an error string that merely
    // *contains* "not found" (with the model name in context) under some
    // statuses — treat both as the same terminal signal.
    if status.as_u16() == 404 || lower.contains("not found") {
        return BackendError::Terminal {
            kind: TerminalKind::UnknownModel,
            message,
        };
    }

    match status.as_u16() {
        401 | 403 => BackendError::Terminal {
            kind: TerminalKind::Auth,
            message,
        },
        400 => {
            if lower.contains("does not support tools") {
                BackendError::Terminal {
                    kind: TerminalKind::SchemaRejected,
                    message,
                }
            } else {
                BackendError::Terminal {
                    kind: TerminalKind::BadRequest,
                    message,
                }
            }
        }
        // Cloud sends no Retry-After header on a 429, so `retry_after` is None
        // and the loop falls back to its own backoff.
        429 => BackendError::Transient {
            kind: TransientKind::RateLimit,
            retry_after: None,
        },
        s if (500..600).contains(&s) => BackendError::Transient {
            kind: TransientKind::ServerError,
            retry_after: None,
        },
        _ => BackendError::Terminal {
            kind: TerminalKind::Other,
            message,
        },
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::{
        OllamaBackend, ThinkLevel, classify_transport_error, extract_error_message, map_stop_reason,
    };
    use crate::model::{
        BackendError, ContentBlock, Message, ModelBackend, SamplingParams, StopReason,
        TerminalKind, ToolCallRequest, TransientKind, TurnRequest, UserBlock,
    };
    use serde_json::{Value, json};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, Request, ResponseTemplate};

    // ---- small helpers -----------------------------------------------------

    fn params() -> SamplingParams {
        SamplingParams {
            max_tokens: 1024,
            temperature: Some(0.0),
            stop_sequences: vec!["STOP".to_string()],
        }
    }

    fn user_hi() -> Vec<Message> {
        vec![Message::User {
            content: vec![UserBlock::Text("hi".to_string())],
        }]
    }

    async fn mount_success(server: &MockServer, body: &str) {
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_string(body))
            .mount(server)
            .await;
    }

    fn simple_req<'a>(
        messages: &'a [Message],
        tools: &'a [Value],
        params: &'a SamplingParams,
    ) -> TurnRequest<'a> {
        TurnRequest {
            system: None,
            messages,
            tools,
            params,
        }
    }

    // ---- (a) happy path: thinking + content + two tool calls --------------

    #[tokio::test]
    async fn maps_thinking_content_and_two_tool_calls() {
        let server = MockServer::start().await;
        let body = json!({
            "message": {
                "role": "assistant",
                "content": "here you go",
                "thinking": "let me reason",
                "tool_calls": [
                    {"function": {"name": "read_file", "arguments": {"path": "a.rs"}}},
                    {"function": {"name": "list_dir", "arguments": {"path": "."}}}
                ]
            },
            "done_reason": "stop",
            "prompt_eval_count": 42,
            "eval_count": 7
        })
        .to_string();
        mount_success(&server, &body).await;

        let backend = OllamaBackend::new("qwen3.6", server.uri());
        let messages = user_hi();
        let tools: Vec<Value> = vec![];
        let p = params();
        let req = simple_req(&messages, &tools, &p);

        let turn = backend.turn(&req).await.expect("turn ok");
        // Reasoning FIRST, then Text, then the two tool calls.
        assert_eq!(turn.content.len(), 4);
        match &turn.content[0] {
            ContentBlock::Reasoning { text, opaque } => {
                assert_eq!(text, "let me reason");
                assert!(opaque.is_none());
            }
            other => panic!("expected Reasoning first, got {other:?}"),
        }
        assert_eq!(turn.text(), "here you go");
        let calls = turn.tool_calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "ollama-call-0");
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].input, json!({"path": "a.rs"}));
        assert_eq!(calls[1].id, "ollama-call-1");
        assert_eq!(calls[1].name, "list_dir");
        // tool_calls presence beats done_reason "stop".
        assert!(matches!(turn.stop_reason, StopReason::ToolUse));
        assert_eq!(turn.usage.input_tokens, 42);
        assert_eq!(turn.usage.output_tokens, 7);
        assert_eq!(turn.usage.cache_read_tokens, None);
        assert_eq!(turn.usage.cache_write_tokens, None);
        assert_eq!(turn.usage.reasoning_tokens, None);
    }

    // ---- (b) request-shape capture ----------------------------------------

    #[allow(clippy::too_many_lines)]
    #[tokio::test]
    async fn outgoing_request_shape_is_correct() {
        let server = MockServer::start().await;
        mount_success(
            &server,
            &json!({
                "message": {"role": "assistant", "content": "ok"},
                "done_reason": "stop",
                "prompt_eval_count": 0,
                "eval_count": 0
            })
            .to_string(),
        )
        .await;

        let messages = vec![Message::User {
            content: vec![UserBlock::Text("hello".to_string())],
        }];
        let tools: Vec<Value> = vec![json!({
            "name": "echo",
            "description": "echoes input back",
            "input_schema": {"type": "object", "properties": {}}
        })];
        let p = SamplingParams {
            max_tokens: 512,
            temperature: Some(0.3),
            stop_sequences: vec!["END".to_string()],
        };
        let req = TurnRequest {
            system: Some("you are a harness"),
            messages: &messages,
            tools: &tools,
            params: &p,
        };

        // High think + a generously large num_ctx (so the guard doesn't trip)
        // + an api key (so the Bearer header is present).
        let backend = OllamaBackend::new("glm-5.2:cloud", server.uri())
            .with_api_key("sk-cloud".to_string())
            .with_num_ctx(100_000)
            .with_think(ThinkLevel::High);
        backend.turn(&req).await.expect("turn ok");

        let received = server.received_requests().await.expect("requests captured");
        assert_eq!(received.len(), 1);
        let r: &Request = &received[0];

        // Bearer header present with the key.
        assert_eq!(
            r.headers.get("authorization").and_then(|v| v.to_str().ok()),
            Some("Bearer sk-cloud")
        );

        let parsed: Value = serde_json::from_slice(&r.body).expect("json body");
        assert_eq!(parsed["model"], "glm-5.2:cloud");
        // stream:false ALWAYS explicit.
        assert_eq!(parsed["stream"], false);
        // think serialized as the "high" string.
        assert_eq!(parsed["think"], "high");

        // System becomes the FIRST message.
        let msgs = parsed["messages"].as_array().expect("messages array");
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "you are a harness");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[1]["content"], "hello");

        // input_schema -> parameters rename under function-tool shape.
        assert_eq!(parsed["tools"][0]["type"], "function");
        assert_eq!(parsed["tools"][0]["function"]["name"], "echo");
        assert_eq!(
            parsed["tools"][0]["function"]["description"],
            "echoes input back"
        );
        assert_eq!(
            parsed["tools"][0]["function"]["parameters"],
            json!({"type": "object", "properties": {}})
        );
        assert!(
            parsed["tools"][0]["function"].get("input_schema").is_none(),
            "input_schema must be renamed to parameters"
        );

        // options placement.
        assert_eq!(parsed["options"]["num_predict"], 512);
        assert_eq!(parsed["options"]["temperature"], 0.3);
        assert_eq!(parsed["options"]["stop"], json!(["END"]));
        assert_eq!(parsed["options"]["num_ctx"], 100_000);
    }

    // ---- (b) think=Off serializes to false; no key => no Bearer -----------

    #[tokio::test]
    async fn think_off_serializes_false_and_no_key_omits_bearer() {
        let server = MockServer::start().await;
        mount_success(
            &server,
            &json!({
                "message": {"role": "assistant", "content": "ok"},
                "done_reason": "stop", "prompt_eval_count": 0, "eval_count": 0
            })
            .to_string(),
        )
        .await;

        let backend = OllamaBackend::new("qwen3.6", server.uri()).with_think(ThinkLevel::Off);
        let messages = user_hi();
        let tools: Vec<Value> = vec![];
        let p = params();
        let req = simple_req(&messages, &tools, &p);
        backend.turn(&req).await.expect("turn ok");

        let received = server.received_requests().await.unwrap();
        let r = &received[0];
        assert!(
            r.headers.get("authorization").is_none(),
            "no api key => no Authorization header"
        );
        let parsed: Value = serde_json::from_slice(&r.body).unwrap();
        assert_eq!(parsed["think"], false);
    }

    // ---- (b) optional fields omitted when unset ---------------------------

    #[tokio::test]
    async fn optional_fields_omitted_when_unset() {
        let server = MockServer::start().await;
        mount_success(
            &server,
            &json!({
                "message": {"role": "assistant", "content": "ok"},
                "done_reason": "stop", "prompt_eval_count": 0, "eval_count": 0
            })
            .to_string(),
        )
        .await;

        let messages = user_hi();
        let tools: Vec<Value> = vec![];
        let p = SamplingParams {
            max_tokens: 16,
            temperature: None,
            stop_sequences: vec![],
        };
        let req = simple_req(&messages, &tools, &p);

        // No api key, no num_ctx, no think, no tools, no temperature/stop.
        let backend = OllamaBackend::new("qwen3.6", server.uri());
        backend.turn(&req).await.expect("turn ok");

        let received = server.received_requests().await.unwrap();
        let body_text = std::str::from_utf8(&received[0].body).unwrap();
        assert!(!body_text.contains("\"think\""));
        assert!(!body_text.contains("\"tools\""));
        assert!(!body_text.contains("\"temperature\""));
        assert!(!body_text.contains("\"stop\""));
        assert!(!body_text.contains("\"num_ctx\""));
        // Required fields still present, stream explicit.
        assert!(body_text.contains("\"model\""));
        assert!(body_text.contains("\"stream\":false"));
        assert!(body_text.contains("\"num_predict\""));
    }

    // ---- (c) tool-result round-trip ---------------------------------------

    #[tokio::test]
    async fn tool_result_resolves_name_and_prefixes_error() {
        let server = MockServer::start().await;
        mount_success(
            &server,
            &json!({
                "message": {"role": "assistant", "content": "done"},
                "done_reason": "stop", "prompt_eval_count": 1, "eval_count": 1
            })
            .to_string(),
        )
        .await;

        let messages = vec![
            Message::Assistant {
                content: vec![ContentBlock::ToolCall(ToolCallRequest {
                    id: "ollama-call-0".to_string(),
                    name: "read_file".to_string(),
                    input: json!({"path": "a.rs"}),
                })],
            },
            Message::User {
                content: vec![UserBlock::ToolResult {
                    call_id: "ollama-call-0".to_string(),
                    content: "boom".to_string(),
                    is_error: true,
                }],
            },
        ];
        let tools: Vec<Value> = vec![];
        let p = params();
        let req = simple_req(&messages, &tools, &p);

        let backend = OllamaBackend::new("qwen3.6", server.uri());
        backend.turn(&req).await.expect("turn ok");

        let received = server.received_requests().await.unwrap();
        let parsed: Value = serde_json::from_slice(&received[0].body).unwrap();
        let msgs = parsed["messages"].as_array().unwrap();
        // [assistant tool call, tool result]
        assert_eq!(msgs[0]["role"], "assistant");
        assert_eq!(msgs[0]["tool_calls"][0]["type"], "function");
        assert_eq!(msgs[0]["tool_calls"][0]["function"]["name"], "read_file");
        assert_eq!(msgs[0]["tool_calls"][0]["function"]["index"], 0);
        assert_eq!(
            msgs[0]["tool_calls"][0]["function"]["arguments"],
            json!({"path": "a.rs"})
        );
        let tool_msg = &msgs[1];
        assert_eq!(tool_msg["role"], "tool");
        assert_eq!(tool_msg["tool_name"], "read_file");
        assert_eq!(tool_msg["content"], "ERROR: boom");
    }

    #[tokio::test]
    async fn unresolvable_call_id_is_protocol_error_and_sends_nothing() {
        let server = MockServer::start().await;
        // No mount: if a request were sent, it would 404; we assert zero were.

        let messages = vec![Message::User {
            content: vec![UserBlock::ToolResult {
                call_id: "ollama-call-99".to_string(),
                content: "orphan".to_string(),
                is_error: false,
            }],
        }];
        let tools: Vec<Value> = vec![];
        let p = params();
        let req = simple_req(&messages, &tools, &p);

        let backend = OllamaBackend::new("qwen3.6", server.uri());
        let err = backend.turn(&req).await.expect_err("must fail");
        match err {
            BackendError::Protocol { message, raw } => {
                assert!(message.contains("ollama-call-99"));
                assert!(raw.is_none());
            }
            other => panic!("expected Protocol, got {other:?}"),
        }
        let received = server.received_requests().await.unwrap();
        assert_eq!(received.len(), 0, "nothing may be sent");
    }

    #[tokio::test]
    async fn parallel_duplicate_name_calls_keep_result_order() {
        let server = MockServer::start().await;
        mount_success(
            &server,
            &json!({
                "message": {"role": "assistant", "content": "done"},
                "done_reason": "stop", "prompt_eval_count": 1, "eval_count": 1
            })
            .to_string(),
        )
        .await;

        let messages = vec![
            Message::Assistant {
                content: vec![
                    ContentBlock::ToolCall(ToolCallRequest {
                        id: "ollama-call-0".to_string(),
                        name: "search".to_string(),
                        input: json!({"q": "first"}),
                    }),
                    ContentBlock::ToolCall(ToolCallRequest {
                        id: "ollama-call-1".to_string(),
                        name: "search".to_string(),
                        input: json!({"q": "second"}),
                    }),
                ],
            },
            // Results supplied in call-1, call-0 order — the wire must keep
            // THIS order (result order == call order is the only disambiguator
            // for duplicate names).
            Message::User {
                content: vec![
                    UserBlock::ToolResult {
                        call_id: "ollama-call-1".to_string(),
                        content: "for second".to_string(),
                        is_error: false,
                    },
                    UserBlock::ToolResult {
                        call_id: "ollama-call-0".to_string(),
                        content: "for first".to_string(),
                        is_error: false,
                    },
                ],
            },
        ];
        let tools: Vec<Value> = vec![];
        let p = params();
        let req = simple_req(&messages, &tools, &p);

        let backend = OllamaBackend::new("qwen3.6", server.uri());
        backend.turn(&req).await.expect("turn ok");

        let received = server.received_requests().await.unwrap();
        let parsed: Value = serde_json::from_slice(&received[0].body).unwrap();
        let msgs = parsed["messages"].as_array().unwrap();
        // msgs[0] = assistant; msgs[1], msgs[2] = the two tool results in order.
        assert_eq!(msgs[1]["role"], "tool");
        assert_eq!(msgs[1]["tool_name"], "search");
        assert_eq!(msgs[1]["content"], "for second");
        assert_eq!(msgs[2]["role"], "tool");
        assert_eq!(msgs[2]["tool_name"], "search");
        assert_eq!(msgs[2]["content"], "for first");
    }

    // ---- (d) thinking echo -------------------------------------------------

    #[tokio::test]
    async fn assistant_reasoning_echoes_into_thinking_field() {
        let server = MockServer::start().await;
        mount_success(
            &server,
            &json!({
                "message": {"role": "assistant", "content": "ok"},
                "done_reason": "stop", "prompt_eval_count": 1, "eval_count": 1
            })
            .to_string(),
        )
        .await;

        let messages = vec![Message::Assistant {
            content: vec![
                ContentBlock::Reasoning {
                    text: "prior thought".to_string(),
                    opaque: None,
                },
                ContentBlock::Text("prior answer".to_string()),
            ],
        }];
        let tools: Vec<Value> = vec![];
        let p = params();
        let req = simple_req(&messages, &tools, &p);

        let backend = OllamaBackend::new("qwen3.6", server.uri());
        backend.turn(&req).await.expect("turn ok");

        let received = server.received_requests().await.unwrap();
        let parsed: Value = serde_json::from_slice(&received[0].body).unwrap();
        let msg = &parsed["messages"][0];
        assert_eq!(msg["role"], "assistant");
        assert_eq!(msg["content"], "prior answer");
        assert_eq!(msg["thinking"], "prior thought");
    }

    // ---- (e) done_reason mappings -----------------------------------------

    #[test]
    fn stop_reason_mapping_covers_arms() {
        // tool-call presence beats any done_reason.
        assert!(matches!(
            map_stop_reason(true, Some("length")),
            StopReason::ToolUse
        ));
        assert!(matches!(
            map_stop_reason(false, Some("stop")),
            StopReason::EndTurn
        ));
        assert!(matches!(
            map_stop_reason(false, Some("length")),
            StopReason::MaxTokens
        ));
        match map_stop_reason(false, Some("guard")) {
            StopReason::Other(s) => assert_eq!(s, "guard"),
            other => panic!("expected Other, got {other:?}"),
        }
        match map_stop_reason(false, None) {
            StopReason::Other(s) => assert!(s.is_empty()),
            other => panic!("expected Other(\"\"), got {other:?}"),
        }
    }

    #[tokio::test]
    async fn length_done_reason_maps_to_max_tokens() {
        let server = MockServer::start().await;
        mount_success(
            &server,
            &json!({
                "message": {"role": "assistant", "content": "truncated"},
                "done_reason": "length", "prompt_eval_count": 3, "eval_count": 9
            })
            .to_string(),
        )
        .await;
        let backend = OllamaBackend::new("qwen3.6", server.uri());
        let messages = user_hi();
        let tools: Vec<Value> = vec![];
        let p = params();
        let req = simple_req(&messages, &tools, &p);
        let turn = backend.turn(&req).await.expect("turn ok");
        assert!(matches!(turn.stop_reason, StopReason::MaxTokens));
        // content-only response: exactly one Text block, no Reasoning.
        assert_eq!(turn.content.len(), 1);
        assert_eq!(turn.text(), "truncated");
    }

    #[tokio::test]
    async fn empty_content_and_thinking_produce_no_blocks() {
        let server = MockServer::start().await;
        mount_success(
            &server,
            &json!({
                "message": {"role": "assistant", "content": "", "thinking": ""},
                "done_reason": "stop", "prompt_eval_count": 0, "eval_count": 0
            })
            .to_string(),
        )
        .await;
        let backend = OllamaBackend::new("qwen3.6", server.uri());
        let messages = user_hi();
        let tools: Vec<Value> = vec![];
        let p = params();
        let req = simple_req(&messages, &tools, &p);
        let turn = backend.turn(&req).await.expect("turn ok");
        assert_eq!(turn.content.len(), 0);
        assert!(matches!(turn.stop_reason, StopReason::EndTurn));
    }

    // ---- (f) error mapping arms -------------------------------------------

    async fn error_turn(status: u16, body: &str) -> BackendError {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(status).set_body_string(body))
            .mount(&server)
            .await;
        let backend = OllamaBackend::new("qwen3.6", server.uri());
        let messages = user_hi();
        let tools: Vec<Value> = vec![];
        let p = params();
        let req = simple_req(&messages, &tools, &p);
        backend.turn(&req).await.expect_err("must fail")
    }

    #[tokio::test]
    async fn maps_404_to_unknown_model() {
        let err = error_turn(404, r#"{"error":"model 'bogus' not found"}"#).await;
        match err {
            BackendError::Terminal { kind, message } => {
                assert_eq!(kind, TerminalKind::UnknownModel);
                assert!(message.contains("bogus"));
            }
            other => panic!("expected UnknownModel, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn maps_not_found_folklore_body_under_non_404() {
        // A 400 whose body merely contains "not found" still classifies as
        // UnknownModel (folklore string beats the status code here).
        let err = error_turn(400, r#"{"error":"model qwen3.6 not found, pull it first"}"#).await;
        assert!(matches!(
            err,
            BackendError::Terminal {
                kind: TerminalKind::UnknownModel,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn maps_401_and_403_to_auth() {
        for status in [401u16, 403] {
            let err = error_turn(status, r#"{"error":"unauthorized"}"#).await;
            assert!(matches!(
                err,
                BackendError::Terminal {
                    kind: TerminalKind::Auth,
                    ..
                }
            ));
        }
    }

    #[tokio::test]
    async fn maps_400_does_not_support_tools_to_schema_rejected() {
        let err = error_turn(400, r#"{"error":"model does not support tools"}"#).await;
        assert!(matches!(
            err,
            BackendError::Terminal {
                kind: TerminalKind::SchemaRejected,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn maps_400_generic_to_bad_request() {
        let err = error_turn(400, r#"{"error":"invalid options.num_predict"}"#).await;
        match err {
            BackendError::Terminal { kind, message } => {
                assert_eq!(kind, TerminalKind::BadRequest);
                assert!(message.contains("num_predict"));
            }
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn maps_429_to_rate_limit_without_retry_after() {
        let err = error_turn(429, r#"{"error":"rate limited"}"#).await;
        assert!(err.is_retryable());
        match err {
            BackendError::Transient { kind, retry_after } => {
                assert_eq!(kind, TransientKind::RateLimit);
                assert!(retry_after.is_none(), "cloud sends no Retry-After");
            }
            other => panic!("expected Transient RateLimit, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn maps_500_to_server_error() {
        let err = error_turn(500, r#"{"error":"internal"}"#).await;
        assert!(matches!(
            err,
            BackendError::Transient {
                kind: TransientKind::ServerError,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn maps_unlabeled_status_to_terminal_other() {
        // A plain (non-folklore) body also exercises the raw-body fallback in
        // extract_error_message.
        let err = error_turn(418, "i'm a teapot").await;
        match err {
            BackendError::Terminal { kind, message } => {
                assert_eq!(kind, TerminalKind::Other);
                assert_eq!(message, "i'm a teapot");
            }
            other => panic!("expected Terminal Other, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn malformed_success_body_maps_to_protocol_with_raw() {
        let server = MockServer::start().await;
        mount_success(&server, "not even json").await;
        let backend = OllamaBackend::new("qwen3.6", server.uri());
        let messages = user_hi();
        let tools: Vec<Value> = vec![];
        let p = params();
        let req = simple_req(&messages, &tools, &p);
        let err = backend.turn(&req).await.expect_err("must fail");
        match err {
            BackendError::Protocol { message, raw } => {
                assert!(message.contains("not parseable"));
                assert_eq!(raw.as_deref(), Some("not even json"));
            }
            other => panic!("expected Protocol, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn connect_refused_maps_to_transient_network() {
        // Nothing listening on 127.0.0.1:1 — the canonical refused-port trick.
        let backend = OllamaBackend::new("qwen3.6", "http://127.0.0.1:1");
        let messages = user_hi();
        let tools: Vec<Value> = vec![];
        let p = params();
        let req = simple_req(&messages, &tools, &p);
        let err = backend.turn(&req).await.expect_err("connect must fail");
        assert!(err.is_retryable());
        assert!(matches!(
            err,
            BackendError::Transient {
                kind: TransientKind::Network,
                ..
            }
        ));
    }

    #[test]
    fn transport_error_classification_covers_timeout_and_network() {
        assert!(matches!(
            classify_transport_error(true),
            BackendError::Transient {
                kind: TransientKind::Timeout,
                ..
            }
        ));
        assert!(matches!(
            classify_transport_error(false),
            BackendError::Transient {
                kind: TransientKind::Network,
                ..
            }
        ));
    }

    #[test]
    fn extract_error_message_falls_back_to_raw_body() {
        assert_eq!(extract_error_message(r#"{"error":"boom"}"#), "boom");
        // Empty error string => raw body fallback.
        assert_eq!(extract_error_message(r#"{"error":""}"#), r#"{"error":""}"#);
        // Non-object body => raw body fallback.
        assert_eq!(extract_error_message("plain text"), "plain text");
    }

    // ---- ThinkLevel serialization -----------------------------------------

    #[test]
    fn think_level_serializes_to_wire_forms() {
        assert_eq!(serde_json::to_value(ThinkLevel::Off).unwrap(), json!(false));
        assert_eq!(serde_json::to_value(ThinkLevel::On).unwrap(), json!(true));
        assert_eq!(serde_json::to_value(ThinkLevel::Low).unwrap(), json!("low"));
        assert_eq!(
            serde_json::to_value(ThinkLevel::Medium).unwrap(),
            json!("medium")
        );
        assert_eq!(
            serde_json::to_value(ThinkLevel::High).unwrap(),
            json!("high")
        );
        assert_eq!(serde_json::to_value(ThinkLevel::Max).unwrap(), json!("max"));
    }

    // ---- (g) context guard -------------------------------------------------

    #[tokio::test]
    async fn context_guard_trips_before_sending() {
        let server = MockServer::start().await;
        // No mount: assert zero requests reach the server.

        let big = "x".repeat(4000); // ~1000 est tokens, well over num_ctx=100.
        let messages = vec![Message::User {
            content: vec![UserBlock::Text(big)],
        }];
        let tools: Vec<Value> = vec![];
        let p = params();
        let req = simple_req(&messages, &tools, &p);

        let backend = OllamaBackend::new("qwen3.6", server.uri()).with_num_ctx(100);
        let err = backend.turn(&req).await.expect_err("guard must trip");
        assert!(matches!(err, BackendError::ContextLengthExceeded));
        assert!(!err.is_retryable());

        let received = server.received_requests().await.unwrap();
        assert_eq!(received.len(), 0, "guard must fire before any HTTP request");
    }

    #[tokio::test]
    async fn no_guard_when_num_ctx_unset() {
        let server = MockServer::start().await;
        mount_success(
            &server,
            &json!({
                "message": {"role": "assistant", "content": "ok"},
                "done_reason": "stop", "prompt_eval_count": 1, "eval_count": 1
            })
            .to_string(),
        )
        .await;

        // Same oversized request, but no num_ctx => no guard => it sends.
        let big = "x".repeat(4000);
        let messages = vec![Message::User {
            content: vec![UserBlock::Text(big)],
        }];
        let tools: Vec<Value> = vec![];
        let p = params();
        let req = simple_req(&messages, &tools, &p);

        let backend = OllamaBackend::new("qwen3.6", server.uri());
        let turn = backend.turn(&req).await.expect("turn ok");
        assert_eq!(turn.text(), "ok");
        let received = server.received_requests().await.unwrap();
        assert_eq!(received.len(), 1);
    }

    // ---- trailing-slash base URL is tolerated -----------------------------

    #[tokio::test]
    async fn base_url_with_trailing_slash_is_normalized() {
        let server = MockServer::start().await;
        mount_success(
            &server,
            &json!({
                "message": {"role": "assistant", "content": "ok"},
                "done_reason": "stop", "prompt_eval_count": 1, "eval_count": 1
            })
            .to_string(),
        )
        .await;
        let base = format!("{}/", server.uri());
        let backend = OllamaBackend::new("qwen3.6", base);
        let messages = user_hi();
        let tools: Vec<Value> = vec![];
        let p = params();
        let req = simple_req(&messages, &tools, &p);
        let turn = backend.turn(&req).await.expect("turn ok");
        assert_eq!(turn.text(), "ok");
    }
}
