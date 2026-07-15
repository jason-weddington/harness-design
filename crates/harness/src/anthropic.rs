//! Anthropic implementation of [`crate::model::ModelBackend`] — the first
//! backend behind the anti-corruption boundary.
//!
//! ## Scope
//!
//! Non-streaming only (this slice). A single [`ModelBackend::turn`] call POSTs
//! `{base_url}/v1/messages` with `stream: false`, awaits the full JSON
//! response, and translates the provider-native shape into the normalized
//! types in [`crate::model`]. SSE/streaming, retry/backoff, persistence, and
//! non-Anthropic providers are out of scope here.
//!
//! ## Design pins (don't re-derive — implement as written)
//!
//! - **No community Anthropic SDK.** `reqwest` + `serde` only. The whole
//!   point of [`crate::model::ModelBackend`] is the anti-corruption boundary;
//!   layering a third-party SDK between the wire and the boundary just adds
//!   another shape to translate (and a transitive supply-chain surface to
//!   audit) for no semantic gain.
//! - **rustls (ring), not OpenSSL.** `reqwest` is pulled with
//!   `default-features = false` and `features = ["json", "rustls-tls"]` so
//!   the build never depends on a system OpenSSL. See `deny.toml` for the
//!   matching license-allow comment.
//! - **Deterministic field order.** [`RequestBody`] declares its fields in a
//!   fixed order (`model`, `max_tokens`, `temperature`, `stop_sequences`,
//!   `system`, `tools`, `messages`) and serde emits them in that order — the
//!   discipline that keeps the prompt cache byte-stable across turns.
//! - **No api-key leakage in `Debug`.** [`AnthropicBackend`] deliberately
//!   does **not** derive `Debug`; the api key only travels into the
//!   `x-api-key` header.
//! - **The backend only classifies; the loop reacts.** Errors map to
//!   [`BackendError`] variants per the trait's contract; nothing here retries
//!   or backs off.
//!
//! ## Testing
//!
//! Tests use `wiremock` — a local HTTP mock server — so the suite never
//! touches a live key or external network. The mock server is bound to
//! [`AnthropicBackend`] via [`AnthropicBackend::with_base_url`], the same
//! override real production code could use to point at a corporate proxy.

use std::time::Duration;

use async_trait::async_trait;
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::model::{
    AssistantTurn, BackendError, ContentBlock, Message, ModelBackend, StopReason, TerminalKind,
    ToolCallRequest, TransientKind, TurnRequest, Usage, UserBlock,
};

/// Anthropic's public API origin — the default backend target.
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";

/// Anthropic's `anthropic-version` header value. Pinned: the messages API is
/// a versioned surface, and silent upgrades break wire-shape assumptions.
const ANTHROPIC_VERSION: &str = "2023-06-01";

// ============================================================================
// Public adapter
// ============================================================================

/// Anthropic-backed [`ModelBackend`].
///
/// Construct with [`Self::new`] and (optionally) [`Self::with_base_url`] —
/// the base-URL override is what tests use to point at a `wiremock` server
/// instead of `api.anthropic.com`.
///
/// Does not derive [`Debug`] on purpose: the api key must not show up in a
/// formatter chain (panic messages, `dbg!`, structured logs).
pub struct AnthropicBackend {
    client: Client,
    model: String,
    api_key: String,
    base_url: String,
}

impl AnthropicBackend {
    /// Build a backend pinned to `model`, authenticated with `api_key`, and
    /// pointed at the default Anthropic origin.
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            model: model.into(),
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
        }
    }

    /// Override the base URL — points the backend at a different origin
    /// (a `wiremock` server in tests, or a corporate proxy in production).
    /// A trailing slash on `url` is tolerated; it is stripped at request
    /// time so `{base}/v1/messages` is always a single-slash URL.
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
}

#[async_trait]
impl ModelBackend for AnthropicBackend {
    async fn turn(&self, req: &TurnRequest<'_>) -> Result<AssistantTurn, BackendError> {
        let body = build_request_body(&self.model, req);
        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| map_reqwest_error(&e))?;

        let status = response.status();
        let retry_after = parse_retry_after(response.headers().get("retry-after"));
        let body_text = response.text().await.map_err(|e| map_reqwest_error(&e))?;

        if !status.is_success() {
            return Err(map_error_status(status, &body_text, retry_after));
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
// Request side — wire shapes that mirror Anthropic's `/v1/messages` body
// ============================================================================

/// Outgoing request body for `POST /v1/messages`.
///
/// **Field order is load-bearing.** serde emits struct fields in declaration
/// order, and the prompt cache hashes the serialized bytes — reordering any
/// of these is a silent cache miss. The order encoded here matches the order
/// in the spec.
#[derive(Serialize)]
struct RequestBody<'a> {
    model: &'a str,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<&'a [String]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<Vec<SystemBlock<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [Value]>,
    messages: Vec<WireMessage<'a>>,
}

/// Anthropic prompt-cache breakpoint marker. Serializes to exactly
/// `{"type":"ephemeral"}` — the only cache ttl Anthropic exposes today. Two
/// `Some(ephemeral)` instances are constructed per request (one static on the
/// system block, one rolling on the last content block of the last message),
/// which is why this derives `Copy`.
#[derive(Clone, Copy, Serialize)]
struct CacheControl {
    #[serde(rename = "type")]
    kind: &'static str,
}

impl CacheControl {
    /// The single cache breakpoint marker we send — a 5-minute ephemeral
    /// cache. Constructed once per breakpoint site, not cached globally,
    /// because it is cheap and the call sites are few.
    const fn ephemeral() -> Self {
        Self { kind: "ephemeral" }
    }
}

/// One block of the `system` content-block array. Anthropic accepts `system`
/// either as a bare string or as an array of typed content blocks; we use the
/// array form so the single text block can carry a `cache_control` breakpoint.
///
/// **Field order is load-bearing within the block.** `cache_control` is
/// declared LAST (after `text`) so a breakpoint-carrying block has a fixed
/// byte layout regardless of the skip-serializing-if-none discipline.
#[derive(Serialize)]
struct SystemBlock<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    text: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
}

#[derive(Serialize)]
struct WireMessage<'a> {
    role: &'static str,
    content: Vec<WireContentBlock<'a>>,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WireContentBlock<'a> {
    Text {
        text: &'a str,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    ToolUse {
        id: &'a str,
        name: &'a str,
        input: &'a Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    ToolResult {
        tool_use_id: &'a str,
        content: &'a str,
        is_error: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    Thinking {
        thinking: &'a str,
        signature: &'a str,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

impl WireContentBlock<'_> {
    /// Attach a prompt-cache breakpoint to this block. Called on the LAST
    /// content block of the LAST message to roll the cache forward as the
    /// conversation grows.
    fn set_cache_breakpoint(&mut self) {
        let cc = Some(CacheControl::ephemeral());
        match self {
            WireContentBlock::Text { cache_control, .. }
            | WireContentBlock::ToolUse { cache_control, .. }
            | WireContentBlock::ToolResult { cache_control, .. }
            | WireContentBlock::Thinking { cache_control, .. } => *cache_control = cc,
        }
    }
}

fn build_request_body<'a>(model: &'a str, req: &'a TurnRequest<'a>) -> RequestBody<'a> {
    // STATIC breakpoint: a single system text block carrying cache_control.
    // Anthropic's canonical cache order is tools -> system -> messages, so a
    // breakpoint on the system block also covers the tools that precede it —
    // no separate tool-level breakpoint is added.
    let system = req.system.map(|s| {
        vec![SystemBlock {
            kind: "text",
            text: s,
            cache_control: Some(CacheControl::ephemeral()),
        }]
    });

    let mut messages: Vec<WireMessage<'a>> = req.messages.iter().map(map_message).collect();

    // ROLLING breakpoint: cache the growing conversation prefix by marking
    // the LAST content block of the LAST message. Guarded so an empty
    // messages slice or an empty content vec is a no-op (no panic, no
    // breakpoint attached) — Anthropic rejects empty content anyway.
    if let Some(last_msg) = messages.last_mut()
        && let Some(last_block) = last_msg.content.last_mut()
    {
        last_block.set_cache_breakpoint();
    }

    RequestBody {
        model,
        max_tokens: req.params.max_tokens,
        temperature: req.params.temperature,
        stop_sequences: if req.params.stop_sequences.is_empty() {
            None
        } else {
            Some(req.params.stop_sequences.as_slice())
        },
        system,
        tools: if req.tools.is_empty() {
            None
        } else {
            Some(req.tools)
        },
        messages,
    }
}

fn map_message(m: &Message) -> WireMessage<'_> {
    match m {
        Message::User { content } => WireMessage {
            role: "user",
            content: content
                .iter()
                .map(|b| match b {
                    UserBlock::Text(t) => WireContentBlock::Text {
                        text: t,
                        cache_control: None,
                    },
                    UserBlock::ToolResult {
                        call_id,
                        content,
                        is_error,
                    } => WireContentBlock::ToolResult {
                        tool_use_id: call_id,
                        content,
                        is_error: *is_error,
                        cache_control: None,
                    },
                })
                .collect(),
        },
        Message::Assistant { content } => WireMessage {
            role: "assistant",
            content: content
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text(t) => Some(WireContentBlock::Text {
                        text: t,
                        cache_control: None,
                    }),
                    ContentBlock::ToolCall(c) => Some(WireContentBlock::ToolUse {
                        id: &c.id,
                        name: &c.name,
                        input: &c.input,
                        cache_control: None,
                    }),
                    // Reasoning blocks are only sent back when the provider
                    // gave us an opaque signature to echo (cache continuity).
                    // Without a signature, Anthropic rejects a thinking block
                    // — so drop it rather than try to forge one.
                    ContentBlock::Reasoning {
                        text,
                        opaque: Some(sig),
                    } => Some(WireContentBlock::Thinking {
                        thinking: text,
                        signature: sig,
                        cache_control: None,
                    }),
                    ContentBlock::Reasoning { opaque: None, .. } => None,
                })
                .collect(),
        },
    }
}

// ============================================================================
// Response side — wire shapes that Anthropic's `/v1/messages` returns
// ============================================================================

#[derive(Deserialize)]
struct ResponseBody {
    content: Vec<ResponseBlock>,
    stop_reason: Option<String>,
    usage: ResponseUsage,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ResponseBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    Thinking {
        thinking: String,
        #[serde(default)]
        signature: Option<String>,
    },
}

// Field names mirror Anthropic's wire shape verbatim — we don't get to
// rename them away from the shared `_tokens` suffix.
#[allow(clippy::struct_field_names)]
#[derive(Deserialize)]
struct ResponseUsage {
    input_tokens: u32,
    output_tokens: u32,
    #[serde(default)]
    cache_read_input_tokens: Option<u32>,
    #[serde(default)]
    cache_creation_input_tokens: Option<u32>,
}

fn map_response(body: ResponseBody) -> AssistantTurn {
    AssistantTurn {
        content: body
            .content
            .into_iter()
            .map(|b| match b {
                ResponseBlock::Text { text } => ContentBlock::Text(text),
                ResponseBlock::ToolUse { id, name, input } => {
                    ContentBlock::ToolCall(ToolCallRequest { id, name, input })
                }
                ResponseBlock::Thinking {
                    thinking,
                    signature,
                } => ContentBlock::Reasoning {
                    text: thinking,
                    opaque: signature,
                },
            })
            .collect(),
        stop_reason: map_stop_reason(body.stop_reason.as_deref()),
        usage: Usage {
            input_tokens: body.usage.input_tokens,
            output_tokens: body.usage.output_tokens,
            cache_read_tokens: body.usage.cache_read_input_tokens,
            cache_write_tokens: body.usage.cache_creation_input_tokens,
            // Anthropic does not currently break out reasoning tokens
            // separately in `usage` — leave it `None` (absent ≠ zero).
            reasoning_tokens: None,
        },
    }
}

fn map_stop_reason(s: Option<&str>) -> StopReason {
    match s {
        Some("end_turn") => StopReason::EndTurn,
        Some("tool_use") => StopReason::ToolUse,
        Some("max_tokens") => StopReason::MaxTokens,
        Some("stop_sequence") => StopReason::StopSequence,
        Some(other) => StopReason::Other(other.to_string()),
        // Absent stop_reason is undocumented but conceivable in error/edge
        // cases; classify rather than panic.
        None => StopReason::Other(String::new()),
    }
}

// ============================================================================
// Error mapping
// ============================================================================

#[derive(Deserialize)]
struct ErrorBody {
    error: ErrorInner,
}

#[derive(Deserialize)]
struct ErrorInner {
    #[serde(rename = "type", default)]
    error_type: String,
    #[serde(default)]
    message: String,
}

/// Reqwest's pre-response errors (connect refused, DNS, timeout, …) → the
/// transient bucket the loop knows how to retry.
fn map_reqwest_error(e: &reqwest::Error) -> BackendError {
    let kind = if e.is_timeout() {
        TransientKind::Timeout
    } else {
        // is_connect / is_request / is_body / is_decode all collapse to a
        // network-class transient: from the loop's perspective the request
        // never produced a usable response, and retrying is the right shape.
        TransientKind::Network
    };
    BackendError::Transient {
        kind,
        retry_after: None,
    }
}

/// Translate a non-2xx response into the appropriate [`BackendError`].
fn map_error_status(
    status: StatusCode,
    body_text: &str,
    retry_after: Option<Duration>,
) -> BackendError {
    let parsed: Option<ErrorBody> = serde_json::from_str(body_text).ok();
    let message = parsed
        .as_ref()
        .map(|e| e.error.message.clone())
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| body_text.to_string());
    let error_type = parsed
        .as_ref()
        .map(|e| e.error.error_type.as_str())
        .unwrap_or_default();

    // Model-not-found is type-tagged regardless of whether the provider
    // returned 400 or 404 in a given moment, so check the tag first.
    if error_type == "not_found_error" {
        return BackendError::Terminal {
            kind: TerminalKind::UnknownModel,
            message,
        };
    }

    match status.as_u16() {
        429 => BackendError::Transient {
            kind: TransientKind::RateLimit,
            retry_after,
        },
        529 => BackendError::Transient {
            kind: TransientKind::Overloaded,
            retry_after,
        },
        401 | 403 => BackendError::Terminal {
            kind: TerminalKind::Auth,
            message,
        },
        400 => classify_bad_request(&message),
        404 => BackendError::Terminal {
            kind: TerminalKind::UnknownModel,
            message,
        },
        s if (500..600).contains(&s) => BackendError::Transient {
            kind: TransientKind::ServerError,
            retry_after,
        },
        _ => BackendError::Terminal {
            kind: TerminalKind::Other,
            message,
        },
    }
}

/// 400-class disambiguation: context-length overflow vs. a generic bad
/// request. Anthropic's "prompt is too long" / "exceeds the context window"
/// signals come through as 400s with a textual message; the loop needs the
/// distinct [`BackendError::ContextLengthExceeded`] variant so it knows to
/// prune/compact before retrying (the backend never decides that itself).
fn classify_bad_request(message: &str) -> BackendError {
    let lower = message.to_lowercase();
    let is_context_overflow = lower.contains("prompt is too long")
        || lower.contains("context length")
        || lower.contains("context window")
        || lower.contains("exceeds the context")
        || lower.contains("maximum context")
        || (lower.contains("token") && (lower.contains("exceed") || lower.contains("too many")));

    if is_context_overflow {
        BackendError::ContextLengthExceeded
    } else {
        BackendError::Terminal {
            kind: TerminalKind::BadRequest,
            message: message.to_string(),
        }
    }
}

/// Parse a `Retry-After` header value as an integer-seconds duration.
/// HTTP-date forms are not supported — Anthropic emits seconds in practice;
/// anything else maps to `None` and the loop falls back to its own backoff.
fn parse_retry_after(header: Option<&reqwest::header::HeaderValue>) -> Option<Duration> {
    header
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(Duration::from_secs)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::{AnthropicBackend, classify_bad_request, map_stop_reason, parse_retry_after};
    use crate::model::{
        BackendError, ContentBlock, Message, ModelBackend, SamplingParams, StopReason,
        TerminalKind, ToolCallRequest, TransientKind, TurnRequest, UserBlock,
    };
    use reqwest::header::HeaderValue;
    use serde_json::{Value, json};
    use std::time::Duration;
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
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_string(body))
            .mount(server)
            .await;
    }

    // ---- (a) happy path: text + tool_use ----------------------------------

    #[tokio::test]
    async fn maps_text_and_tool_use_response_to_assistant_turn() {
        let server = MockServer::start().await;
        let body = json!({
            "content": [
                {"type": "text", "text": "thinking out loud"},
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "read_file",
                    "input": {"path": "src/lib.rs"}
                }
            ],
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 42,
                "output_tokens": 7,
                "cache_read_input_tokens": 16,
                "cache_creation_input_tokens": 4
            }
        })
        .to_string();
        mount_success(&server, &body).await;

        let backend =
            AnthropicBackend::new("claude-sonnet-5", "sk-test").with_base_url(server.uri());
        let messages = user_hi();
        let tools: Vec<Value> = vec![];
        let p = params();
        let req = TurnRequest {
            system: Some("be helpful"),
            messages: &messages,
            tools: &tools,
            params: &p,
        };

        let turn = backend.turn(&req).await.expect("turn ok");
        assert_eq!(turn.text(), "thinking out loud");
        assert!(matches!(turn.stop_reason, StopReason::ToolUse));
        let calls = turn.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "toolu_01");
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].input, json!({"path": "src/lib.rs"}));
        assert_eq!(turn.usage.input_tokens, 42);
        assert_eq!(turn.usage.output_tokens, 7);
        assert_eq!(turn.usage.cache_read_tokens, Some(16));
        assert_eq!(turn.usage.cache_write_tokens, Some(4));
        assert_eq!(turn.usage.reasoning_tokens, None);
    }

    // ---- thinking block round-trip on the response side -------------------

    #[tokio::test]
    async fn maps_thinking_block_to_reasoning_with_signature() {
        let server = MockServer::start().await;
        let body = json!({
            "content": [
                {"type": "thinking", "thinking": "let me think", "signature": "sig-xyz"}
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        })
        .to_string();
        mount_success(&server, &body).await;

        let backend = AnthropicBackend::new("m", "k").with_base_url(server.uri());
        let messages = user_hi();
        let tools: Vec<Value> = vec![];
        let p = params();
        let req = TurnRequest {
            system: None,
            messages: &messages,
            tools: &tools,
            params: &p,
        };

        let turn = backend.turn(&req).await.expect("turn ok");
        assert_eq!(turn.content.len(), 1);
        match &turn.content[0] {
            ContentBlock::Reasoning { text, opaque } => {
                assert_eq!(text, "let me think");
                assert_eq!(opaque.as_deref(), Some("sig-xyz"));
            }
            _ => panic!("expected Reasoning block"),
        }
        assert!(matches!(turn.stop_reason, StopReason::EndTurn));
    }

    // ---- (b) 429 → Transient{RateLimit} with parsed retry-after -----------

    #[tokio::test]
    async fn maps_429_to_transient_rate_limit_with_retry_after() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(429)
                    .insert_header("retry-after", "5")
                    .set_body_string(
                        r#"{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}"#,
                    ),
            )
            .mount(&server)
            .await;

        let backend = AnthropicBackend::new("m", "k").with_base_url(server.uri());
        let messages = user_hi();
        let tools: Vec<Value> = vec![];
        let p = params();
        let req = TurnRequest {
            system: None,
            messages: &messages,
            tools: &tools,
            params: &p,
        };

        let err = backend.turn(&req).await.expect_err("must fail");
        assert!(err.is_retryable(), "rate-limit must be retryable");
        match err {
            BackendError::Transient { kind, retry_after } => {
                assert_eq!(kind, TransientKind::RateLimit);
                assert_eq!(retry_after, Some(Duration::from_secs(5)));
            }
            other => panic!("expected Transient, got {other:?}"),
        }
    }

    // ---- (c) 401 → Terminal{Auth} -----------------------------------------

    #[tokio::test]
    async fn maps_401_to_terminal_auth() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(401).set_body_string(
                r#"{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}"#,
            ))
            .mount(&server)
            .await;

        let backend = AnthropicBackend::new("m", "k").with_base_url(server.uri());
        let messages = user_hi();
        let tools: Vec<Value> = vec![];
        let p = params();
        let req = TurnRequest {
            system: None,
            messages: &messages,
            tools: &tools,
            params: &p,
        };

        let err = backend.turn(&req).await.expect_err("must fail");
        assert!(!err.is_retryable());
        match err {
            BackendError::Terminal { kind, message } => {
                assert_eq!(kind, TerminalKind::Auth);
                assert!(message.contains("invalid x-api-key"));
            }
            other => panic!("expected Terminal{{Auth}}, got {other:?}"),
        }
    }

    // ---- (d) malformed body → Protocol ------------------------------------

    #[tokio::test]
    async fn maps_malformed_success_body_to_protocol_error() {
        let server = MockServer::start().await;
        // 200 with a body that doesn't match ResponseBody.
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not even json"))
            .mount(&server)
            .await;

        let backend = AnthropicBackend::new("m", "k").with_base_url(server.uri());
        let messages = user_hi();
        let tools: Vec<Value> = vec![];
        let p = params();
        let req = TurnRequest {
            system: None,
            messages: &messages,
            tools: &tools,
            params: &p,
        };

        let err = backend.turn(&req).await.expect_err("must fail");
        match err {
            BackendError::Protocol { message, raw } => {
                assert!(message.contains("not parseable"));
                assert_eq!(raw.as_deref(), Some("not even json"));
            }
            other => panic!("expected Protocol, got {other:?}"),
        }
    }

    // ---- (e) request-shape: capture outgoing body and assert fields -------

    // The request-shape test is intentionally exhaustive: it captures one
    // outgoing body and asserts headers + every top-level field's order +
    // every block-mapping rule in one place. Splitting it would dilute that
    // single source-of-truth, so accept the line count.
    #[allow(clippy::too_many_lines)]
    #[tokio::test]
    async fn outgoing_request_body_has_expected_shape_and_field_order() {
        let server = MockServer::start().await;
        // A minimal success body so we get past the response parse and can
        // inspect the captured request.
        mount_success(
            &server,
            &json!({
                "content": [],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 0, "output_tokens": 0}
            })
            .to_string(),
        )
        .await;

        let messages = vec![
            Message::User {
                content: vec![
                    UserBlock::Text("hello".to_string()),
                    UserBlock::ToolResult {
                        call_id: "toolu_prev".to_string(),
                        content: "ok".to_string(),
                        is_error: false,
                    },
                ],
            },
            Message::Assistant {
                content: vec![
                    ContentBlock::Text("ack".to_string()),
                    ContentBlock::ToolCall(ToolCallRequest {
                        id: "toolu_curr".to_string(),
                        name: "echo".to_string(),
                        input: json!({"x": 1}),
                    }),
                    // No opaque signature → must NOT appear on the wire.
                    ContentBlock::Reasoning {
                        text: "internal".to_string(),
                        opaque: None,
                    },
                    // With a signature → must appear as a thinking block.
                    ContentBlock::Reasoning {
                        text: "trace".to_string(),
                        opaque: Some("sig-1".to_string()),
                    },
                ],
            },
        ];
        let tools: Vec<Value> = vec![json!({
            "name": "echo",
            "description": "echoes input back",
            "input_schema": {"type": "object"}
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

        let backend =
            AnthropicBackend::new("claude-sonnet-5", "sk-test").with_base_url(server.uri());
        backend.turn(&req).await.expect("turn ok");

        let received = server.received_requests().await.expect("requests captured");
        assert_eq!(received.len(), 1);
        let r: &Request = &received[0];

        // Headers were set.
        assert_eq!(
            r.headers
                .get("x-api-key")
                .map(HeaderValue::to_str)
                .and_then(Result::ok),
            Some("sk-test")
        );
        assert_eq!(
            r.headers
                .get("anthropic-version")
                .map(HeaderValue::to_str)
                .and_then(Result::ok),
            Some("2023-06-01")
        );

        let body_text = std::str::from_utf8(&r.body).expect("utf-8 body");

        // Deterministic top-level field order — the prompt-cache pin.
        let model_idx = body_text.find("\"model\"").expect("model present");
        let max_tokens_idx = body_text
            .find("\"max_tokens\"")
            .expect("max_tokens present");
        let temperature_idx = body_text
            .find("\"temperature\"")
            .expect("temperature present");
        let stop_seqs_idx = body_text
            .find("\"stop_sequences\"")
            .expect("stop_sequences present");
        let system_idx = body_text.find("\"system\"").expect("system present");
        let tools_idx = body_text.find("\"tools\"").expect("tools present");
        let messages_idx = body_text.find("\"messages\"").expect("messages present");
        assert!(model_idx < max_tokens_idx);
        assert!(max_tokens_idx < temperature_idx);
        assert!(temperature_idx < stop_seqs_idx);
        assert!(stop_seqs_idx < system_idx);
        assert!(system_idx < tools_idx);
        assert!(tools_idx < messages_idx);

        // Parsed-value spot checks.
        let parsed: Value = serde_json::from_str(body_text).expect("json");
        assert_eq!(parsed["model"], "claude-sonnet-5");
        assert_eq!(parsed["max_tokens"], 512);
        assert_eq!(parsed["temperature"], 0.3);
        assert_eq!(parsed["stop_sequences"], json!(["END"]));
        // system is now a one-element content-block array carrying the STATIC
        // cache breakpoint.
        assert_eq!(parsed["system"][0]["type"], "text");
        assert_eq!(parsed["system"][0]["text"], "you are a harness");
        assert_eq!(parsed["system"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(parsed["tools"][0]["name"], "echo");

        // Messages mapping.
        let msgs = parsed["messages"].as_array().expect("messages array");
        assert_eq!(msgs.len(), 2);

        // User message: text + tool_result, no role drift.
        assert_eq!(msgs[0]["role"], "user");
        let u_content = msgs[0]["content"].as_array().expect("user content");
        assert_eq!(u_content.len(), 2);
        assert_eq!(u_content[0]["type"], "text");
        assert_eq!(u_content[0]["text"], "hello");
        assert_eq!(u_content[1]["type"], "tool_result");
        assert_eq!(u_content[1]["tool_use_id"], "toolu_prev");
        assert_eq!(u_content[1]["content"], "ok");
        assert_eq!(u_content[1]["is_error"], false);

        // Assistant message: text + tool_use + thinking; the unsigned
        // Reasoning block was dropped.
        assert_eq!(msgs[1]["role"], "assistant");
        let a_content = msgs[1]["content"].as_array().expect("assistant content");
        assert_eq!(a_content.len(), 3, "unsigned Reasoning must be dropped");
        assert_eq!(a_content[0]["type"], "text");
        assert_eq!(a_content[0]["text"], "ack");
        assert_eq!(a_content[1]["type"], "tool_use");
        assert_eq!(a_content[1]["id"], "toolu_curr");
        assert_eq!(a_content[1]["name"], "echo");
        assert_eq!(a_content[1]["input"], json!({"x": 1}));
        assert_eq!(a_content[2]["type"], "thinking");
        assert_eq!(a_content[2]["thinking"], "trace");
        assert_eq!(a_content[2]["signature"], "sig-1");

        // ROLLING breakpoint: only the LAST content block of the LAST message
        // carries cache_control. In this test the last message is the
        // assistant message, so the rolling breakpoint lands on a_content[2]
        // (the surviving thinking block) — that is deliberate for this
        // serialization unit test and MUST NOT be "fixed"; production's last
        // pre-model message is always a user text/tool_result block.
        assert_eq!(a_content[2]["cache_control"]["type"], "ephemeral");
        // No other block carries a breakpoint.
        assert_eq!(u_content[0]["cache_control"], Value::Null);
        assert_eq!(u_content[1]["cache_control"], Value::Null);
        assert_eq!(a_content[0]["cache_control"], Value::Null);
        assert_eq!(a_content[1]["cache_control"], Value::Null);
        // Budget: static system + rolling last-block = exactly 2.
        let cc_count = body_text.matches("\"cache_control\"").count();
        assert_eq!(cc_count, 2, "expected exactly 2 cache_control breakpoints");
    }

    // ---- request-shape: optional fields are omitted when absent -----------

    #[tokio::test]
    async fn outgoing_request_omits_optional_fields_when_unset() {
        let server = MockServer::start().await;
        mount_success(
            &server,
            &json!({
                "content": [],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 0, "output_tokens": 0}
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
        let req = TurnRequest {
            system: None,
            messages: &messages,
            tools: &tools,
            params: &p,
        };

        let backend = AnthropicBackend::new("m", "k").with_base_url(server.uri());
        backend.turn(&req).await.expect("turn ok");

        let received = server.received_requests().await.unwrap();
        let body_text = std::str::from_utf8(&received[0].body).unwrap();
        assert!(!body_text.contains("\"temperature\""));
        assert!(!body_text.contains("\"stop_sequences\""));
        assert!(!body_text.contains("\"system\""));
        assert!(!body_text.contains("\"tools\""));
        // Required fields still present.
        assert!(body_text.contains("\"model\""));
        assert!(body_text.contains("\"max_tokens\""));
        assert!(body_text.contains("\"messages\""));
    }

    // ---- rolling-breakpoint guards: empty content and empty messages -------

    // A last message whose blocks all filter out (here: a single unsigned
    // Reasoning block) maps to an EMPTY content vec. The rolling-breakpoint
    // guard must skip attaching and not panic, and the body must contain ZERO
    // cache_control occurrences (no system, no rolling).
    #[tokio::test]
    async fn rolling_breakpoint_skipped_when_last_message_content_is_empty() {
        let server = MockServer::start().await;
        mount_success(
            &server,
            &json!({
                "content": [],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 0, "output_tokens": 0}
            })
            .to_string(),
        )
        .await;

        let messages = vec![Message::Assistant {
            content: vec![ContentBlock::Reasoning {
                text: "x".to_string(),
                opaque: None,
            }],
        }];
        let tools: Vec<Value> = vec![];
        let p = params();
        let req = TurnRequest {
            system: None,
            messages: &messages,
            tools: &tools,
            params: &p,
        };

        let backend = AnthropicBackend::new("m", "k").with_base_url(server.uri());
        backend.turn(&req).await.expect("turn ok");

        let received = server.received_requests().await.unwrap();
        let body_text = std::str::from_utf8(&received[0].body).unwrap();
        assert!(
            !body_text.contains("\"cache_control\""),
            "no breakpoint expected when last content is empty, got: {body_text}"
        );
    }

    // An empty messages slice must not panic and must not attach a rolling
    // breakpoint. (Anthropic rejects this request server-side; we only assert
    // our own serialization does not crash or invent a breakpoint.)
    #[tokio::test]
    async fn rolling_breakpoint_skipped_when_messages_slice_is_empty() {
        let server = MockServer::start().await;
        mount_success(
            &server,
            &json!({
                "content": [],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 0, "output_tokens": 0}
            })
            .to_string(),
        )
        .await;

        let messages: Vec<Message> = vec![];
        let tools: Vec<Value> = vec![];
        let p = params();
        let req = TurnRequest {
            system: None,
            messages: &messages,
            tools: &tools,
            params: &p,
        };

        let backend = AnthropicBackend::new("m", "k").with_base_url(server.uri());
        backend.turn(&req).await.expect("turn ok");

        let received = server.received_requests().await.unwrap();
        let body_text = std::str::from_utf8(&received[0].body).unwrap();
        assert!(
            !body_text.contains("\"cache_control\""),
            "no breakpoint expected with empty messages, got: {body_text}"
        );
    }

    // ---- two-breakpoint production-realistic last message ------------------

    // Production's last pre-model message is always a user text/tool_result
    // block (the loop alternates user -> assistant -> user). This test pins
    // the production shape: a last-message user text block carries the rolling
    // breakpoint, and the whole body contains EXACTLY two cache_control
    // occurrences (static system + rolling last-block).
    #[tokio::test]
    async fn two_breakpoints_attach_for_production_realistic_last_message() {
        let server = MockServer::start().await;
        mount_success(
            &server,
            &json!({
                "content": [],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 0, "output_tokens": 0}
            })
            .to_string(),
        )
        .await;

        let messages = vec![
            Message::User {
                content: vec![UserBlock::Text("first turn".to_string())],
            },
            Message::Assistant {
                content: vec![ContentBlock::Text("reply".to_string())],
            },
            Message::User {
                content: vec![UserBlock::Text("second turn".to_string())],
            },
        ];
        let tools: Vec<Value> = vec![json!({
            "name": "echo",
            "description": "echoes input back",
            "input_schema": {"type": "object"}
        })];
        let p = params();
        let req = TurnRequest {
            system: Some("you are a harness"),
            messages: &messages,
            tools: &tools,
            params: &p,
        };

        let backend = AnthropicBackend::new("m", "k").with_base_url(server.uri());
        backend.turn(&req).await.expect("turn ok");

        let received = server.received_requests().await.unwrap();
        let body_text = std::str::from_utf8(&received[0].body).unwrap();
        let parsed: Value = serde_json::from_str(body_text).expect("json");

        // STATIC breakpoint on the system block.
        assert_eq!(parsed["system"][0]["cache_control"]["type"], "ephemeral");

        // ROLLING breakpoint on the last message's last (only) content block.
        let msgs = parsed["messages"].as_array().expect("messages array");
        let last = msgs.last().expect("last message present");
        let last_content = last["content"].as_array().expect("last content array");
        let last_block = last_content.last().expect("last block present");
        assert_eq!(last_block["cache_control"]["type"], "ephemeral");

        // Budget: exactly two breakpoints total.
        let cc_count = body_text.matches("\"cache_control\"").count();
        assert_eq!(cc_count, 2, "expected exactly 2 cache_control breakpoints");
    }

    // ---- status-code arm coverage ----------------------------------------

    #[tokio::test]
    async fn maps_529_to_transient_overloaded() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(529).set_body_string(
                r#"{"type":"error","error":{"type":"overloaded_error","message":"overloaded"}}"#,
            ))
            .mount(&server)
            .await;

        let backend = AnthropicBackend::new("m", "k").with_base_url(server.uri());
        let messages = user_hi();
        let tools: Vec<Value> = vec![];
        let p = params();
        let req = TurnRequest {
            system: None,
            messages: &messages,
            tools: &tools,
            params: &p,
        };
        let err = backend.turn(&req).await.expect_err("must fail");
        assert!(matches!(
            err,
            BackendError::Transient {
                kind: TransientKind::Overloaded,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn maps_503_to_transient_server_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(503).set_body_string(
                r#"{"type":"error","error":{"type":"api_error","message":"upstream down"}}"#,
            ))
            .mount(&server)
            .await;

        let backend = AnthropicBackend::new("m", "k").with_base_url(server.uri());
        let messages = user_hi();
        let tools: Vec<Value> = vec![];
        let p = params();
        let req = TurnRequest {
            system: None,
            messages: &messages,
            tools: &tools,
            params: &p,
        };
        let err = backend.turn(&req).await.expect_err("must fail");
        assert!(matches!(
            err,
            BackendError::Transient {
                kind: TransientKind::ServerError,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn maps_400_with_context_message_to_context_length_exceeded() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(400).set_body_string(
                    r#"{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 250000 tokens > 200000 maximum"}}"#,
                ),
            )
            .mount(&server)
            .await;

        let backend = AnthropicBackend::new("m", "k").with_base_url(server.uri());
        let messages = user_hi();
        let tools: Vec<Value> = vec![];
        let p = params();
        let req = TurnRequest {
            system: None,
            messages: &messages,
            tools: &tools,
            params: &p,
        };
        let err = backend.turn(&req).await.expect_err("must fail");
        assert!(matches!(err, BackendError::ContextLengthExceeded));
        assert!(
            !err.is_retryable(),
            "context overflow is not retryable until pruned"
        );
    }

    #[tokio::test]
    async fn maps_400_generic_to_terminal_bad_request() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(400).set_body_string(
                r#"{"type":"error","error":{"type":"invalid_request_error","message":"messages: at least one message is required"}}"#,
            ))
            .mount(&server)
            .await;

        let backend = AnthropicBackend::new("m", "k").with_base_url(server.uri());
        let messages = user_hi();
        let tools: Vec<Value> = vec![];
        let p = params();
        let req = TurnRequest {
            system: None,
            messages: &messages,
            tools: &tools,
            params: &p,
        };
        let err = backend.turn(&req).await.expect_err("must fail");
        match err {
            BackendError::Terminal { kind, message } => {
                assert_eq!(kind, TerminalKind::BadRequest);
                assert!(message.contains("at least one message"));
            }
            other => panic!("expected Terminal{{BadRequest}}, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn maps_404_to_terminal_unknown_model() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(404).set_body_string(
                r#"{"type":"error","error":{"type":"not_found_error","message":"model: claude-bogus"}}"#,
            ))
            .mount(&server)
            .await;

        let backend = AnthropicBackend::new("claude-bogus", "k").with_base_url(server.uri());
        let messages = user_hi();
        let tools: Vec<Value> = vec![];
        let p = params();
        let req = TurnRequest {
            system: None,
            messages: &messages,
            tools: &tools,
            params: &p,
        };
        let err = backend.turn(&req).await.expect_err("must fail");
        match err {
            BackendError::Terminal { kind, message } => {
                assert_eq!(kind, TerminalKind::UnknownModel);
                assert!(message.contains("claude-bogus"));
            }
            other => panic!("expected Terminal{{UnknownModel}}, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn maps_404_without_type_tag_to_terminal_unknown_model() {
        // The `not_found_error` type tag short-circuits BEFORE the
        // status-code match (see maps_404_to_terminal_unknown_model). A 404
        // that does NOT carry that tag must still land on the `404 =>` arm of
        // `map_error_status` and classify as UnknownModel — a provider that
        // returns a bare 404 (or an unparsable body) must not fall through to
        // the catch-all Terminal{Other}.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(404).set_body_string("no such resource"))
            .mount(&server)
            .await;

        let backend = AnthropicBackend::new("claude-bogus", "k").with_base_url(server.uri());
        let messages = user_hi();
        let tools: Vec<Value> = vec![];
        let p = params();
        let req = TurnRequest {
            system: None,
            messages: &messages,
            tools: &tools,
            params: &p,
        };
        let err = backend.turn(&req).await.expect_err("must fail");
        match err {
            BackendError::Terminal { kind, message } => {
                assert_eq!(
                    kind,
                    TerminalKind::UnknownModel,
                    "a bare 404 must classify as UnknownModel, not Other",
                );
                // The unparsable body falls through as the message verbatim.
                assert_eq!(message, "no such resource");
            }
            other => panic!("expected Terminal{{UnknownModel}} from a bare 404, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn maps_400_with_not_found_type_to_unknown_model() {
        // Some Anthropic responses tag model-not-found as a 400 with
        // `not_found_error`; the type tag must win over the status code.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(400).set_body_string(
                r#"{"type":"error","error":{"type":"not_found_error","message":"unknown model id"}}"#,
            ))
            .mount(&server)
            .await;

        let backend = AnthropicBackend::new("x", "k").with_base_url(server.uri());
        let messages = user_hi();
        let tools: Vec<Value> = vec![];
        let p = params();
        let req = TurnRequest {
            system: None,
            messages: &messages,
            tools: &tools,
            params: &p,
        };
        let err = backend.turn(&req).await.expect_err("must fail");
        assert!(matches!(
            err,
            BackendError::Terminal {
                kind: TerminalKind::UnknownModel,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn maps_418_to_terminal_other() {
        // Anything outside the labeled status arms maps to Terminal{Other}
        // — the catch-all keeps the loop from accidentally retrying a
        // strange terminal failure.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(418).set_body_string("i'm a teapot"))
            .mount(&server)
            .await;

        let backend = AnthropicBackend::new("m", "k").with_base_url(server.uri());
        let messages = user_hi();
        let tools: Vec<Value> = vec![];
        let p = params();
        let req = TurnRequest {
            system: None,
            messages: &messages,
            tools: &tools,
            params: &p,
        };
        let err = backend.turn(&req).await.expect_err("must fail");
        match err {
            BackendError::Terminal { kind, message } => {
                assert_eq!(kind, TerminalKind::Other);
                assert_eq!(message, "i'm a teapot");
            }
            other => panic!("expected Terminal{{Other}}, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn maps_403_to_terminal_auth() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(403).set_body_string(
                r#"{"type":"error","error":{"type":"permission_error","message":"forbidden"}}"#,
            ))
            .mount(&server)
            .await;

        let backend = AnthropicBackend::new("m", "k").with_base_url(server.uri());
        let messages = user_hi();
        let tools: Vec<Value> = vec![];
        let p = params();
        let req = TurnRequest {
            system: None,
            messages: &messages,
            tools: &tools,
            params: &p,
        };
        let err = backend.turn(&req).await.expect_err("must fail");
        assert!(matches!(
            err,
            BackendError::Terminal {
                kind: TerminalKind::Auth,
                ..
            }
        ));
    }

    // ---- network transport failure → Transient{Network} -------------------

    #[tokio::test]
    async fn unreachable_endpoint_maps_to_transient_network() {
        // Point at a port nothing is listening on (127.0.0.1:1 is the
        // canonical refused-port trick). Reqwest's connect-time failure
        // must translate to a Transient{Network} the loop can retry.
        let backend = AnthropicBackend::new("m", "k").with_base_url("http://127.0.0.1:1");
        let messages = user_hi();
        let tools: Vec<Value> = vec![];
        let p = params();
        let req = TurnRequest {
            system: None,
            messages: &messages,
            tools: &tools,
            params: &p,
        };
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

    // ---- trailing-slash base URL is tolerated -----------------------------

    #[tokio::test]
    async fn base_url_with_trailing_slash_is_normalized() {
        let server = MockServer::start().await;
        mount_success(
            &server,
            &json!({
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1}
            })
            .to_string(),
        )
        .await;

        // Deliberately add a trailing slash.
        let base = format!("{}/", server.uri());
        let backend = AnthropicBackend::new("m", "k").with_base_url(base);
        let messages = user_hi();
        let tools: Vec<Value> = vec![];
        let p = params();
        let req = TurnRequest {
            system: None,
            messages: &messages,
            tools: &tools,
            params: &p,
        };
        let turn = backend.turn(&req).await.expect("turn ok");
        assert_eq!(turn.text(), "ok");
    }

    // ---- pure-function coverage ------------------------------------------

    #[test]
    fn map_stop_reason_covers_every_variant() {
        assert!(matches!(
            map_stop_reason(Some("end_turn")),
            StopReason::EndTurn
        ));
        assert!(matches!(
            map_stop_reason(Some("tool_use")),
            StopReason::ToolUse
        ));
        assert!(matches!(
            map_stop_reason(Some("max_tokens")),
            StopReason::MaxTokens
        ));
        assert!(matches!(
            map_stop_reason(Some("stop_sequence")),
            StopReason::StopSequence
        ));
        match map_stop_reason(Some("filter_intervened")) {
            StopReason::Other(s) => assert_eq!(s, "filter_intervened"),
            other => panic!("expected Other, got {other:?}"),
        }
        match map_stop_reason(None) {
            StopReason::Other(s) => assert!(s.is_empty()),
            other => panic!("expected Other(\"\"), got {other:?}"),
        }
    }

    #[test]
    fn classify_bad_request_recognizes_context_overflow_variants() {
        // Each phrase here is one Anthropic has been observed to emit on
        // an over-limit prompt; the loop must see ContextLengthExceeded
        // for every one of them.
        for msg in [
            "prompt is too long: 250000 tokens",
            "request exceeds context length",
            "context window of 200000 tokens",
            "exceeds the context window",
            "input exceeds maximum context for this model",
            "too many tokens in the request",
        ] {
            assert!(
                matches!(
                    classify_bad_request(msg),
                    BackendError::ContextLengthExceeded
                ),
                "expected ContextLengthExceeded for {msg:?}"
            );
        }

        // A generic bad request must NOT be misclassified as context overflow.
        match classify_bad_request("messages: at least one message is required") {
            BackendError::Terminal { kind, .. } => assert_eq!(kind, TerminalKind::BadRequest),
            other => panic!("expected Terminal{{BadRequest}}, got {other:?}"),
        }
    }

    #[test]
    fn parse_retry_after_parses_int_seconds_and_rejects_garbage() {
        let h = HeaderValue::from_static("7");
        assert_eq!(parse_retry_after(Some(&h)), Some(Duration::from_secs(7)));

        let h_spaced = HeaderValue::from_static(" 12 ");
        assert_eq!(
            parse_retry_after(Some(&h_spaced)),
            Some(Duration::from_secs(12))
        );

        let h_bad = HeaderValue::from_static("Wed, 21 Oct 2015 07:28:00 GMT");
        assert_eq!(parse_retry_after(Some(&h_bad)), None);

        assert_eq!(parse_retry_after(None), None);
    }
}
