//! The normalized model-IO contract — the **anti-corruption boundary** every
//! backend (Anthropic, Ollama, …) translates its wire format INTO so the loop
//! never sees provider-native shapes.
//!
//! This module is **types + the [`ModelBackend`] trait only**. No HTTP, no
//! provider impl, no loop engine. Each backend adapter (a later item) is
//! responsible for translating its wire format into the value types here; the
//! loop only ever reads these normalized shapes.
//!
//! ## Design pins (don't re-derive — implement as written)
//!
//! - **Illegal states unrepresentable, by role.** [`ContentBlock`] (what an
//!   assistant produces) and [`UserBlock`] (what the user/harness produces)
//!   are distinct enums. [`Message`] is `User { content: Vec<UserBlock> }`
//!   xor `Assistant { content: Vec<ContentBlock> }` — the type system makes it
//!   impossible for an assistant message to hold a tool result, or a user
//!   message to hold a tool call.
//! - **No `System` role.** The system prompt rides on [`TurnRequest`] instead
//!   of as a message. There is no `Message::System`.
//! - **Cheap append from turn to history.** [`From<AssistantTurn>`] for
//!   [`Message`] is the loop's hot path — it moves `content` into
//!   [`Message::Assistant`] rather than re-mapping per variant. The
//!   per-turn metadata ([`AssistantTurn::stop_reason`] / [`AssistantTurn::usage`])
//!   is intentionally dropped because the chat history doesn't need it.
//! - **Determinism.** Any map uses [`std::collections::BTreeMap`], never
//!   `HashMap`, so serialized JSON is byte-stable — the discipline that keeps
//!   the prompt cache hitting. (Today's value types don't need a map; the
//!   rule is a project-wide pin for when one is added.)
//! - **Errors classify, the loop reacts.** [`BackendError`] separates
//!   transient (retryable) from terminal (not) from context-length (only
//!   retryable after the request is mutated — prune/compact — which is the
//!   loop's job) from protocol (the provider returned a shape we can't parse).
//!   The backend never decides to retry; it just labels.
//! - **Model id fixed at construction.** [`ModelBackend`] takes no model
//!   parameter on [`ModelBackend::turn`]; pick the model when you build the
//!   backend. That keeps the per-call surface narrow and makes routing a
//!   construction-time decision the loop doesn't have to re-check every step.
//!
//! ## Scope
//!
//! Out of scope here (and tracked separately): the Anthropic HTTP adapter,
//! the loop engine + finish tool, streaming/SSE, prompt caching wiring,
//! retry/backoff policy, and persistence — [`BackendError`] only labels; the
//! loop will decide how to react.

use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

// ===== Tool-call request ==============================================

/// A request from the model to invoke a tool. The `id` is the
/// provider-assigned identifier that pairs the call with its later
/// [`UserBlock::ToolResult`]; `name` is the tool's registered name; `input` is
/// the raw JSON the model produced for that tool's schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallRequest {
    /// Provider-assigned id — what a later `ToolResult.call_id` must match.
    pub id: String,
    /// The tool's registered name (the registry key).
    pub name: String,
    /// The tool's arguments, as the model produced them.
    pub input: Value,
}

// ===== Content blocks =================================================

/// One unit of assistant-produced content.
///
/// This is the **assistant lane**: the model can emit prose
/// ([`Self::Text`]), an internal reasoning trace ([`Self::Reasoning`] — kept
/// because some providers require the opaque hash echoed back to preserve
/// caching), or a tool invocation ([`Self::ToolCall`]). It deliberately can
/// NOT carry a tool result — that lives on the user lane.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContentBlock {
    /// Visible assistant prose.
    Text(String),
    /// An internal-reasoning block. `opaque` is a provider-specific blob
    /// (e.g. a signature/hash) that some backends require echoed back on the
    /// next turn for cache continuity.
    Reasoning {
        text: String,
        opaque: Option<String>,
    },
    /// The model wants to call a tool.
    ToolCall(ToolCallRequest),
}

/// One unit of user/harness-produced content.
///
/// This is the **user lane**: free-form input ([`Self::Text`]) or the result
/// of a previous tool invocation ([`Self::ToolResult`]). It deliberately can
/// NOT carry a tool call — only the assistant emits those.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum UserBlock {
    /// Plain user input.
    Text(String),
    /// The result of a tool call, identified by `call_id` (matching a prior
    /// [`ToolCallRequest::id`]). `is_error` is the steering signal — `true`
    /// means "the call surfaced a problem the model should react to", not
    /// "the harness crashed".
    ToolResult {
        call_id: String,
        content: String,
        is_error: bool,
    },
}

// ===== Message ========================================================

/// A single message in the conversation, role-precise so each role can only
/// carry its legal content. There is intentionally **no `System` variant** —
/// the system prompt rides on [`TurnRequest::system`] instead.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Message {
    /// User/harness input. Carries only [`UserBlock`]s.
    User { content: Vec<UserBlock> },
    /// Assistant output. Carries only [`ContentBlock`]s.
    Assistant { content: Vec<ContentBlock> },
}

// ===== Stop reason & usage ============================================

/// Why the model stopped producing this turn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopReason {
    /// Model decided it was finished.
    EndTurn,
    /// Model emitted at least one tool call and is waiting for results.
    ToolUse,
    /// Hit the per-turn `max_tokens` cap.
    MaxTokens,
    /// Hit one of the configured stop sequences.
    StopSequence,
    /// Anything else the backend wants to label (provider-specific).
    Other(String),
}

/// Token accounting for a single turn.
///
/// `input_tokens` / `output_tokens` are required (every provider returns
/// them in some form). The cache and reasoning fields are `Option` because
/// they vary across providers — **absent is not the same as zero**. A `None`
/// here means "this provider didn't report it", not "it was zero".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_read_tokens: Option<u32>,
    pub cache_write_tokens: Option<u32>,
    pub reasoning_tokens: Option<u32>,
}

// ===== Assistant turn =================================================

/// One full assistant response — `content` plus the per-turn metadata
/// ([`Self::stop_reason`], [`Self::usage`]) the loop uses for control flow
/// and accounting.
///
/// Appending a turn to history is intentionally cheap: see
/// [`From<AssistantTurn>`] for [`Message`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssistantTurn {
    pub content: Vec<ContentBlock>,
    pub stop_reason: StopReason,
    pub usage: Usage,
}

impl AssistantTurn {
    /// Concatenate every [`ContentBlock::Text`] block, in order. Reasoning
    /// and tool-call blocks are skipped — this is the "what did the model
    /// actually say to the user" view.
    pub fn text(&self) -> String {
        let mut out = String::new();
        for block in &self.content {
            if let ContentBlock::Text(t) = block {
                out.push_str(t);
            }
        }
        out
    }

    /// Every [`ContentBlock::ToolCall`] in the turn, in order. The loop
    /// iterates this to dispatch the requested calls.
    pub fn tool_calls(&self) -> Vec<&ToolCallRequest> {
        self.content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolCall(call) => Some(call),
                ContentBlock::Text(_) | ContentBlock::Reasoning { .. } => None,
            })
            .collect()
    }
}

impl From<AssistantTurn> for Message {
    /// The loop's hot path: append a turn to history. We move `content`
    /// into [`Message::Assistant`] (no per-variant remap, no clone), and
    /// intentionally drop `stop_reason` / `usage` — those are *turn
    /// metadata*, not chat history.
    fn from(turn: AssistantTurn) -> Self {
        Self::Assistant {
            content: turn.content,
        }
    }
}

// ===== Request side ===================================================

/// Per-call sampling knobs. Provider-agnostic; each backend translates these
/// into its wire format.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamplingParams {
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    pub stop_sequences: Vec<String>,
}

/// One full request to the backend — what [`ModelBackend::turn`] takes.
///
/// All fields are borrowed: the loop owns the conversation, system prompt,
/// tool list, and params, and lends them to the backend for the duration of
/// a single call. The model id is **not** here — it's fixed at backend
/// construction (see [`ModelBackend`]).
#[derive(Debug, Clone, Copy)]
pub struct TurnRequest<'a> {
    pub system: Option<&'a str>,
    pub messages: &'a [Message],
    pub tools: &'a [Value],
    pub params: &'a SamplingParams,
}

// ===== Errors =========================================================

/// Why a transient (retryable) backend failure happened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransientKind {
    /// HTTP 429 / explicit rate-limit response.
    RateLimit,
    /// Provider reported the model as overloaded (HTTP 529 on Anthropic).
    Overloaded,
    /// HTTP 5xx that isn't an `Overloaded` variant.
    ServerError,
    /// TCP/connection-level failure (DNS, refused, reset).
    Network,
    /// Request-level timeout.
    Timeout,
}

/// Why a terminal (non-retryable) backend failure happened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalKind {
    /// HTTP 401/403 — credentials are missing, invalid, or insufficient.
    Auth,
    /// HTTP 400 — request was malformed in a way the provider rejected.
    BadRequest,
    /// The configured model id is not a model the provider knows.
    UnknownModel,
    /// The tool schema we sent was rejected by the provider.
    SchemaRejected,
    /// Anything else the backend wants to label as terminal.
    Other,
}

/// The classified failure surface of a [`ModelBackend::turn`] call.
///
/// **The loop never sees provider-native errors** — each backend translates
/// its wire failures into one of these four variants:
///
/// - [`Self::Transient`] — retrying the same request might work.
///   [`Self::is_retryable`] returns `true` for exactly this variant.
/// - [`Self::ContextLengthExceeded`] — the request itself is too big. This
///   is **only retryable after the request is mutated** (prune/compact),
///   which is the loop's job; the backend can't do it. Hence
///   [`Self::is_retryable`] is `false` for this variant.
/// - [`Self::Terminal`] — the request will never work as-is (auth, schema,
///   unknown model, …). Surface up.
/// - [`Self::Protocol`] — the provider returned a shape we can't parse. The
///   raw payload (if we have it) is attached for triage; this is a bug in
///   the adapter, not a retryable failure.
///
/// Not `Serialize`/`Deserialize` on purpose: this is a runtime error, not
/// something the run record persists.
#[derive(Debug, Error)]
pub enum BackendError {
    /// A retryable failure. `retry_after` is the provider's hint (e.g. the
    /// `Retry-After` header) when present.
    #[error("transient backend failure ({kind:?}; retry_after={retry_after:?})")]
    Transient {
        kind: TransientKind,
        retry_after: Option<Duration>,
    },
    /// The request exceeded the model's context window. Retryable **only
    /// after** the request is mutated (prune/compact); that's the loop's
    /// job, not the backend's.
    #[error("context length exceeded")]
    ContextLengthExceeded,
    /// A non-retryable failure: the request will never succeed as-is.
    #[error("terminal backend failure ({kind:?}): {message}")]
    Terminal { kind: TerminalKind, message: String },
    /// The provider returned a shape the adapter couldn't parse. This is an
    /// adapter bug; `raw` is the unparsed payload (when available) for
    /// triage.
    #[error("protocol error: {message}")]
    Protocol {
        message: String,
        raw: Option<String>,
    },
}

impl BackendError {
    /// `true` only for [`Self::Transient`].
    ///
    /// `ContextLengthExceeded` is deliberately **not** retryable here — see
    /// the variant doc. The loop will detect it, mutate the request
    /// (prune/compact), and try again; the backend itself can't.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::Transient { .. })
    }
}

// ===== Trait ==========================================================

/// The anti-corruption boundary between the agent loop and any specific
/// model provider.
///
/// Every backend adapter (Anthropic, Ollama, …) implements this trait by
/// translating its wire format **into** the normalized value types in this
/// module. The loop never sees provider-native shapes — it consumes
/// [`AssistantTurn`] / produces [`TurnRequest`] / classifies failures via
/// [`BackendError`].
///
/// **The model id is fixed at backend construction**, not per call. Pick the
/// model when you build the backend; the loop decides *which* backend to
/// call, not *what model* a backend uses internally. That keeps the
/// per-call surface narrow.
///
/// **Dyn-compatibility:** the loop will eventually hold backends as
/// `Arc<dyn ModelBackend>` for routing, so the trait must be object-safe.
/// A bare `async fn` in a trait is not yet dyn-compatible on this
/// toolchain, so we use the [`mod@async_trait`] crate (same pattern as
/// [`crate::tool::Tool`]).
#[async_trait]
pub trait ModelBackend: Send + Sync {
    /// Execute one model turn against `req`. The backend is responsible for
    /// every wire-format translation; the loop only sees the normalized
    /// types defined in this module.
    async fn turn(&self, req: &TurnRequest<'_>) -> Result<AssistantTurn, BackendError>;
}

// =======================================================================
// Tests
// =======================================================================

#[cfg(test)]
mod tests {
    use super::{
        AssistantTurn, BackendError, ContentBlock, Message, ModelBackend, SamplingParams,
        StopReason, TerminalKind, ToolCallRequest, TransientKind, TurnRequest, Usage, UserBlock,
    };
    use async_trait::async_trait;
    use serde::{Serialize, de::DeserializeOwned};
    use serde_json::{Value, json};
    use std::time::Duration;

    fn round_trip<T>(value: &T)
    where
        T: Serialize + DeserializeOwned + PartialEq + std::fmt::Debug,
    {
        let json = serde_json::to_string(value).expect("serialize");
        let back: T = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(value, &back, "round-trip should be lossless");
    }

    // ---- helpers ----

    fn sample_tool_call() -> ToolCallRequest {
        ToolCallRequest {
            id: "call-1".to_string(),
            name: "read_file".to_string(),
            input: json!({ "path": "src/lib.rs" }),
        }
    }

    fn sample_usage() -> Usage {
        Usage {
            input_tokens: 1024,
            output_tokens: 256,
            cache_read_tokens: Some(800),
            cache_write_tokens: Some(128),
            reasoning_tokens: Some(64),
        }
    }

    fn sample_assistant_turn() -> AssistantTurn {
        AssistantTurn {
            content: vec![
                ContentBlock::Text("starting up".to_string()),
                ContentBlock::Reasoning {
                    text: "let me look at the file".to_string(),
                    opaque: Some("sig-abc".to_string()),
                },
                ContentBlock::ToolCall(sample_tool_call()),
                ContentBlock::Text(" — and then continue".to_string()),
            ],
            stop_reason: StopReason::ToolUse,
            usage: sample_usage(),
        }
    }

    // ---- serde round-trips for every value type ----

    #[test]
    fn round_trip_tool_call_request() {
        round_trip(&sample_tool_call());
    }

    #[test]
    fn round_trip_content_block_all_variants() {
        round_trip(&ContentBlock::Text("hello".to_string()));
        round_trip(&ContentBlock::Reasoning {
            text: "thinking".to_string(),
            opaque: Some("sig-xyz".to_string()),
        });
        round_trip(&ContentBlock::Reasoning {
            text: "thinking".to_string(),
            opaque: None,
        });
        round_trip(&ContentBlock::ToolCall(sample_tool_call()));
    }

    #[test]
    fn round_trip_user_block_all_variants() {
        round_trip(&UserBlock::Text("please do the thing".to_string()));
        round_trip(&UserBlock::ToolResult {
            call_id: "call-1".to_string(),
            content: "ok: 3 lines".to_string(),
            is_error: false,
        });
        round_trip(&UserBlock::ToolResult {
            call_id: "call-1".to_string(),
            content: "file not found".to_string(),
            is_error: true,
        });
    }

    #[test]
    fn round_trip_message_both_variants() {
        let user = Message::User {
            content: vec![
                UserBlock::Text("hi".to_string()),
                UserBlock::ToolResult {
                    call_id: "c1".to_string(),
                    content: "ok".to_string(),
                    is_error: false,
                },
            ],
        };
        round_trip(&user);

        let assistant = Message::Assistant {
            content: vec![
                ContentBlock::Text("sure".to_string()),
                ContentBlock::ToolCall(sample_tool_call()),
            ],
        };
        round_trip(&assistant);
    }

    #[test]
    fn round_trip_stop_reason_all_variants() {
        round_trip(&StopReason::EndTurn);
        round_trip(&StopReason::ToolUse);
        round_trip(&StopReason::MaxTokens);
        round_trip(&StopReason::StopSequence);
        round_trip(&StopReason::Other("filter_intervened".to_string()));
    }

    #[test]
    fn round_trip_usage_with_and_without_optionals() {
        round_trip(&sample_usage());
        round_trip(&Usage {
            input_tokens: 10,
            output_tokens: 0,
            cache_read_tokens: None,
            cache_write_tokens: None,
            reasoning_tokens: None,
        });
    }

    #[test]
    fn round_trip_assistant_turn() {
        round_trip(&sample_assistant_turn());
    }

    #[test]
    fn round_trip_sampling_params() {
        round_trip(&SamplingParams {
            max_tokens: 1024,
            temperature: Some(0.7),
            stop_sequences: vec!["END".to_string(), "STOP".to_string()],
        });
        round_trip(&SamplingParams {
            max_tokens: 256,
            temperature: None,
            stop_sequences: vec![],
        });
    }

    // ---- absent != zero for Option<u32> in Usage ----

    #[test]
    fn usage_absent_is_distinct_from_zero() {
        // None and Some(0) must round-trip distinctly — this is the
        // "absent != zero" contract for provider-varying fields.
        let absent = Usage {
            input_tokens: 1,
            output_tokens: 1,
            cache_read_tokens: None,
            cache_write_tokens: None,
            reasoning_tokens: None,
        };
        let zero = Usage {
            input_tokens: 1,
            output_tokens: 1,
            cache_read_tokens: Some(0),
            cache_write_tokens: Some(0),
            reasoning_tokens: Some(0),
        };
        assert_ne!(absent, zero);

        let absent_s = serde_json::to_string(&absent).expect("ser");
        let zero_s = serde_json::to_string(&zero).expect("ser");
        assert_ne!(absent_s, zero_s);
    }

    // ---- AssistantTurn helpers ----

    #[test]
    fn text_concatenates_only_text_blocks_in_order() {
        let turn = sample_assistant_turn();
        // Reasoning + ToolCall blocks are skipped; the two Text blocks
        // concatenate in declaration order.
        assert_eq!(turn.text(), "starting up — and then continue");
    }

    #[test]
    fn text_empty_when_no_text_blocks() {
        let turn = AssistantTurn {
            content: vec![
                ContentBlock::Reasoning {
                    text: "x".to_string(),
                    opaque: None,
                },
                ContentBlock::ToolCall(sample_tool_call()),
            ],
            stop_reason: StopReason::ToolUse,
            usage: sample_usage(),
        };
        assert_eq!(turn.text(), "");
    }

    #[test]
    fn tool_calls_returns_each_call_in_order() {
        let c1 = ToolCallRequest {
            id: "id-1".to_string(),
            name: "a".to_string(),
            input: json!({}),
        };
        let c2 = ToolCallRequest {
            id: "id-2".to_string(),
            name: "b".to_string(),
            input: json!({"x": 1}),
        };
        let turn = AssistantTurn {
            content: vec![
                ContentBlock::Text("intro".to_string()),
                ContentBlock::ToolCall(c1.clone()),
                ContentBlock::Reasoning {
                    text: "midway".to_string(),
                    opaque: None,
                },
                ContentBlock::ToolCall(c2.clone()),
            ],
            stop_reason: StopReason::ToolUse,
            usage: sample_usage(),
        };
        let calls = turn.tool_calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0], &c1);
        assert_eq!(calls[1], &c2);
    }

    #[test]
    fn tool_calls_empty_when_no_tool_use() {
        let turn = AssistantTurn {
            content: vec![ContentBlock::Text("just words".to_string())],
            stop_reason: StopReason::EndTurn,
            usage: sample_usage(),
        };
        assert!(turn.tool_calls().is_empty());
    }

    // ---- From<AssistantTurn> for Message ----

    #[test]
    fn from_assistant_turn_yields_assistant_message_with_identical_content() {
        let turn = sample_assistant_turn();
        let content_clone = turn.content.clone();
        let msg: Message = turn.into();
        match msg {
            Message::Assistant { content } => {
                assert_eq!(
                    content, content_clone,
                    "Message::Assistant.content must equal the turn's content"
                );
            }
            Message::User { .. } => panic!("expected Assistant variant"),
        }
    }

    // ---- BackendError: is_retryable across every variant ----

    #[test]
    fn is_retryable_is_true_only_for_transient() {
        let transient_kinds = [
            TransientKind::RateLimit,
            TransientKind::Overloaded,
            TransientKind::ServerError,
            TransientKind::Network,
            TransientKind::Timeout,
        ];
        for kind in transient_kinds {
            let with_hint = BackendError::Transient {
                kind,
                retry_after: Some(Duration::from_secs(2)),
            };
            assert!(with_hint.is_retryable(), "Transient {kind:?} retryable");
            let without_hint = BackendError::Transient {
                kind,
                retry_after: None,
            };
            assert!(without_hint.is_retryable());
            // Touch Display so coverage sees it.
            let _ = format!("{with_hint}");
        }

        // ContextLengthExceeded is deliberately NOT retryable here — the
        // loop must mutate the request first.
        let ctx = BackendError::ContextLengthExceeded;
        assert!(!ctx.is_retryable());
        assert!(format!("{ctx}").contains("context length"));

        for kind in [
            TerminalKind::Auth,
            TerminalKind::BadRequest,
            TerminalKind::UnknownModel,
            TerminalKind::SchemaRejected,
            TerminalKind::Other,
        ] {
            let term = BackendError::Terminal {
                kind,
                message: format!("term: {kind:?}"),
            };
            assert!(!term.is_retryable(), "Terminal {kind:?} not retryable");
            assert!(format!("{term}").contains("terminal"));
        }

        let proto = BackendError::Protocol {
            message: "unparsable".to_string(),
            raw: Some("{not json".to_string()),
        };
        assert!(!proto.is_retryable());
        let proto_no_raw = BackendError::Protocol {
            message: "schema drift".to_string(),
            raw: None,
        };
        assert!(!proto_no_raw.is_retryable());
        assert!(format!("{proto}").contains("protocol"));
        assert!(format!("{proto_no_raw}").contains("protocol"));
    }

    #[test]
    fn backend_error_implements_std_error() {
        // The trait bound is what callers will actually rely on, so prove
        // the wiring through a generic function.
        fn assert_error<E: std::error::Error>(_e: &E) {}
        let e = BackendError::ContextLengthExceeded;
        assert_error(&e);
    }

    // ---- TurnRequest borrows wire cleanly ----

    #[test]
    fn turn_request_borrows_compose() {
        let messages = vec![
            Message::User {
                content: vec![UserBlock::Text("hi".to_string())],
            },
            Message::Assistant {
                content: vec![ContentBlock::Text("hello".to_string())],
            },
        ];
        let tools: Vec<Value> = vec![json!({"name": "echo"})];
        let params = SamplingParams {
            max_tokens: 256,
            temperature: Some(0.0),
            stop_sequences: vec![],
        };
        let req = TurnRequest {
            system: Some("you are a harness"),
            messages: &messages,
            tools: &tools,
            params: &params,
        };
        assert_eq!(req.system, Some("you are a harness"));
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.tools.len(), 1);
        assert_eq!(req.params.max_tokens, 256);
    }

    // ---- ModelBackend is dyn-compatible ----

    /// A tiny in-memory backend used to prove the trait is object-safe and
    /// the async wiring compiles.
    struct EchoBackend;

    #[async_trait]
    impl ModelBackend for EchoBackend {
        async fn turn(&self, req: &TurnRequest<'_>) -> Result<AssistantTurn, BackendError> {
            // Echo back the last user text block, if any — otherwise empty.
            let echoed = req
                .messages
                .iter()
                .rev()
                .find_map(|m| match m {
                    Message::User { content } => content.iter().find_map(|b| match b {
                        UserBlock::Text(t) => Some(t.clone()),
                        UserBlock::ToolResult { .. } => None,
                    }),
                    Message::Assistant { .. } => None,
                })
                .unwrap_or_default();
            Ok(AssistantTurn {
                content: vec![ContentBlock::Text(echoed)],
                stop_reason: StopReason::EndTurn,
                usage: Usage {
                    input_tokens: 0,
                    output_tokens: 0,
                    cache_read_tokens: None,
                    cache_write_tokens: None,
                    reasoning_tokens: None,
                },
            })
        }
    }

    #[tokio::test]
    async fn model_backend_is_object_safe_and_invocable() {
        // Build a `dyn ModelBackend` — this is the routing shape the loop
        // will use. If the trait weren't object-safe, this line wouldn't
        // compile, which is the real assertion.
        let backend: std::sync::Arc<dyn ModelBackend> = std::sync::Arc::new(EchoBackend);

        let messages = vec![Message::User {
            content: vec![UserBlock::Text("ping".to_string())],
        }];
        let tools: Vec<Value> = vec![];
        let params = SamplingParams {
            max_tokens: 16,
            temperature: None,
            stop_sequences: vec![],
        };
        let req = TurnRequest {
            system: None,
            messages: &messages,
            tools: &tools,
            params: &params,
        };
        let turn = backend.turn(&req).await.expect("turn ok");
        assert_eq!(turn.text(), "ping");
        assert!(matches!(turn.stop_reason, StopReason::EndTurn));
        assert!(turn.tool_calls().is_empty());

        // And the From impl appends to history without per-variant remap.
        let msg: Message = turn.into();
        assert!(matches!(msg, Message::Assistant { .. }));
    }
}
