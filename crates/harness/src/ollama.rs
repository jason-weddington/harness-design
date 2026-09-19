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
//! the OpenAI-compat `/v1` path, and capability discovery via `/api/tags` and
//! `/api/ps` are out of scope here. Context-length resolution IS in scope:
//! the shared [`resolve_num_ctx`] (built on the `POST /api/show` probe
//! [`resolve_context_length`]) is used by talos (`crates/talos/src/main.rs`)
//! AND both eval runners (`examples/coding_eval.rs`,
//! `examples/mined_eval.rs`) — one implementation, so the `num_ctx`
//! precedence cannot drift between the shipped lane and the measured lane,
//! and a local model's window is set from the model's own advertised
//! `{arch}.context_length` rather than a hardcoded fallback.
//!
//! Note: a non-localhost, non-cloud Ollama daemon (e.g. `http://jason-desktop:11434`,
//! a LAN address) is not matched by [`is_local_ollama_url`] and is therefore
//! never probed — it receives no `num_ctx` and inherits Ollama's own low
//! default (documented as 2048 with silent oldest-message dropping). Setting
//! `OLLAMA_NUM_CTX` is the operator's remedy for that topology.
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
//!   takes the model id (e.g. `glm-5.3:cloud` on cloud, a `qwen3.6`-family tag
//!   locally); the wire never carries a literal.
//! - **No api-key leakage in `Debug`.** [`OllamaBackend`] deliberately does
//!   **not** derive `Debug`; the api key only travels into the
//!   `Authorization` header.
//! - **The backend only classifies; the loop reacts.** Failures map to
//!   [`BackendError`] variants; nothing here retries or backs off.
//!
//! ## `prompt_eval_cached_count` (Ollama ≥ 0.33.3)
//!
//! Ollama daemons 0.33.3+ report `prompt_eval_cached_count` alongside
//! `prompt_eval_count` — the prompt tokens served from the KV prefix cache.
//! Empirically (2026-09-19, a 29,721-token prompt):
//!
//! - first call: `prompt_eval_count` 29721 / `prompt_eval_cached_count` 0 →
//!   `input_tokens` 29721, `cache_read_tokens` Some(0);
//! - second call: 29721 / 29696 → `input_tokens` 25, `cache_read_tokens`
//!   Some(29696).
//!
//! `prompt_eval_count` is the TOTAL prompt tokens (cached INCLUDED);
//! the normalized [`Usage::input_tokens`] carries the UNCACHED remainder
//! (`prompt_eval_count − prompt_eval_cached_count`, saturating), matching the
//! Anthropic/Bedrock convention so `input * rate_in + cache_read *
//! rate_cached` pricing never double-counts. A daemon older than 0.33.3 omits
//! the field entirely: `input_tokens` = the total, `cache_read_tokens` =
//! None — not reported is not the same as zero hits.
//!
//! Ambiguity note: when the report is inconsistent
//! (`prompt_eval_cached_count` > `prompt_eval_count`), [`map_response`]
//! clamps `input_tokens` to 0 and emits a stderr warning.
//! `input_tokens` 0 + `cache_read_tokens` Some(n) is then indistinguishable
//! in the durable record between a fully-cached prompt and a clamped
//! inconsistent report — the stderr line is the only discriminator
//! (persisting it as a structured transcript event is deferred;
//! `map_response` has no writer).
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

impl ThinkLevel {
    /// The `OLLAMA_THINK` env spelling of this level.
    ///
    /// These are the ENV spellings talos's `OLLAMA_THINK` variable accepts —
    /// DELIBERATELY NOT the Ollama wire forms the hand-written `Serialize`
    /// impl emits (`off`/`on` serialize to the booleans `false`/`true` on
    /// the wire; the graded levels are the same strings in both). Recording
    /// the env spelling on a run record is what keeps the record honest
    /// about what the operator actually set.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::On => "on",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Max => "max",
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
        // an error: `prompt_eval_count` is the TOTAL prompt tokens for the
        // request (cached tokens INCLUDED — see the `prompt_eval_cached_count`
        // split in `map_response`), while `estimate_prompt_tokens` is a
        // chars/4 approximation, so a post-hoc mismatch would signal estimate
        // error, not context loss. And the real dropped-context failure
        // (ollama/ollama#11885) produces NO response-side signal at all —
        // hence the client-side pre-flight guard. The guard only runs when
        // `num_ctx` is set;
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
// Context-length resolution via /api/show
// ============================================================================

/// Timeout for the `POST /api/show` probe.
///
/// A hung daemon must not wedge eval startup; `reqwest` 0.12 has NO default
/// request timeout. `OllamaBackend::new` uses a bare `Client::new()` — that is
/// deliberately not copied here.
const SHOW_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// The result of a successful [`resolve_context_length`] call.
///
/// Carries provenance (not a bare `u32`) so a wrong-key resolution is
/// detectable after the fact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedContextLength {
    /// The advertised context length.
    pub value: u32,
    /// The literal string read from `model_info["general.architecture"]`.
    pub architecture: String,
    /// The literal key that was looked up: `"{architecture}.context_length"`.
    pub key: String,
}

/// Error returned by [`resolve_context_length`] when the probe fails.
#[derive(Debug, thiserror::Error)]
#[error("could not resolve context length for model `{model}` from {url}: {kind}")]
pub struct ContextLengthError {
    pub model: String,
    pub url: String,
    pub kind: ContextLengthErrorKind,
}

/// The specific failure reason within a [`ContextLengthError`].
#[derive(Debug, thiserror::Error)]
pub enum ContextLengthErrorKind {
    /// The HTTP request itself failed (connect refused, DNS, timeout, etc.).
    /// Carries `reqwest::Error::to_string()` — never a hand-written string.
    #[error("transport failure: {0}")]
    Transport(String),
    /// The daemon replied with a non-2xx status.
    /// `body` is the verbatim response body text, un-parsed.
    #[error("HTTP {status}: {body}")]
    Status { status: u16, body: String },
    /// The response was 200 but its payload did not have the expected shape.
    /// The message names the failing JSON path.
    #[error("malformed /api/show payload: {0}")]
    Malformed(String),
}

/// Resolve the advertised context length for `model` from the Ollama daemon at
/// `base_url` by querying `POST /api/show`.
///
/// Resolution path:
/// 1. Parse the 200 body as JSON.
/// 2. Take the object at `model_info`.
/// 3. Read `model_info["general.architecture"]` as string `arch`.
/// 4. Read `model_info["{arch}.context_length"]` and convert to `u32`.
///
/// There is NO fallback scan over `model_info` for any key ending in
/// `.context_length` — real payloads contain decoy keys
/// (e.g. `gptoss.rope.scaling.original_context_length`) that would produce
/// wrong values.
///
/// The returned value also arms the pre-flight guard via `with_num_ctx`
/// (see `OllamaBackend::with_num_ctx`), so a small advertised window converts
/// silent truncation into a hard `BackendError::ContextLengthExceeded` — loud
/// by design.
///
/// # Errors
///
/// Returns [`ContextLengthError`] on any of:
/// - transport failure (connect refused, DNS, timeout)
/// - non-2xx HTTP status
/// - malformed payload (missing/wrong-typed fields)
/// - zero context length
///
/// ```no_run
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let r = harness::ollama::resolve_context_length(
///     "http://localhost:11434",
///     "qwen3.6:35b",
///     None,
/// ).await?;
/// println!("num_ctx={} (arch={} key={})", r.value, r.architecture, r.key);
/// # Ok(())
/// # }
/// ```
pub async fn resolve_context_length(
    base_url: &str,
    model: &str,
    api_key: Option<&str>,
) -> Result<ResolvedContextLength, ContextLengthError> {
    let url = format!("{}/api/show", base_url.trim_end_matches('/'));

    // Convenience closure — builds an error with the same model/url context.
    let mk_err = |kind| ContextLengthError {
        model: model.to_string(),
        url: url.clone(),
        kind,
    };

    let client = reqwest::Client::builder()
        .timeout(SHOW_TIMEOUT)
        .build()
        .map_err(|e| mk_err(ContextLengthErrorKind::Transport(e.to_string())))?;

    let mut builder = client.post(&url).json(&serde_json::json!({"model": model}));
    if let Some(key) = api_key {
        builder = builder.header("authorization", format!("Bearer {key}"));
    }

    let response = builder
        .send()
        .await
        .map_err(|e| mk_err(ContextLengthErrorKind::Transport(e.to_string())))?;

    let status = response.status();
    let body_text = response
        .text()
        .await
        .map_err(|e| mk_err(ContextLengthErrorKind::Transport(e.to_string())))?;

    if !status.is_success() {
        return Err(mk_err(ContextLengthErrorKind::Status {
            status: status.as_u16(),
            body: body_text,
        }));
    }

    let payload: serde_json::Value = serde_json::from_str(&body_text).map_err(|_| {
        mk_err(ContextLengthErrorKind::Malformed(
            "body not JSON".to_string(),
        ))
    })?;

    let model_info = payload.get("model_info").ok_or_else(|| {
        mk_err(ContextLengthErrorKind::Malformed(
            "model_info missing".to_string(),
        ))
    })?;

    let model_info = model_info.as_object().ok_or_else(|| {
        mk_err(ContextLengthErrorKind::Malformed(
            "model_info not an object".to_string(),
        ))
    })?;

    let arch_val = model_info.get("general.architecture").ok_or_else(|| {
        mk_err(ContextLengthErrorKind::Malformed(
            "model_info.general.architecture missing".to_string(),
        ))
    })?;

    let arch = arch_val.as_str().ok_or_else(|| {
        mk_err(ContextLengthErrorKind::Malformed(
            "model_info.general.architecture not a string".to_string(),
        ))
    })?;

    let ctx_key = format!("{arch}.context_length");

    let ctx_val = model_info.get(&ctx_key).ok_or_else(|| {
        mk_err(ContextLengthErrorKind::Malformed(format!(
            "model_info.{ctx_key} missing"
        )))
    })?;

    let value = match ctx_val {
        serde_json::Value::Number(n) => {
            let u = n.as_u64().ok_or_else(|| {
                mk_err(ContextLengthErrorKind::Malformed(format!(
                    "model_info.{ctx_key} not a u32"
                )))
            })?;
            u32::try_from(u).map_err(|_| {
                mk_err(ContextLengthErrorKind::Malformed(format!(
                    "model_info.{ctx_key} not a u32"
                )))
            })?
        }
        _ => {
            return Err(mk_err(ContextLengthErrorKind::Malformed(format!(
                "model_info.{ctx_key} not a u32"
            ))));
        }
    };

    if value == 0 {
        return Err(mk_err(ContextLengthErrorKind::Malformed(format!(
            "model_info.{ctx_key} is zero"
        ))));
    }

    Ok(ResolvedContextLength {
        value,
        architecture: arch.to_string(),
        key: ctx_key,
    })
}

/// Audit floor for a resolved `num_ctx`. This is **not** a default or
/// fallback — [`resolve_num_ctx`] never assigns it as the value. It is used
/// only to flag a probe result that sits below it (a `warning` line and the
/// `BELOW-FLOOR` provenance variant in the `desc`); the run still proceeds
/// with the verbatim advertised value.
pub const MIN_EXPECTED_NUM_CTX: u32 = 32_768;

/// Whether `base_url` names a local Ollama daemon — the URL contains
/// `localhost` or `127.0.0.1`. The single gate used by [`resolve_num_ctx`]
/// to decide whether the `POST /api/show` probe fires; non-local URLs
/// (cloud, LAN hostnames) get no `num_ctx` and inherit Ollama's own
/// default.
pub fn is_local_ollama_url(base_url: &str) -> bool {
    base_url.contains("localhost") || base_url.contains("127.0.0.1")
}

/// How a [`NumCtxResolution`] was produced — the provenance of the value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumCtxSource {
    /// `OLLAMA_NUM_CTX` was set to a non-empty value that parses as `u32` —
    /// used verbatim, no HTTP request.
    Explicit,
    /// The value came from a `POST /api/show` probe of a local daemon.
    Probe,
    /// No env value and a non-local base URL — the backend gets no `num_ctx`
    /// (Ollama's own default applies), no HTTP request.
    Default,
}

impl NumCtxSource {
    /// The stable string form used in structured stderr / run records.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Explicit => "explicit",
            Self::Probe => "probe",
            Self::Default => "default",
        }
    }
}

/// The outcome of [`resolve_num_ctx`]: the value to pin (if any), its
/// provenance, a provenance description for run headers / records, and an
/// optional warning line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumCtxResolution {
    /// The `num_ctx` to pin via [`OllamaBackend::with_num_ctx`], or `None`
    /// (leave `num_ctx` unset — Ollama's own default applies).
    pub value: Option<u32>,
    /// How `value` was obtained.
    pub source: NumCtxSource,
    /// Human-readable provenance, e.g.
    /// `num_ctx=262144 (resolved: arch=qwen35moe key=qwen35moe.context_length)`.
    pub desc: String,
    /// A pre-formatted warning line (below-floor probe result). The library
    /// only RETURNS this string — printing is the caller's job.
    pub warning: Option<String>,
}

/// Resolve the `num_ctx` to pin for an Ollama backend — the SINGLE
/// implementation of the `OLLAMA_NUM_CTX` precedence shared by talos and
/// both eval runners (one function, so the shipped lane and the measured
/// lane cannot drift).
///
/// Five branches, in exactly this order:
/// 0. `raw_env` is `Some(s)` and `s.trim()` is empty → treated as UNSET
///    (falls through to 3/4/5).
/// 1. `raw_env` is `Some(s)`, `s.trim()` non-empty, parses as `u32` →
///    [`NumCtxSource::Explicit`]; NO HTTP request is made.
/// 2. `raw_env` is `Some(s)`, `s.trim()` non-empty, does NOT parse as `u32`
///    → `Err` naming `OLLAMA_NUM_CTX` and the untrimmed raw string; NO HTTP
///    request is made.
/// 3. unset/empty AND [`is_local_ollama_url`] → probe via
///    [`resolve_context_length`]; on `Ok` with a value at or above
///    [`MIN_EXPECTED_NUM_CTX`] → [`NumCtxSource::Probe`], no warning.
/// 4. same as 3 but the value is below the floor →
///    [`NumCtxSource::Probe`] with a `warning` (BELOW-FLOOR); the run still
///    proceeds with the verbatim advertised value.
/// 5. unset/empty AND NOT local → [`NumCtxSource::Default`], `value: None`,
///    NO HTTP request is made.
///
/// # Errors
///
/// Branch 2 always fails. In branches 3/4 a probe failure is propagated
/// verbatim as `resolve_context_length`'s error string (which names both
/// the model id and the `{base_url}/api/show` url) — NEVER a constant, a
/// default, or `None`.
///
/// ```no_run
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let r = harness::ollama::resolve_num_ctx(
///     "http://localhost:11434",
///     "qwen3.6:35b",
///     None,
///     None,
/// ).await?;
/// println!("{} (source={})", r.desc, r.source.as_str());
/// # Ok(())
/// # }
/// ```
pub async fn resolve_num_ctx(
    base_url: &str,
    model: &str,
    api_key: Option<&str>,
    raw_env: Option<&str>,
) -> Result<NumCtxResolution, String> {
    // Branches 0/1/2: an env value, when present and non-blank after trim,
    // is decisive — a parse error fails loudly BEFORE any HTTP request.
    // An empty/whitespace-only value (branch 0) falls through as UNSET.
    if let Some(s) = raw_env
        && !s.trim().is_empty()
    {
        return match s.parse::<u32>() {
            Ok(n) => Ok(NumCtxResolution {
                value: Some(n),
                source: NumCtxSource::Explicit,
                desc: format!("num_ctx={n} (explicit OLLAMA_NUM_CTX)"),
                warning: None,
            }),
            Err(_) => Err(format!("OLLAMA_NUM_CTX must be a valid u32, got `{s}`")),
        };
    }

    // Branches 3/4: local daemon → probe; fail-loud, never a fallback.
    if is_local_ollama_url(base_url) {
        let resolved = resolve_context_length(base_url, model, api_key)
            .await
            .map_err(|e| e.to_string())?;
        let v = resolved.value;
        let a = resolved.architecture;
        let k = resolved.key;
        if v >= MIN_EXPECTED_NUM_CTX {
            Ok(NumCtxResolution {
                value: Some(v),
                source: NumCtxSource::Probe,
                desc: format!("num_ctx={v} (resolved: arch={a} key={k})"),
                warning: None,
            })
        } else {
            Ok(NumCtxResolution {
                value: Some(v),
                source: NumCtxSource::Probe,
                desc: format!("num_ctx={v} (resolved, BELOW-FLOOR: arch={a} key={k})"),
                warning: Some(format!(
                    "WARNING: resolved num_ctx={v} for `{model}` (arch={a}, key={k}) is \
                     BELOW the {MIN_EXPECTED_NUM_CTX} sanity floor; the run may be \
                     truncation-invalid — set OLLAMA_NUM_CTX to override"
                )),
            })
        }
    } else {
        // Branch 5: non-local, no env value → no `num_ctx`, no HTTP request.
        Ok(NumCtxResolution {
            value: None,
            source: NumCtxSource::Default,
            desc: "num_ctx=default".to_string(),
            warning: None,
        })
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
    // Ollama ≥ 0.33.3: prompt tokens served from the KV prefix cache. Older
    // daemons omit the field entirely — `#[serde(default)]` keeps the whole
    // body deserializable, same pattern as the fields above.
    #[serde(default)]
    prompt_eval_cached_count: Option<u32>,
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

/// Build the stderr warning emitted when Ollama reports
/// `prompt_eval_cached_count` greater than `prompt_eval_count` — a
/// self-contradictory usage report (cached tokens cannot exceed the total).
///
/// Pure and `String`-returning so the overflow branch is unit-testable
/// without capturing stderr (pattern: `inert_precondition_warning`).
fn ollama_usage_inconsistency_warning(total: u32, cached: u32) -> String {
    format!(
        "warning: ollama usage inconsistency — prompt_eval_cached_count \
         {cached} > prompt_eval_count {total}; clamping input_tokens to 0"
    )
}

/// Translate Ollama's native chat response into an [`AssistantTurn`].
///
/// Usage mapping (`prompt_eval_cached_count` split — see the module doc):
/// `prompt_eval_count` is the TOTAL prompt tokens (cached included). One
/// total rule with three branches:
///
/// - `Some(c)` with `c <= total`: `input_tokens = total - c` (the uncached
///   remainder), `cache_read_tokens = Some(c)`.
/// - `Some(c)` with `c > total` (including `total == 0` — both wire fields
///   are independent `#[serde(default)] Option<u32>`, so an absent
///   `prompt_eval_count` plus a present cached count lands here): the report
///   is inconsistent, so `input_tokens` is clamped to 0,
///   `cache_read_tokens = Some(c)`, and exactly one stderr warning is
///   emitted via [`ollama_usage_inconsistency_warning`]. Never an `Err` —
///   the return type has no error channel.
/// - `None` (daemon < 0.33.3 omits the field): `input_tokens = total`,
///   `cache_read_tokens = None` (not reported ≠ zero).
///
/// In all three branches `cache_write_tokens` and `reasoning_tokens` stay
/// `None` — Ollama reports neither (the reported exception is
/// `prompt_eval_cached_count`, daemons ≥ 0.33.3; absent ≠ zero).
///
/// Persisted-seam consequence: because `input_tokens` here is the UNCACHED
/// remainder, `RunStats.input_tokens` / `BudgetConsumed.tokens` and
/// `Event::ModelCall.prompt_tokens` carry that remainder for Ollama turns —
/// a cached second call contributes 25, not 29,721, to the persisted token
/// budget (mirroring Anthropic/Bedrock, whose `input_tokens` already exclude
/// cache reads). The raw-input invariant `input + cache_read == total` keeps
/// eval's `total_raw_input_tokens` unchanged.
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

    // One total rule: `prompt_eval_count` is the TOTAL prompt tokens (cached
    // included); split it by the reported prefix-cache hits.
    let total = body.prompt_eval_count.unwrap_or(0);
    let usage = match body.prompt_eval_cached_count {
        // Consistent report: input is the uncached remainder, cache_read is
        // the reported prefix-cache hits.
        Some(c) if c <= total => Usage {
            input_tokens: total - c,
            output_tokens: body.eval_count.unwrap_or(0),
            cache_read_tokens: Some(c),
            cache_write_tokens: None,
            reasoning_tokens: None,
        },
        // Inconsistent report (cached > total, including total == 0 when
        // prompt_eval_count was absent). Clamp rather than panic or Err —
        // map_response returns AssistantTurn, so Err is impossible by
        // signature — and warn exactly once on stderr.
        Some(c) => {
            eprintln!("{}", ollama_usage_inconsistency_warning(total, c));
            Usage {
                input_tokens: 0,
                output_tokens: body.eval_count.unwrap_or(0),
                cache_read_tokens: Some(c),
                cache_write_tokens: None,
                reasoning_tokens: None,
            }
        }
        // Field not reported (daemon < 0.33.3): the whole prompt was
        // evaluated — input is the total, cache_read is None (not reported
        // ≠ zero). Byte-for-byte today's behaviour.
        None => Usage {
            input_tokens: total,
            output_tokens: body.eval_count.unwrap_or(0),
            cache_read_tokens: None,
            cache_write_tokens: None,
            reasoning_tokens: None,
        },
    };

    AssistantTurn {
        content: blocks,
        stop_reason: map_stop_reason(has_tool_calls, body.done_reason.as_deref()),
        usage,
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
        OllamaBackend, ThinkLevel, classify_transport_error, extract_error_message,
        map_stop_reason, ollama_usage_inconsistency_warning,
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

    // ---- (a2) cached prompt tokens → cache_read_tokens ---------------------
    //
    // The empirically observed second-call numbers: a 29,721-token prompt
    // where 29,696 tokens were prefix-cache hits → 25 uncached.

    #[tokio::test]
    async fn maps_cached_prompt_tokens_into_cache_read() {
        let server = MockServer::start().await;
        let body = json!({
            "message": {"role": "assistant", "content": "ok"},
            "done_reason": "stop",
            "prompt_eval_count": 29721,
            "prompt_eval_cached_count": 29696,
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
        assert_eq!(turn.usage.input_tokens, 25);
        assert_eq!(turn.usage.cache_read_tokens, Some(29696));
        assert_eq!(turn.usage.output_tokens, 7);
        assert_eq!(turn.usage.cache_write_tokens, None);
        assert_eq!(turn.usage.reasoning_tokens, None);
        // Raw-input invariant: uncached + cached == total prompt tokens.
        assert_eq!(
            turn.usage
                .input_tokens
                .saturating_add(turn.usage.cache_read_tokens.unwrap_or(0)),
            29721
        );
    }

    // ---- (a3) inconsistent report: cached > total → clamp + warn -----------

    #[tokio::test]
    async fn clamps_input_when_cached_exceeds_total_and_warns() {
        let server = MockServer::start().await;
        let body = json!({
            "message": {"role": "assistant", "content": "ok"},
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "prompt_eval_cached_count": 25,
            "eval_count": 1
        })
        .to_string();
        mount_success(&server, &body).await;

        let backend = OllamaBackend::new("qwen3.6", server.uri());
        let messages = user_hi();
        let tools: Vec<Value> = vec![];
        let p = params();
        let req = simple_req(&messages, &tools, &p);

        // Clamped, not rejected: map_response returns AssistantTurn, so an
        // inconsistent usage report can never be an Err.
        let turn = backend.turn(&req).await.expect("turn ok");
        assert_eq!(turn.usage.input_tokens, 0);
        assert_eq!(turn.usage.cache_read_tokens, Some(25));
        assert_eq!(turn.usage.output_tokens, 1);
        assert_eq!(turn.usage.cache_write_tokens, None);
        assert_eq!(turn.usage.reasoning_tokens, None);

        // The warning string is testable without capturing stderr (repo
        // convention: the emission itself is review-verifiable, the pure fn's
        // return value is asserted).
        let warning = ollama_usage_inconsistency_warning(10, 25);
        assert!(
            warning.starts_with("warning: ollama usage inconsistency — prompt_eval_cached_count ")
        );
        assert!(warning.contains("25 > prompt_eval_count 10"));
        assert!(warning.ends_with("; clamping input_tokens to 0"));
    }

    // ---- (a4) cached field present with zero hits → Some(0), not None ------

    #[tokio::test]
    async fn zero_cached_count_is_some_zero() {
        let server = MockServer::start().await;
        // Empirical FIRST-call wire shape: the field is PRESENT with zero hits
        // (distinct from an old daemon omitting the field).
        let body = json!({
            "message": {"role": "assistant", "content": "ok"},
            "done_reason": "stop",
            "prompt_eval_count": 29721,
            "prompt_eval_cached_count": 0,
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
        assert_eq!(turn.usage.input_tokens, 29721);
        assert_eq!(turn.usage.cache_read_tokens, Some(0));
        assert_eq!(turn.usage.output_tokens, 7);
        assert_eq!(turn.usage.cache_write_tokens, None);
    }

    // ---- (a5) cached field absent → None, not zero --------------------------

    #[tokio::test]
    async fn absent_cached_count_stays_none_not_zero() {
        let server = MockServer::start().await;
        // Daemon < 0.33.3 wire shape: NO prompt_eval_cached_count key.
        let body = json!({
            "message": {"role": "assistant", "content": "ok"},
            "done_reason": "stop",
            "prompt_eval_count": 29721,
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
        // Input is the TOTAL (not an uncached remainder) and cache_read is
        // None — "not reported" is distinguishable from "not cached" (Some(0)).
        assert_eq!(turn.usage.input_tokens, 29721);
        assert_eq!(turn.usage.cache_read_tokens, None);
        assert_eq!(turn.usage.output_tokens, 7);
        assert_eq!(turn.usage.cache_write_tokens, None);
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

    /// The `OLLAMA_THINK` env spellings (`as_str`) and the Ollama wire forms
    /// (the hand-written `Serialize`) are pinned apart in ONE place: the
    /// env vocabulary is `off|on|low|medium|high|max` strings, while `off`
    /// and `on` serialize to the booleans `false`/`true` on the wire.
    #[test]
    fn think_level_as_str_is_the_env_spelling_not_the_wire_form() {
        assert_eq!(ThinkLevel::Off.as_str(), "off");
        assert_eq!(ThinkLevel::On.as_str(), "on");
        assert_eq!(ThinkLevel::Low.as_str(), "low");
        assert_eq!(ThinkLevel::Medium.as_str(), "medium");
        assert_eq!(ThinkLevel::High.as_str(), "high");
        assert_eq!(ThinkLevel::Max.as_str(), "max");
        // Re-pinned here: `as_str("off")` and the wire form (`false`) must
        // never be conflated — the record carries the env spelling.
        assert_eq!(serde_json::to_value(ThinkLevel::Off).unwrap(), json!(false));
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

    // ---- (h) context-length resolution ------------------------------------

    use super::{
        ContextLengthErrorKind, MIN_EXPECTED_NUM_CTX, NumCtxResolution, NumCtxSource,
        is_local_ollama_url, resolve_context_length, resolve_num_ctx,
    };

    /// Mount a successful `/api/show` response with the given `model_info` payload.
    async fn mount_show(server: &MockServer, model_info: Value) {
        Mock::given(method("POST"))
            .and(path("/api/show"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(json!({"model_info": model_info}).to_string()),
            )
            .mount(server)
            .await;
    }

    /// Mount a `/api/show` response that returns the given status and body.
    async fn mount_show_error(server: &MockServer, status: u16, body: &str) {
        Mock::given(method("POST"))
            .and(path("/api/show"))
            .respond_with(ResponseTemplate::new(status).set_body_string(body))
            .mount(server)
            .await;
    }

    /// Mount a `/api/show` response with a raw body (for testing non-JSON bodies).
    async fn mount_show_raw(server: &MockServer, body: &str) {
        Mock::given(method("POST"))
            .and(path("/api/show"))
            .respond_with(ResponseTemplate::new(200).set_body_string(body))
            .mount(server)
            .await;
    }

    // Request path is /api/show, method is POST, body equals {"model":"qwen3.6:35b"}.
    #[tokio::test]
    async fn show_request_shape_is_correct() {
        let server = MockServer::start().await;
        mount_show(
            &server,
            json!({"general.architecture": "qwen35moe", "qwen35moe.context_length": 262_144_u32}),
        )
        .await;

        resolve_context_length(&server.uri(), "qwen3.6:35b", None)
            .await
            .expect("resolve ok");

        let received = server.received_requests().await.expect("requests captured");
        assert_eq!(received.len(), 1);
        let r: &Request = &received[0];
        assert_eq!(r.method.as_str(), "POST");
        assert_eq!(r.url.path(), "/api/show");
        let parsed: Value = serde_json::from_slice(&r.body).expect("json body");
        assert_eq!(parsed, json!({"model": "qwen3.6:35b"}));
    }

    // Bearer header present when api_key is Some.
    #[tokio::test]
    async fn show_bearer_present_when_api_key_set() {
        let server = MockServer::start().await;
        mount_show(
            &server,
            json!({"general.architecture": "qwen35moe", "qwen35moe.context_length": 262_144_u32}),
        )
        .await;

        resolve_context_length(&server.uri(), "qwen3.6:35b", Some("sk-test"))
            .await
            .expect("resolve ok");

        let received = server.received_requests().await.expect("requests captured");
        assert_eq!(received.len(), 1);
        let r: &Request = &received[0];
        assert_eq!(
            r.headers.get("authorization").and_then(|v| v.to_str().ok()),
            Some("Bearer sk-test")
        );
    }

    // No authorization header when api_key is None.
    #[tokio::test]
    async fn show_no_bearer_when_no_api_key() {
        let server = MockServer::start().await;
        mount_show(
            &server,
            json!({"general.architecture": "qwen35moe", "qwen35moe.context_length": 262_144_u32}),
        )
        .await;

        resolve_context_length(&server.uri(), "qwen3.6:35b", None)
            .await
            .expect("resolve ok");

        let received = server.received_requests().await.expect("requests captured");
        let r: &Request = &received[0];
        assert!(
            r.headers.get("authorization").is_none(),
            "no api key => no Authorization header"
        );
    }

    // Pinned decoy regression: gpt-oss shape. The decoy key
    // gptoss.rope.scaling.original_context_length must NOT be returned.
    #[tokio::test]
    async fn show_gptoss_decoy_regression() {
        let server = MockServer::start().await;
        mount_show(
            &server,
            json!({
                "general.architecture": "gptoss",
                "gptoss.context_length": 131_072_u32,
                "gptoss.rope.scaling.original_context_length": 4096_u32
            }),
        )
        .await;

        let r = resolve_context_length(&server.uri(), "gpt-oss:20b", None)
            .await
            .expect("resolve ok");
        assert_eq!(r.value, 131_072);
        assert_eq!(r.architecture, "gptoss");
        assert_eq!(r.key, "gptoss.context_length");
    }

    // Pinned qwen35moe shape.
    #[tokio::test]
    async fn show_qwen35moe_pinned() {
        let server = MockServer::start().await;
        mount_show(
            &server,
            json!({
                "general.architecture": "qwen35moe",
                "qwen35moe.context_length": 262_144_u32
            }),
        )
        .await;

        let r = resolve_context_length(&server.uri(), "qwen3.6:35b", None)
            .await
            .expect("resolve ok");
        assert_eq!(r.value, 262_144);
        assert_eq!(r.architecture, "qwen35moe");
        assert_eq!(r.key, "qwen35moe.context_length");
    }

    // 404 from /api/show yields Status { status: 404 }, and Display contains
    // the model id, url, and "404".
    #[tokio::test]
    async fn show_404_yields_status_error() {
        let server = MockServer::start().await;
        mount_show_error(&server, 404, r#"{"error":"model 'x' not found"}"#).await;

        let err = resolve_context_length(&server.uri(), "x", None)
            .await
            .expect_err("must fail");
        assert!(matches!(
            err.kind,
            ContextLengthErrorKind::Status { status: 404, .. }
        ));
        let display = err.to_string();
        assert!(display.contains('x'), "should contain model id");
        assert!(display.contains("404"), "should contain status code");
        // url is in the error struct; to_string() goes through the #[error] format
        assert!(display.contains(&server.uri()) || display.contains("api/show"));
    }

    // Unreachable endpoint yields Transport, never a panic.
    #[tokio::test]
    async fn show_unreachable_yields_transport_error() {
        // Port 1 on localhost requires root and will always be refused —
        // the canonical closed-port trick (mirrors existing
        // `connect_refused_maps_to_transient_network`).
        let err = resolve_context_length("http://127.0.0.1:1", "model", None)
            .await
            .expect_err("must fail");
        assert!(
            matches!(err.kind, ContextLengthErrorKind::Transport(_)),
            "expected Transport, got {:?}",
            err.kind
        );
    }

    // (a) 200 body that is not JSON.
    #[tokio::test]
    async fn show_malformed_not_json() {
        let server = MockServer::start().await;
        mount_show_raw(&server, "not json at all").await;
        let err = resolve_context_length(&server.uri(), "m", None)
            .await
            .expect_err("must fail");
        match &err.kind {
            ContextLengthErrorKind::Malformed(msg) => {
                assert!(msg.contains("body not JSON"), "got: {msg}");
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    // (b) JSON with no model_info key.
    #[tokio::test]
    async fn show_malformed_no_model_info() {
        let server = MockServer::start().await;
        mount_show_raw(&server, r#"{"other_field": 1}"#).await;
        let err = resolve_context_length(&server.uri(), "m", None)
            .await
            .expect_err("must fail");
        match &err.kind {
            ContextLengthErrorKind::Malformed(msg) => {
                assert!(msg.contains("model_info missing"), "got: {msg}");
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    // (c) model_info present but not an object.
    #[tokio::test]
    async fn show_malformed_model_info_not_object() {
        let server = MockServer::start().await;
        mount_show_raw(&server, r#"{"model_info": []}"#).await;
        let err = resolve_context_length(&server.uri(), "m", None)
            .await
            .expect_err("must fail");
        match &err.kind {
            ContextLengthErrorKind::Malformed(msg) => {
                assert!(msg.contains("model_info not an object"), "got: {msg}");
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    // (d) model_info object but no general.architecture.
    #[tokio::test]
    async fn show_malformed_no_general_architecture() {
        let server = MockServer::start().await;
        mount_show_raw(&server, r#"{"model_info": {"other.key": 1}}"#).await;
        let err = resolve_context_length(&server.uri(), "m", None)
            .await
            .expect_err("must fail");
        match &err.kind {
            ContextLengthErrorKind::Malformed(msg) => {
                assert!(
                    msg.contains("model_info.general.architecture missing"),
                    "got: {msg}"
                );
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    // (e) general.architecture present but not a string.
    #[tokio::test]
    async fn show_malformed_architecture_not_string() {
        let server = MockServer::start().await;
        mount_show_raw(&server, r#"{"model_info": {"general.architecture": 42}}"#).await;
        let err = resolve_context_length(&server.uri(), "m", None)
            .await
            .expect_err("must fail");
        match &err.kind {
            ContextLengthErrorKind::Malformed(msg) => {
                assert!(
                    msg.contains("model_info.general.architecture not a string"),
                    "got: {msg}"
                );
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    // (f) context_length key absent.
    #[tokio::test]
    async fn show_malformed_context_length_missing() {
        let server = MockServer::start().await;
        mount_show_raw(
            &server,
            r#"{"model_info": {"general.architecture": "myarch"}}"#,
        )
        .await;
        let err = resolve_context_length(&server.uri(), "m", None)
            .await
            .expect_err("must fail");
        match &err.kind {
            ContextLengthErrorKind::Malformed(msg) => {
                assert!(
                    msg.contains("model_info.myarch.context_length missing"),
                    "got: {msg}"
                );
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    // (g) context_length is -1 (not representable as u32).
    #[tokio::test]
    async fn show_malformed_context_length_negative() {
        let server = MockServer::start().await;
        mount_show_raw(
            &server,
            r#"{"model_info": {"general.architecture": "myarch", "myarch.context_length": -1}}"#,
        )
        .await;
        let err = resolve_context_length(&server.uri(), "m", None)
            .await
            .expect_err("must fail");
        match &err.kind {
            ContextLengthErrorKind::Malformed(msg) => {
                assert!(
                    msg.contains("model_info.myarch.context_length not a u32"),
                    "got: {msg}"
                );
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    // (h) context_length is a string "262144" (not a number).
    #[tokio::test]
    async fn show_malformed_context_length_string() {
        let server = MockServer::start().await;
        mount_show_raw(
            &server,
            r#"{"model_info": {"general.architecture": "myarch", "myarch.context_length": "262144"}}"#,
        )
        .await;
        let err = resolve_context_length(&server.uri(), "m", None)
            .await
            .expect_err("must fail");
        match &err.kind {
            ContextLengthErrorKind::Malformed(msg) => {
                assert!(
                    msg.contains("model_info.myarch.context_length not a u32"),
                    "got: {msg}"
                );
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    // (i) context_length is 4294967296 (exceeds u32::MAX).
    #[tokio::test]
    async fn show_malformed_context_length_too_large() {
        let server = MockServer::start().await;
        mount_show_raw(
            &server,
            r#"{"model_info": {"general.architecture": "myarch", "myarch.context_length": 4294967296}}"#,
        )
        .await;
        let err = resolve_context_length(&server.uri(), "m", None)
            .await
            .expect_err("must fail");
        match &err.kind {
            ContextLengthErrorKind::Malformed(msg) => {
                assert!(
                    msg.contains("model_info.myarch.context_length not a u32"),
                    "got: {msg}"
                );
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    // (j) context_length is 0 — Malformed, not Ok(0).
    #[tokio::test]
    async fn show_malformed_context_length_zero() {
        let server = MockServer::start().await;
        mount_show_raw(
            &server,
            r#"{"model_info": {"general.architecture": "myarch", "myarch.context_length": 0}}"#,
        )
        .await;
        let err = resolve_context_length(&server.uri(), "m", None)
            .await
            .expect_err("must fail");
        match &err.kind {
            ContextLengthErrorKind::Malformed(msg) => {
                assert!(
                    msg.contains("model_info.myarch.context_length is zero"),
                    "got: {msg}"
                );
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    // Trailing slash on base_url is tolerated — resolves against /api/show
    // with a single slash.
    #[tokio::test]
    async fn show_trailing_slash_base_url_is_tolerated() {
        let server = MockServer::start().await;
        mount_show(
            &server,
            json!({
                "general.architecture": "qwen35moe",
                "qwen35moe.context_length": 262_144_u32
            }),
        )
        .await;

        let base_with_slash = format!("{}/", server.uri());
        let r = resolve_context_length(&base_with_slash, "qwen3.6:35b", None)
            .await
            .expect("resolve ok");
        assert_eq!(r.value, 262_144);
    }

    // Below-floor value (8192) is returned verbatim — no clamp.
    #[tokio::test]
    async fn show_below_floor_value_returned_verbatim() {
        let server = MockServer::start().await;
        mount_show(
            &server,
            json!({
                "general.architecture": "qwen3",
                "qwen3.context_length": 8192_u32
            }),
        )
        .await;

        let r = resolve_context_length(&server.uri(), "qwen3:1b", None)
            .await
            .expect("resolve ok");
        assert_eq!(r.value, 8192);
        assert_eq!(r.architecture, "qwen3");
        assert_eq!(r.key, "qwen3.context_length");
    }

    // ---- resolve_num_ctx: the shared five-branch precedence --------------

    // (0) set-but-empty / whitespace-only is UNSET, not a parse error.
    #[tokio::test]
    async fn num_ctx_empty_env_is_unset_not_parse_error() {
        for raw in ["", "   "] {
            let r = resolve_num_ctx("https://ollama.com", "m", None, Some(raw))
                .await
                .expect("empty env value must fall through, not Err");
            assert_eq!(r.value, None);
            assert_eq!(r.source, NumCtxSource::Default);
            assert_eq!(r.desc, "num_ctx=default");
            assert!(r.warning.is_none());
        }
    }

    // (1) explicit valid u32 is used verbatim. No mock is mounted: an
    // accidental probe would 404 and yield Err, so Ok proves no probe fired.
    #[tokio::test]
    async fn num_ctx_explicit_value_is_used_without_probe() {
        let server = MockServer::start().await;
        let r = resolve_num_ctx(&server.uri(), "m", None, Some("65536"))
            .await
            .expect("explicit value must succeed without a probe");
        assert_eq!(r.value, Some(65_536));
        assert_eq!(r.source, NumCtxSource::Explicit);
        assert_eq!(r.desc, "num_ctx=65536 (explicit OLLAMA_NUM_CTX)");
        assert!(r.warning.is_none());
    }

    // (2) non-u32 env value is a parse error naming the UNTRIMMED string.
    #[tokio::test]
    async fn num_ctx_bad_env_is_parse_error_with_untrimmed_raw() {
        for raw in ["not-a-number", " 12x "] {
            let err = resolve_num_ctx("https://ollama.com", "m", None, Some(raw))
                .await
                .expect_err("non-u32 env value must be Err");
            assert!(
                err.contains(raw),
                "error must name the raw string `{raw}`: {err}"
            );
            assert!(
                err.contains("OLLAMA_NUM_CTX"),
                "error must name the var: {err}"
            );
        }
    }

    // (3) unset + local + probe >= floor -> Probe, no warning.
    #[tokio::test]
    async fn num_ctx_local_probe_above_floor_is_probe() {
        let server = MockServer::start().await;
        mount_show(
            &server,
            json!({"general.architecture": "qwen35moe", "qwen35moe.context_length": 262_144_u32}),
        )
        .await;
        let r = resolve_num_ctx(&server.uri(), "qwen3.6:35b", None, None)
            .await
            .expect("probe must resolve");
        assert_eq!(r.value, Some(262_144));
        assert_eq!(r.source, NumCtxSource::Probe);
        assert!(r.desc.contains("resolved: arch="), "desc: {}", r.desc);
        assert!(r.desc.contains("key="), "desc: {}", r.desc);
        assert!(r.warning.is_none());
    }

    // (4) unset + local + probe < floor -> Probe + warning.
    #[tokio::test]
    async fn num_ctx_local_probe_below_floor_warns() {
        let server = MockServer::start().await;
        mount_show(
            &server,
            json!({"general.architecture": "qwen3", "qwen3.context_length": 8_192_u32}),
        )
        .await;
        let r = resolve_num_ctx(&server.uri(), "qwen3:1b", None, None)
            .await
            .expect("below floor is data, not a failure");
        assert_eq!(r.value, Some(8_192));
        assert_eq!(r.source, NumCtxSource::Probe);
        assert!(r.desc.contains("BELOW-FLOOR"), "desc: {}", r.desc);
        let w = r.warning.expect("below-floor must carry a warning");
        assert!(
            w.contains(&format!("BELOW the {MIN_EXPECTED_NUM_CTX} sanity floor")),
            "warning: {w}"
        );
        assert!(w.contains("set OLLAMA_NUM_CTX to override"), "warning: {w}");
    }

    // (5) unset + non-local -> Default, no probe, no HTTP.
    #[tokio::test]
    async fn num_ctx_non_local_unset_is_default() {
        let r = resolve_num_ctx("https://ollama.com", "m", None, None)
            .await
            .expect("non-local unset must succeed");
        assert_eq!(r.value, None);
        assert_eq!(r.source, NumCtxSource::Default);
        assert_eq!(r.desc, "num_ctx=default");
        assert!(r.warning.is_none());
    }

    // err: probe failure propagates as Err naming BOTH the model and the url.
    #[tokio::test]
    async fn num_ctx_probe_failure_is_err_naming_model_and_url() {
        let server = MockServer::start().await;
        mount_show_error(&server, 404, "{}").await;
        let err = resolve_num_ctx(&server.uri(), "m", None, None)
            .await
            .expect_err("probe failure must be Err, never a fallback");
        assert!(err.contains('m'), "must name the model: {err}");
        assert!(
            err.contains(&server.uri()),
            "must name the server uri: {err}"
        );
    }

    // is_local_ollama_url: the single probe gate.
    #[test]
    fn is_local_ollama_url_gate() {
        assert!(is_local_ollama_url("http://localhost:11434"));
        assert!(is_local_ollama_url("http://127.0.0.1:11434"));
        assert!(!is_local_ollama_url("https://ollama.com"));
        assert!(!is_local_ollama_url("http://jason-desktop:11434"));
    }

    // NumCtxSource::as_str: the stable string mapping.
    #[test]
    fn num_ctx_source_as_str_mapping() {
        assert_eq!(NumCtxSource::Explicit.as_str(), "explicit");
        assert_eq!(NumCtxSource::Probe.as_str(), "probe");
        assert_eq!(NumCtxSource::Default.as_str(), "default");
        let r = NumCtxResolution {
            value: Some(1),
            source: NumCtxSource::Default,
            desc: "num_ctx=default".to_string(),
            warning: None,
        };
        assert_eq!(r.source.as_str(), "default");
    }
}
