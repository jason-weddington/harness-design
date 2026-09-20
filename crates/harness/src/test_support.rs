//! Crate-wide support utilities for TEST code.
//!
//! This module is compiled under `#[cfg(test)]` OR the default-off, dev-only
//! `test-support` feature (see `crates/harness/Cargo.toml`) — so a consumer
//! crate's dev-test target can drive the same scripted fakes this crate's own
//! suite does. It is compiled in NEITHER shape by a default-featured build, so
//! nothing here ships in a release build.
//!
//! It exists so that the scripted [`MockBackend`] can be shared across test
//! suites without each re-deriving a fake backend.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Mutex;

use async_trait::async_trait;

use crate::exec::{ChangeObserver, TreeObservation};
use crate::model::{
    AssistantTurn, BackendError, Message, ModelBackend, OutputCapResolution, TerminalKind,
    TurnRequest,
};

/// A scripted [`ModelBackend`] for tests.
///
/// It replays a pre-set queue of per-turn outcomes — each
/// `Result<AssistantTurn, BackendError>` is handed back, in order, by one call
/// to [`ModelBackend::turn`]. This lets a test drive the agent loop through an
/// exact trajectory (tool calls, plain text, errors) with no network.
///
/// If the loop draws more turns than were scripted (an "over-draw"), `turn`
/// returns a terminal [`BackendError`] rather than silently looping — so a test
/// that miscounts iterations fails loudly instead of hanging.
///
/// **Test-only:** the module is compiled under `#[cfg(test)]` or the
/// default-off `test-support` feature and never in a default-featured build,
/// so this type does not exist in a production build.
pub struct MockBackend {
    script: Mutex<VecDeque<Result<AssistantTurn, BackendError>>>,
    calls: Mutex<u32>,
    /// Snapshot of the `messages` slice passed to the most recent `turn`
    /// call — lets a test assert on the history the loop actually built.
    last_messages: Mutex<Vec<Message>>,
    /// One entry per `turn` call, in order, capturing the `system` prompt
    /// the loop sent that turn (owned copy). Tests use this to prove the
    /// engine renders the system prompt ONCE and re-sends the byte-identical
    /// string every iteration — a prompt-cache correctness invariant.
    systems_seen: Mutex<Vec<Option<String>>>,
    /// One entry per `turn` call, in order, capturing the full `messages`
    /// slice passed that turn. Unlike [`Self::last_messages`] (which
    /// overwrites on every call), this accumulates — so tests can assert on
    /// the messages the loop sent on the FIRST turn of a multi-turn script,
    /// which is the key assertion for crash-resume and fresh-context resume.
    messages_seen: Mutex<Vec<Vec<Message>>>,
    /// One entry per `turn` call, in order, capturing the FULL `req.tools`
    /// slice (owned copies of the JSON schemas) the loop sent that call —
    /// lets tests assert on the tool union actually advertised to the model,
    /// which is the output-capability boundary the loop controls.
    tools_seen: Mutex<Vec<Vec<serde_json::Value>>>,
    /// One entry per `turn` call, in order, capturing the
    /// `req.params.max_tokens` the loop sent that call — lets tests pin the
    /// per-iteration output-cap resolution.
    params_seen: Mutex<Vec<u32>>,
    /// How many times [`ModelBackend::output_cap`] has been called.
    output_cap_calls: Mutex<u32>,
    /// Optional test override for [`ModelBackend::output_cap`]. `None`
    /// (the default) inherits the trait fallback.
    output_cap_override: Option<Box<dyn Fn(Option<u32>) -> OutputCapResolution + Send + Sync>>,
    /// Optional test override for [`ModelBackend::context_limit`]. `None`
    /// (the default) inherits the trait's `None` default — the same gate
    /// that keeps compaction off for every real backend that advertises no
    /// limit, so engine tests arm/disarm the compaction path without a live
    /// daemon.
    context_limit_override: Option<u32>,
}

impl MockBackend {
    /// Build a backend from an explicit sequence of per-turn outcomes
    /// (`Ok(turn)` or `Err(backend_error)`), consumed front-to-back.
    pub fn new(script: Vec<Result<AssistantTurn, BackendError>>) -> Self {
        Self {
            script: Mutex::new(script.into()),
            calls: Mutex::new(0),
            last_messages: Mutex::new(Vec::new()),
            systems_seen: Mutex::new(Vec::new()),
            messages_seen: Mutex::new(Vec::new()),
            tools_seen: Mutex::new(Vec::new()),
            params_seen: Mutex::new(Vec::new()),
            output_cap_calls: Mutex::new(0),
            output_cap_override: None,
            context_limit_override: None,
        }
    }

    /// Convenience constructor for the common all-success case: every scripted
    /// turn is wrapped in `Ok`.
    pub fn from_turns(turns: Vec<AssistantTurn>) -> Self {
        Self::new(turns.into_iter().map(Ok).collect())
    }

    /// How many times [`ModelBackend::turn`] has been called so far — used by
    /// tests to assert the iteration count.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned (another test panicked while
    /// holding it — a fail-loudly, test-only posture).
    pub fn calls(&self) -> u32 {
        *self.calls.lock().expect("calls lock poisoned")
    }

    /// The `messages` the loop sent on the most recent `turn` call.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned (another test panicked while
    /// holding it — a fail-loudly, test-only posture).
    pub fn last_messages(&self) -> Vec<Message> {
        self.last_messages
            .lock()
            .expect("last_messages lock poisoned")
            .clone()
    }

    /// One entry per `turn` call, in order: the `system` prompt string the
    /// loop sent that turn. Tests assert every entry is equal to prove the
    /// engine renders the prompt exactly once and re-sends byte-identical
    /// bytes.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned (another test panicked while
    /// holding it — a fail-loudly, test-only posture).
    pub fn systems_seen(&self) -> Vec<Option<String>> {
        self.systems_seen
            .lock()
            .expect("systems_seen lock poisoned")
            .clone()
    }

    /// One entry per `turn` call, in order: the full `messages` slice the
    /// loop sent that turn. Unlike [`Self::last_messages`], this accumulates
    /// across turns so tests can assert on the FIRST turn's messages (the
    /// key assertion for crash-resume and fresh-context resume).
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned (another test panicked while
    /// holding it — a fail-loudly, test-only posture).
    #[must_use]
    pub fn messages_seen(&self) -> Vec<Vec<Message>> {
        self.messages_seen
            .lock()
            .expect("messages_seen lock poisoned")
            .clone()
    }

    /// One entry per `turn` call, in order: the `tools` slice the loop
    /// advertised that turn (owned JSON schema copies). Tests assert on this
    /// to pin the tool union — including `vec![]` for a turn where NOTHING
    /// must be callable.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned (another test panicked while
    /// holding it — a fail-loudly, test-only posture).
    #[must_use]
    pub fn tools_seen(&self) -> Vec<Vec<serde_json::Value>> {
        self.tools_seen
            .lock()
            .expect("tools_seen lock poisoned")
            .clone()
    }

    /// One entry per `turn` call, in order: the `req.params.max_tokens` the
    /// loop sent that call. Tests assert on this to pin the per-iteration
    /// output-cap resolution (derived caps move turn to turn).
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned (another test panicked while
    /// holding it — a fail-loudly, test-only posture).
    pub fn params_seen(&self) -> Vec<u32> {
        self.params_seen
            .lock()
            .expect("params_seen lock poisoned")
            .clone()
    }

    /// How many times [`ModelBackend::output_cap`] has been called on this
    /// mock — zero proves the operator override short-circuits resolution.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned (another test panicked while
    /// holding it — a fail-loudly, test-only posture).
    pub fn output_cap_calls(&self) -> u32 {
        *self
            .output_cap_calls
            .lock()
            .expect("output_cap_calls lock poisoned")
    }

    /// Script [`ModelBackend::output_cap`] with a closure over the
    /// `prompt_tokens` argument. Without this, the mock inherits the trait
    /// fallback resolution.
    #[must_use]
    pub fn with_output_cap_override(
        mut self,
        f: Box<dyn Fn(Option<u32>) -> OutputCapResolution + Send + Sync>,
    ) -> Self {
        self.output_cap_override = Some(f);
        self
    }

    /// Arm [`ModelBackend::context_limit`] with a pinned window — the
    /// mocked equivalent of `OllamaBackend::with_num_ctx`. Without this, the
    /// mock inherits the `None` default and the engine's compaction path is
    /// off (exactly as it is for a real unpinned backend).
    #[must_use]
    pub fn with_context_limit_override(mut self, limit: u32) -> Self {
        self.context_limit_override = Some(limit);
        self
    }
}

#[async_trait]
impl ModelBackend for MockBackend {
    fn output_cap(&self, prompt_tokens: Option<u32>) -> OutputCapResolution {
        *self
            .output_cap_calls
            .lock()
            .expect("output_cap_calls lock poisoned") += 1;
        match &self.output_cap_override {
            Some(f) => f(prompt_tokens),
            None => crate::model::OutputCapResolution {
                max_tokens: crate::model::DEFAULT_MAX_TOKENS,
                source: crate::model::MaxTokensSource::Fallback,
            },
        }
    }

    fn context_limit(&self) -> Option<u32> {
        self.context_limit_override
    }

    async fn turn(&self, req: &TurnRequest<'_>) -> Result<AssistantTurn, BackendError> {
        *self.calls.lock().expect("calls lock poisoned") += 1;
        self.params_seen
            .lock()
            .expect("params_seen lock poisoned")
            .push(req.params.max_tokens);
        let msgs = req.messages.to_vec();
        self.last_messages
            .lock()
            .expect("last_messages lock poisoned")
            .clone_from(&msgs);
        self.systems_seen
            .lock()
            .expect("systems_seen lock poisoned")
            .push(req.system.map(str::to_string));
        self.messages_seen
            .lock()
            .expect("messages_seen lock poisoned")
            .push(msgs);
        self.tools_seen
            .lock()
            .expect("tools_seen lock poisoned")
            .push(req.tools.to_vec());
        let next = self
            .script
            .lock()
            .expect("script lock poisoned")
            .pop_front();
        next.unwrap_or_else(|| {
            Err(BackendError::Terminal {
                kind: TerminalKind::Other,
                message: "MockBackend script exhausted (over-drawn)".to_string(),
            })
        })
    }
}

/// A scripted [`ChangeObserver`] for tests — the observation-side twin of
/// [`MockBackend`].
///
/// It replays a pre-set queue of [`TreeObservation`]s — the first is consumed
/// by the engine's run-start baseline capture, the second (when scripted) by
/// the finish-time current-tree observation. This lets a test drive BOTH leg-3
/// sites deterministically through a provider with no filesystem and no
/// network, proving that both sides route through
/// [`crate::engine::RunConfig::with_change_observer`] rather than falling back
/// to the default [`crate::exec::GitTreeObserver`].
///
/// If the engine observes more often than scripted (an "over-draw"), `observe`
/// returns `TreeObservation::Unobservable` with a fixed reason rather than
/// panicking — mirroring [`MockBackend`]'s fail-loudly-over-draw posture.
///
/// **Test-only:** the module is compiled under `#[cfg(test)]` or the
/// default-off `test-support` feature and never in a default-featured build,
/// so this type does not exist in a production build.
#[derive(Debug)]
pub struct StubChangeObserver {
    script: Mutex<VecDeque<TreeObservation>>,
    calls: Mutex<u32>,
}

impl StubChangeObserver {
    /// Build an observer from an explicit sequence of observations, consumed
    /// front-to-back: the first feeds the baseline, the second the finish-time
    /// observation, and so on.
    pub fn new(script: Vec<TreeObservation>) -> Self {
        Self {
            script: Mutex::new(script.into()),
            calls: Mutex::new(0),
        }
    }

    /// How many times [`ChangeObserver::observe`] has been called so far.
    /// A test asserting `calls() == 2` (1 baseline + 1 finish) proves BOTH
    /// sides route through the provider.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned (another test panicked while
    /// holding it — a fail-loudly, test-only posture).
    pub fn calls(&self) -> u32 {
        *self.calls.lock().expect("calls lock poisoned")
    }
}

#[async_trait]
impl ChangeObserver for StubChangeObserver {
    async fn observe(&self, _root: &Path) -> TreeObservation {
        *self.calls.lock().expect("calls lock poisoned") += 1;
        self.script
            .lock()
            .expect("script lock poisoned")
            .pop_front()
            .unwrap_or_else(|| TreeObservation::Unobservable {
                reason: "StubChangeObserver script exhausted (over-drawn)".to_string(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ContentBlock, StopReason, Usage};

    /// `tools_seen` must capture the `req.tools` slice verbatim, one entry
    /// per `turn` call — the assertion seam a consumer crate uses to pin the
    /// tool union the loop advertises (including the empty union).
    #[tokio::test]
    async fn tools_seen_captures_the_advertised_tool_union_per_call() {
        let tools = vec![
            serde_json::json!({"name": "add_pointer", "input_schema": {}}),
            serde_json::json!({"name": "create_map", "input_schema": {}}),
        ];
        let turn = AssistantTurn {
            content: vec![ContentBlock::Text("hi".to_string())],
            stop_reason: StopReason::EndTurn,
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
                cache_read_tokens: None,
                cache_write_tokens: None,
                reasoning_tokens: None,
            },
        };
        let backend = MockBackend::from_turns(vec![turn]);
        let messages = [Message::User {
            content: vec![crate::model::UserBlock::Text("hi".to_string())],
        }];
        let params = crate::model::SamplingParams {
            max_tokens: 8,
            temperature: None,
            stop_sequences: vec![],
        };
        let request = TurnRequest {
            system: None,
            messages: &messages,
            tools: &tools,
            params: &params,
        };

        backend.turn(&request).await.expect("scripted turn");

        assert_eq!(backend.calls(), 1);
        let seen = backend.tools_seen();
        assert_eq!(seen.len(), 1, "one entry per turn call");
        assert_eq!(seen[0], tools, "the tool union is captured verbatim");
        assert_eq!(backend.params_seen(), vec![8]);
        assert_eq!(backend.systems_seen(), vec![None::<String>]);
    }
}
