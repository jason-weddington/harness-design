//! Crate-wide, **test-only** support utilities.
//!
//! This module is compiled only under `#[cfg(test)]`. It exists so that the
//! scripted [`MockBackend`] can be shared across modules' test suites (the
//! engine loop here, and Item F's eval harness later) without each re-deriving
//! a fake backend. Nothing here ships in a release build.

use std::collections::VecDeque;
use std::sync::Mutex;

use async_trait::async_trait;

use crate::model::{AssistantTurn, BackendError, Message, ModelBackend, TerminalKind, TurnRequest};

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
/// **Test-only:** the whole module is `#[cfg(test)]`, so this type never exists
/// in a non-test build.
pub(crate) struct MockBackend {
    script: Mutex<VecDeque<Result<AssistantTurn, BackendError>>>,
    calls: Mutex<u32>,
    /// Snapshot of the `messages` slice passed to the most recent `turn`
    /// call — lets a test assert on the history the loop actually built.
    last_messages: Mutex<Vec<Message>>,
}

impl MockBackend {
    /// Build a backend from an explicit sequence of per-turn outcomes
    /// (`Ok(turn)` or `Err(backend_error)`), consumed front-to-back.
    pub(crate) fn new(script: Vec<Result<AssistantTurn, BackendError>>) -> Self {
        Self {
            script: Mutex::new(script.into()),
            calls: Mutex::new(0),
            last_messages: Mutex::new(Vec::new()),
        }
    }

    /// Convenience constructor for the common all-success case: every scripted
    /// turn is wrapped in `Ok`.
    pub(crate) fn from_turns(turns: Vec<AssistantTurn>) -> Self {
        Self::new(turns.into_iter().map(Ok).collect())
    }

    /// How many times [`ModelBackend::turn`] has been called so far — used by
    /// tests to assert the iteration count.
    pub(crate) fn calls(&self) -> u32 {
        *self.calls.lock().expect("calls lock poisoned")
    }

    /// The `messages` the loop sent on the most recent `turn` call.
    pub(crate) fn last_messages(&self) -> Vec<Message> {
        self.last_messages
            .lock()
            .expect("last_messages lock poisoned")
            .clone()
    }
}

#[async_trait]
impl ModelBackend for MockBackend {
    async fn turn(&self, req: &TurnRequest<'_>) -> Result<AssistantTurn, BackendError> {
        *self.calls.lock().expect("calls lock poisoned") += 1;
        *self
            .last_messages
            .lock()
            .expect("last_messages lock poisoned") = req.messages.to_vec();
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
