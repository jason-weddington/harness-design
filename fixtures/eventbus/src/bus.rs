//! The event bus — publishes events to registered subscribers and records
//! each delivery in an append-only log.

use crate::model::{Event, SubId};
use crate::registry::Registry;

/// An in-memory pub/sub event bus.
///
/// Wraps a [`Registry`] for subscription management and maintains a
/// `delivery_log` recording every delivered `(SubId, payload)` entry in
/// delivery order.
///
/// `subscribe`, `subscribers_for`, `delivery_log()`, and `unsubscribe` are
/// implemented; `publish` is a `todo!()` stub whose contract is described in
/// its doc comment and in `task.json`.
pub struct EventBus {
    registry: Registry,
    delivery_log: Vec<(SubId, i64)>,
}

impl EventBus {
    /// Create an empty event bus.
    pub fn new() -> Self {
        Self {
            registry: Registry::new(),
            delivery_log: Vec::new(),
        }
    }

    /// Subscribe to `topic` on this bus; returns the assigned `SubId`.
    pub fn subscribe(&mut self, topic: &str) -> SubId {
        self.registry.subscribe(topic)
    }

    /// Return the subscriber ids for `topic` in subscription order.
    pub fn subscribers_for(&self, topic: &str) -> Vec<SubId> {
        self.registry.subscribers_for(topic)
    }

    /// Return the delivery log as a slice of `(SubId, payload)` entries in
    /// delivery order.
    pub fn delivery_log(&self) -> &[(SubId, i64)] {
        &self.delivery_log
    }

    /// Publish `event` to every subscriber of `event.topic`, in subscription
    /// order.
    ///
    /// Appends exactly one `(SubId, event.payload)` entry to the delivery log
    /// per delivered subscriber and returns the delivered count. Publishing to
    /// a topic with no subscribers returns 0 and appends nothing.
    #[allow(unused_variables)]
    pub fn publish(&mut self, event: &Event) -> usize {
        todo!()
    }

    /// Unsubscribe the subscriber with the given `id` from this bus. Returns
    /// `true` if the id was known and removed, `false` otherwise.
    pub fn unsubscribe(&mut self, id: SubId) -> bool {
        self.registry.unsubscribe(id)
    }
}
