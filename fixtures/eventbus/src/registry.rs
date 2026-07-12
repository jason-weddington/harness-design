//! Subscription registry — maps topics to ordered subscriber lists.

use crate::model::SubId;
use std::collections::HashMap;

/// A registry of subscriptions: maps each topic to the ordered list of
/// subscriber ids that subscribed to it.
///
/// `subscribe` and `subscribers_for` are implemented; `unsubscribe` is a
/// `todo!()` stub whose contract is described in its doc comment and in
/// `task.json`.
pub struct Registry {
    /// Next id to allocate; incremented before assignment so ids start at 1.
    next_id: u64,
    /// topic → subscriber ids in subscription (append) order.
    by_topic: HashMap<String, Vec<SubId>>,
}

impl Registry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            next_id: 0,
            by_topic: HashMap::new(),
        }
    }

    /// Subscribe to `topic`. Allocates a fresh, monotonic, never-reused
    /// `SubId` (ids start at 1) and appends it to the topic's subscriber list.
    /// Returns the assigned id.
    pub fn subscribe(&mut self, topic: &str) -> SubId {
        self.next_id += 1;
        let id = SubId(self.next_id);
        self.by_topic
            .entry(topic.to_string())
            .or_default()
            .push(id.clone());
        id
    }

    /// Return the subscriber ids for `topic` in subscription order.
    pub fn subscribers_for(&self, topic: &str) -> Vec<SubId> {
        self.by_topic.get(topic).cloned().unwrap_or_default()
    }

    /// Unsubscribe the subscriber with the given `id`.
    ///
    /// Returns `true` and removes exactly that subscriber if the id is known
    /// (preserving the relative order of surviving subscribers). Returns
    /// `false` and changes nothing if the id is unknown.
    #[allow(unused_variables)]
    pub fn unsubscribe(&mut self, id: SubId) -> bool {
        todo!()
    }
}
