//! Event-bus data model — a coding-eval fixture.
//!
//! Fully implemented (no stubs). `SubId` is the monotonic, never-reused
//! subscription identifier allocated by [`crate::registry::Registry`].
//! [`Event`] is the unit published to the bus.

/// A monotonic, never-reused subscription identifier.
///
/// Ids are allocated by `Registry::subscribe` starting at 1 and strictly
/// increasing; an id is never recycled, even after the subscriber is
/// removed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubId(pub u64);

/// An event published to the bus: a topic string plus an integer payload.
///
/// `publish` delivers `event` to every subscriber of `event.topic` and
/// records `(SubId, event.payload)` entries in the delivery log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event {
    pub topic: String,
    pub payload: i64,
}
