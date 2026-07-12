//! Sealed holdout tests for the event bus (`publish` + `unsubscribe`).
//!
//! These tests cover ordered delivery via `delivery_log()`, the delivered-
//! count return value, no-subscriber and topic-isolation cases, and
//! unsubscribe order preservation. They live under `holdout/` in the
//! committed source and are copied into the crate's `tests/` directory only
//! when verifying solvability / running the eval:
//!
//!   cp -r fixtures/eventbus/holdout/. fixtures/eventbus/
//!   cd fixtures/eventbus && cargo test
//!
//! All tests are green only under correct `publish` (bus.rs) AND
//! `unsubscribe` (registry.rs) implementations.

use eventbus::{Event, EventBus, SubId};

/// `publish` delivers to every subscriber of the event's topic IN SUBSCRIPTION
/// ORDER and returns the delivered count.
#[test]
fn publish_delivers_in_subscription_order_and_returns_count() {
    let mut bus = EventBus::new();
    let s1 = bus.subscribe("news");
    let s2 = bus.subscribe("news");
    let s3 = bus.subscribe("news");
    let n = bus.publish(&Event { topic: "news".to_string(), payload: 42 });
    assert_eq!(n, 3, "publish returns the delivered count");
    assert_eq!(
        bus.delivery_log(),
        &[
            (SubId(s1.0), 42),
            (SubId(s2.0), 42),
            (SubId(s3.0), 42),
        ],
        "delivery log must record one entry per subscriber in subscription order"
    );
}

/// Publishing to a topic with no subscribers returns 0 and appends nothing.
#[test]
fn publish_to_empty_topic_returns_zero_and_appends_nothing() {
    let mut bus = EventBus::new();
    let _ = bus.subscribe("news");
    let n = bus.publish(&Event { topic: "weather".to_string(), payload: 7 });
    assert_eq!(n, 0, "no subscribers → delivered count 0");
    assert!(bus.delivery_log().is_empty(), "no delivery log entries");
}

/// Distinct topics are isolated: a publish to topic A never delivers to a
/// B-subscriber.
#[test]
fn topics_are_isolated() {
    let mut bus = EventBus::new();
    let _ = bus.subscribe("a");
    let b_sub = bus.subscribe("b");
    let n = bus.publish(&Event { topic: "a".to_string(), payload: 100 });
    assert_eq!(n, 1, "only the a-subscriber is delivered to");
    assert!(
        !bus
            .delivery_log()
            .iter()
            .any(|(id, _)| *id == b_sub),
        "publish to topic a must not deliver to a b-subscriber"
    );
}

/// Unsubscribe of a known id returns true and removes exactly that subscriber
/// while PRESERVING the relative order of survivors (verified by unsubscribing
/// a MIDDLE subscriber, then checking subsequent delivery order).
#[test]
fn unsubscribe_middle_preserves_survivor_order() {
    let mut bus = EventBus::new();
    let _ = bus.subscribe("news"); // SubId(1)
    let s2 = bus.subscribe("news"); // SubId(2)
    let _ = bus.subscribe("news"); // SubId(3)
    let _ = bus.subscribe("news"); // SubId(4)
    // Unsubscribe the MIDDLE subscriber (s2).
    assert!(bus.unsubscribe(s2), "known id → true");
    // Publish and verify the surviving subscribers are delivered in order.
    let n = bus.publish(&Event { topic: "news".to_string(), payload: 9 });
    assert_eq!(n, 3, "one subscriber removed");
    assert_eq!(
        bus.delivery_log(),
        &[
            (SubId(1), 9),
            (SubId(3), 9),
            (SubId(4), 9),
        ],
        "surviving subscribers must preserve their original relative order"
    );
}

/// Unsubscribe of an unknown id returns false and changes nothing.
#[test]
fn unsubscribe_unknown_id_returns_false_and_no_op() {
    let mut bus = EventBus::new();
    let _ = bus.subscribe("news"); // SubId(1)
    assert!(!bus.unsubscribe(SubId(999)), "unknown id → false");
    // Publish still delivers to the one remaining subscriber.
    let n = bus.publish(&Event { topic: "news".to_string(), payload: 5 });
    assert_eq!(n, 1, "subscriber count unchanged");
    assert_eq!(bus.delivery_log(), &[(SubId(1), 5)]);
}

/// A re-subscribe after unsubscribe gets a fresh, never-reused id and
/// receives subsequent events.
#[test]
fn resubscribe_after_unsubscribe_gets_fresh_id() {
    let mut bus = EventBus::new();
    let s1 = bus.subscribe("news"); // SubId(1)
    let _ = bus.subscribe("news"); // SubId(2)
    assert!(bus.unsubscribe(s1), "remove SubId(1)");
    // Re-subscribe: the new id must be 3 (never reuse 1 or 2).
    let s3 = bus.subscribe("news");
    assert_eq!(s3, SubId(3), "re-subscribe must get a fresh never-reused id");
    let n = bus.publish(&Event { topic: "news".to_string(), payload: 8 });
    assert_eq!(n, 2, "SubId(2) and SubId(3) receive the event");
    assert_eq!(bus.delivery_log(), &[(SubId(2), 8), (SubId(3), 8)]);
}