// NOTES: Tier 7 / hard — multi-file dispatch-scale

//! A multi-file pub/sub event bus — a coding-eval fixture.
//!
//! This crate is committed in a DELIBERATELY HALF-BUILT state across TWO
//! files: `src/registry.rs` has `unsubscribe` as a `todo!()` stub and
//! `src/bus.rs` has `publish` as a `todo!()` stub. The visible tests below
//! exercise only the already-implemented surface (`subscribe`,
//! `subscribers_for`, `delivery_log`). The sealed holdout suite is the sole
//! grader for the stubbed work.
//!
//! The two stubs sit in different files, so a correct solution is a coherent
//! cross-file edit: the delivery log lives in `bus.rs`, the subscription ids
//! live in `registry.rs`, and the `SubId` newtype lives in `model.rs`.

pub mod bus;
pub mod model;
pub mod registry;

pub use bus::EventBus;
pub use model::{Event, SubId};
pub use registry::Registry;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subscribe_returns_strictly_increasing_ids() {
        let mut bus = EventBus::new();
        let a = bus.subscribe("topic");
        let b = bus.subscribe("topic");
        let c = bus.subscribe("other");
        assert_eq!(a, SubId(1));
        assert_eq!(b, SubId(2));
        assert_eq!(c, SubId(3));
    }

    #[test]
    fn fresh_bus_has_empty_delivery_log() {
        let bus = EventBus::new();
        assert!(bus.delivery_log().is_empty());
    }
}
