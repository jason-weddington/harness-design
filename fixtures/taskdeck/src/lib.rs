//! A minimal task-tracker — a coding-eval fixture.
//!
//! This crate is committed in a DELIBERATELY HALF-BUILT state:
//! `src/commands.rs` contains two unimplemented stubs (`complete` and `purge`)
//! whose contracts are described in their doc comments. The eval harness presents
//! this crate to a model-under-test and judges whether it can finish the
//! implementation correctly.

pub mod model;
pub mod store;
pub mod commands;
pub mod archive;
