//! Concrete tools the agent can invoke.
//!
//! Each submodule ships one [`Tool`](crate::tool::Tool) implementation. The
//! trait, [`ToolResult`](crate::tool::ToolResult), and
//! [`ToolRegistry`](crate::tool::ToolRegistry) themselves live in
//! [`crate::tool`]; this module just gathers the concrete tools.

pub mod edit_file;
