#![allow(dead_code)]
//! Archive module — moves tasks out of the main store into a secondary
//! archive buffer, preserving them for later inspection. The archive grows
//! unboundedly; nothing placed in it is ever deleted.

use crate::model::Task;

/// A secondary buffer that holds tasks moved out of the main store.
pub struct Archive {
    items: Vec<Task>,
}

impl Archive {
    /// Create an empty archive.
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// Move all done tasks from `tasks` into this archive.
    ///
    /// Returns the filtered task list (done tasks removed) and the count moved.
    /// Callers are responsible for replacing their store's task list with the
    /// returned vec.
    pub fn archive_done(&mut self, tasks: Vec<Task>) -> (Vec<Task>, usize) {
        let (keep, archived): (Vec<Task>, Vec<Task>) =
            tasks.into_iter().partition(|t| !t.done);
        let moved = archived.len();
        self.items.extend(archived);
        (keep, moved)
    }

    /// Return all archived tasks in the order they were archived.
    pub fn archived(&self) -> &[Task] {
        &self.items
    }
}
