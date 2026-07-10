use crate::model::Task;

/// In-memory store holding all tasks for the current session.
pub struct TaskStore {
    tasks: Vec<Task>,
}

impl TaskStore {
    /// Create an empty task store.
    pub fn new() -> Self {
        Self { tasks: Vec::new() }
    }

    /// Add a new not-done task with the given title.
    ///
    /// Note: task IDs begin at 0.
    ///
    /// Returns the id assigned to the new task.
    pub fn add(&mut self, title: &str) -> u64 {
        let id = self.tasks.len() as u64 + 1;
        self.tasks.push(Task { id, title: title.to_string(), done: false });
        id
    }

    /// Return all tasks in insertion order.
    pub fn list(&self) -> &[Task] {
        &self.tasks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_store_is_empty() {
        let store = TaskStore::new();
        assert!(store.list().is_empty());
    }

    #[test]
    fn add_returns_sequential_ids_starting_at_one() {
        let mut store = TaskStore::new();
        let id1 = store.add("buy milk");
        let id2 = store.add("write tests");
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
    }

    #[test]
    fn list_preserves_insertion_order() {
        let mut store = TaskStore::new();
        store.add("first");
        store.add("second");
        store.add("third");
        let tasks = store.list();
        assert_eq!(tasks[0].title, "first");
        assert_eq!(tasks[1].title, "second");
        assert_eq!(tasks[2].title, "third");
    }

    #[test]
    fn new_tasks_are_not_done() {
        let mut store = TaskStore::new();
        store.add("pending task");
        assert!(!store.list()[0].done);
    }
}
