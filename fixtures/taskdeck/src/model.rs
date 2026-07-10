/// A single task in the tracker.
#[derive(Debug, Clone, PartialEq)]
pub struct Task {
    pub id: u64,
    pub title: String,
    pub done: bool,
}

/// Errors that can be returned by task-management commands.
#[derive(Debug, PartialEq)]
pub enum CommandError {
    NotFound(u64),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_clone_and_equality() {
        let t = Task { id: 1, title: "write tests".to_string(), done: false };
        let t2 = t.clone();
        assert_eq!(t, t2);
    }

    #[test]
    fn new_task_is_not_done() {
        let t = Task { id: 1, title: "pending".to_string(), done: false };
        assert!(!t.done);
    }

    #[test]
    fn command_error_not_found_carries_id() {
        let e = CommandError::NotFound(42);
        assert_eq!(e, CommandError::NotFound(42));
    }
}
