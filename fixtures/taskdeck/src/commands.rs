use crate::model::CommandError;
use crate::store::TaskStore;

impl TaskStore {
    /// Mark the task with the given `id` as done.
    ///
    /// Returns `Ok(())` on success.
    /// Returns `Err(CommandError::NotFound(id))` if no task has that id.
    /// If the task is already done, returns `Ok(())` (idempotent — no error).
    #[allow(unused_variables)]
    pub fn complete(&mut self, id: u64) -> Result<(), CommandError> {
        todo!()
    }

    /// Remove all done tasks from the store, preserving the relative order of
    /// surviving (not-done) tasks.
    ///
    /// Returns the number of tasks removed.
    /// A second call when no done tasks remain returns 0 (idempotent).
    pub fn purge(&mut self) -> usize {
        todo!()
    }
}
