use taskdeck::model::CommandError;
use taskdeck::store::TaskStore;

/// complete on a nonexistent id must return Err(NotFound(_)), not panic.
#[test]
fn complete_nonexistent_id_returns_not_found_error() {
    let mut store = TaskStore::new();
    assert!(
        matches!(store.complete(999), Err(CommandError::NotFound(_))),
        "expected NotFound for id 999, got something else"
    );
}

/// purge must preserve the relative order of surviving tasks and only remove done ones.
#[test]
fn purge_preserves_not_done_tasks_in_insertion_order() {
    let mut store = TaskStore::new();
    let id1 = store.add("keep alpha");
    let id2 = store.add("mark done");
    let id3 = store.add("keep gamma");

    store.complete(id2).expect("complete should succeed for existing id");
    let removed = store.purge();

    assert_eq!(removed, 1, "one done task should be removed");
    let tasks = store.list();
    assert_eq!(tasks.len(), 2, "two not-done tasks should survive");
    assert_eq!(tasks[0].id, id1);
    assert_eq!(tasks[0].title, "keep alpha");
    assert!(!tasks[0].done);
    assert_eq!(tasks[1].id, id3);
    assert_eq!(tasks[1].title, "keep gamma");
    assert!(!tasks[1].done);
}

/// purge returns the count removed; a second immediate call returns 0.
#[test]
fn purge_second_call_returns_zero() {
    let mut store = TaskStore::new();
    store.add("task one");
    store.add("task two");
    store.complete(1).unwrap();

    let first = store.purge();
    let second = store.purge();
    assert_eq!(first, 1);
    assert_eq!(second, 0, "no done tasks remain so second purge removes 0");
}

/// Completing a task then purging removes exactly that task and no others.
#[test]
fn complete_then_purge_removes_exactly_the_completed_task() {
    let mut store = TaskStore::new();
    let id_a = store.add("alpha");
    let id_b = store.add("beta");
    let id_c = store.add("gamma");

    store.complete(id_b).expect("complete beta");
    let removed = store.purge();

    assert_eq!(removed, 1);
    let ids: Vec<u64> = store.list().iter().map(|t| t.id).collect();
    assert!(ids.contains(&id_a), "alpha must survive");
    assert!(!ids.contains(&id_b), "beta (done) must be purged");
    assert!(ids.contains(&id_c), "gamma must survive");
}
