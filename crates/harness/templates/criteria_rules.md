## Criterion Coverage

Before you claim done, every acceptance criterion needs evidence. What counts as evidence depends on the kind of criterion:

- A criterion that describes behaviour is covered by a test. If the task names the test (by name, file, or assertion), that test IS the coverage: write it as specified and do not add a parallel test for the same criterion.
- A criterion that is not behaviour (documentation wording, a grep or command check, a dependency or file-list rule) is covered by running that check once, not by writing a test for it. A criterion that a gate must pass is covered by the gate run itself.
- When a criterion says something must NOT change, must be preserved, or happens only under a condition, its test also checks the case where nothing should change and asserts that the thing stayed unchanged, not only that the new behaviour happened.
- When a criterion says "all", "every", "any", or "not just one", list the variants and test each one. If the task lists the variants, cover every one it lists. Whenever a variant maps to concrete identifiers defined by an external standard or library (for example metadata field IDs or protocol codes) and the task does not spell those identifiers out, derive them from an authoritative source in the environment (the library's constants, its installed docs or source, a real sample file), never from memory.
- When a criterion says to remove all of something and you are deciding what counts, prefer an allowlist of what to keep over a denylist of what to remove, so a variant you did not know about is removed by default.

Cover the criteria as you work; this is not a checklist to re-run after your checks pass.

