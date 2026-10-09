# Widget double counting

## Acceptance criteria

- `Widget.count()` returns twice the number of items it holds.
- An empty widget still counts zero.

## Approach

Write a test that captures the acceptance criteria FIRST, before you change any
other code, and run it. Confirm it fails, and that it fails for the reason you
expect — a test that passes before you have implemented anything is testing
nothing. Then implement until it passes.

If the task genuinely has no test-shaped outcome (a pure refactor, a dependency
bump, a docs or config change), state that in one line and go straight to the
change.
## Verification

Run the following command to verify the task is complete:

    P=$(shuf -i 20000-60000 -n 1); echo "port $P" && printf 'a\\b\n' && true

Once your new test passes and this command is green, call finish(done) immediately.
Note that this command may already be green before you start — a green gate on
untouched code means you have not begun, not that you are done.
Do not re-verify individual acceptance criteria with extra reads or commands
after a passing check — the passing check IS the verification, and every
additional step spends your iteration budget without adding evidence.