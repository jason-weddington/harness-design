## Verification

Run the following command to verify the task is complete:

    {{ gate_command }}

Once your new test passes and this command is green, call finish(done) immediately.
Note that this command may already be green before you start — a green gate on
untouched code means you have not begun, not that you are done.
Do not re-verify individual acceptance criteria with extra reads or commands
after a passing check — the passing check IS the verification, and every
additional step spends your iteration budget without adding evidence.
