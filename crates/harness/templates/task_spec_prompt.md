## {{ title }}

{{ description }}

## Acceptance Criteria

{% for ac in acceptance_criteria -%}
- {{ ac }}
{% endfor %}
## Files to Modify

The paths listed below are starting points for navigation — they are a map, not
a complete inventory. Use the bash tool (e.g., `grep`, `find`) to locate code
within and around these files. Edit exactly what the task requires and no more;
prefer edit_file for mutations (its unique-match contract is safer than sed -i).

{% for f in files_to_modify -%}
- `{{ f.path }}`: {{ f.change }}
{% endfor %}
{% include "test_first_approach.md" %}
## Verification

Run the following command to verify the task is complete:

    {{ gate_command }}

Once your new test passes and this command is green, call finish(done) immediately.
Note that this command may already be green before you start — a green gate on
untouched code means you have not begun, not that you are done.
Do not re-verify individual acceptance criteria with extra reads or commands
after a passing check — the passing check IS the verification, and every
additional step spends your iteration budget without adding evidence.
