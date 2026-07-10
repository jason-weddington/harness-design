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
## Verification

Run the following command to verify the task is complete:

    {{ gate_command }}

As soon as this command passes, call finish(done) immediately. Do not re-verify
individual acceptance criteria with extra reads or commands after a passing
check — the passing check IS the verification, and every additional step spends
your iteration budget without adding evidence.
