## {{ title }}

{{ description }}

## Acceptance Criteria

{% for ac in acceptance_criteria -%}
- {{ ac }}
{% endfor %}
## Files to Modify

The paths listed below are the navigation layer for this task. No search tool
exists in this environment, so these paths are your map — read and edit exactly
these files (and nothing else unless the description above says otherwise).

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
