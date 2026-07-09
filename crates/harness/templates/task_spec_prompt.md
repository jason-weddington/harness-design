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
