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
{% if include_test_first %}{% include "test_first_approach.md" %}{% endif %}{% if include_criteria_rules %}{% include "criteria_rules.md" %}{% endif %}{% include "verification_section.md" %}
