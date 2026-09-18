## Question

{{ question }}

## Result schema

Your `result` payload must conform to this JSON Schema, shown exactly as it
was supplied to the run:

```json
{{ schema_text }}
```

## Rules

You are answering a question about the workspace — the deliverable is data,
not a change. Read whatever you need to read, reason it through, and report
what you found.

You must not modify the workspace. This run is read-only: a working tree that
changed since the run started makes the answer unacceptable, and the harness
rejects it. If you have already changed something, revert it before you
finish.

End the run by calling the `finish` tool with disposition `answer` — a
`finish(answer)` call — supplying a `result` that matches the schema above.
The harness validates it and feeds any validation errors back to you rather
than terminating the run.
