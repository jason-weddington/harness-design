## Ralph outer-loop iteration

You are ONE iteration of an outer "ralph" loop. Each iteration runs with a FRESH context — you do not see prior assistant turns, and you cannot rely on anything you "said" last time. All durable state lives OUTSIDE your context window: the codebase on disk, the git history, and the progress notes injected below. Treat this as a cold start every single time.

## Objective

{{ objective }}

## Progress notes so far

{{ notes }}

## What to do this iteration

Read the progress notes above (they are the running journal appended by prior iterations — empty on the very first pass). Then do exactly one thing — not one thing plus a quick refactor — toward the objective. Make a real, grounded change on disk: edit a file, fix a bug, add a test, or land one cohesive step of the work. Do not batch; one iteration buys one unit of progress.

After making your change, append a short progress note to the file named `{{ notes_file }}` describing what you did this iteration and what the next iteration should pick up, so the next fresh-context iteration can continue without re-deriving your reasoning. Then call `finish` with your disposition.