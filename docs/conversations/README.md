# Conversation Summaries

Compact records of reasoning, decisions, and actions from significant development
sessions. Use these when a conversation produced architectural choices, experiment
pivots, or follow-up direction that should outlive the chat.

Detailed experiment tables and metrics belong in
[`docs/EXPERIMENTS.md`](../EXPERIMENTS.md). Brief
milestones belong in [`docs/CHANGELOG.md`](../CHANGELOG.md).

## When To Write One

Write a summary when the session:

- introduced or changed an experiment line, package, or benchmark;
- recorded key decisions or rejected approaches worth remembering;
- set a clear next research direction.

Skip summaries for small bug fixes, one-off runs, or changes already fully
captured elsewhere.

## File Naming

Create a new file in this folder:

```text
docs/conversations/YYYY-MM-DD-<short-topic>.md
```

Use a date range when the work spans multiple days, e.g.
`2026-05-27-28-spotlight-hierarchy.md`.

## Template

Follow the structure in
[`2026-05-27-28-spotlight-hierarchy.md`](2026-05-27-28-spotlight-hierarchy.md):

1. **Title and date range**
2. **Initial problem** — what was broken or unclear
3. **Key decisions** — bullet list of durable choices
4. **Experiment progression** — short subsections for major steps (E9, E10, …)
5. **Current state** — what is true now
6. **Follow-up ideas** — likely next steps, not a backlog dump

Keep it compact: decisions and rationale, not a transcript. Link to experiment
docs and artifacts instead of duplicating tables or full result dumps.

## After Writing

- Add or update a brief entry in `docs/CHANGELOG.md` if the session changed repo
  direction or shipped a milestone.
- Put detailed metrics and run tables in `docs/EXPERIMENTS.md`.
