---
name: daily-memory-journal
description: Read and maintain the Codex important memory journal at ~/.codex/memories/important-memory.md. Use when the user asks to continue earlier Codex work, read the important memory document, resume a previous session, review what Codex finished on a prior day, or avoid repeating earlier mistakes. This skill pairs with a nightly summarizer that appends one section per day from ~/.codex/sessions logs.
---

# Daily Memory Journal

Use this skill when the user asks to resume prior work, read the important memory document, continue from a previous day, review Codex's recent work, or inspect the nightly memory summary.

## Read Order

Open `~/.codex/memories/important-memory.md` first.

Check these sections in order:

- `## Manual Memory` for stable rules, recurring pitfalls, and user preferences.
- The newest `### YYYY-MM-DD` block under `## Daily Summaries` for recent execution history.
- An older dated block only if the user names a specific date or task.

## How To Use The Memory

- Before resuming work, cite the relevant daily block briefly and continue from the recorded state instead of rediscovering it.
- Before retrying a workflow that failed earlier, check the memory for prior error signatures, resume points, path assumptions, and environment choices.
- When you learn a stable lesson that should outlive a single day, add or refine one concise bullet under `## Manual Memory`.

## Boundaries

- Do not rewrite older daily sections unless the user asks or the automation clearly produced incorrect output.
- Keep `## Manual Memory` short and durable; daily details belong in the dated blocks.
- If the memory file is missing or stale, state that clearly and regenerate or repair it with `scripts/build_daily_memory.py`.

## Bundled Scripts

- `scripts/build_daily_memory.py`
  Use to summarize one local calendar day from `~/.codex/sessions` into `~/.codex/memories/important-memory.md`.
- `scripts/run_midnight_summary.sh`
  Wrapper for unattended execution. Defaults to summarizing the previous local day.
- `scripts/install_midnight_cron.sh`
  Installs one user crontab entry for `00:00` local time. The scheduled run summarizes the previous day, so a job triggered at `00:00` on March 17, 2026 writes the block for March 16, 2026.
