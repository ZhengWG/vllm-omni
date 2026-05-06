# vllm-omni CI watcher

This folder contains tooling to monitor the upstream
[`vllm-project/vllm-omni`](https://github.com/vllm-project/vllm-omni) CI from
this fork (`ZhengWG/vllm-omni`), without requiring a long-lived process.

## Files

- `automation-prompt.md` — the prompt body and configuration to paste into a
  **Cursor Automation** (Scheduled trigger). This is the recommended way to
  get persistent CI monitoring; each tick spins up a fresh Cloud Agent that
  does one polling iteration and exits.
- `poll.sh` — a self-contained `bash` polling loop, useful only when you have
  a long-lived host (e.g. a Cursor "My Machines" worker, a personal devbox,
  or a laptop you keep open). On a normal Cursor-hosted Cloud Agent VM the
  process dies the moment the VM is reclaimed.

## How alerts are delivered

The Automation maintains a single tracking GitHub Issue **in this repository**
(`ZhengWG/vllm-omni`), labeled `ci-watch`, titled
`[ci-watch] vllm-project/vllm-omni`. The issue body holds the last-known
upstream CI state as a fenced JSON block. On every tick the agent diffs the
current upstream state against the stored state and, only if something
changed, posts a comment summarising the change. When everything is steady,
no comment is written and no notification fires.

To switch to Slack/email/webhook later, replace Step 4 of the prompt with the
desired side-effect (a single `curl` to a webhook is enough).

## Resetting state

Close the tracking issue. The next Automation run will create a fresh one and
re-baseline.

## Why this lives in `ZhengWG/vllm-omni` and not upstream

The Automation needs to write a tracking Issue + comments somewhere. Doing
that on `vllm-project/vllm-omni` would pollute the upstream issue tracker.
This fork is the natural home: same repo lineage, same `gh` token can both
read upstream and write here, and any noise stays out of the public project.
