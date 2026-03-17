# Repository Instructions

This repository can be operated against a remote CFFF DSW instance, but Codex itself should run on the local machine.

## Remote DSW Workflow

- When a task mentions `DSW`, `CFFF`, `cloud`, `remote`, or deployment on the university instance, assume the remote host is reachable through the local SSH alias `cfff`.
- Do not try to install or run Codex inside the remote instance unless the user explicitly asks for a local-model-only setup.
- Prefer the helper scripts in [`script/cfff-shell.sh`](/home/syr/code/lingbot-va/script/cfff-shell.sh), [`script/cfff-run.sh`](/home/syr/code/lingbot-va/script/cfff-run.sh), and [`script/cfff-sync.sh`](/home/syr/code/lingbot-va/script/cfff-sync.sh) instead of ad hoc SSH commands.
- For long-running remote training or deployment tasks, prefer [`script/cfff-job-start.sh`](/home/syr/code/lingbot-va/script/cfff-job-start.sh) so stdout/stderr are persisted to `logs/cfff-jobs/<job-name>/<timestamp>.log`.
- Use [`script/cfff-log-show.sh`](/home/syr/code/lingbot-va/script/cfff-log-show.sh) to read the latest remote logs and [`script/cfff-log-follow.sh`](/home/syr/code/lingbot-va/script/cfff-log-follow.sh) to follow them interactively.
- Default remote base path is `/cpfs01/projects-HDD/cfff-4a2485d4a88d_HDD/ct_24210860031/812/SYR/code`, and the scripts append the local repo name. For this repository, the default remote project root is `/cpfs01/projects-HDD/cfff-4a2485d4a88d_HDD/ct_24210860031/812/SYR/code/lingbot-va`. Override with `CFFF_REMOTE_ROOT=/remote/path` or `CFFF_REMOTE_BASE=/remote/base`.
- `script/cfff-run.sh` and `script/cfff-shell.sh` auto-load remote Conda from common install paths and activate `base` by default. Override with `CFFF_CONDA_ENV=<env>` or `CFFF_CONDA_SH=/remote/path/to/conda.sh`.
- Before long-running remote jobs, prefer `tmux` on the remote instance.
- Ask before destructive remote operations such as deleting files, killing sessions, resetting git state, or overwriting remote outputs.

## First Remote Checks

When starting a fresh remote task, prefer this order:

1. `./script/cfff-run.sh "whoami && hostname && pwd"`
2. `./script/cfff-run.sh "nvidia-smi"`
3. `./script/cfff-sync.sh` if local changes need to be sent to the remote instance

## Extra Reference

If a task needs the full human-oriented workflow, read [`doc/cfff_dsw_workflow.md`](/home/syr/code/lingbot-va/doc/cfff_dsw_workflow.md).
