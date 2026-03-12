# CFFF DSW Remote Workflow

This repository is developed locally, while model execution or deployment can happen on the remote CFFF DSW instance.

## Core Rule

- Run `codex` on the local machine.
- Reach the remote instance through the local SSH alias `cfff`.
- Do not open a second Codex inside the remote machine for normal cloud work.

## Default Assumptions

- Local repository root: `/home/syr/code/lingbot-va`
- Remote SSH alias: `cfff`
- Default remote base path: `/cpfs01/projects-HDD/cfff-4a2485d4a88d_HDD/ct_24210860031/812/SYR/code`
- Default remote repository root for this repo: `/cpfs01/projects-HDD/cfff-4a2485d4a88d_HDD/ct_24210860031/812/SYR/code/lingbot-va`

If the remote checkout lives somewhere else, override it when calling the helper scripts:

```bash
CFFF_REMOTE_ROOT=/your/remote/project ./script/cfff-sync.sh
```

If you want to keep the same repo-name-based convention but change only the base path:

```bash
CFFF_REMOTE_BASE=/your/remote/code-root ./script/cfff-sync.sh
```

## Helper Scripts

- `./script/cfff-shell.sh`
  - Open an interactive shell on the remote instance, land in the remote project root, and auto-load Conda.
- `./script/cfff-run.sh "<command>"`
  - Run one remote command from the local machine inside the remote project root with Conda auto-loaded.
- `./script/cfff-sync.sh`
  - Sync the local repository to the remote project root with `rsync`.

By default, `cfff-run.sh` and `cfff-shell.sh` activate the remote `base` Conda environment when Conda is found in a common install path such as `~/miniconda3`, `~/anaconda3`, or `/opt/conda`.

Override this behavior when needed:

```bash
CFFF_CONDA_ENV=cosmos ./script/cfff-run.sh "python -V"
CFFF_CONDA_ENV='' ./script/cfff-run.sh "python -V"
CFFF_CONDA_SH=/home/ct_24210860031/miniconda3/etc/profile.d/conda.sh ./script/cfff-shell.sh
```

## Standard Session

1. Confirm the remote host is reachable:

```bash
./script/cfff-run.sh "whoami && hostname && pwd"
```

2. Check GPU visibility:

```bash
./script/cfff-run.sh "nvidia-smi"
```

3. Sync local code to the remote instance:

```bash
./script/cfff-sync.sh
```

4. Run remote commands from the local machine:

```bash
./script/cfff-run.sh "python -V"
./script/cfff-run.sh "bash evaluation/robotwin/launch_server.sh"
```

5. For long-running jobs, use `tmux` on the remote side:

```bash
./script/cfff-run.sh "tmux new -d -s lingbot-server 'cd ~/code/lingbot-va && bash evaluation/robotwin/launch_server.sh'"
./script/cfff-run.sh "tmux ls"
```

6. If inspection is easier interactively, attach a remote shell:

```bash
./script/cfff-shell.sh
```

## Recommended Pattern For New Codex Sessions

Open Codex in the local repository root:

```bash
cd /home/syr/code/lingbot-va
codex
```

Then prompt it with something like:

```text
按仓库里的 CFFF DSW 工作流操作。先检查远端 GPU 和 Python 环境，再把当前仓库同步到远端，最后指导我启动推理服务。
```

Because `AGENTS.md` is present at the repository root, the new session should already know that:

- remote work goes through `ssh cfff`
- `codex` stays local
- helper scripts are the preferred interface

## Notes

- If `ssh cfff` stops working, fix SSH first before debugging the repository.
- If the remote home directory is reset by the platform, rerun your SSH key setup and resync the repository.
- Avoid running destructive commands on the remote instance without confirming with the user first.
