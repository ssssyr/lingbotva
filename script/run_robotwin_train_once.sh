#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MARKER="# codex robotwin auto-start 20260319"

tmp_cron="$(mktemp)"
crontab -l 2>/dev/null | grep -F -v "${MARKER}" > "${tmp_cron}" || true
crontab "${tmp_cron}"
rm -f "${tmp_cron}"

exec "${REPO_ROOT}/script/start_robotwin_train_when_ready.sh"
