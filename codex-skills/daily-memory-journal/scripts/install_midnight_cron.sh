#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_midnight_summary.sh"
LOG_FILE="${HOME}/.codex/memories/daily-memory-cron.log"
MARKER="# codex daily-memory-journal"
TMP_FILE="$(mktemp)"

cleanup() {
    rm -f "${TMP_FILE}"
}
trap cleanup EXIT

crontab -l 2>/dev/null | grep -v "${MARKER}" > "${TMP_FILE}" || true
printf '0 0 * * * /bin/bash %q >> %q 2>&1 %s\n' "${RUNNER}" "${LOG_FILE}" "${MARKER}" >> "${TMP_FILE}"
crontab "${TMP_FILE}"
echo "Installed midnight cron entry for daily-memory-journal."
