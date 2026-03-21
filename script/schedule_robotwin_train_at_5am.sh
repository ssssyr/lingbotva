#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${REPO_ROOT}"

sleep_seconds="$(
python - <<'PY'
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

tz = ZoneInfo("Asia/Shanghai")
now = datetime.now(tz)
target = now.replace(hour=5, minute=0, second=0, microsecond=0)
if target <= now:
    target += timedelta(days=1)
print(int((target - now).total_seconds()))
print(target.isoformat())
PY
)"

delay="$(printf '%s\n' "${sleep_seconds}" | sed -n '1p')"
target_time="$(printf '%s\n' "${sleep_seconds}" | sed -n '2p')"

printf '[auto-start] %s scheduled_target=%s sleep_seconds=%s\n' "$(date -Iseconds)" "${target_time}" "${delay}"
sleep "${delay}"
exec "${REPO_ROOT}/script/start_robotwin_train_when_ready.sh"
