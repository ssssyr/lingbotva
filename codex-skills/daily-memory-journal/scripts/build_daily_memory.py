#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

try:
    from zoneinfo import ZoneInfo
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"zoneinfo is required: {exc}") from exc


DEFAULT_SESSIONS_ROOT = Path("~/.codex/sessions").expanduser()
DEFAULT_MEMORY_FILE = Path("~/.codex/memories/important-memory.md").expanduser()

BEGIN_FMT = "<!-- BEGIN DAILY SUMMARY {day} -->"
END_FMT = "<!-- END DAILY SUMMARY {day} -->"

ERROR_LINE_REGEX = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)\b")

PATCH_FILE_REGEX = re.compile(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$", re.MULTILINE)


@dataclass
class TurnSummary:
    turn_id: str
    user_messages: list[str] = field(default_factory=list)
    agent_updates: list[str] = field(default_factory=list)
    final_message: str = ""
    aborted: bool = False
    commands: list[str] = field(default_factory=list)
    touched_files: list[str] = field(default_factory=list)
    error_signatures: list[str] = field(default_factory=list)


@dataclass
class SessionSummary:
    session_id: str
    path: Path
    cwd: str = ""
    started_at: datetime | None = None
    turns: list[TurnSummary] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize one day of Codex sessions into the important memory journal."
    )
    parser.add_argument(
        "--date",
        help="Local calendar day to summarize in YYYY-MM-DD. Defaults to yesterday in the selected timezone.",
    )
    parser.add_argument(
        "--timezone",
        default="Asia/Shanghai",
        help="IANA timezone used for the default date calculation. Default: Asia/Shanghai.",
    )
    parser.add_argument(
        "--sessions-root",
        default=str(DEFAULT_SESSIONS_ROOT),
        help=f"Root directory for Codex session logs. Default: {DEFAULT_SESSIONS_ROOT}",
    )
    parser.add_argument(
        "--memory-file",
        default=str(DEFAULT_MEMORY_FILE),
        help=f"Markdown memory file to update. Default: {DEFAULT_MEMORY_FILE}",
    )
    parser.add_argument(
        "--max-command-samples",
        type=int,
        default=4,
        help="Maximum number of distinct commands to show per turn.",
    )
    parser.add_argument(
        "--max-file-samples",
        type=int,
        default=6,
        help="Maximum number of touched files to show per turn.",
    )
    return parser.parse_args()


def parse_iso8601(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def local_day_from_args(args: argparse.Namespace) -> date:
    if args.date:
        return date.fromisoformat(args.date)
    tz = ZoneInfo(args.timezone)
    return datetime.now(tz).date() - timedelta(days=1)


def session_files_for_day(root: Path, day: date) -> list[Path]:
    day_root = root / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}"
    if not day_root.exists():
        return []
    return sorted(day_root.glob("rollout-*.jsonl"))


def clean_text(text: str, limit: int = 320) -> str:
    text = text.replace("\r", "")
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    lines = [line.strip(" -\t") for line in text.splitlines()]
    lines = [line for line in lines if line]
    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def normalize_error(text: str) -> str:
    text = text.replace("\\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:180]


def extract_error_signatures(text: str) -> list[str]:
    signatures: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = normalize_error(raw_line)
        if not line:
            continue
        keep = False
        if "Traceback" in line:
            line = "Traceback"
            keep = True
        elif ERROR_LINE_REGEX.search(line):
            keep = True
        elif "已终止" in line or "OOM" in line or "Killed" in line:
            keep = True
        elif "No such file" in line or "permission denied" in line.lower():
            keep = True
        if not keep:
            continue
        if "# Working with the user" in line:
            continue
        if line not in seen:
            seen.add(line)
            signatures.append(line)
    return signatures


def load_json_or_none(value: str) -> Any:
    try:
        return json.loads(value)
    except Exception:
        return None


def ensure_turn(session: SessionSummary, turn_id: str | None) -> TurnSummary:
    if session.turns and (turn_id is None or session.turns[-1].turn_id == turn_id):
        return session.turns[-1]
    if turn_id is None:
        turn_id = f"unknown-{len(session.turns) + 1}"
    turn = TurnSummary(turn_id=turn_id)
    session.turns.append(turn)
    return turn


def parse_function_call(session: SessionSummary, payload: dict[str, Any]) -> None:
    turn = ensure_turn(session, None)
    name = payload.get("name", "")
    args = load_json_or_none(payload.get("arguments", "")) or {}
    if name == "exec_command":
        cmd = str(args.get("cmd", "")).strip()
        if cmd:
            turn.commands.append(normalize_command(cmd))
        workdir = str(args.get("workdir", "")).strip()
        if workdir:
            session.cwd = session.cwd or workdir
    elif name == "apply_patch":
        patch_text = payload.get("arguments", "")
        turn.touched_files.extend(PATCH_FILE_REGEX.findall(patch_text))
    else:
        label = name or "tool_call"
        if args:
            turn.commands.append(
                f"{label}: {clean_text(json.dumps(args, ensure_ascii=False), limit=120)}"
            )
        else:
            turn.commands.append(label)


def parse_function_output(session: SessionSummary, payload: dict[str, Any]) -> None:
    turn = ensure_turn(session, None)
    output = str(payload.get("output", ""))
    turn.error_signatures.extend(extract_error_signatures(output))


def parse_session_file(path: Path) -> SessionSummary:
    session = SessionSummary(session_id=path.stem, path=path)
    current_turn_id: str | None = None
    for raw_line in path.open(encoding="utf-8"):
        event = json.loads(raw_line)
        event_type = event.get("type")
        payload = event.get("payload", {})

        if event_type == "session_meta":
            session.session_id = payload.get("id", session.session_id)
            session.cwd = payload.get("cwd", session.cwd)
            session.started_at = parse_iso8601(payload.get("timestamp"))
            continue

        if event_type == "turn_context":
            session.cwd = payload.get("cwd", session.cwd)
            current_turn_id = payload.get("turn_id", current_turn_id)
            ensure_turn(session, current_turn_id)
            continue

        if event_type == "event_msg":
            subtype = payload.get("type")
            if subtype == "task_started":
                current_turn_id = payload.get("turn_id", current_turn_id)
                ensure_turn(session, current_turn_id)
            elif subtype == "user_message":
                turn = ensure_turn(session, current_turn_id)
                message = str(payload.get("message", "")).strip()
                if message:
                    turn.user_messages.append(message)
                    turn.error_signatures.extend(extract_error_signatures(message))
            elif subtype == "agent_message":
                turn = ensure_turn(session, current_turn_id)
                message = str(payload.get("message", "")).strip()
                if message:
                    turn.agent_updates.append(message)
            elif subtype == "task_complete":
                turn = ensure_turn(session, payload.get("turn_id", current_turn_id))
                turn.final_message = str(payload.get("last_agent_message", "")).strip()
                turn.error_signatures.extend(extract_error_signatures(turn.final_message))
            elif subtype == "turn_aborted":
                turn = ensure_turn(session, payload.get("turn_id", current_turn_id))
                turn.aborted = True
            continue

        if event_type == "response_item":
            subtype = payload.get("type")
            if subtype == "function_call":
                parse_function_call(session, payload)
            elif subtype == "function_call_output":
                parse_function_output(session, payload)

    for turn in session.turns:
        turn.commands = unique_preserve_order(turn.commands)
        turn.touched_files = unique_preserve_order(turn.touched_files)
        turn.error_signatures = unique_preserve_order(turn.error_signatures)
    return session


def unique_preserve_order(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def concise_turn_title(turn: TurnSummary) -> str:
    if turn.user_messages:
        return clean_text(turn.user_messages[0], limit=110)
    if turn.final_message:
        return clean_text(turn.final_message, limit=110)
    return turn.turn_id


def concise_outcome(turn: TurnSummary) -> str:
    if turn.final_message:
        return clean_text(turn.final_message, limit=240)
    if turn.aborted:
        return "Turn was interrupted before completion."
    if turn.agent_updates:
        return clean_text(turn.agent_updates[-1], limit=240)
    return "No final assistant message recorded."


def normalize_command(command: str) -> str:
    text = " ".join(part.strip() for part in command.splitlines() if part.strip())
    if not text:
        return ""
    first_segment = re.split(r"\s*(?:\||&&|\|\|)\s*", text, maxsplit=1)[0].strip()
    if "python - <<'PY'" in first_segment or "python - <<" in first_segment:
        return "python heredoc"
    if first_segment.startswith("write_stdin"):
        return "write_stdin"
    if first_segment.startswith("update_plan"):
        return "update_plan"
    try:
        parts = shlex.split(first_segment)
    except ValueError:
        return clean_text(first_segment, limit=70)
    if not parts:
        return ""
    return " ".join(parts[:3])


def session_touched_files(session: SessionSummary) -> list[str]:
    return unique_preserve_order(
        [path for turn in session.turns for path in turn.touched_files]
    )


def session_error_signatures(session: SessionSummary) -> list[str]:
    return unique_preserve_order(
        [sig for turn in session.turns for sig in turn.error_signatures]
    )


def session_command_samples(session: SessionSummary) -> list[str]:
    counter = Counter(cmd for turn in session.turns for cmd in turn.commands)
    return [cmd for cmd, _count in counter.most_common(6)]


def significant_turns(session: SessionSummary) -> list[TurnSummary]:
    if not session.turns:
        return []
    picked: list[TurnSummary] = []

    def add(turn: TurnSummary) -> None:
        if turn not in picked:
            picked.append(turn)

    add(session.turns[0])
    for turn in session.turns:
        if turn.touched_files or turn.aborted or turn.error_signatures:
            add(turn)
    if len(session.turns) > 1:
        add(session.turns[-1])
    if len(session.turns) > 2:
        add(session.turns[-2])

    if len(picked) <= 5:
        return picked
    trimmed = picked[:2] + picked[-3:]
    result: list[TurnSummary] = []
    seen_ids: set[str] = set()
    for turn in trimmed:
        if turn.turn_id in seen_ids:
            continue
        seen_ids.add(turn.turn_id)
        result.append(turn)
    return result


def summarize_day(sessions: list[SessionSummary], target_day: date, args: argparse.Namespace) -> str:
    all_turns = [turn for session in sessions for turn in session.turns]
    workdirs = unique_preserve_order([session.cwd for session in sessions if session.cwd])
    touched_files = unique_preserve_order(
        [path for turn in all_turns for path in turn.touched_files]
    )
    command_counter = Counter(cmd for turn in all_turns for cmd in turn.commands)
    error_signatures = unique_preserve_order(
        [sig for turn in all_turns for sig in turn.error_signatures]
    )

    top_commands = [item for item, _count in command_counter.most_common(8)]
    successful_turns = sum(1 for turn in all_turns if turn.final_message and not turn.aborted)
    aborted_turns = sum(1 for turn in all_turns if turn.aborted)

    lines: list[str] = []
    lines.append(BEGIN_FMT.format(day=target_day.isoformat()))
    lines.append(f"### {target_day.isoformat()}")
    lines.append("")
    lines.append("- Sessions: {}".format(len(sessions)))
    lines.append("- Turns: {}".format(len(all_turns)))
    lines.append("- Completed turns: {}".format(successful_turns))
    lines.append("- Aborted turns: {}".format(aborted_turns))
    if workdirs:
        lines.append("- Workspaces: {}".format(", ".join(workdirs[:6])))
    if touched_files:
        lines.append("- Touched files: {}".format(", ".join(touched_files[:12])))
    if top_commands:
        lines.append("- Frequent commands: {}".format(" | ".join(top_commands[:8])))
    lines.append("")

    if error_signatures:
        lines.append("#### Error Watchlist")
        lines.append("")
        for item in error_signatures[:10]:
            lines.append(f"- {item}")
        lines.append("")

    lines.append("#### Session Timeline")
    lines.append("")

    for index, session in enumerate(sessions, start=1):
        started = session.started_at.astimezone(ZoneInfo(args.timezone)).strftime("%H:%M") if session.started_at else "unknown"
        lines.append(
            f"##### Session {index} | {started} | {session.cwd or 'unknown cwd'}"
        )
        lines.append("")
        lines.append(f"- Goal: {concise_turn_title(session.turns[0]) if session.turns else 'unknown'}")
        lines.append(
            f"- End state: {concise_outcome(session.turns[-1]) if session.turns else 'No turns recorded.'}"
        )
        lines.append(f"- Turn count: {len(session.turns)}")
        session_files = session_touched_files(session)
        if session_files:
            lines.append(f"- Files: {', '.join(session_files[:10])}")
        session_commands = session_command_samples(session)
        if session_commands:
            lines.append(f"- Commands: {' | '.join(session_commands[:6])}")
        session_errors = session_error_signatures(session)
        if session_errors:
            lines.append(f"- Errors: {' | '.join(session_errors[:6])}")
        lines.append("- Notable turns:")
        for turn_index, turn in enumerate(significant_turns(session), start=1):
            title = concise_turn_title(turn)
            outcome = concise_outcome(turn)
            lines.append(f"  - Turn {turn_index}: {title}")
            lines.append(f"    Outcome: {outcome}")
            if turn.touched_files:
                files = ", ".join(turn.touched_files[: args.max_file_samples])
                lines.append(f"    Files: {files}")
            if turn.commands:
                commands = " | ".join(turn.commands[: args.max_command_samples])
                lines.append(f"    Commands: {commands}")
            if turn.error_signatures:
                errors = " | ".join(turn.error_signatures[:6])
                lines.append(f"    Errors: {errors}")
        lines.append("")

    if not sessions:
        lines.append("No Codex sessions were found for this day.")
        lines.append("")

    lines.append(END_FMT.format(day=target_day.isoformat()))
    lines.append("")
    return "\n".join(lines)


def default_memory_template() -> str:
    return """# Codex Important Memory

This file is updated by the daily-memory-journal automation. New daily sections are inserted at the top of `## Daily Summaries`.

## Manual Memory

- Keep only durable lessons here: stable paths, recurring pitfalls, preferred workflows, and user preferences.
- When a daily summary reveals a lesson that should persist beyond one day, condense it into one short bullet here.

## Daily Summaries

"""


def upsert_daily_block(memory_text: str, target_day: date, block: str) -> str:
    begin = re.escape(BEGIN_FMT.format(day=target_day.isoformat()))
    end = re.escape(END_FMT.format(day=target_day.isoformat()))
    pattern = re.compile(begin + r".*?" + end + r"\n?", re.DOTALL)
    if pattern.search(memory_text):
        return pattern.sub(block, memory_text)

    anchor = "## Daily Summaries\n"
    if anchor not in memory_text:
        memory_text = memory_text.rstrip() + "\n\n## Daily Summaries\n\n"
    return memory_text.replace(anchor, anchor + "\n" + block, 1)


def main() -> None:
    args = parse_args()
    target_day = local_day_from_args(args)
    sessions_root = Path(args.sessions_root).expanduser()
    memory_file = Path(args.memory_file).expanduser()

    files = session_files_for_day(sessions_root, target_day)
    sessions = [parse_session_file(path) for path in files]
    block = summarize_day(sessions, target_day, args)

    memory_file.parent.mkdir(parents=True, exist_ok=True)
    if memory_file.exists():
        memory_text = memory_file.read_text(encoding="utf-8")
    else:
        memory_text = default_memory_template()

    updated = upsert_daily_block(memory_text, target_day, block)
    memory_file.write_text(updated, encoding="utf-8")
    print(f"Wrote daily summary for {target_day.isoformat()} to {memory_file}")
    print(f"Sessions summarized: {len(sessions)}")


if __name__ == "__main__":
    main()
