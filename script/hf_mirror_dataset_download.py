#!/usr/bin/env python3

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import requests
from huggingface_hub import hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face dataset through a mirror endpoint."
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=20)
    parser.add_argument("--page-pause", type=float, default=0.2)
    return parser.parse_args()


def rewrite_url(url: str, endpoint: str) -> str:
    endpoint_parts = urlparse(endpoint)
    parsed = urlparse(url)
    return urlunparse(
        (
            endpoint_parts.scheme,
            endpoint_parts.netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )


def get_cache_paths(local_dir: Path) -> tuple[Path, Path]:
    cache_dir = local_dir / ".cache" / "hf_mirror_download"
    return cache_dir / "manifest.txt", cache_dir / "state.json"


def load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text())


def save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True))


def compute_retry_sleep(exc: Exception, attempt: int) -> int:
    response = getattr(exc, "response", None)
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(int(retry_after), 1)
            except ValueError:
                pass
        if response.status_code == 429:
            return min(30 * attempt, 600)
    return min(2 ** (attempt - 1), 30)


def list_repo_files(
    repo_id: str,
    local_dir: Path,
    endpoint: str,
    revision: str,
    timeout: float,
    retries: int,
    page_pause: float,
) -> list[str]:
    session = requests.Session()
    manifest_path, state_path = get_cache_paths(local_dir)
    state = load_state(state_path)
    saved_next_url = state.get("next_url")
    url = saved_next_url or f"{endpoint.rstrip('/')}/api/datasets/{repo_id}/tree/{revision}"
    params = None if saved_next_url else {
        "recursive": "true",
        "expand": "false",
        "limit": "1000",
    }
    headers = {"Accept": "application/json"}
    count = int(state.get("count", 0))
    listing_complete = bool(state.get("listing_complete", False))

    if saved_next_url and manifest_path.exists():
        print(f"[list] resuming from count={count}", flush=True)
    elif not listing_complete:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("")
        save_state(
            state_path,
            {"count": 0, "listing_complete": False, "next_url": url},
        )
        count = 0

    while url and not listing_complete:
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                response = session.get(url, params=params, headers=headers, timeout=timeout)
                response.raise_for_status()
                payload = response.json()
                page_files = [
                    item["path"] for item in payload if item.get("type") == "file" and "path" in item
                ]
                next_url = response.links.get("next", {}).get("url")
                url = rewrite_url(next_url, endpoint) if next_url else None
                params = None
                if page_files:
                    with manifest_path.open("a", encoding="utf-8") as fh:
                        for path in page_files:
                            fh.write(path)
                            fh.write("\n")
                count += len(page_files)
                listing_complete = url is None
                save_state(
                    state_path,
                    {
                        "count": count,
                        "listing_complete": listing_complete,
                        "next_url": url,
                    },
                )
                last_error = None
                break
            except Exception as exc:  # pragma: no cover - best effort CLI utility
                last_error = exc
                if attempt == retries:
                    raise
                sleep_s = compute_retry_sleep(exc, attempt)
                print(
                    f"[list] attempt {attempt}/{retries} failed for {url}: {exc}; retrying in {sleep_s}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(sleep_s)
        if last_error is not None:
            raise last_error
        print(f"[list] collected {count} files so far", flush=True)
        if url and page_pause > 0:
            time.sleep(page_pause)

    files = [line.strip() for line in manifest_path.read_text().splitlines() if line.strip()]
    print(f"[list] manifest loaded with {len(files)} files", flush=True)
    return files


def download_one(
    repo_id: str,
    filename: str,
    local_dir: str,
    endpoint: str,
    revision: str,
    retries: int,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                local_dir=local_dir,
                endpoint=endpoint,
                revision=revision,
            )
        except Exception as exc:  # pragma: no cover - best effort CLI utility
            last_error = exc
            if attempt == retries:
                raise
            sleep_s = min(2 ** (attempt - 1), 8)
            print(
                f"[download] attempt {attempt}/{retries} failed for {filename}: {exc}; retrying in {sleep_s}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(sleep_s)
    assert last_error is not None
    raise last_error


def main() -> int:
    args = parse_args()
    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[start] repo={args.repo_id} endpoint={args.endpoint} target={local_dir} workers={args.max_workers}",
        flush=True,
    )
    files = list_repo_files(
        repo_id=args.repo_id,
        local_dir=local_dir,
        endpoint=args.endpoint,
        revision=args.revision,
        timeout=args.timeout,
        retries=args.retries,
        page_pause=args.page_pause,
    )
    print(f"[list] total files={len(files)}", flush=True)

    pending_files = []
    skipped_existing = 0
    for filename in files:
        if (local_dir / filename).exists():
            skipped_existing += 1
            continue
        pending_files.append(filename)

    if skipped_existing:
        print(
            f"[download] skipping {skipped_existing} existing files",
            flush=True,
        )
    print(f"[download] pending files={len(pending_files)}", flush=True)

    completed = 0
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        future_map = {
            pool.submit(
                download_one,
                args.repo_id,
                filename,
                str(local_dir),
                args.endpoint,
                args.revision,
                args.retries,
            ): filename
            for filename in pending_files
        }
        for future in as_completed(future_map):
            filename = future_map[future]
            try:
                future.result()
                completed += 1
                if (
                    completed == 1
                    or completed % 100 == 0
                    or completed == len(pending_files)
                ):
                    print(
                        f"[download] completed {completed}/{len(pending_files)}",
                        flush=True,
                    )
            except Exception as exc:  # pragma: no cover - best effort CLI utility
                failures.append((filename, str(exc)))
                print(f"[download] FAILED {filename}: {exc}", file=sys.stderr, flush=True)

    if failures:
        print(f"[done] failures={len(failures)}", file=sys.stderr, flush=True)
        for filename, error in failures[:20]:
            print(f"[done] sample failure {filename}: {error}", file=sys.stderr, flush=True)
        return 1

    print("[done] success", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
