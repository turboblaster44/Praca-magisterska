"""
Run every statistics script in this directory tree and snapshot the outputs
into stat_output/<name>/ for later comparison between runs.

Usage:
    python statistics/run_all.py <run_name>
    python statistics/run_all.py                  # uses timestamp as name

Each script still writes its outputs next to itself (so they can be re-run
individually). After each succeeds, generated *.png / *.csv / *.txt files are
copied into stat_output/<run_name>/<relative_path>/, mirroring the source
folder structure. A run_info.txt with the start datetime + per-script results
is written at the root of the run folder.
"""

import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter


STATS_DIR = Path(__file__).resolve().parent
ROOT = STATS_DIR.parent
OUTPUT_ROOT = ROOT / "stat_output"
THIS_FILE = Path(__file__).resolve()
ARTIFACT_EXTS = {".png", ".csv", ".txt"}


def discover() -> list[Path]:
    return sorted(
        p for p in STATS_DIR.rglob("*.py")
        if p.resolve() != THIS_FILE
        and p.name != "__init__.py"
        and not p.name.startswith("_")
    )


def run_one(script: Path) -> tuple[bool, float, str]:
    start = perf_counter()
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    elapsed = perf_counter() - start
    tail = (result.stdout + result.stderr).strip().splitlines()[-3:]
    return result.returncode == 0, elapsed, "\n  ".join(tail)


def snapshot_outputs(script: Path, run_dir: Path) -> list[Path]:
    src_dir = script.parent
    rel = src_dir.relative_to(STATS_DIR)
    dst_dir = run_dir / rel
    dst_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for f in src_dir.iterdir():
        if f.is_file() and f.suffix.lower() in ARTIFACT_EXTS:
            shutil.copy2(f, dst_dir / f.name)
            copied.append(dst_dir / f.name)
    return copied


def main() -> None:
    started = datetime.now()
    run_name = sys.argv[1] if len(sys.argv) > 1 else started.strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run name : {run_name}")
    print(f"Run dir  : {run_dir.relative_to(ROOT)}")
    print(f"Started  : {started.isoformat(timespec='seconds')}\n")

    scripts = discover()
    print(f"Found {len(scripts)} statistics scripts\n")

    results = []
    for i, script in enumerate(scripts, 1):
        rel = script.relative_to(ROOT)
        print(f"[{i}/{len(scripts)}] {rel}")
        ok, secs, tail = run_one(script)
        mark = "OK " if ok else "FAIL"
        print(f"  -> {mark} ({secs:.1f}s)")
        if tail:
            print(f"  {tail}")

        artifacts = snapshot_outputs(script, run_dir) if ok else []
        results.append((rel, ok, secs, len(artifacts)))

    finished = datetime.now()
    passed = sum(1 for _, ok, _, _ in results if ok)
    failed = len(results) - passed

    info_lines = [
        f"run_name : {run_name}",
        f"started  : {started.isoformat(timespec='seconds')}",
        f"finished : {finished.isoformat(timespec='seconds')}",
        f"duration : {(finished - started).total_seconds():.1f}s",
        f"summary  : {passed} passed, {failed} failed",
        "",
        "scripts:",
    ]
    for rel, ok, secs, n in results:
        mark = "OK  " if ok else "FAIL"
        info_lines.append(f"  [{mark}] {rel}  ({secs:.1f}s, {n} artifacts)")

    (run_dir / "run_info.txt").write_text("\n".join(info_lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Summary: {passed} passed, {failed} failed")
    print(f"Artifacts snapshot: {run_dir.relative_to(ROOT)}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
