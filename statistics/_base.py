"""Shared helpers for statistics scripts: project root resolution + CLI."""

import argparse
import sys
from pathlib import Path


def _resolve_root() -> Path:
    here = Path(__file__).resolve()
    for ancestor in (here, *here.parents):
        if (ancestor.parent / "config.py").is_file():
            return ancestor.parent
    raise RuntimeError("Could not locate project root (no config.py found).")


ROOT = _resolve_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def stats_argparser(
    default_input: str | Path,
    default_output_dir: str | Path,
    description: str = "",
) -> argparse.ArgumentParser:
    """ArgumentParser with --input / --output-dir defaulting to script's config-driven values."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--input", default=str(default_input),
                   help=f"Input CSV (default: {default_input})")
    p.add_argument("--output-dir", default=str(default_output_dir),
                   help=f"Output directory (default: {default_output_dir})")
    return p


def ensure_dir(path: str | Path) -> Path:
    """Create *path* (and parents) and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
