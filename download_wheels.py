#!/usr/bin/env python3
"""
Download pip wheels for offline install (air-gapped lab PCs).

Run this on a machine *with* internet, then copy the whole LIBS folder
(including wheels/) to the offline PC and run:  python install.py

Examples:
    # Windows lab PC with Python 3.12 (typical Anaconda):
    python download_wheels.py --platform win_amd64 --python 3.12

    # Also pull macOS / Linux wheels for the current machine:
    python download_wheels.py --platform win_amd64 --python 3.12 --also-current

    python download_wheels.py --help
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIREMENTS = ROOT / "requirements.txt"
DEFAULT_WHEELS = ROOT / "wheels"

# Short tags → pip --platform values
PLATFORM_ALIASES = {
    "win": "win_amd64",
    "windows": "win_amd64",
    "win_amd64": "win_amd64",
    "win_arm64": "win_arm64",
    "mac": "macosx_11_0_arm64",
    "macos": "macosx_11_0_arm64",
    "macos_arm": "macosx_11_0_arm64",
    "macos_intel": "macosx_10_9_x86_64",
    "linux": "manylinux2014_x86_64",
    "manylinux": "manylinux2014_x86_64",
}


def py_tag(version: str) -> tuple[str, str]:
    """'3.12' → ('312', 'cp312')."""
    parts = version.strip().split(".")
    if len(parts) < 2:
        raise SystemExit(f"Bad --python {version!r}; use e.g. 3.12")
    major, minor = parts[0], parts[1]
    short = f"{major}{minor}"
    return short, f"cp{short}"


def download_for_platform(
    *,
    wheels_dir: Path,
    platform: str,
    python: str,
) -> None:
    short, abi = py_tag(python)
    wheels_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "-r",
        str(REQUIREMENTS),
        "-d",
        str(wheels_dir),
        "--python-version",
        short,
        "--platform",
        platform,
        "--implementation",
        "cp",
        "--abi",
        abi,
        "--only-binary",
        ":all:",
    ]
    print(f"  → {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(
            f"pip download failed for platform={platform} python={python}.\n"
            "Check network, platform tag, and that wheels exist on PyPI."
        )


def download_current(wheels_dir: Path) -> None:
    """Wheels for this interpreter (no cross-platform flags)."""
    wheels_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "-r",
        str(REQUIREMENTS),
        "-d",
        str(wheels_dir),
    ]
    print(f"  → {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit("pip download failed for current platform.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download dependency wheels for offline LIBS install."
    )
    parser.add_argument(
        "--platform",
        action="append",
        default=[],
        metavar="TAG",
        help=(
            "Target platform (repeatable). Aliases: win, macos_arm, macos_intel, linux. "
            "Default: win_amd64 (lab Windows PCs)."
        ),
    )
    parser.add_argument(
        "--python",
        default="3.12",
        help="Target CPython version for cross-platform wheels (default: 3.12)",
    )
    parser.add_argument(
        "--also-current",
        action="store_true",
        help="Also download wheels for this machine's Python",
    )
    parser.add_argument(
        "--wheels-dir",
        type=Path,
        default=DEFAULT_WHEELS,
        help=f"Output directory (default: {DEFAULT_WHEELS})",
    )
    args = parser.parse_args()

    if not REQUIREMENTS.is_file():
        raise SystemExit(f"Missing {REQUIREMENTS}")

    platforms = args.platform or ["win_amd64"]
    resolved: list[str] = []
    for p in platforms:
        key = p.strip().lower().replace("-", "_")
        resolved.append(PLATFORM_ALIASES.get(key, p.strip()))

    wheels_dir = (
        args.wheels_dir if args.wheels_dir.is_absolute() else ROOT / args.wheels_dir
    )

    print()
    print("LIBS — download wheels for offline install")
    print("-" * 40)
    print(f"  Requirements: {REQUIREMENTS}")
    print(f"  Output:       {wheels_dir}")
    print(f"  Platforms:    {', '.join(resolved)}")
    print(f"  Python:       {args.python}")
    print()

    for plat in resolved:
        print(f"Downloading for {plat} (CPython {args.python})…")
        download_for_platform(
            wheels_dir=wheels_dir, platform=plat, python=args.python
        )
        print()

    if args.also_current:
        print("Downloading for current interpreter…")
        download_current(wheels_dir)
        print()

    n = len(list(wheels_dir.glob("*.whl"))) + len(list(wheels_dir.glob("*.tar.gz")))
    size_mb = sum(f.stat().st_size for f in wheels_dir.iterdir() if f.is_file()) / (
        1024 * 1024
    )
    print(f"  ✓ {n} package file(s) in {wheels_dir} ({size_mb:.0f} MB)")
    print()
    print("  Copy the whole LIBS folder (with wheels/) to the offline PC, then:")
    print("    python install.py")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        sys.exit(130)
