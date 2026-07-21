#!/usr/bin/env python3
"""
Easy installer for LIBS Spectrum Explorer.

Creates a virtual environment, installs dependencies from requirements.txt,
and downloads the NIST emission-line library only if missing (prompts first).

Usage:
    python3 install.py
    python3 install.py --skip-nist
    python3 install.py --venv .venv
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
NIST_LIBRARY = ROOT / "nist_lines" / "libs_line_library.csv"
MIN_PYTHON = (3, 10)


def info(msg: str) -> None:
    print(f"  → {msg}")


def ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def fail(msg: str, code: int = 1) -> None:
    print(f"  ✗ {msg}", file=sys.stderr)
    sys.exit(code)


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    info(" ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd or ROOT)
    if result.returncode != 0:
        fail(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def python_ok() -> None:
    if sys.version_info < MIN_PYTHON:
        fail(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required "
            f"(found {sys.version_info.major}.{sys.version_info.minor})"
        )
    ok(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")


def venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def create_venv(venv_dir: Path) -> Path:
    py = venv_python(venv_dir)
    if py.is_file():
        ok(f"Virtual environment already exists: {venv_dir}")
        return py

    info(f"Creating virtual environment at {venv_dir}")
    run([sys.executable, "-m", "venv", str(venv_dir)])
    if not py.is_file():
        fail(f"venv created but Python not found at {py}")
    ok(f"Created {venv_dir}")
    return py


def install_requirements(py: Path) -> None:
    if not REQUIREMENTS.is_file():
        fail(f"Missing {REQUIREMENTS}")

    info("Upgrading pip")
    run([str(py), "-m", "pip", "install", "--upgrade", "pip"])

    info(f"Installing dependencies from {REQUIREMENTS.name}")
    run([str(py), "-m", "pip", "install", "-r", str(REQUIREMENTS)])
    ok("Dependencies installed")


def find_nist_library() -> Path | None:
    """Return path to libs_line_library.csv if present and non-empty."""
    if NIST_LIBRARY.is_file() and NIST_LIBRARY.stat().st_size > 0:
        return NIST_LIBRARY
    return None


def prompt_yes_no(question: str, *, default: bool = True) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    try:
        answer = input(f"  ? {question}{suffix}").strip().lower()
    except EOFError:
        return default
    if not answer:
        return default
    return answer in ("y", "yes")


def download_nist(py: Path) -> None:
    script = ROOT / "download_nist_lines.py"
    if not script.is_file():
        fail(f"Missing {script}")

    info("Downloading NIST emission lines (this may take a while)…")
    run([str(py), str(script)])
    if find_nist_library() is None:
        fail(f"Download finished but library not found at {NIST_LIBRARY}")
    ok(f"NIST line library ready: {NIST_LIBRARY}")


def maybe_download_nist(py: Path, *, skip: bool, force: bool) -> None:
    existing = find_nist_library()

    if force:
        info("Re-downloading NIST emission lines (--force-nist)…")
        download_nist(py)
        return

    if existing is not None:
        return

    if skip:
        info("Skipping NIST download (--skip-nist)")
        return

    print()
    print("  Element matching needs the NIST ASD line library.")
    print("  Download from NIST now? (internet required; can take several minutes)")
    if not prompt_yes_no("Download NIST lines now?", default=True):
        info("Skipped. Run later: python download_nist_lines.py")
        return

    download_nist(py)


def print_next_steps(venv_dir: Path) -> None:
    if sys.platform == "win32":
        activate = f"{venv_dir}\\Scripts\\activate"
        gui = f"{venv_dir}\\Scripts\\python libs_gui.py"
    else:
        activate = f"source {venv_dir}/bin/activate"
        gui = f"{venv_dir}/bin/python libs_gui.py"

    print()
    print("=" * 60)
    print("  LIBS Spectrum Explorer is ready.")
    print("=" * 60)
    print()
    print("  Activate the environment:")
    print(f"    {activate}")
    print()
    print("  Or launch the GUI directly:")
    print(f"    {gui}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install LIBS Spectrum Explorer (venv + deps + optional NIST lines)."
    )
    parser.add_argument(
        "--venv",
        type=Path,
        default=DEFAULT_VENV,
        help=f"Virtual environment path (default: {DEFAULT_VENV})",
    )
    parser.add_argument(
        "--skip-nist",
        action="store_true",
        help="Skip downloading the NIST line library if missing",
    )
    parser.add_argument(
        "--force-nist",
        action="store_true",
        help="Re-download NIST lines even if already present",
    )
    args = parser.parse_args()

    print()
    print("LIBS Spectrum Explorer — installer")
    print("-" * 40)

    python_ok()

    # Check NIST first so the user sees the status early
    existing = find_nist_library()
    if existing is not None:
        size_mb = existing.stat().st_size / (1024 * 1024)
        ok(f"NIST line library found: {existing} ({size_mb:.1f} MB)")
    else:
        info(f"NIST line library not found at {NIST_LIBRARY}")

    venv_dir = args.venv if args.venv.is_absolute() else ROOT / args.venv
    py = create_venv(venv_dir)
    install_requirements(py)
    maybe_download_nist(py, skip=args.skip_nist, force=args.force_nist)

    print_next_steps(venv_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        sys.exit(130)
