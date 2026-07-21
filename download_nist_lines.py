#!/usr/bin/env python3
"""
Download NIST ASD emission lines for LIBS element matching.

Queries https://physics.nist.gov/cgi-bin/ASD/lines1.pl for each element,
ions I–III, wavelength 180–1022 nm. Saves per-element CSVs and a merged library.

Be polite: default delay between requests is 1.5 s.
Cite NIST ASD in any resulting work:
  https://physics.nist.gov/PhysRefData/ASD/lines_form.html
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import time
from pathlib import Path

import requests

NIST_URL = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl"

# Stable / commonly tabulated elements H–U (skip most synthetics)
ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb",
    "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Sm",
    "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl",
    "Pb", "Bi", "Th", "U",
]

# Max ion stage for LIBS-relevant queries (H/He special-cased below)
DEFAULT_MAX_ION = 3

ION_ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV"}

HEADERS = {
    "User-Agent": (
        "LIBS-line-downloader/1.0 "
        "(research; local LIBS matching; +https://physics.nist.gov/PhysRefData/ASD/)"
    ),
    "Accept": "text/plain,text/csv,text/html,*/*",
    "Referer": "https://physics.nist.gov/PhysRefData/ASD/lines_form.html",
}


def max_ion_for(element: str) -> int:
    if element == "H":
        return 1
    if element == "He":
        return 2
    return DEFAULT_MAX_ION


def spectrum_query(element: str) -> str:
    n = max_ion_for(element)
    if n == 1:
        return f"{element} I"
    return f"{element} I-{ION_ROMAN[n]}"


def nist_params(spectra: str, low_nm: float, high_nm: float) -> dict:
    # Parameters matched to a working browser/CSV query.
    # show_av=2: vacuum <200 nm, air 200–2000 nm, vacuum >2000 nm
    # line_out=1: only lines with observed wavelengths / intensities
    return {
        "spectra": spectra,
        "limits_type": "0",
        "low_w": str(low_nm),
        "upp_w": str(high_nm),
        "unit": "1",
        "de": "0",
        "format": "2",
        "line_out": "1",
        "remove_js": "on",
        "en_unit": "0",
        "output": "0",
        "bibrefs": "1",
        "page_size": "9999",
        "show_obs_wl": "1",
        "show_calc_wl": "1",
        "unc_out": "1",
        "order_out": "0",
        "show_av": "2",
        "tsb_value": "0",
        "A_out": "0",
        "intens_out": "on",
        "allowed_out": "1",
        "forbid_out": "1",
        "conf_out": "on",
        "term_out": "on",
        "enrg_out": "on",
        "J_out": "on",
        "g_out": "on",
        "submit": "Retrieve Data",
    }


def clean_cell(value: str) -> str:
    """Strip NIST Excel-formula quoting: '=""394.4""' -> '394.4'."""
    if value is None:
        return ""
    v = value.strip()
    m = re.fullmatch(r'="+(.*?)"+', v)
    if m:
        return m.group(1)
    if v.startswith('="') and v.endswith('"'):
        return v[2:-1].strip('"')
    return v


def parse_wavelength(raw: str) -> float | None:
    """Parse NIST wavelength strings that may include +, ?, [], () markers."""
    v = clean_cell(raw)
    if not v:
        return None
    # Strip uncertainty / blend markers: 194.851+, [46912.38], (15020464), 604.753?
    v = v.strip()
    v = re.sub(r"[\[\]\(\)]", "", v)
    v = v.rstrip("+-?")
    v = v.strip()
    m = re.match(r"^([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)", v)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def parse_intensity(raw: str) -> float | None:
    """Best-effort numeric intensity (strips letter codes like '24g')."""
    raw = clean_cell(raw)
    if not raw:
        return None
    m = re.match(r"^([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)", raw.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def wavelength_medium_from_header(header: str) -> str:
    h = header.lower()
    if "vac" in h:
        return "vacuum"
    if "air" in h:
        return "air"
    return "unknown"


def parse_nist_csv(text: str, element: str) -> list[dict]:
    """Parse NIST CSV that may switch vacuum/air header mid-file."""
    if "No lines are available" in text or "Input Error" in text:
        return []
    if "challenge-platform" in text or "Just a moment" in text:
        raise RuntimeError("Blocked by Cloudflare challenge; retry later or from a browser session.")

    lines = text.splitlines()
    rows: list[dict] = []
    fieldnames: list[str] | None = None
    medium = "unknown"

    for line in lines:
        if not line.strip():
            continue
        # New header block (vacuum vs air sections)
        if "obs_wl" in line or line.startswith("element,"):
            reader = csv.reader(io.StringIO(line))
            fieldnames = next(reader)
            # Prefer observed wavelength column for medium tag
            obs_cols = [c for c in fieldnames if c.startswith("obs_wl")]
            medium = wavelength_medium_from_header(obs_cols[0] if obs_cols else line)
            continue
        if fieldnames is None:
            continue

        reader = csv.reader(io.StringIO(line))
        try:
            values = next(reader)
        except StopIteration:
            continue
        if len(values) < 3:
            continue

        raw = {fieldnames[i]: clean_cell(values[i]) if i < len(values) else "" for i in range(len(fieldnames))}

        # Wavelength: prefer observed, else Ritz
        obs = ""
        ritz = ""
        for key, val in raw.items():
            lk = key.lower()
            if lk.startswith("obs_wl") and val:
                obs = val
            if lk.startswith("ritz_wl") and val:
                ritz = val
        wl = parse_wavelength(obs) if obs else None
        if wl is None:
            wl = parse_wavelength(ritz) if ritz else None
        if wl is None:
            continue
        wavelength_nm = wl
        observed_nm = parse_wavelength(obs) if obs else None
        ritz_nm = parse_wavelength(ritz) if ritz else None

        sp = raw.get("sp_num") or raw.get("spectr") or ""
        try:
            ion_stage = int(sp) if sp else None
        except ValueError:
            ion_stage = None

        el = raw.get("element") or element
        intens_raw = raw.get("intens", "")
        aki_raw = raw.get("Aki(s^-1)", "") or raw.get("Aki", "")

        rows.append(
            {
                "element": el,
                "ion_stage": ion_stage if ion_stage is not None else "",
                "species": f"{el} {ION_ROMAN.get(ion_stage, str(ion_stage))}" if ion_stage else el,
                "wavelength_nm": wavelength_nm,
                "observed_nm": observed_nm if observed_nm is not None else "",
                "ritz_nm": ritz_nm if ritz_nm is not None else "",
                "wavelength_medium": medium,
                "intensity": parse_intensity(intens_raw) if intens_raw else "",
                "intensity_raw": intens_raw,
                "Aki": float(aki_raw) if aki_raw not in ("",) and re.match(r"^[0-9.eE+-]+$", aki_raw) else "",
                "accuracy": raw.get("Acc", ""),
                "Ei_cm-1": raw.get("Ei(cm-1)", ""),
                "Ek_cm-1": raw.get("Ek(cm-1)", ""),
                "conf_i": raw.get("conf_i", ""),
                "term_i": raw.get("term_i", ""),
                "J_i": raw.get("J_i", ""),
                "conf_k": raw.get("conf_k", ""),
                "term_k": raw.get("term_k", ""),
                "J_k": raw.get("J_k", ""),
            }
        )
    return rows


def fetch_element(
    session: requests.Session,
    element: str,
    low_nm: float,
    high_nm: float,
    timeout: float = 180.0,
) -> tuple[str, list[dict]]:
    """
    Download NIST lines for an element (ions I–III by default).

    Tries the combined I–III query first (line_out=1: observed lines).
    If that is empty — common for some elements (e.g. Nb) where NIST only
    publishes Handbook-style relative intensities — falls back to per-ion
    queries, then to line_out=0 (all tabulated lines including Ritz).
    """
    spectra = spectrum_query(element)
    params = nist_params(spectra, low_nm, high_nm)
    resp = session.get(NIST_URL, params=params, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    text = resp.content.decode("utf-8", errors="replace")
    rows = parse_nist_csv(text, element)

    if not rows:
        rows = _fetch_ions(session, element, low_nm, high_nm, line_out="1", timeout=timeout)
    if not rows:
        # Nb I–III (and a few others) have no line_out=1 ASD rows but do have
        # relative-intensity / Ritz lines under line_out=0.
        rows = _fetch_ions(session, element, low_nm, high_nm, line_out="0", timeout=timeout)
        if rows:
            text = f"# fallback line_out=0 for {element} ({len(rows)} lines)\n"

    return text, rows


def _fetch_ions(
    session: requests.Session,
    element: str,
    low_nm: float,
    high_nm: float,
    *,
    line_out: str,
    timeout: float,
) -> list[dict]:
    rows: list[dict] = []
    for ion in range(1, max_ion_for(element) + 1):
        sp = f"{element} {ION_ROMAN[ion]}"
        params = nist_params(sp, low_nm, high_nm)
        params["line_out"] = line_out
        r = session.get(NIST_URL, params=params, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        part = parse_nist_csv(r.content.decode("utf-8", errors="replace"), element)
        for row in part:
            if row["ion_stage"] == "":
                row["ion_stage"] = ion
                row["species"] = sp
            try:
                stage = int(row["ion_stage"]) if row["ion_stage"] != "" else ion
            except (TypeError, ValueError):
                stage = ion
            # Drop rare NIST artefacts (e.g. sp_num = atomic number)
            if stage < 1 or stage > 6:
                continue
            row["ion_stage"] = stage
            row["species"] = f"{element} {ION_ROMAN.get(stage, stage)}"
            rows.append(row)
        time.sleep(0.5)
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


_LIBRARY_FIELDS = [
    "element",
    "ion_stage",
    "species",
    "wavelength_nm",
    "observed_nm",
    "ritz_nm",
    "wavelength_medium",
    "intensity",
    "intensity_raw",
    "Aki",
    "accuracy",
    "Ei_cm-1",
    "Ek_cm-1",
    "conf_i",
    "term_i",
    "J_i",
    "conf_k",
    "term_k",
    "J_k",
]


def merge_clean_library(clean_dir: Path, merged: Path) -> int:
    """Rebuild libs_line_library.csv from every non-empty clean/*.csv."""
    all_rows: list[dict] = []
    for path in sorted(clean_dir.glob("*.csv")):
        if path.stat().st_size == 0:
            continue
        with path.open(newline="", encoding="utf-8") as f:
            all_rows.extend(csv.DictReader(f))
    if not all_rows:
        return 0
    with merged.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_LIBRARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in sorted(
            all_rows,
            key=lambda r: (float(r["wavelength_nm"]), str(r.get("element", ""))),
        ):
            w.writerow(row)
    return len(all_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk-download NIST ASD lines for LIBS")
    parser.add_argument("--out", type=Path, default=Path("nist_lines"), help="Output directory")
    parser.add_argument("--low", type=float, default=180.0, help="Lower wavelength (nm)")
    parser.add_argument("--high", type=float, default=1022.0, help="Upper wavelength (nm)")
    parser.add_argument("--delay", type=float, default=1.5, help="Seconds between element queries")
    parser.add_argument(
        "--elements",
        nargs="*",
        default=None,
        help="Subset of element symbols (default: full list)",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip elements with existing CSV")
    args = parser.parse_args()

    out_dir = args.out
    raw_dir = out_dir / "raw"
    clean_dir = out_dir / "clean"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    elements = args.elements or ELEMENTS
    session = requests.Session()
    ok, fail, empty = [], [], []

    print(f"Downloading {len(elements)} elements, {args.low}–{args.high} nm → {out_dir}")
    for i, el in enumerate(elements, 1):
        clean_path = clean_dir / f"{el}.csv"
        raw_path = raw_dir / f"{el}.csv"
        if args.skip_existing and clean_path.exists() and clean_path.stat().st_size > 0:
            print(f"[{i}/{len(elements)}] {el}: skip existing")
            continue

        spectra = spectrum_query(el)
        print(f"[{i}/{len(elements)}] {spectra} ...", end=" ", flush=True)
        try:
            raw_text, rows = fetch_element(session, el, args.low, args.high)
            raw_path.write_text(raw_text, encoding="utf-8")
            write_rows(clean_path, rows)
            if rows:
                ok.append(el)
                print(f"{len(rows)} lines")
            else:
                empty.append(el)
                print("no lines")
        except Exception as exc:
            fail.append((el, str(exc)))
            print(f"FAILED: {exc}")
        if i < len(elements):
            time.sleep(args.delay)

    merged = out_dir / "libs_line_library.csv"
    n = merge_clean_library(clean_dir, merged)

    print("\nDone.")
    print(f"  OK: {len(ok)}  empty: {len(empty)}  failed: {len(fail)}")
    if empty:
        print(f"  Empty: {', '.join(empty)}")
    if fail:
        print("  Failures:")
        for el, msg in fail:
            print(f"    {el}: {msg}")
    print(f"  Merged library: {merged} ({n} lines)")
    print("Cite: https://physics.nist.gov/PhysRefData/ASD/lines_form.html")


if __name__ == "__main__":
    main()
