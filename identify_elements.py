#!/usr/bin/env python3
"""
Read a LIBS spectrum (.txt + optional .cfg) and match peaks to NIST ASD lines.

Example:
  .venv/bin/python identify_elements.py docs/stone-9b.txt
  .venv/bin/python identify_elements.py docs/stone-9b.txt --plot
"""

from __future__ import annotations

import argparse
import configparser
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from matplotlib_config import apply_matplotlib_config


# ---------------------------------------------------------------------------
# Spectrum I/O
# ---------------------------------------------------------------------------


@dataclass
class SpectrumMeta:
    path: Path
    cfg_path: Path | None = None
    n_conditioning_shots: int | None = None
    n_accumulations: int | None = None
    laser_energy_mJ: float | None = None
    qs_delay_us: float | None = None
    integration_time_us: float | None = None
    integration_delay_us: float | None = None
    wavelength_ranges: list[str] | None = None


@dataclass
class Spectrum:
    wavelength_nm: np.ndarray
    intensity: np.ndarray
    meta: SpectrumMeta


def _parse_cfg(cfg_path: Path) -> SpectrumMeta:
    raw = cfg_path.read_text(encoding="utf-8", errors="replace")
    # Config uses "Key = value" — ConfigParser needs section headers (already present)
    parser = configparser.ConfigParser()
    parser.optionxform = str  # keep key case
    parser.read_string(raw)

    def get_int(section: str, key: str) -> int | None:
        if parser.has_option(section, key):
            return int(parser.get(section, key))
        return None

    def get_float(section: str, key: str) -> float | None:
        if parser.has_option(section, key):
            return float(parser.get(section, key))
        return None

    ranges: list[str] = []
    if parser.has_section("Spectrometer"):
        for key, val in parser.items("Spectrometer"):
            if key.lower().startswith("wavelength ranges"):
                ranges.append(val.strip().strip('"'))

    return SpectrumMeta(
        path=cfg_path,  # overwritten by caller
        cfg_path=cfg_path,
        n_conditioning_shots=get_int("Measurement", "No. of conditioning shots"),
        n_accumulations=get_int("Measurement", "No. of accumulations"),
        laser_energy_mJ=get_float("Laser", "Energy (mJ)"),
        qs_delay_us=get_float("Laser", "QS delay (us)"),
        integration_time_us=get_float("Spectrometer", "Integration time (us)"),
        integration_delay_us=get_float("Spectrometer", "Integration delay (us)"),
        wavelength_ranges=ranges or None,
    )


def load_spectrum(txt_path: Path, cfg_path: Path | None = None) -> Spectrum:
    """Load tab-separated wavelength_nm, intensity spectrum."""
    txt_path = Path(txt_path)
    if cfg_path is None:
        candidate = txt_path.with_suffix(".cfg")
        cfg_path = candidate if candidate.exists() else None

    data = np.loadtxt(txt_path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected 2 columns in {txt_path}")
    wl = data[:, 0].astype(float)
    inten = data[:, 1].astype(float)

    if cfg_path is not None:
        meta = _parse_cfg(cfg_path)
        meta.path = txt_path
    else:
        meta = SpectrumMeta(path=txt_path)

    return Spectrum(wavelength_nm=wl, intensity=inten, meta=meta)


def combine_spectra(
    spectra: list[Spectrum],
    *,
    mode: str = "mean",
    label: str | None = None,
) -> Spectrum:
    """
    Combine several spectra onto a common wavelength grid.

    ``mode``:
      - ``"mean"`` / ``"average"`` — arithmetic mean (typical for multi-spot LIBS)
      - ``"sum"`` — intensity sum (boosts weak lines; scales with N)

    If wavelength axes match, intensities are stacked directly. Otherwise each
    spectrum is linearly interpolated onto the first spectrum's wavelength grid.
    """
    if not spectra:
        raise ValueError("No spectra to combine")
    if len(spectra) == 1:
        return spectra[0]

    mode_l = mode.strip().lower()
    if mode_l in ("average", "avg"):
        mode_l = "mean"
    if mode_l not in ("mean", "sum"):
        raise ValueError(f"Unknown combine mode: {mode!r} (use 'mean' or 'sum')")

    ref = spectra[0]
    wl = ref.wavelength_nm.astype(float)
    stack: list[np.ndarray] = []
    for spec in spectra:
        if spec.wavelength_nm.shape == wl.shape and np.allclose(
            spec.wavelength_nm, wl, rtol=0.0, atol=1e-6
        ):
            stack.append(spec.intensity.astype(float))
        else:
            # Guard against non-monotonic axes from some instrument exports
            order = np.argsort(spec.wavelength_nm)
            x = spec.wavelength_nm[order].astype(float)
            y = spec.intensity[order].astype(float)
            stack.append(np.interp(wl, x, y, left=y[0], right=y[-1]))

    arr = np.vstack(stack)
    if mode_l == "sum":
        combined = arr.sum(axis=0)
        op = "sum"
    else:
        combined = arr.mean(axis=0)
        op = "mean"

    names = [s.meta.path.stem for s in spectra]
    stem = label or f"{op}_of_{len(spectra)}"
    # Carry forward shared instrument metadata when present on the first file
    meta = SpectrumMeta(
        path=Path(f"{stem}.txt"),
        cfg_path=ref.meta.cfg_path,
        n_conditioning_shots=ref.meta.n_conditioning_shots,
        n_accumulations=ref.meta.n_accumulations,
        laser_energy_mJ=ref.meta.laser_energy_mJ,
        qs_delay_us=ref.meta.qs_delay_us,
        integration_time_us=ref.meta.integration_time_us,
        integration_delay_us=ref.meta.integration_delay_us,
        wavelength_ranges=list(ref.meta.wavelength_ranges or [])
        + [f"combined:{op}:{','.join(names)}"],
    )
    return Spectrum(wavelength_nm=wl.copy(), intensity=combined, meta=meta)


def write_spectrum(path: Path, spectrum: Spectrum) -> Path:
    """
    Write a spectrum as tab-separated ``wavelength_nm\\tintensity`` (LIBS .txt).

    Compatible with :func:`load_spectrum`. Parent directories are created as needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for wl, inten in zip(
            spectrum.wavelength_nm.astype(float),
            spectrum.intensity.astype(float),
            strict=True,
        ):
            f.write(f"{wl:.6f}\t{inten:.6f}\n")
    return path.resolve()


# ---------------------------------------------------------------------------
# Peak finding
# ---------------------------------------------------------------------------


@dataclass
class Peak:
    wavelength_nm: float
    intensity: float
    prominence: float
    index: int
    manual: bool = False


def find_spectrum_peaks(
    spectrum: Spectrum,
    *,
    prominence: float | None = None,
    min_prominence_frac: float = 0.02,
    distance: int = 4,
    height: float | None = None,
) -> list[Peak]:
    """
    Find emission peaks.

    If prominence is None, use max(absolute floor, frac * max intensity).
    """
    y = spectrum.intensity
    ymax = float(np.max(y))
    if prominence is None:
        prominence = max(200.0, min_prominence_frac * ymax)

    kw: dict = {"prominence": prominence, "distance": distance}
    if height is not None:
        kw["height"] = height

    idx, props = find_peaks(y, **kw)
    peaks = [
        Peak(
            wavelength_nm=float(spectrum.wavelength_nm[i]),
            intensity=float(y[i]),
            prominence=float(props["prominences"][j]),
            index=int(i),
            manual=False,
        )
        for j, i in enumerate(idx)
    ]
    peaks.sort(key=lambda p: p.prominence, reverse=True)
    return peaks


def make_peak_at_wavelength(spectrum: Spectrum, wavelength_nm: float) -> Peak:
    """Snap to the nearest spectrum sample and build a manual Peak."""
    idx = int(np.argmin(np.abs(spectrum.wavelength_nm - wavelength_nm)))
    inten = float(spectrum.intensity[idx])
    wl = float(spectrum.wavelength_nm[idx])
    lo = max(0, idx - 25)
    hi = min(len(spectrum.intensity), idx + 26)
    baseline = float(np.percentile(spectrum.intensity[lo:hi], 10))
    prom = max(inten - baseline, abs(inten) * 0.25, 50.0)
    return Peak(
        wavelength_nm=wl,
        intensity=inten,
        prominence=float(prom),
        index=idx,
        manual=True,
    )


def merge_peaks(auto_peaks: list[Peak], manual_peaks: list[Peak], *, min_sep_nm: float = 0.05) -> list[Peak]:
    """Combine auto + manual peaks; drop manuals that duplicate an auto peak."""
    merged = list(auto_peaks)
    for mp in manual_peaks:
        if any(abs(mp.wavelength_nm - p.wavelength_nm) < min_sep_nm for p in merged):
            continue
        merged.append(mp)
    merged.sort(key=lambda p: p.prominence, reverse=True)
    return merged


# ---------------------------------------------------------------------------
# NIST library + matching
# ---------------------------------------------------------------------------


@dataclass
class LibraryLine:
    element: str
    ion_stage: int | None
    species: str
    wavelength_nm: float
    intensity: float | None
    aki: float | None


def load_line_library(path: Path) -> list[LibraryLine]:
    lines: list[LibraryLine] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                wl = float(row["wavelength_nm"])
            except (KeyError, ValueError, TypeError):
                continue
            ion = row.get("ion_stage") or ""
            try:
                ion_stage = int(float(ion)) if ion not in ("", None) else None
            except ValueError:
                ion_stage = None
            intens = row.get("intensity") or ""
            aki = row.get("Aki") or ""
            try:
                intens_f = float(intens) if intens not in ("", None) else None
            except ValueError:
                intens_f = None
            try:
                aki_f = float(aki) if aki not in ("", None) else None
            except ValueError:
                aki_f = None
            lines.append(
                LibraryLine(
                    element=row.get("element", "").strip(),
                    ion_stage=ion_stage,
                    species=row.get("species") or row.get("element", ""),
                    wavelength_nm=wl,
                    intensity=intens_f,
                    aki=aki_f,
                )
            )
    return lines


def strong_library_lines(
    library: list[LibraryLine],
    elements: list[str] | set[str],
    *,
    wl_min: float,
    wl_max: float,
    max_per_element: int = 50,
    libs_diagnostics: bool = False,
) -> dict[str, list[LibraryLine]]:
    """
    Strongest NIST lines per element inside ``[wl_min, wl_max]``.

    By default, ranking uses tabulated NIST intensity, then Aki.

    When ``libs_diagnostics`` is True, prefer typical LIBS-observable lines:
    neutrals (I) in the optical window with tabulated intensity, plus strong
    singly-ionized (II) lines (e.g. Ca II IR triplet). That avoids ranking on
    Cs III / deep-UV monsters while still allowing ion diagnostics that are
    common in gated air LIBS.
    """
    wanted = {e for e in elements if e}
    if not wanted or not library:
        return {}

    by_el: dict[str, list[LibraryLine]] = {e: [] for e in wanted}
    for line in library:
        if line.element not in wanted:
            continue
        if line.wavelength_nm < wl_min or line.wavelength_nm > wl_max:
            continue
        by_el[line.element].append(line)

    def _rank_raw(line: LibraryLine) -> tuple[float, float]:
        return (line.intensity or 0.0, line.aki or 0.0)

    def _libs_choose(lines: list[LibraryLine], n: int) -> list[LibraryLine]:
        neutrals = [
            L
            for L in lines
            if (L.ion_stage or 0) == 1
            and L.intensity is not None
            and L.intensity > 0
            and 250.0 <= L.wavelength_nm <= 850.0
        ]
        ions = [
            L
            for L in lines
            if (L.ion_stage or 0) == 2
            and L.intensity is not None
            and L.intensity > 0
        ]
        ranked_n = sorted(neutrals, key=_rank_raw, reverse=True)
        ranked_i = sorted(ions, key=_rank_raw, reverse=True)
        if not ranked_n and not ranked_i:
            pool = [L for L in lines if (L.ion_stage or 1) <= 2] or list(lines)
            return sorted(pool, key=_rank_raw, reverse=True)[:n]
        # Neutrals first (presence / primary), but reserve ion slots so
        # Ca II 854/866 etc. are not crowded out by UV Ca II / Ca I only.
        n_ion = min(len(ranked_i), max(0, min(4, n // 2))) if ranked_i else 0
        n_neu = n - n_ion
        chosen: list[LibraryLine] = []
        seen: set[float] = set()

        def _add(src: list[LibraryLine], limit: int) -> None:
            for L in src:
                if len(chosen) >= limit:
                    return
                key = round(L.wavelength_nm, 4)
                if key in seen:
                    continue
                seen.add(key)
                chosen.append(L)

        _add(ranked_n, n_neu)
        _add(ranked_i, n)
        _add(ranked_n, n)
        return chosen

    out: dict[str, list[LibraryLine]] = {}
    for el in elements:  # preserve caller order
        if el not in by_el:
            continue
        if libs_diagnostics:
            out[el] = _libs_choose(by_el[el], max_per_element)
        else:
            ranked = sorted(by_el[el], key=_rank_raw, reverse=True)
            out[el] = ranked[:max_per_element]
    return out


def elements_in_wavelength_range(
    library: list[LibraryLine],
    *,
    wl_min: float,
    wl_max: float,
) -> list[str]:
    """Sorted unique element symbols with at least one NIST line in range."""
    found: set[str] = set()
    for line in library:
        if wl_min <= line.wavelength_nm <= wl_max and line.element:
            found.add(line.element)
    return sorted(found, key=lambda e: (len(e), e))


def browse_library_lines(
    library: list[LibraryLine],
    *,
    element: str | None = None,
    ion_stage: int | None = None,
    wl_min: float,
    wl_max: float,
    max_lines: int = 80,
) -> list[LibraryLine]:
    """
    NIST lines for interactive browsing within a wavelength window.

    Optionally filter by element and ion stage. Ranked by NIST intensity, then Aki.
    """
    lines: list[LibraryLine] = []
    for line in library:
        if line.wavelength_nm < wl_min or line.wavelength_nm > wl_max:
            continue
        if element and line.element != element:
            continue
        if ion_stage is not None and line.ion_stage != ion_stage:
            continue
        lines.append(line)
    lines.sort(
        key=lambda L: (L.intensity or 0.0, L.aki or 0.0, -L.wavelength_nm),
        reverse=True,
    )
    return lines[:max_lines]


@dataclass
class Match:
    peak: Peak
    line: LibraryLine
    delta_nm: float


@dataclass
class ElementHit:
    element: str
    n_peaks: int
    score: float
    confidence: float  # 0–100% identification confidence
    matches: list[Match]
    #: User-added from periodic table (kept across rematch when still absent)
    manual: bool = False


def _has_nearby_peak(
    peak_wls_sorted: np.ndarray,
    wavelength_nm: float,
    tol_nm: float,
) -> bool:
    if peak_wls_sorted.size == 0:
        return False
    i = int(np.searchsorted(peak_wls_sorted, wavelength_nm))
    for j in (i - 1, i):
        if 0 <= j < peak_wls_sorted.size:
            if abs(float(peak_wls_sorted[j]) - wavelength_nm) <= tol_nm:
                return True
    return False


def match_library_lines(
    library: list[LibraryLine],
    *,
    wl_min: float,
    wl_max: float,
    max_per_element: int = 10,
    libs_diagnostics: bool = True,
) -> list[LibraryLine]:
    """
    Subset of ``library`` used for peak→line assignment.

    Keeps only the top ``max_per_element`` LIBS diagnostics per element
    (see ``strong_library_lines(..., libs_diagnostics=True)``). Browse /
    click-inspect should keep using the full NIST catalog.
    """
    if not library or max_per_element <= 0:
        return []
    elements = sorted({L.element for L in library if L.element})
    strong = strong_library_lines(
        library,
        elements,
        wl_min=wl_min,
        wl_max=wl_max,
        max_per_element=max_per_element,
        libs_diagnostics=libs_diagnostics,
    )
    out: list[LibraryLine] = []
    for lines in strong.values():
        out.extend(lines)
    return out


def augment_match_library_near_peaks(
    match_lib: list[LibraryLine],
    library: list[LibraryLine],
    peaks: list[Peak],
    *,
    tol_nm: float,
) -> list[LibraryLine]:
    """
    Add I/II catalog lines within ``tol_nm`` of each observed peak.

    Top-N diagnostic lists are dominated by UV/optical entries, so strong
    IR LIBS features (Ca II 854/866 nm, …) would otherwise never be
    assignable even when they sit on obvious peaks. Unsupported dense-line
    species are still suppressed later by the diagnostic presence prior.
    """
    if not peaks or not library:
        return list(match_lib)
    existing = {(L.element, round(L.wavelength_nm, 4)) for L in match_lib}
    lib_wl = np.asarray([L.wavelength_nm for L in library], dtype=float)
    order = np.argsort(lib_wl)
    lib_wl_sorted = lib_wl[order]
    lib_sorted = [library[i] for i in order]
    extra: list[LibraryLine] = []
    for peak in peaks:
        lo = int(np.searchsorted(lib_wl_sorted, peak.wavelength_nm - tol_nm))
        hi = int(np.searchsorted(lib_wl_sorted, peak.wavelength_nm + tol_nm))
        for k in range(lo, hi):
            line = lib_sorted[k]
            stage = line.ion_stage or 1
            if stage > 2:
                continue
            if line.intensity is None or line.intensity <= 0:
                continue
            key = (line.element, round(line.wavelength_nm, 4))
            if key in existing:
                continue
            existing.add(key)
            extra.append(line)
    if not extra:
        return list(match_lib)
    return list(match_lib) + extra


def element_diagnostic_support(
    peaks: list[Peak],
    library: list[LibraryLine],
    *,
    wl_min: float | None = None,
    wl_max: float | None = None,
    tol_nm: float = 0.10,
    n_diagnostics: int = 5,
    primary_out: dict[str, bool] | None = None,
    primary_wavelength_out: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Fraction of each element's strongest LIBS diagnostics that have a nearby peak.

    Used as a presence prior: if V's top diagnostics (e.g. 411.2, 437.9 nm)
    are absent, V should not win contested overlaps against elements that
    *do* show their strong lines (e.g. Pb 405.8 nm).

    If ``primary_out`` / ``primary_wavelength_out`` are provided, they are
    cleared and filled with whether each element's #1 diagnostic has a nearby
    peak, and that diagnostic wavelength.
    """
    if primary_out is not None:
        primary_out.clear()
    if primary_wavelength_out is not None:
        primary_wavelength_out.clear()
    if not peaks or not library:
        return {}
    peak_wls = np.array(sorted({p.wavelength_nm for p in peaks}), dtype=float)
    if wl_min is None:
        wl_min = float(peak_wls.min()) - 1.0
    if wl_max is None:
        wl_max = float(peak_wls.max()) + 1.0

    elements = sorted({L.element for L in library if L.element})
    strong = strong_library_lines(
        library,
        elements,
        wl_min=wl_min,
        wl_max=wl_max,
        max_per_element=n_diagnostics,
        libs_diagnostics=True,
    )
    out: dict[str, float] = {}
    for el, lines in strong.items():
        if not lines:
            out[el] = 0.0
            if primary_out is not None:
                primary_out[el] = False
            continue
        hits = sum(
            1 for L in lines if _has_nearby_peak(peak_wls, L.wavelength_nm, tol_nm)
        )
        out[el] = float(hits) / float(len(lines))
        if primary_out is not None:
            primary_out[el] = _has_nearby_peak(
                peak_wls, lines[0].wavelength_nm, tol_nm
            )
        if primary_wavelength_out is not None:
            primary_wavelength_out[el] = float(lines[0].wavelength_nm)
    return out


def match_peaks(
    peaks: list[Peak],
    library: list[LibraryLine],
    *,
    tol_nm: float = 0.12,
    prefer_strong_library: bool = True,
    use_diagnostic_prior: bool = True,
    diagnostic_tol_nm: float | None = None,
    n_diagnostics: int = 5,
    match_max_per_element: int | None = 10,
    match_libs_diagnostics: bool = True,
    diagnostic_support_out: dict[str, float] | None = None,
    primary_diagnostic_out: dict[str, bool] | None = None,
    primary_wavelength_out: dict[str, float] | None = None,
) -> list[Match]:
    """
    Assign each peak to the best nearby NIST line (Δλ + NIST intensity).

    By default only each element's top ``match_max_per_element`` LIBS
    diagnostics are considered (``libs_diagnostics`` ranking). Pass
    ``match_max_per_element=None`` to search the full library. Browse NIST
    should keep using the unfiltered catalog.

    When ``use_diagnostic_prior`` is True (default), candidates from elements
    whose strongest diagnostic lines are missing from the spectrum are
    heavily down-weighted (and lose to any supported competitor). That stops
    dense, high-NIST-I species (e.g. V) from stealing peaks when their primary
    lines are absent. Top diagnostics of *supported* elements also get a
    relaxed Δλ so modest wavelength offsets still claim the right peak.

    If ``diagnostic_support_out`` / ``primary_diagnostic_out`` /
    ``primary_wavelength_out`` are provided, they are cleared and filled for
    ``score_elements``.
    """
    if not peaks or not library:
        if diagnostic_support_out is not None:
            diagnostic_support_out.clear()
        if primary_diagnostic_out is not None:
            primary_diagnostic_out.clear()
        if primary_wavelength_out is not None:
            primary_wavelength_out.clear()
        return []

    peak_span_lo = min(p.wavelength_nm for p in peaks) - 1.0
    peak_span_hi = max(p.wavelength_nm for p in peaks) + 1.0

    # Restrict assignment to top-N LIBS diagnostics (Browse stays on full library).
    match_lib = library
    if match_max_per_element is not None:
        match_lib = match_library_lines(
            library,
            wl_min=peak_span_lo,
            wl_max=peak_span_hi,
            max_per_element=int(match_max_per_element),
            libs_diagnostics=match_libs_diagnostics,
        )
        if not match_lib:
            match_lib = library
        else:
            # Peak-local I/II lines (Ca II IR triplet, …) omitted from top-N
            match_lib = augment_match_library_near_peaks(
                match_lib, library, peaks, tol_nm=tol_nm
            )

    lib_wl = np.array([L.wavelength_nm for L in match_lib])
    order = np.argsort(lib_wl)
    lib_wl_sorted = lib_wl[order]
    lib_sorted = [match_lib[i] for i in order]

    # Same-element / same-ion neighbors for multiplet consistency scoring
    by_el_ion: dict[tuple[str, int], list[LibraryLine]] = {}
    for line in match_lib:
        key = (line.element, int(line.ion_stage or 1))
        by_el_ion.setdefault(key, []).append(line)

    peak_wls_sorted = np.array(sorted({p.wavelength_nm for p in peaks}), dtype=float)

    support: dict[str, float] = {}
    primary: dict[str, bool] = {}
    primary_wl: dict[str, float] = {}
    top_diag_wls: dict[str, set[float]] = {}
    if use_diagnostic_prior:
        # Slightly tighter than peak-match tol: dense spectra otherwise give
        # spurious "support" to rare earths from chance overlaps.
        d_tol = float(diagnostic_tol_nm) if diagnostic_tol_nm is not None else min(
            max(tol_nm, 0.08), 0.10
        )
        # Support / primary gates use the full library's diagnostic ranking so
        # presence tests stay consistent even when assignment is top-N only.
        support = element_diagnostic_support(
            peaks,
            library,
            wl_min=peak_span_lo,
            wl_max=peak_span_hi,
            tol_nm=d_tol,
            n_diagnostics=n_diagnostics,
            primary_out=primary,
            primary_wavelength_out=primary_wl,
        )
        # Soft Δλ boost for lines in the match (top-N) set.
        strong = strong_library_lines(
            library,
            sorted(support.keys()),
            wl_min=peak_span_lo,
            wl_max=peak_span_hi,
            max_per_element=(
                int(match_max_per_element)
                if match_max_per_element is not None
                else n_diagnostics
            ),
            libs_diagnostics=match_libs_diagnostics,
        )
        for el, lines in strong.items():
            top_diag_wls[el] = {round(L.wavelength_nm, 4) for L in lines}

    if diagnostic_support_out is not None:
        diagnostic_support_out.clear()
        diagnostic_support_out.update(support)
    if primary_diagnostic_out is not None:
        primary_diagnostic_out.clear()
        primary_diagnostic_out.update(primary)
    if primary_wavelength_out is not None:
        primary_wavelength_out.clear()
        primary_wavelength_out.update(primary_wl)

    def line_strength(line: LibraryLine) -> float:
        s = 1.0
        if line.intensity is not None and line.intensity > 0:
            # Soft-cap: NIST relative I can be huge for obscure Fe lines and
            # drown real LIBS diagnostics (Ca II IR vs Fe I 866).
            s += min(float(line.intensity), 2.0e3)
        if line.aki is not None and line.aki > 0:
            s += min(line.aki / 1e6, 5e4)
        return s

    def multiplet_hits(line: LibraryLine) -> int:
        """Sibling same-element/ion lines within 40 nm that also have peaks."""
        sibs = by_el_ion.get((line.element, int(line.ion_stage or 1)), [])
        if len(sibs) < 2:
            return 0
        n_hit = 0
        for s in sibs:
            if abs(s.wavelength_nm - line.wavelength_nm) < 0.02:
                continue
            if abs(s.wavelength_nm - line.wavelength_nm) > 40.0:
                continue
            if _has_nearby_peak(peak_wls_sorted, s.wavelength_nm, tol_nm):
                n_hit += 1
        return n_hit

    def presence_scale(line: LibraryLine) -> float:
        """1 = full credit; near 0 = element's diagnostics are missing."""
        if not use_diagnostic_prior:
            return 1.0
        el = line.element
        s = float(support.get(el, 0.0))
        is_top_diag = round(line.wavelength_nm, 4) in top_diag_wls.get(el, ())
        if is_top_diag:
            # Contested peak that is itself a primary diagnostic — keep fair weight
            return 0.05 + 0.95 * max(s, 0.55)
        if s <= 0.0:
            # No optical diagnostics — still allow clear ion multiplets
            # (Ca II 854+866) so they are not left unassigned.
            return 1e-8
        return 0.05 + 0.95 * s

    def candidate_score(delta: float, line: LibraryLine) -> float:
        # Squared Δλ so near-exact matches win; intensity breaks near-ties.
        abs_d = abs(delta)
        n_multi = multiplet_hits(line) if prefer_strong_library else 0
        ion_multi = n_multi >= 1 and (line.ion_stage or 1) == 2
        if prefer_strong_library and use_diagnostic_prior:
            el = line.element
            s = float(support.get(el, 0.0))
            is_top_diag = round(line.wavelength_nm, 4) in top_diag_wls.get(el, ())
            # Calibration / wavelength offsets often put the observed peak
            # 0.05–0.08 nm off a real diagnostic (e.g. Pb I 405.78). Shrink
            # effective Δλ so primary lines of *present* elements beat closer
            # secondary overlaps from other species.
            if is_top_diag and s > 0.0:
                abs_d *= 0.35
            # Ion multiplet with confirming sibling peaks (Ca II 854+866)
            elif ion_multi:
                abs_d *= 0.35
            elif s > 0.35 and (line.ion_stage or 1) <= 2:
                abs_d *= 0.55
        d_term = (abs_d / 0.05) ** 2
        if prefer_strong_library:
            scale = presence_scale(line)
            if scale < 1e-6:
                if ion_multi:
                    # Self-supported ion multiplet (no optical Ca I needed)
                    scale = 0.60
                else:
                    # Hard reject unsupported overlap candidates
                    return d_term + 1e6
            strength = max(line_strength(line) * scale, 1.0)
            multi_term = -0.85 * min(n_multi, 3)
            return d_term - 0.45 * np.log10(strength) + multi_term
        return d_term

    _REJECT = 1e5  # scores with +1e6 hard-reject land above this

    matches: list[Match] = []
    for peak in peaks:
        lo = int(np.searchsorted(lib_wl_sorted, peak.wavelength_nm - tol_nm))
        hi = int(np.searchsorted(lib_wl_sorted, peak.wavelength_nm + tol_nm))
        best_ok: tuple[float, LibraryLine, float] | None = None
        best_any: tuple[float, LibraryLine, float] | None = None
        for k in range(lo, hi):
            line = lib_sorted[k]
            d = line.wavelength_nm - peak.wavelength_nm
            sc = candidate_score(d, line)
            if best_any is None or sc < best_any[0]:
                best_any = (sc, line, d)
            if sc < _REJECT and (best_ok is None or sc < best_ok[0]):
                best_ok = (sc, line, d)
        # Prefer a supported / diagnostic candidate. Only fall back to an
        # unsupported overlap when *no* supported element has a line nearby
        # (keeps matrix lines like Fe I 526.9 when NIST "top" lines are UV).
        best = best_ok if best_ok is not None else best_any
        if best is not None:
            _, line, d = best
            matches.append(Match(peak=peak, line=line, delta_nm=d))
    return matches


# Elements that often show only one strong LIBS line (e.g. H-α) still deserve ranking.
SINGLE_PEAK_ELEMENTS = frozenset({"H"})

# Air / plasma background — do not demand a geological primary diagnostic.
ATMOSPHERE_ELEMENTS = frozenset({"H", "C", "N", "O"})

# Dense-line / rare species that often get chance hits in busy spectra (e.g. soil
# CRMs). Require the #1 LIBS diagnostic to be present, plus modest support.
STRICT_PRIMARY_ELEMENTS = frozenset(
    {
        # Lanthanides / actinides
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Sc",
        "Y",
        "Th",
        "U",
        # Dense-line / uncommon metals that steal peaks without their primary
        "Cs",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Ru",
        "Rh",
        "Pd",
        "Hg",
        "Nb",
        "Mo",
        "Sn",
        "Sb",
        "Ge",
        "In",
        "Ga",
        "Te",
    }
)


def _ion_multiplet_cluster(matches: list[Match], *, window_nm: float = 40.0) -> bool:
    """True if ≥2 stage-II lines cluster within ``window_nm`` (e.g. Ca II IR)."""
    ions = [m for m in matches if (m.line.ion_stage or 1) == 2]
    if len(ions) < 2:
        return False
    wls = [float(m.line.wavelength_nm) for m in ions]
    return (max(wls) - min(wls)) <= window_nm


def score_elements(
    matches: list[Match],
    *,
    min_peaks: int = 2,
    single_peak_elements: frozenset[str] | set[str] = SINGLE_PEAK_ELEMENTS,
    diagnostic_support: dict[str, float] | None = None,
    primary_diagnostic: dict[str, bool] | None = None,
    primary_wavelength: dict[str, float] | None = None,
) -> list[ElementHit]:
    """
    Rank elements and assign a 0–100% confidence for each.

    Most elements need ``min_peaks`` matched peaks (default 2). Hydrogen is
    allowed with a single peak (usually H-α), since other Balmer lines are
    often weak or absent in air LIBS.

    When ``diagnostic_support`` / ``primary_diagnostic`` are provided (from
    ``match_peaks``), incidental overlaps in dense spectra are filtered:
    rare-earth / uncommon species must actually match their #1 diagnostic;
    ordinary elements need the primary or ≥4 confirming peaks.
    Clear ion multiplets (e.g. Ca II 854+866 nm) are kept even when optical
    neutrals are weak or absent.
    """
    by_el: dict[str, list[Match]] = {}
    for m in matches:
        by_el.setdefault(m.line.element, []).append(m)

    hits: list[ElementHit] = []
    for el, ms in by_el.items():
        required = 1 if el in single_peak_elements else min_peaks
        if len(ms) < required:
            continue
        support = (
            float(diagnostic_support.get(el, 0.0))
            if diagnostic_support is not None
            else None
        )
        has_primary = (
            bool(primary_diagnostic.get(el, False))
            if primary_diagnostic is not None
            else None
        )
        p_wl = (
            primary_wavelength.get(el) if primary_wavelength is not None else None
        )
        multiplet_ok = _ion_multiplet_cluster(ms)

        # Weak incidental matches with zero diagnostic backbone → skip
        if (
            support is not None
            and el not in single_peak_elements
            and support <= 0.0
            and len(ms) < 4
            and not multiplet_ok
        ):
            continue

        # Dense-spectrum false IDs (Tm, Eu, Pr, Mo, …): must *assign* a peak
        # to the #1 diagnostic (not merely have some other peak nearby).
        if has_primary is not None and el in STRICT_PRIMARY_ELEMENTS:
            if not has_primary:
                continue
            if support is not None and support < 0.4:
                continue
            if p_wl is not None:
                claimed = [
                    m
                    for m in ms
                    if abs(m.line.wavelength_nm - p_wl) <= 0.02
                ]
                if not claimed or min(abs(m.delta_nm) for m in claimed) > 0.08:
                    continue

        # Ordinary elements: primary diagnostic, or enough confirming peaks
        # (matrix lines whose NIST "top" λ may sit outside the observed set).
        if (
            has_primary is not None
            and el not in ATMOSPHERE_ELEMENTS
            and el not in single_peak_elements
            and el not in STRICT_PRIMARY_ELEMENTS
            and not has_primary
            and len(ms) < 4
            and not multiplet_ok
        ):
            continue

        score = 0.0
        qualities: list[float] = []
        prominences: list[float] = []
        nist_logs: list[float] = []
        for m in ms:
            lib_boost = 1.0
            inten = m.line.intensity
            if inten is not None and inten > 0:
                lib_boost += 0.1 * np.log10(1.0 + inten)
                nist_logs.append(float(np.log10(1.0 + inten)))
            else:
                nist_logs.append(0.0)
            # Strong NIST lines often sit 0.05–0.1 nm off after calibration;
            # do not punish those IDs as harshly as weak overlaps.
            delta_ref = 0.08 if (inten is not None and inten >= 1e4) else 0.05
            q = 1.0 / (1.0 + abs(m.delta_nm) / delta_ref)
            qualities.append(q)
            prominences.append(float(np.log1p(m.peak.prominence)))
            score += np.log1p(m.peak.prominence) * q * lib_boost

        n = len(ms)
        mean_q = float(np.mean(qualities))
        # More confirming peaks → higher confidence (diminishing returns)
        coverage = 1.0 - float(np.exp(-(n - required + 1) / 3.0))
        # Typical strong LIBS peaks: log1p(prominence) ~ 6–10
        strength = min(1.0, float(np.mean(prominences)) / 9.0)
        # Matching high-NIST-I lines (true diagnostics) raises confidence
        nist_weight = min(1.0, float(np.mean(nist_logs)) / 5.0)
        confidence = 100.0 * (
            0.30 * coverage
            + 0.30 * mean_q
            + 0.15 * strength
            + 0.25 * nist_weight
        )
        # Trace IDs: few peaks but on the element's strongest NIST lines
        # (e.g. Pb I 405.78 / 368.35) deserve a clear ranking boost.
        median_inten = float(
            np.median(
                [
                    (m.line.intensity if m.line.intensity is not None else 0.0)
                    for m in ms
                ]
            )
        )
        if median_inten >= 5e4:
            confidence += 12.0
        if support is not None and support > 0.0:
            confidence += 6.0 * support
        if has_primary:
            confidence += 5.0
        if multiplet_ok:
            confidence += 8.0
        confidence = float(np.clip(confidence, 0.0, 99.0))

        hits.append(
            ElementHit(
                element=el,
                n_peaks=n,
                score=float(score),
                confidence=confidence,
                matches=ms,
            )
        )
    hits.sort(key=lambda h: (h.confidence, h.score), reverse=True)
    return hits


@dataclass
class Candidate:
    line: LibraryLine
    delta_nm: float
    score: float


def candidates_near_wavelength(
    wavelength_nm: float,
    library: list[LibraryLine],
    *,
    tol_nm: float = 0.15,
    max_results: int = 25,
    prefer_element: str | None = None,
    prefer_wavelength_nm: float | None = None,
) -> list[Candidate]:
    """
    Return NIST lines near a clicked wavelength.

    Sort order:
      1. Exact Match assignment (``prefer_wavelength_nm``), if any
      2. Other lines from the matched element (``prefer_element``)
      3. Remaining lines by strongest NIST intensity, then smallest |Δλ|
    """
    if not library:
        return []

    lib_wl = np.array([L.wavelength_nm for L in library])
    order = np.argsort(lib_wl)
    lib_wl_sorted = lib_wl[order]
    lib_sorted = [library[i] for i in order]

    lo = int(np.searchsorted(lib_wl_sorted, wavelength_nm - tol_nm))
    hi = int(np.searchsorted(lib_wl_sorted, wavelength_nm + tol_nm))

    prefer_el = (prefer_element or "").strip() or None

    cands: list[Candidate] = []
    for k in range(lo, hi):
        line = lib_sorted[k]
        d = line.wavelength_nm - wavelength_nm
        strength = 1.0
        if line.intensity is not None and line.intensity > 0:
            strength += line.intensity
        if line.aki is not None and line.aki > 0:
            strength += min(line.aki / 1e6, 5e4)
        # Keep score for callers that still use it; primary ordering is below.
        score = (abs(d) / 0.05) ** 2 - 0.45 * np.log10(strength)
        cands.append(Candidate(line=line, delta_nm=d, score=float(score)))

    def _sort_key(c: Candidate) -> tuple:
        is_exact = (
            prefer_wavelength_nm is not None
            and abs(c.line.wavelength_nm - prefer_wavelength_nm) < 1e-3
        )
        is_pref_el = prefer_el is not None and c.line.element == prefer_el
        if is_exact:
            pref_rank = 0
        elif is_pref_el:
            pref_rank = 1
        else:
            pref_rank = 2
        inten = c.line.intensity if c.line.intensity is not None else -1.0
        return (pref_rank, -float(inten), abs(c.delta_nm))

    cands.sort(key=_sort_key)
    return cands[:max_results]


def nearest_peak(peaks: list[Peak], wavelength_nm: float, *, max_dist_nm: float = 1.0) -> Peak | None:
    """Find the closest detected peak to a wavelength, or None if too far."""
    if not peaks:
        return None
    best = min(peaks, key=lambda p: abs(p.wavelength_nm - wavelength_nm))
    if abs(best.wavelength_nm - wavelength_nm) > max_dist_nm:
        return None
    return best


# ---------------------------------------------------------------------------
# Reporting / plot
# ---------------------------------------------------------------------------


def print_report(
    spectrum: Spectrum,
    peaks: list[Peak],
    hits: list[ElementHit],
    *,
    top_elements: int = 15,
    top_lines_per_element: int = 5,
) -> None:
    meta = spectrum.meta
    print(f"Spectrum: {meta.path.name}")
    if meta.cfg_path:
        print(f"Config:   {meta.cfg_path.name}")
        if meta.laser_energy_mJ is not None:
            print(f"  Laser: {meta.laser_energy_mJ:g} mJ")
        if meta.qs_delay_us is not None:
            print(f"  QS delay: {meta.qs_delay_us:g} µs")
        if meta.integration_time_us is not None:
            print(
                f"  Gate: {meta.integration_time_us:g} µs  "
                f"delay {meta.integration_delay_us} µs"
                if meta.integration_delay_us is not None
                else f"  Gate: {meta.integration_time_us:g} µs"
            )
        if meta.n_accumulations is not None:
            print(f"  Accumulations: {meta.n_accumulations}")
    print(
        f"Range: {spectrum.wavelength_nm.min():.2f}–{spectrum.wavelength_nm.max():.2f} nm  "
        f"({len(spectrum.wavelength_nm)} points)"
    )
    print(f"Peaks found: {len(peaks)}")
    print()
    print(f"{'Rank':<5} {'Element':<8} {'#peaks':>6} {'Conf%':>8}")
    print("-" * 32)
    for i, hit in enumerate(hits[:top_elements], 1):
        print(f"{i:<5} {hit.element:<8} {hit.n_peaks:>6} {hit.confidence:>7.0f}%")

    print()
    print("Top matched lines by element:")
    for hit in hits[:top_elements]:
        print(f"\n  {hit.element}  (confidence={hit.confidence:.0f}%, n={hit.n_peaks})")
        ms = sorted(hit.matches, key=lambda m: m.peak.prominence, reverse=True)
        for m in ms[:top_lines_per_element]:
            lib_i = f"{m.line.intensity:.0f}" if m.line.intensity is not None else "-"
            print(
                f"    peak {m.peak.wavelength_nm:8.3f} nm (I={m.peak.intensity:8.0f})  "
                f"→ {m.line.species:<7} {m.line.wavelength_nm:8.3f} nm  "
                f"Δ={m.delta_nm:+.3f}  NIST_I={lib_i}"
            )


def plot_spectrum(
    spectrum: Spectrum,
    peaks: list[Peak],
    hits: list[ElementHit],
    *,
    out_path: Path | None = None,
    top_label_elements: int = 8,
    max_labels: int = 25,
) -> None:
    apply_matplotlib_config()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(spectrum.wavelength_nm, spectrum.intensity, color="#1a1a1a", label="spectrum")

    # Label strongest peaks for top elements
    top_els = {h.element for h in hits[:top_label_elements]}
    labeled = 0
    for hit in hits[:top_label_elements]:
        for m in sorted(hit.matches, key=lambda x: x.peak.prominence, reverse=True)[:4]:
            if labeled >= max_labels:
                break
            p = m.peak
            ax.plot(p.wavelength_nm, p.intensity, "o", ms=3.5, color="#c0392b")
            ax.annotate(
                f"{m.line.species}\n{p.wavelength_nm:.1f}",
                (p.wavelength_nm, p.intensity),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=7,
                color="#8b1a1a",
            )
            labeled += 1

    # Mark unlabeled peaks lightly
    for p in peaks:
        ax.axvline(p.wavelength_nm, color="#c0392b", alpha=0.08, lw=0.6)

    title = spectrum.meta.path.stem
    ax.set_title(f"LIBS identification — {title}")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity (counts)")
    ax.set_xlim(spectrum.wavelength_nm.min(), spectrum.wavelength_nm.max())
    if top_els:
        ax.text(
            0.01,
            0.98,
            "Candidates: " + ", ".join(h.element for h in hits[:top_label_elements]),
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="#ccc"),
        )
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path)
        print(f"Wrote plot: {out_path}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Identify elements in a LIBS spectrum via NIST lines")
    parser.add_argument("spectrum", type=Path, help="Spectrum .txt (wavelength<TAB>intensity)")
    parser.add_argument("--cfg", type=Path, default=None, help="Optional .cfg (default: same stem)")
    parser.add_argument(
        "--library",
        type=Path,
        default=Path("nist_lines/libs_line_library.csv"),
        help="NIST line library CSV",
    )
    parser.add_argument("--tol", type=float, default=0.12, help="Wavelength match tolerance (nm)")
    parser.add_argument("--prominence", type=float, default=None, help="Peak prominence threshold")
    parser.add_argument("--prominence-frac", type=float, default=0.015, help="Prominence as fraction of max")
    parser.add_argument("--top", type=int, default=15, help="Top elements to report")
    parser.add_argument("--plot", action="store_true", help="Show/save annotated spectrum plot")
    parser.add_argument("--plot-out", type=Path, default=None, help="Save plot to this path")
    args = parser.parse_args()

    spectrum = load_spectrum(args.spectrum, args.cfg)
    peaks = find_spectrum_peaks(
        spectrum,
        prominence=args.prominence,
        min_prominence_frac=args.prominence_frac,
    )
    library = load_line_library(args.library)
    support: dict[str, float] = {}
    primary: dict[str, bool] = {}
    primary_wl: dict[str, float] = {}
    matches = match_peaks(
        peaks,
        library,
        tol_nm=args.tol,
        diagnostic_support_out=support,
        primary_diagnostic_out=primary,
        primary_wavelength_out=primary_wl,
    )
    hits = score_elements(
        matches,
        min_peaks=2,
        diagnostic_support=support,
        primary_diagnostic=primary,
        primary_wavelength=primary_wl,
    )
    print_report(spectrum, peaks, hits, top_elements=args.top)

    if args.plot or args.plot_out:
        out = args.plot_out
        if out is None and args.plot:
            out = args.spectrum.with_name(args.spectrum.stem + "_identified.png")
        plot_spectrum(spectrum, peaks, hits, out_path=out)


if __name__ == "__main__":
    main()
