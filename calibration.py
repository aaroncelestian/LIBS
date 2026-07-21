"""
CRM univariate LIBS calibration.

Workflow:
  1. Load standard spectra (+ optional .cfg)
  2. Enter known concentrations for elements of interest
  3. Local baseline subtract + integrate diagnostic lines
  4. Soft-flag overlapping NIST neighbors
  5. Fit intensity → concentration curves (linear / optional quadratic)
  6. Apply to an unknown spectrum (multi-line average)
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from identify_elements import LibraryLine, Spectrum, load_spectrum


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class DiagnosticLine:
    """A candidate calibrant wavelength for one element."""

    element: str
    wavelength_nm: float
    species: str = ""
    enabled: bool = True
    overlap_warning: str | None = None


@dataclass
class StandardSample:
    """One CRM / standard spectrum with known concentrations."""

    sample_id: str
    spectrum: Spectrum
    concentrations: dict[str, float | None] = field(default_factory=dict)
    atmosphere: str = "unknown"

    @property
    def path(self) -> Path:
        return self.spectrum.meta.path


@dataclass
class CurveFit:
    """Univariate calibration fit for one diagnostic line."""

    element: str
    wavelength_nm: float
    degree: int  # 1 = linear, 2 = quadratic
    coeffs: list[float]  # highest power first (np.polyfit order)
    r_squared: float
    n_points: int
    intensities: list[float]
    concentrations: list[float]
    sample_ids: list[str]


@dataclass
class ElementPrediction:
    element: str
    concentration: float
    std: float | None
    n_lines: int
    line_predictions: list[tuple[float, float]]  # (wavelength_nm, C)


@dataclass
class CalibrationSet:
    """In-memory calibration session."""

    standards: list[StandardSample] = field(default_factory=list)
    elements: list[str] = field(default_factory=list)
    #: Subset of ``elements`` to fit / predict / report / plot.
    #: Empty means “all of ``elements``”.
    quantify_elements: list[str] = field(default_factory=list)
    diagnostic_lines: list[DiagnosticLine] = field(default_factory=list)
    half_width_nm: float = 0.20
    baseline_pad_nm: float = 0.40
    overlap_tol_nm: float = 0.12
    fit_degree: int = 1
    min_standards: int = 2
    atmosphere: str = "unknown"
    #: Label for concentration values (e.g. "wt%", "ppm"). Metadata only —
    #: the fit does not convert units; keep CRM and unknown reporting consistent.
    concentration_unit: str = "wt%"
    fits: list[CurveFit] = field(default_factory=list)

    def active_elements(self) -> list[str]:
        """Elements selected for quantification (fit / predict / report)."""
        if not self.quantify_elements:
            return list(self.elements)
        wanted = set(self.quantify_elements)
        return [e for e in self.elements if e in wanted]


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------


def subtract_local_baseline(
    wavelength_nm: np.ndarray,
    intensity: np.ndarray,
    center_nm: float,
    *,
    half_width_nm: float = 0.20,
    pad_nm: float = 0.40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Local continuum via edge-percentile baseline in a window around ``center_nm``.

    Returns (wl_slice, intensity_corrected, baseline_on_slice).
    """
    lo = center_nm - half_width_nm - pad_nm
    hi = center_nm + half_width_nm + pad_nm
    mask = (wavelength_nm >= lo) & (wavelength_nm <= hi)
    if not np.any(mask):
        empty = np.array([], dtype=float)
        return empty, empty, empty

    wl = wavelength_nm[mask]
    y = intensity[mask].astype(float)

    # Use samples outside the integration core as baseline anchors
    core = (wl >= center_nm - half_width_nm) & (wl <= center_nm + half_width_nm)
    wings = ~core
    if np.count_nonzero(wings) >= 4:
        level = float(np.percentile(y[wings], 10))
    else:
        level = float(np.percentile(y, 10))

    # Mild linear tilt from left/right wing medians when available
    left = y[wl < center_nm - half_width_nm]
    right = y[wl > center_nm + half_width_nm]
    if len(left) >= 2 and len(right) >= 2:
        y_l = float(np.median(left))
        y_r = float(np.median(right))
        x_l = float(np.median(wl[wl < center_nm - half_width_nm]))
        x_r = float(np.median(wl[wl > center_nm + half_width_nm]))
        if abs(x_r - x_l) > 1e-9:
            slope = (y_r - y_l) / (x_r - x_l)
            baseline = y_l + slope * (wl - x_l)
        else:
            baseline = np.full_like(wl, level)
    else:
        baseline = np.full_like(wl, level)

    return wl, y - baseline, baseline


def integrate_peak_area(
    spectrum: Spectrum,
    center_nm: float,
    *,
    half_width_nm: float = 0.20,
    pad_nm: float = 0.40,
) -> float:
    """Net peak area (trapezoid) after local baseline subtraction."""
    wl, y_corr, _ = subtract_local_baseline(
        spectrum.wavelength_nm,
        spectrum.intensity,
        center_nm,
        half_width_nm=half_width_nm,
        pad_nm=pad_nm,
    )
    if len(wl) < 2:
        return 0.0
    core = (wl >= center_nm - half_width_nm) & (wl <= center_nm + half_width_nm)
    if np.count_nonzero(core) < 2:
        return float(max(np.max(y_corr), 0.0)) if len(y_corr) else 0.0
    area = float(np.trapezoid(np.clip(y_corr[core], 0.0, None), wl[core]))
    return max(area, 0.0)


def peak_net_height(
    spectrum: Spectrum,
    center_nm: float,
    *,
    half_width_nm: float = 0.20,
    pad_nm: float = 0.40,
) -> float:
    """Net peak height at nearest sample after local baseline (secondary metric)."""
    wl, y_corr, _ = subtract_local_baseline(
        spectrum.wavelength_nm,
        spectrum.intensity,
        center_nm,
        half_width_nm=half_width_nm,
        pad_nm=pad_nm,
    )
    if len(wl) == 0:
        return 0.0
    idx = int(np.argmin(np.abs(wl - center_nm)))
    return float(max(y_corr[idx], 0.0))


# ---------------------------------------------------------------------------
# Overlap flagging + diagnostic line suggestions
# ---------------------------------------------------------------------------


def flag_line_overlaps(
    lines: list[DiagnosticLine],
    library: list[LibraryLine],
    *,
    tol_nm: float = 0.12,
    min_intensity: float = 50.0,
) -> list[DiagnosticLine]:
    """
    Soft-flag diagnostic lines that have strong NIST neighbors from other elements.

    Does not disable lines; sets ``overlap_warning`` text.
    """
    if not library:
        return lines

    lib_wl = np.array([L.wavelength_nm for L in library])
    order = np.argsort(lib_wl)
    lib_wl_s = lib_wl[order]
    lib_s = [library[i] for i in order]

    out: list[DiagnosticLine] = []
    for d in lines:
        lo = int(np.searchsorted(lib_wl_s, d.wavelength_nm - tol_nm))
        hi = int(np.searchsorted(lib_wl_s, d.wavelength_nm + tol_nm))
        offenders: list[str] = []
        for k in range(lo, hi):
            L = lib_s[k]
            if L.element == d.element:
                continue
            strength = L.intensity if L.intensity is not None else 0.0
            if strength < min_intensity and (L.aki is None or L.aki < 1e6):
                continue
            offenders.append(f"{L.species} {L.wavelength_nm:.3f} nm")
        warn = None
        if offenders:
            shown = ", ".join(offenders[:4])
            extra = f" (+{len(offenders) - 4} more)" if len(offenders) > 4 else ""
            warn = f"Possible overlap: {shown}{extra}"
        out.append(
            DiagnosticLine(
                element=d.element,
                wavelength_nm=d.wavelength_nm,
                species=d.species,
                enabled=d.enabled,
                overlap_warning=warn,
            )
        )
    return out


def suggest_diagnostic_lines(
    library: list[LibraryLine],
    elements: list[str],
    *,
    wl_min: float,
    wl_max: float,
    max_per_element: int = 8,
    overlap_tol_nm: float = 0.12,
) -> list[DiagnosticLine]:
    """Strongest NIST lines per element in range, with overlap soft-flags."""
    from identify_elements import strong_library_lines

    by_el = strong_library_lines(
        library,
        elements,
        wl_min=wl_min,
        wl_max=wl_max,
        max_per_element=max_per_element,
    )
    lines: list[DiagnosticLine] = []
    for el in elements:
        for L in by_el.get(el, []):
            lines.append(
                DiagnosticLine(
                    element=el,
                    wavelength_nm=float(L.wavelength_nm),
                    species=L.species or el,
                    enabled=True,
                )
            )
    return flag_line_overlaps(lines, library, tol_nm=overlap_tol_nm)


def seed_lines_from_matches(
    matches_by_element: dict[str, list[tuple[float, str]]],
    library: list[LibraryLine],
    *,
    max_per_element: int = 8,
    overlap_tol_nm: float = 0.12,
) -> list[DiagnosticLine]:
    """
    Build diagnostic lines from Identify matches.

    ``matches_by_element`` maps element → list of (wavelength_nm, species).
    """
    lines: list[DiagnosticLine] = []
    for el, pairs in matches_by_element.items():
        seen: set[float] = set()
        for wl, species in pairs[:max_per_element]:
            key = round(wl, 3)
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                DiagnosticLine(
                    element=el,
                    wavelength_nm=float(wl),
                    species=species or el,
                    enabled=True,
                )
            )
    return flag_line_overlaps(lines, library, tol_nm=overlap_tol_nm)


# ---------------------------------------------------------------------------
# Fitting + prediction
# ---------------------------------------------------------------------------


def fit_calibration_curve(
    intensities: list[float],
    concentrations: list[float],
    *,
    sample_ids: list[str] | None = None,
    element: str = "",
    wavelength_nm: float = 0.0,
    degree: int = 1,
) -> CurveFit | None:
    """
    Fit C = f(I) with numpy polyfit. Returns None if too few points,
    non-finite data, no intensity contrast, or the least-squares solve fails.
    """
    if degree not in (1, 2):
        raise ValueError("degree must be 1 or 2")
    xs: list[float] = []
    ys: list[float] = []
    ids: list[str] = []
    sid = sample_ids or [str(i) for i in range(len(intensities))]
    for i, (inten, conc) in enumerate(zip(intensities, concentrations)):
        if inten is None or conc is None:
            continue
        try:
            xi = float(inten)
            yi = float(conc)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(xi) or not np.isfinite(yi):
            continue
        # Zero/negative net area is common for missing lines — keep only if some
        # standards have signal; all-zero still rejected via ptp check below.
        xs.append(xi)
        ys.append(yi)
        ids.append(sid[i] if i < len(sid) else str(i))
    n = len(xs)
    if n < degree + 1:
        return None

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return None

    # Identical (or nearly identical) intensities → singular design matrix
    x_span = float(np.ptp(x))
    x_scale = float(np.max(np.abs(x)))
    if x_span <= 0.0 or (x_scale > 0 and x_span / x_scale < 1e-14):
        return None
    if float(np.ptp(y)) == 0.0 and degree >= 1:
        # Flat concentrations still allow a horizontal fit, but polyfit can
        # be numerically ugly with bad x; allow it only with scaled x.
        pass

    # Scale intensity for numerical stability (LIBS areas can be huge or tiny)
    scale = x_scale if x_scale > 0 else 1.0
    x_n = x / scale
    try:
        with np.errstate(all="ignore"):
            coeffs_n = np.polyfit(x_n, y, degree)
    except np.linalg.LinAlgError:
        return None
    if coeffs_n is None or not np.all(np.isfinite(coeffs_n)):
        return None

    # Map normalized coeffs back to original intensity scale
    if degree == 1:
        coeffs = np.array([coeffs_n[0] / scale, coeffs_n[1]], dtype=float)
    else:
        coeffs = np.array(
            [
                coeffs_n[0] / (scale**2),
                coeffs_n[1] / scale,
                coeffs_n[2],
            ],
            dtype=float,
        )
    if not np.all(np.isfinite(coeffs)):
        return None

    y_hat = np.polyval(coeffs, x)
    if not np.all(np.isfinite(y_hat)):
        return None
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    if not np.isfinite(r2):
        r2 = 0.0

    return CurveFit(
        element=element,
        wavelength_nm=wavelength_nm,
        degree=degree,
        coeffs=[float(c) for c in coeffs],
        r_squared=float(r2),
        n_points=n,
        intensities=xs,
        concentrations=ys,
        sample_ids=ids,
    )


def build_fits(
    cal: CalibrationSet,
    *,
    skipped: list[str] | None = None,
) -> list[CurveFit]:
    """
    Fit every enabled diagnostic line that has enough standards.

    Unfittable lines are skipped (not raised). Optional ``skipped`` collects
    human-readable reasons.
    """
    active = set(cal.active_elements())
    fits: list[CurveFit] = []
    notes = skipped if skipped is not None else []

    def _note(msg: str) -> None:
        notes.append(msg)

    for dline in cal.diagnostic_lines:
        if not dline.enabled:
            continue
        if dline.element not in active:
            continue
        label = f"{dline.element} {dline.wavelength_nm:.3f} nm"
        intensities: list[float] = []
        concentrations: list[float] = []
        sample_ids: list[str] = []
        for std in cal.standards:
            conc = std.concentrations.get(dline.element)
            if conc is None:
                continue
            try:
                inten = integrate_peak_area(
                    std.spectrum,
                    dline.wavelength_nm,
                    half_width_nm=cal.half_width_nm,
                    pad_nm=cal.baseline_pad_nm,
                )
            except Exception as exc:
                _note(f"{label}: integration failed on {std.sample_id} ({exc})")
                inten = float("nan")
            intensities.append(float(inten) if inten is not None else float("nan"))
            concentrations.append(float(conc))
            sample_ids.append(std.sample_id)

        finite_n = sum(
            1
            for a, b in zip(intensities, concentrations)
            if np.isfinite(a) and np.isfinite(b)
        )
        need = max(cal.min_standards, cal.fit_degree + 1)
        if finite_n < need:
            _note(
                f"{label}: only {finite_n} usable standard(s) "
                f"(need ≥{need} with concentration + finite peak area)"
            )
            continue

        x_arr = np.asarray(
            [a for a, b in zip(intensities, concentrations) if np.isfinite(a) and np.isfinite(b)],
            dtype=float,
        )
        if len(x_arr) and float(np.ptp(x_arr)) <= 0.0:
            _note(
                f"{label}: no intensity contrast across standards "
                "(same/zero net peak area — try another line or check λ / window)"
            )
            continue

        try:
            fit = fit_calibration_curve(
                intensities,
                concentrations,
                sample_ids=sample_ids,
                element=dline.element,
                wavelength_nm=dline.wavelength_nm,
                degree=cal.fit_degree,
            )
        except np.linalg.LinAlgError:
            fit = None
        except Exception as exc:
            _note(f"{label}: fit error ({exc})")
            continue

        if fit is None:
            _note(
                f"{label}: fit failed (singular/ill-conditioned data — "
                "check peak areas and concentrations)"
            )
            continue
        fits.append(fit)

    cal.fits = fits
    return fits


def predict_from_fit(fit: CurveFit, intensity: float) -> float:
    return float(np.polyval(fit.coeffs, intensity))


def predict_concentrations(
    cal: CalibrationSet,
    unknown: Spectrum,
    *,
    elements: list[str] | None = None,
) -> list[ElementPrediction]:
    """
    Apply fitted curves to an unknown; average enabled lines per element.

    ``elements`` limits the report (e.g. 2–3 of interest). Default:
    ``cal.active_elements()`` (checked quantify set).
    """
    if not cal.fits:
        build_fits(cal)

    report = list(elements) if elements is not None else cal.active_elements()
    report_set = set(report)

    by_el: dict[str, list[tuple[float, float]]] = {}
    for fit in cal.fits:
        if fit.element not in report_set:
            continue
        inten = integrate_peak_area(
            unknown,
            fit.wavelength_nm,
            half_width_nm=cal.half_width_nm,
            pad_nm=cal.baseline_pad_nm,
        )
        c = predict_from_fit(fit, inten)
        by_el.setdefault(fit.element, []).append((fit.wavelength_nm, c))

    preds: list[ElementPrediction] = []
    for el in report:
        pairs = by_el.get(el, [])
        if not pairs:
            continue
        vals = [c for _, c in pairs]
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        preds.append(
            ElementPrediction(
                element=el,
                concentration=mean,
                std=std,
                n_lines=len(vals),
                line_predictions=pairs,
            )
        )
    return preds


# ---------------------------------------------------------------------------
# CSV I/O (concentrations)
# ---------------------------------------------------------------------------


def load_concentrations_csv(path: Path) -> dict[str, dict[str, float | None]]:
    """
    Load concentrations keyed by standard_id → element → value.

    Expected header: standard_id, Element1, Element2, ...
    Blank cells → None (element not calibrated for that standard).
    """
    path = Path(path)
    out: dict[str, dict[str, float | None]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Empty CSV: {path}")
        fields = [h.strip() for h in reader.fieldnames]
        id_key = fields[0]
        elements = [h for h in fields[1:] if h]
        for row in reader:
            sid = (row.get(id_key) or "").strip()
            if not sid:
                continue
            concs: dict[str, float | None] = {}
            for el in elements:
                raw = (row.get(el) or "").strip()
                if raw == "":
                    concs[el] = None
                else:
                    concs[el] = float(raw)
            out[sid] = concs
    return out


def save_concentrations_csv(
    path: Path,
    standards: list[StandardSample],
    elements: list[str],
) -> None:
    """Write standard_id + element concentration columns."""
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["standard_id", *elements])
        for std in standards:
            row: list[str] = [std.sample_id]
            for el in elements:
                v = std.concentrations.get(el)
                row.append("" if v is None else f"{v:g}")
            writer.writerow(row)


def apply_concentrations(
    standards: list[StandardSample],
    table: dict[str, dict[str, float | None]],
    *,
    match_by: str = "sample_id",
) -> list[str]:
    """
    Merge CSV concentrations into standards.

    match_by: 'sample_id' or 'stem' (spectrum filename stem).
    Returns list of unmatched CSV keys.
    """
    index: dict[str, StandardSample] = {}
    for s in standards:
        if match_by == "stem":
            index[s.path.stem] = s
        index[s.sample_id] = s

    unmatched: list[str] = []
    for key, concs in table.items():
        std = index.get(key)
        if std is None:
            unmatched.append(key)
            continue
        for el, val in concs.items():
            std.concentrations[el] = val
    return unmatched


# ---------------------------------------------------------------------------
# Acquisition-condition consistency
# ---------------------------------------------------------------------------


def _values_agree(vals: list[float], *, rtol: float = 1e-3, atol: float = 1e-6) -> bool:
    """True if all finite values match within relative/absolute tolerance."""
    finite = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if len(finite) <= 1:
        return True
    ref = finite[0]
    return all(abs(v - ref) <= max(atol, rtol * abs(ref)) for v in finite[1:])


def acquisition_mismatch_warnings(
    standards: list[StandardSample],
    *,
    rtol: float = 1e-3,
    atol: float = 1e-6,
) -> list[str]:
    """
    Warn when CRM standards do not share identical acquisition settings
    from their .cfg files (laser energy, gate delay, integration time,
    accumulations).
    """
    if len(standards) < 2:
        return []

    warnings: list[str] = []
    missing_cfg = [s.sample_id for s in standards if s.spectrum.meta.cfg_path is None]
    if missing_cfg:
        warnings.append(
            "No .cfg for: "
            + ", ".join(missing_cfg[:8])
            + ("…" if len(missing_cfg) > 8 else "")
            + " — cannot verify acquisition match."
        )

    checks: list[tuple[str, str]] = [
        ("Laser energy (mJ)", "laser_energy_mJ"),
        ("QS delay (µs)", "qs_delay_us"),
        ("Integration time / gate width (µs)", "integration_time_us"),
        ("Integration delay (µs)", "integration_delay_us"),
        ("Accumulations", "n_accumulations"),
    ]

    for label, attr in checks:
        pairs: list[tuple[str, float | None]] = []
        for s in standards:
            m = s.spectrum.meta
            if m.cfg_path is None:
                continue
            raw = getattr(m, attr, None)
            pairs.append((s.sample_id, None if raw is None else float(raw)))

        if len(pairs) < 2:
            continue

        missing_param = [sid for sid, v in pairs if v is None]
        present = [(sid, v) for sid, v in pairs if v is not None]
        if missing_param and present:
            warnings.append(
                f"{label}: missing in cfg for {', '.join(missing_param[:6])}"
                + ("…" if len(missing_param) > 6 else "")
            )
        if len(present) < 2:
            continue
        vals = [v for _, v in present]
        if not _values_agree(vals, rtol=rtol, atol=atol):
            detail = ", ".join(f"{sid}={v:g}" for sid, v in present[:8])
            extra = f" (+{len(present) - 8} more)" if len(present) > 8 else ""
            warnings.append(f"{label} differs across standards: {detail}{extra}")

    return warnings


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------


def calibration_set_to_dict(cal: CalibrationSet) -> dict:
    return {
        "version": 1,
        "atmosphere": cal.atmosphere,
        "concentration_unit": cal.concentration_unit,
        "half_width_nm": cal.half_width_nm,
        "baseline_pad_nm": cal.baseline_pad_nm,
        "overlap_tol_nm": cal.overlap_tol_nm,
        "fit_degree": cal.fit_degree,
        "min_standards": cal.min_standards,
        "elements": list(cal.elements),
        "quantify_elements": list(cal.quantify_elements),
        "diagnostic_lines": [
            {
                "element": d.element,
                "wavelength_nm": d.wavelength_nm,
                "species": d.species,
                "enabled": d.enabled,
                "overlap_warning": d.overlap_warning,
            }
            for d in cal.diagnostic_lines
        ],
        "standards": [
            {
                "sample_id": s.sample_id,
                "spectrum_path": str(s.path),
                "cfg_path": str(s.spectrum.meta.cfg_path) if s.spectrum.meta.cfg_path else None,
                "atmosphere": s.atmosphere,
                "concentrations": {
                    k: v for k, v in s.concentrations.items()
                },
            }
            for s in cal.standards
        ],
        "fits": [
            {
                "element": f.element,
                "wavelength_nm": f.wavelength_nm,
                "degree": f.degree,
                "coeffs": f.coeffs,
                "r_squared": f.r_squared,
                "n_points": f.n_points,
                "intensities": f.intensities,
                "concentrations": f.concentrations,
                "sample_ids": f.sample_ids,
            }
            for f in cal.fits
        ],
    }


def save_calibration_set(cal: CalibrationSet, path: Path) -> None:
    path = Path(path)
    path.write_text(json.dumps(calibration_set_to_dict(cal), indent=2), encoding="utf-8")


def load_calibration_set(path: Path) -> CalibrationSet:
    """Reload spectra from paths stored in the JSON session file."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    elements = list(data.get("elements") or [])
    quantify = list(data.get("quantify_elements") or [])
    if not quantify:
        quantify = list(elements)
    cal = CalibrationSet(
        elements=elements,
        quantify_elements=quantify,
        half_width_nm=float(data.get("half_width_nm", 0.20)),
        baseline_pad_nm=float(data.get("baseline_pad_nm", 0.40)),
        overlap_tol_nm=float(data.get("overlap_tol_nm", 0.12)),
        fit_degree=int(data.get("fit_degree", 1)),
        min_standards=int(data.get("min_standards", 2)),
        atmosphere=str(data.get("atmosphere") or "unknown"),
        concentration_unit=str(data.get("concentration_unit") or "wt%"),
    )
    for d in data.get("diagnostic_lines") or []:
        cal.diagnostic_lines.append(
            DiagnosticLine(
                element=d["element"],
                wavelength_nm=float(d["wavelength_nm"]),
                species=d.get("species") or d["element"],
                enabled=bool(d.get("enabled", True)),
                overlap_warning=d.get("overlap_warning"),
            )
        )
    for s in data.get("standards") or []:
        txt = Path(s["spectrum_path"])
        cfg = Path(s["cfg_path"]) if s.get("cfg_path") else None
        spectrum = load_spectrum(txt, cfg)
        concs_raw = s.get("concentrations") or {}
        concs: dict[str, float | None] = {}
        for k, v in concs_raw.items():
            concs[k] = None if v is None else float(v)
        cal.standards.append(
            StandardSample(
                sample_id=s.get("sample_id") or txt.stem,
                spectrum=spectrum,
                concentrations=concs,
                atmosphere=s.get("atmosphere") or cal.atmosphere,
            )
        )
    for f in data.get("fits") or []:
        cal.fits.append(
            CurveFit(
                element=f["element"],
                wavelength_nm=float(f["wavelength_nm"]),
                degree=int(f["degree"]),
                coeffs=[float(c) for c in f["coeffs"]],
                r_squared=float(f["r_squared"]),
                n_points=int(f["n_points"]),
                intensities=[float(x) for x in f.get("intensities") or []],
                concentrations=[float(x) for x in f.get("concentrations") or []],
                sample_ids=list(f.get("sample_ids") or []),
            )
        )
    return cal


def add_standard_from_path(
    cal: CalibrationSet,
    txt_path: Path,
    *,
    cfg_path: Path | None = None,
    sample_id: str | None = None,
    atmosphere: str | None = None,
) -> StandardSample:
    """Load a spectrum and append it to the calibration set."""
    spectrum = load_spectrum(Path(txt_path), cfg_path)
    sid = sample_id or spectrum.meta.path.stem
    # Preserve element columns already in use
    concs = {el: None for el in cal.elements}
    std = StandardSample(
        sample_id=sid,
        spectrum=spectrum,
        concentrations=concs,
        atmosphere=atmosphere or cal.atmosphere,
    )
    cal.standards.append(std)
    return std


def ensure_element_columns(cal: CalibrationSet, elements: list[str]) -> None:
    """Set concentration columns; new elements default into the quantify set."""
    old_elements = set(cal.elements)
    old_q = set(cal.quantify_elements) if cal.quantify_elements else set(cal.elements)
    cal.elements = list(elements)
    cal.quantify_elements = [
        e for e in cal.elements if (e in old_q) or (e not in old_elements)
    ]
    for std in cal.standards:
        for el in cal.elements:
            std.concentrations.setdefault(el, None)
        extra = [k for k in std.concentrations if k not in cal.elements]
        for k in extra:
            del std.concentrations[k]


def save_predictions_csv(
    path: Path,
    preds: list[ElementPrediction],
    *,
    unit: str = "",
) -> None:
    """Write quantified element predictions (selected subset)."""
    path = Path(path)
    conc_col = f"concentration_{unit}" if unit else "concentration"
    std_col = f"std_{unit}" if unit else "std"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["element", conc_col, std_col, "n_lines"])
        for p in preds:
            writer.writerow(
                [
                    p.element,
                    f"{p.concentration:g}",
                    "" if p.std is None else f"{p.std:g}",
                    p.n_lines,
                ]
            )
