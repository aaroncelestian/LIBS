"""
CRM univariate LIBS calibration.

Workflow:
  1. Load standard spectra (+ optional .cfg)
  2. Enter known concentrations for elements of interest
  3. Local baseline + Gaussian/Voigt peak fit (with λ shift) or net area
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
    #: If set, keep for Curves QC but exclude from Quant (e.g. "negative I→C slope")
    rejected: str | None = None

    @property
    def usable(self) -> bool:
        return not self.rejected


def usable_fits(cal: "CalibrationSet") -> list[CurveFit]:
    """Fits eligible for Quant (non-rejected)."""
    return [f for f in cal.fits if f.usable]


@dataclass
class ElementPrediction:
    element: str
    concentration: float
    std: float | None
    n_lines: int
    line_predictions: list[tuple[float, float]]  # (wavelength_nm, C)
    #: True if any line extrapolated below the I→C zero crossing (floored to 0)
    below_calibration: bool = False


@dataclass
class QuantSpectrumResult:
    """One unknown spectrum after applying CRM fits."""

    index: int  # 1-based spectrum number (display / series axis)
    filename: str
    spectrum_path: str
    predictions: list[ElementPrediction]
    #: (element, wavelength_nm) → measured peak intensity used for I→C
    line_intensities: dict[tuple[str, float], float] = field(default_factory=dict)

    def prediction_for(self, element: str) -> ElementPrediction | None:
        for p in self.predictions:
            if p.element == element:
                return p
        return None

    def concentrations(self) -> dict[str, float]:
        return {p.element: float(p.concentration) for p in self.predictions}


def confidence_interval_95(
    mean: float,
    std: float | None,
    n: int,
) -> tuple[float, float] | None:
    """
    Two-sided 95% CI for the mean from sample std: mean ± t * std / √n.

    Returns None when ``n < 2`` or ``std`` is missing/non-finite.
    """
    if n < 2 or std is None:
        return None
    try:
        s = float(std)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(s) or s < 0.0 or not np.isfinite(mean):
        return None
    from scipy.stats import t as student_t

    tcrit = float(student_t.ppf(0.975, n - 1))
    half = tcrit * s / np.sqrt(n)
    return float(mean - half), float(mean + half)


@dataclass
class CalibrationSet:
    """In-memory calibration session."""

    standards: list[StandardSample] = field(default_factory=list)
    elements: list[str] = field(default_factory=list)
    #: Subset of ``elements`` to fit / predict / report / plot.
    #: Empty means “all of ``elements``”.
    quantify_elements: list[str] = field(default_factory=list)
    diagnostic_lines: list[DiagnosticLine] = field(default_factory=list)
    half_width_nm: float = 0.15
    baseline_pad_nm: float = 0.12
    #: ``snip`` (default), ``linear`` (edge→edge), or ``flat`` (edge mean)
    baseline_method: str = "snip"
    #: SNIP peak-clipping iterations (RamanLab default 40)
    snip_iterations: int = 40
    #: Peak intensity model: ``gaussian`` (default), ``voigt``, or ``net_area``
    peak_model: str = "gaussian"
    #: Allowed |fitted − NIST| shift when peak_model is gaussian/voigt
    shift_tol_nm: float = 0.15
    overlap_tol_nm: float = 0.12
    fit_degree: int = 1
    min_standards: int = 2
    atmosphere: str = "unknown"
    #: Concentration unit for CRM entry and predictions (e.g. "wt%", "ppm").
    #: Convertible mass units are scaled when the user changes the unit in the UI.
    concentration_unit: str = "wt%"
    fits: list[CurveFit] = field(default_factory=list)

    def active_elements(self) -> list[str]:
        """Elements selected for quantification (fit / predict / report)."""
        if not self.quantify_elements:
            return list(self.elements)
        wanted = set(self.quantify_elements)
        return [e for e in self.elements if e in wanted]


# ---------------------------------------------------------------------------
# Concentration units
# ---------------------------------------------------------------------------

# Display labels for the unit combo (order shown in UI).
CONCENTRATION_UNIT_CHOICES: tuple[str, ...] = (
    "wt%",
    "ppm",
    "mg/kg",
    "µg/g",
    "mass frac",
    "at%",
    "oxide wt%",
)

# Factors to absolute mass fraction (0–1). Same physical quantity ↔ convertible.
_MASS_FRAC_PER_UNIT: dict[str, float] = {
    "wt%": 1e-2,
    "ppm": 1e-6,
    "mg/kg": 1e-6,
    "µg/g": 1e-6,
    "ug/g": 1e-6,
    "mass frac": 1.0,
    "mass fraction": 1.0,
    "frac": 1.0,
}


def normalize_concentration_unit(unit: str) -> str:
    """Canonical unit label for comparisons."""
    u = (unit or "").strip()
    aliases = {
        "wt %": "wt%",
        "weight %": "wt%",
        "weight%": "wt%",
        "mass%": "wt%",
        "ug/g": "µg/g",
        "ppm (µg/g)": "ppm",
        "ppm (mg/kg)": "ppm",
        "mass fraction": "mass frac",
        "fraction": "mass frac",
    }
    for a, canon in aliases.items():
        if u.lower() == a.lower():
            return canon
    for choice in CONCENTRATION_UNIT_CHOICES:
        if u.lower() == choice.lower():
            return choice
    return u or "wt%"


def concentration_units_convertible(unit_a: str, unit_b: str) -> bool:
    """True if both units are mass-based and can be linearly converted."""

    def factor_key(u: str) -> str | None:
        n = normalize_concentration_unit(u)
        for k in _MASS_FRAC_PER_UNIT:
            if k.lower() == n.lower():
                return k
        return None

    return factor_key(unit_a) is not None and factor_key(unit_b) is not None


def convert_concentration(
    value: float,
    from_unit: str,
    to_unit: str,
) -> float:
    """
    Convert a concentration between mass-based units.

    Raises ``ValueError`` if either unit is not mass-convertible (e.g. at%).
    """
    if not concentration_units_convertible(from_unit, to_unit):
        raise ValueError(
            f"Cannot convert between {from_unit!r} and {to_unit!r} "
            "(only wt%, ppm, mg/kg, µg/g, mass frac)."
        )
    fu = normalize_concentration_unit(from_unit)
    tu = normalize_concentration_unit(to_unit)
    if fu.lower() == tu.lower():
        return float(value)

    def factor(u: str) -> float:
        n = normalize_concentration_unit(u)
        for k, f in _MASS_FRAC_PER_UNIT.items():
            if k.lower() == n.lower():
                return f
        raise ValueError(u)

    mass_frac = float(value) * factor(fu)
    return mass_frac / factor(tu)


def convert_calibration_concentrations(
    cal: CalibrationSet,
    from_unit: str,
    to_unit: str,
) -> int:
    """
    In-place convert every standard concentration from ``from_unit`` to ``to_unit``.

    Returns the number of values converted. No-op (returns 0) if units match
    or are not convertible.
    """
    if normalize_concentration_unit(from_unit).lower() == normalize_concentration_unit(
        to_unit
    ).lower():
        return 0
    if not concentration_units_convertible(from_unit, to_unit):
        return 0
    n = 0
    for std in cal.standards:
        for el, val in list(std.concentrations.items()):
            if val is None:
                continue
            std.concentrations[el] = convert_concentration(float(val), from_unit, to_unit)
            n += 1
    return n


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------

BASELINE_METHODS: tuple[str, ...] = ("snip", "linear", "flat")


def snip_baseline(y: np.ndarray, n_iter: int = 40) -> np.ndarray:
    """
    SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping) baseline.

    Ported from RamanLab ``peak_fitting_qt6._baseline_snip`` (Morháč LLS
    transform + iterative clipping). Peaks are clipped downward toward the
    continuum; the estimate cannot over-subtract above the data.
    """
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return y.copy()
    n_iter = int(max(1, n_iter))
    y_min = float(np.nanmin(y))
    # Guard non-finite samples
    y_work = np.nan_to_num(y, nan=y_min, posinf=y_min, neginf=y_min)
    z = np.log(np.log(np.sqrt(y_work - y_min + 1.0) + 1.0) + 1.0)
    for p in range(1, n_iter + 1):
        if 2 * p >= len(z):
            break
        z_new = z.copy()
        # Vectorized clip: z[i] ← min(z[i], mean of neighbors ±p)
        z_new[p:-p] = np.minimum(z[p:-p], 0.5 * (z[:-2 * p] + z[2 * p :]))
        z = z_new
    background = (np.exp(np.exp(z) - 1.0) - 1.0) ** 2 + y_min - 1.0
    return np.maximum(background, y_min)


def _window_bounds(
    center_nm: float,
    half_width_nm: float,
    pad_nm: float,
    *,
    edge_frac: float = 0.40,
) -> tuple[float, float, float, float, float, float]:
    """
    Return ``(lo, hi, edge_w, integrate_lo, integrate_hi, edge_frac_clamped)``.

    Layout (wavelength increasing)::

        |--edge--|-- pad remnant + half_width (×2) --|--edge--|
        lo                                              hi

    Net area is integrated between the edge strips (``integrate_lo`` …
    ``integrate_hi``), so peak shoulders that sit in the pad are not cut off.
    """
    half_width_nm = max(float(half_width_nm), 1e-4)
    pad_nm = max(float(pad_nm), 1e-4)
    ef = float(np.clip(edge_frac, 0.15, 1.0))
    edge_w = max(pad_nm * ef, 1e-4)
    lo = center_nm - half_width_nm - pad_nm
    hi = center_nm + half_width_nm + pad_nm
    integrate_lo = lo + edge_w
    integrate_hi = hi - edge_w
    return lo, hi, edge_w, integrate_lo, integrate_hi, ef


def subtract_local_baseline(
    wavelength_nm: np.ndarray,
    intensity: np.ndarray,
    center_nm: float,
    *,
    half_width_nm: float = 0.15,
    pad_nm: float = 0.12,
    method: str = "snip",
    edge_frac: float = 0.40,
    snip_iterations: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Local continuum under a diagnostic line.

    Window is ``center ± (half_width + pad)``.

    ``method``:
      - ``snip`` — RamanLab SNIP peak-clipping (default; best for crowded LIBS)
      - ``linear`` — line between left/right edge-strip levels
      - ``flat`` — constant level from both edge strips

    Returns (wl_slice, intensity_corrected, baseline_on_slice).
    """
    method = (method or "snip").strip().lower()
    if method not in BASELINE_METHODS:
        method = "snip"

    lo, hi, edge_w, _, _, _ = _window_bounds(
        center_nm, half_width_nm, pad_nm, edge_frac=edge_frac
    )
    mask = (wavelength_nm >= lo) & (wavelength_nm <= hi)
    if not np.any(mask):
        empty = np.array([], dtype=float)
        return empty, empty, empty

    wl = np.asarray(wavelength_nm[mask], dtype=float)
    y = np.asarray(intensity[mask], dtype=float)

    if method == "snip":
        baseline = snip_baseline(y, n_iter=snip_iterations)
        return wl, y - baseline, baseline

    left_edge = (wl >= lo) & (wl <= lo + edge_w)
    right_edge = (wl >= hi - edge_w) & (wl <= hi)

    def _edge_level(sel: np.ndarray) -> tuple[float, float] | None:
        if np.count_nonzero(sel) < 2:
            return None
        # Low percentile resists noise spikes; still below typical peak tails
        return float(np.median(wl[sel])), float(np.percentile(y[sel], 20))

    left = _edge_level(left_edge)
    right = _edge_level(right_edge)

    # Fallbacks if an edge is empty (window clipped / sparse sampling)
    if left is None and right is None:
        wings = (wl < center_nm - half_width_nm) | (wl > center_nm + half_width_nm)
        level = float(np.percentile(y[wings] if np.any(wings) else y, 20))
        baseline = np.full_like(wl, level)
    elif left is None:
        baseline = np.full_like(wl, right[1])  # type: ignore[index]
    elif right is None:
        baseline = np.full_like(wl, left[1])
    elif method == "flat":
        baseline = np.full_like(wl, 0.5 * (left[1] + right[1]))
    else:
        x_l, y_l = left
        x_r, y_r = right
        if abs(x_r - x_l) > 1e-9:
            slope = (y_r - y_l) / (x_r - x_l)
            baseline = y_l + slope * (wl - x_l)
        else:
            baseline = np.full_like(wl, 0.5 * (y_l + y_r))

    return wl, y - baseline, baseline


def integrate_peak_area(
    spectrum: Spectrum,
    center_nm: float,
    *,
    half_width_nm: float = 0.15,
    pad_nm: float = 0.12,
    method: str = "snip",
    edge_frac: float = 0.40,
    snip_iterations: int = 40,
) -> float:
    """
    Net peak area (trapezoid) after local baseline subtraction.

    Integrates the full span **between** the left/right edge-anchor strips
    (not only ``center ± half_width``), so peak shoulders are included.
    """
    _lo, _hi, _edge_w, integrate_lo, integrate_hi, _ef = _window_bounds(
        center_nm, half_width_nm, pad_nm, edge_frac=edge_frac
    )
    wl, y_corr, _ = subtract_local_baseline(
        spectrum.wavelength_nm,
        spectrum.intensity,
        center_nm,
        half_width_nm=half_width_nm,
        pad_nm=pad_nm,
        method=method,
        edge_frac=edge_frac,
        snip_iterations=snip_iterations,
    )
    if len(wl) < 2:
        return 0.0
    core = (wl >= integrate_lo) & (wl <= integrate_hi)
    if np.count_nonzero(core) < 2:
        return float(max(np.max(y_corr), 0.0)) if len(y_corr) else 0.0
    area = float(np.trapezoid(np.clip(y_corr[core], 0.0, None), wl[core]))
    return max(area, 0.0)


PEAK_MODELS: tuple[str, ...] = ("gaussian", "voigt", "net_area")


@dataclass
class PeakFitResult:
    """Result of a local emission-line fit (or net-area fallback)."""

    nist_center_nm: float
    fitted_center_nm: float
    delta_nm: float
    amplitude: float
    fwhm_nm: float
    area: float
    peak_model: str
    fit_ok: bool
    fit_wavelength_nm: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    fit_intensity: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    message: str = ""


def _normalize_peak_model(peak_model: str | None) -> str:
    m = (peak_model or "gaussian").strip().lower()
    if m in ("gauss", "g"):
        return "gaussian"
    if m in ("v",):
        return "voigt"
    if m in ("area", "trap", "trapezoid", "net"):
        return "net_area"
    if m not in PEAK_MODELS:
        return "gaussian"
    return m


def _gaussian_profile(wl: np.ndarray, amp: float, center: float, sigma: float) -> np.ndarray:
    s = max(float(sigma), 1e-6)
    return amp * np.exp(-0.5 * ((wl - center) / s) ** 2)


def _voigt_profile(
    wl: np.ndarray, amp: float, center: float, sigma: float, gamma: float
) -> np.ndarray:
    from scipy.special import voigt_profile

    s = max(float(sigma), 1e-6)
    g = max(float(gamma), 1e-9)
    # voigt_profile is area-normalized; scale so peak ≈ amp at center
    core = voigt_profile(wl - center, s, g)
    peak = float(voigt_profile(0.0, s, g))
    if peak <= 0:
        return np.zeros_like(wl)
    return amp * (core / peak)


def _fwhm_gaussian(sigma: float) -> float:
    return float(2.354820045 * max(sigma, 0.0))


def _fwhm_voigt(sigma: float, gamma: float) -> float:
    # Olivero & Longbothum approximation
    fg = _fwhm_gaussian(sigma)
    fl = float(2.0 * max(gamma, 0.0))
    return 0.5346 * fl + float(np.sqrt(0.2166 * fl * fl + fg * fg))


def fit_emission_peak(
    spectrum: Spectrum,
    center_nm: float,
    *,
    half_width_nm: float = 0.15,
    pad_nm: float = 0.12,
    baseline_method: str = "snip",
    peak_model: str = "gaussian",
    shift_tol_nm: float = 0.15,
    edge_frac: float = 0.40,
    snip_iterations: int = 40,
) -> PeakFitResult:
    """
    Fit a Gaussian or Voigt to the local baseline-corrected peak.

    Fitted center is constrained to ``center_nm ± shift_tol_nm``. Area is the
    integral of the fitted profile (above zero). On failure, falls back to
    trapezoid net area with ``fit_ok=False``.
    """
    from scipy.optimize import curve_fit

    model = _normalize_peak_model(peak_model)
    tol = max(float(shift_tol_nm), 1e-4)
    net = integrate_peak_area(
        spectrum,
        center_nm,
        half_width_nm=half_width_nm,
        pad_nm=pad_nm,
        method=baseline_method,
        edge_frac=edge_frac,
        snip_iterations=snip_iterations,
    )

    def _fallback(msg: str) -> PeakFitResult:
        return PeakFitResult(
            nist_center_nm=center_nm,
            fitted_center_nm=center_nm,
            delta_nm=0.0,
            amplitude=0.0,
            fwhm_nm=0.0,
            area=float(net),
            peak_model=model,
            fit_ok=False,
            message=msg,
        )

    if model == "net_area":
        return PeakFitResult(
            nist_center_nm=center_nm,
            fitted_center_nm=center_nm,
            delta_nm=0.0,
            amplitude=0.0,
            fwhm_nm=0.0,
            area=float(net),
            peak_model="net_area",
            fit_ok=True,
            message="net area",
        )

    _lo, _hi, _ew, integrate_lo, integrate_hi, _ef = _window_bounds(
        center_nm, half_width_nm, pad_nm, edge_frac=edge_frac
    )
    wl_all, y_corr, _ = subtract_local_baseline(
        spectrum.wavelength_nm,
        spectrum.intensity,
        center_nm,
        half_width_nm=half_width_nm,
        pad_nm=pad_nm,
        method=baseline_method,
        edge_frac=edge_frac,
        snip_iterations=snip_iterations,
    )
    if len(wl_all) < 5:
        return _fallback("too few samples in window")

    core = (wl_all >= integrate_lo) & (wl_all <= integrate_hi)
    if np.count_nonzero(core) < 5:
        return _fallback("too few samples in integrate span")

    wl = wl_all[core]
    y = np.clip(y_corr[core], 0.0, None)
    if float(np.nanmax(y)) <= 0:
        return _fallback("no positive signal after baseline")

    # Seed from local maximum within shift tolerance of NIST λ
    near = (wl >= center_nm - tol) & (wl <= center_nm + tol)
    if not np.any(near):
        near = np.ones(len(wl), dtype=bool)
    seed_idx = int(np.argmax(y[near]))
    seed_wl = float(wl[near][seed_idx])
    seed_amp = float(y[near][seed_idx])
    span = max(float(integrate_hi - integrate_lo), 1e-3)
    seed_sigma = max(span / 6.0, 0.01)

    c_lo, c_hi = center_nm - tol, center_nm + tol
    amp_hi = max(seed_amp * 5.0, seed_amp + 1.0)
    sig_hi = max(span, 0.05)

    try:
        if model == "voigt":
            seed_gamma = seed_sigma * 0.3

            def model_fn(x, amp, cen, sig, gam):
                return _voigt_profile(x, amp, cen, sig, gam)

            p0 = (seed_amp, seed_wl, seed_sigma, seed_gamma)
            bounds = (
                (0.0, c_lo, 1e-4, 1e-6),
                (amp_hi, c_hi, sig_hi, sig_hi),
            )
            popt, _ = curve_fit(
                model_fn, wl, y, p0=p0, bounds=bounds, maxfev=20000
            )
            amp, cen, sig, gam = (float(v) for v in popt)
            y_fit = _voigt_profile(wl, amp, cen, sig, gam)
            fwhm = _fwhm_voigt(sig, gam)
        else:
            def model_fn(x, amp, cen, sig):
                return _gaussian_profile(x, amp, cen, sig)

            p0 = (seed_amp, seed_wl, seed_sigma)
            bounds = (
                (0.0, c_lo, 1e-4),
                (amp_hi, c_hi, sig_hi),
            )
            popt, _ = curve_fit(
                model_fn, wl, y, p0=p0, bounds=bounds, maxfev=20000
            )
            amp, cen, sig = (float(v) for v in popt)
            y_fit = _gaussian_profile(wl, amp, cen, sig)
            fwhm = _fwhm_gaussian(sig)
    except Exception as exc:
        return _fallback(f"fit failed: {exc}")

    area = float(np.trapezoid(np.clip(y_fit, 0.0, None), wl))
    if not np.isfinite(area) or area < 0:
        return _fallback("non-finite fitted area")

    # Dense curve for QC overlay (on baseline scale: baseline + profile)
    wl_dense = np.linspace(float(wl[0]), float(wl[-1]), max(len(wl) * 4, 80))
    if model == "voigt":
        y_dense = _voigt_profile(wl_dense, amp, cen, sig, gam)
    else:
        y_dense = _gaussian_profile(wl_dense, amp, cen, sig)

    return PeakFitResult(
        nist_center_nm=center_nm,
        fitted_center_nm=cen,
        delta_nm=cen - center_nm,
        amplitude=amp,
        fwhm_nm=fwhm,
        area=max(area, 0.0),
        peak_model=model,
        fit_ok=True,
        fit_wavelength_nm=wl_dense,
        fit_intensity=y_dense,
        message="ok",
    )


def extract_peak_intensity(
    spectrum: Spectrum,
    center_nm: float,
    *,
    half_width_nm: float = 0.15,
    pad_nm: float = 0.12,
    baseline_method: str = "snip",
    peak_model: str = "gaussian",
    shift_tol_nm: float = 0.15,
    edge_frac: float = 0.40,
    snip_iterations: int = 40,
) -> float:
    """
    Peak intensity metric for I→C: fitted area (gaussian/voigt) or net area.

    Always returns a finite non-negative float (0 on empty windows).
    """
    result = fit_emission_peak(
        spectrum,
        center_nm,
        half_width_nm=half_width_nm,
        pad_nm=pad_nm,
        baseline_method=baseline_method,
        peak_model=peak_model,
        shift_tol_nm=shift_tol_nm,
        edge_frac=edge_frac,
        snip_iterations=snip_iterations,
    )
    return float(max(result.area, 0.0))


@dataclass
class PeakIntegrationView:
    """Diagnostic snapshot of local baseline + peak extraction."""

    center_nm: float
    half_width_nm: float
    pad_nm: float
    wavelength_nm: np.ndarray
    intensity: np.ndarray
    baseline: np.ndarray
    corrected: np.ndarray
    area: float
    method: str = "linear"
    edge_lo_nm: float = 0.0
    edge_hi_left_nm: float = 0.0
    edge_lo_right_nm: float = 0.0
    edge_hi_nm: float = 0.0
    integrate_lo_nm: float = 0.0
    integrate_hi_nm: float = 0.0
    peak_model: str = "gaussian"
    fit_ok: bool = False
    fitted_center_nm: float = 0.0
    delta_nm: float = 0.0
    fwhm_nm: float = 0.0
    fit_wavelength_nm: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    #: Fitted profile in raw intensity units (baseline + model)
    fit_intensity: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))


def peak_integration_view(
    spectrum: Spectrum,
    center_nm: float,
    *,
    half_width_nm: float = 0.15,
    pad_nm: float = 0.12,
    method: str = "snip",
    edge_frac: float = 0.40,
    peak_model: str = "gaussian",
    shift_tol_nm: float = 0.15,
    snip_iterations: int = 40,
) -> PeakIntegrationView | None:
    """
    Return arrays for Peak QC: baseline, net span, and optional fitted profile.
    """
    lo, hi, edge_w, integrate_lo, integrate_hi, _ef = _window_bounds(
        center_nm, half_width_nm, pad_nm, edge_frac=edge_frac
    )
    # Slightly wider slice for context when auto-zooming the QC plot
    context = max(0.05, 0.25 * (hi - lo))
    mask = (spectrum.wavelength_nm >= lo - context) & (
        spectrum.wavelength_nm <= hi + context
    )
    if not np.any(mask):
        return None
    wl_raw = np.asarray(spectrum.wavelength_nm[mask], dtype=float)
    y_raw = np.asarray(spectrum.intensity[mask], dtype=float)
    wl, y_corr, baseline = subtract_local_baseline(
        spectrum.wavelength_nm,
        spectrum.intensity,
        center_nm,
        half_width_nm=half_width_nm,
        pad_nm=pad_nm,
        method=method,
        edge_frac=edge_frac,
        snip_iterations=snip_iterations,
    )
    if len(wl) < 2:
        return None

    fit = fit_emission_peak(
        spectrum,
        center_nm,
        half_width_nm=half_width_nm,
        pad_nm=pad_nm,
        baseline_method=method,
        peak_model=peak_model,
        shift_tol_nm=shift_tol_nm,
        edge_frac=edge_frac,
        snip_iterations=snip_iterations,
    )
    # Interpolate baseline onto the (possibly wider) display slice
    baseline_disp = np.interp(wl_raw, wl, baseline, left=baseline[0], right=baseline[-1])
    y_corr_disp = y_raw - baseline_disp

    fit_wl = fit.fit_wavelength_nm
    fit_y_raw = np.array([], dtype=float)
    if fit.fit_ok and len(fit_wl) and len(fit.fit_intensity):
        base_on_fit = np.interp(fit_wl, wl, baseline, left=baseline[0], right=baseline[-1])
        fit_y_raw = base_on_fit + fit.fit_intensity

    return PeakIntegrationView(
        center_nm=center_nm,
        half_width_nm=half_width_nm,
        pad_nm=pad_nm,
        wavelength_nm=wl_raw,
        intensity=y_raw,
        baseline=baseline_disp,
        corrected=y_corr_disp,
        area=float(fit.area),
        method=(method or "linear"),
        edge_lo_nm=lo,
        edge_hi_left_nm=lo + edge_w,
        edge_lo_right_nm=hi - edge_w,
        edge_hi_nm=hi,
        integrate_lo_nm=integrate_lo,
        integrate_hi_nm=integrate_hi,
        peak_model=fit.peak_model,
        fit_ok=fit.fit_ok,
        fitted_center_nm=fit.fitted_center_nm if fit.fit_ok else center_nm,
        delta_nm=fit.delta_nm if fit.fit_ok else 0.0,
        fwhm_nm=fit.fwhm_nm if fit.fit_ok else 0.0,
        fit_wavelength_nm=fit_wl,
        fit_intensity=fit_y_raw,
    )


def peak_net_height(
    spectrum: Spectrum,
    center_nm: float,
    *,
    half_width_nm: float = 0.15,
    pad_nm: float = 0.12,
    method: str = "snip",
    snip_iterations: int = 40,
) -> float:
    """Net peak height at nearest sample after local baseline (secondary metric)."""
    wl, y_corr, _ = subtract_local_baseline(
        spectrum.wavelength_nm,
        spectrum.intensity,
        center_nm,
        half_width_nm=half_width_nm,
        pad_nm=pad_nm,
        method=method,
        snip_iterations=snip_iterations,
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


# Preferred CRM / LIBS calibrant wavelengths (nm), ordered best-first.
# Prefer isolated / strong lines away from the Fe UV forest when possible
# (e.g. Ca II IR triplet, K 766/770, Na D, Li 670).
PREFERRED_CALIBRANT_LINES: dict[str, tuple[tuple[float, str], ...]] = {
    "Ca": (
        (854.209, "Ca II"),
        (866.214, "Ca II"),
        (849.802, "Ca II"),
        (393.366, "Ca II"),
        (396.847, "Ca II"),
        (422.673, "Ca I"),
    ),
    "K": ((766.490, "K I"), (769.896, "K I")),
    "Na": ((588.995, "Na I"), (589.592, "Na I")),
    "Li": ((670.776, "Li I"), (610.354, "Li I")),
    "Mg": (
        (285.213, "Mg I"),
        (279.553, "Mg II"),
        (280.270, "Mg II"),
        (518.360, "Mg I"),
    ),
    "Al": ((396.152, "Al I"), (394.401, "Al I")),
    "Si": ((288.158, "Si I"), (251.611, "Si I"), (252.851, "Si I")),
    "Sr": ((407.771, "Sr II"), (421.552, "Sr II")),
    "Ba": ((455.403, "Ba II"), (493.408, "Ba II")),
    "Cu": ((324.754, "Cu I"), (327.396, "Cu I")),
    "Pb": (
        (405.781, "Pb I"),
        (368.346, "Pb I"),
        (363.957, "Pb I"),
        (280.200, "Pb I"),
    ),
    "Mn": ((403.076, "Mn I"), (279.482, "Mn II")),
    "Cr": ((425.435, "Cr I"), (427.481, "Cr I"), (428.972, "Cr I")),
    "Zn": ((481.053, "Zn I"), (472.215, "Zn I"), (213.857, "Zn I")),
    "Fe": ((371.994, "Fe I"), (373.487, "Fe I"), (404.581, "Fe I")),
    "O": ((777.194, "O I"), (844.636, "O I")),
    "N": ((746.831, "N I"), (744.229, "N I"), (821.634, "N I")),
    "H": ((656.280, "H"),),
    "S": ((921.286, "S I"), (922.809, "S I"), (923.754, "S I")),
    "P": ((253.399, "P I"), (255.328, "P I")),
    "Ti": ((334.941, "Ti II"), (336.121, "Ti II"), (337.280, "Ti II")),
}

# How many lines Suggest offers per element by default.
# 1 is enough for a curve; 2–4 supports multi-line mean + dropping bad λ.
DEFAULT_SUGGEST_LINES_PER_ELEMENT = 4


def _resolve_library_line(
    library: list[LibraryLine],
    element: str,
    wavelength_nm: float,
    *,
    tol_nm: float = 0.05,
) -> LibraryLine | None:
    """Nearest library line for ``element`` within ``tol_nm``, else None."""
    best: LibraryLine | None = None
    best_d = float("inf")
    for line in library:
        if line.element != element:
            continue
        d = abs(float(line.wavelength_nm) - float(wavelength_nm))
        if d <= tol_nm and d < best_d:
            best = line
            best_d = d
    return best


def suggest_diagnostic_lines(
    library: list[LibraryLine],
    elements: list[str],
    *,
    wl_min: float,
    wl_max: float,
    max_per_element: int = 4,
    overlap_tol_nm: float = 0.12,
    identify_matches: dict[str, list[tuple[float, str]]] | None = None,
) -> list[DiagnosticLine]:
    """
    Smart diagnostic-line suggestions for CRM calibration.

    Per element (up to ``max_per_element``, default 4):
      1. Preferred calibrant λ (Ca II IR, K 766/770, …) in range
      2. Identify-tab matches (strongest peaks first), if provided
      3. Strongest remaining NIST I/II lines in range

    One good line is enough for an I→C curve; 2–4 enabled lines give a
    multi-line mean and let you drop bad λ after QC.
    """
    from identify_elements import strong_library_lines

    max_n = max(1, int(max_per_element))
    lines: list[DiagnosticLine] = []

    for el in elements:
        chosen: list[DiagnosticLine] = []
        seen: set[float] = set()

        def _add(wl: float, species: str) -> bool:
            if len(chosen) >= max_n:
                return False
            key = round(float(wl), 3)
            if key in seen:
                return False
            if float(wl) < wl_min - 0.05 or float(wl) > wl_max + 0.05:
                return False
            seen.add(key)
            chosen.append(
                DiagnosticLine(
                    element=el,
                    wavelength_nm=float(wl),
                    species=species or el,
                    enabled=True,
                )
            )
            return True

        # 1) Preferred calibrants
        for pref_wl, pref_sp in PREFERRED_CALIBRANT_LINES.get(el, ()):
            if pref_wl < wl_min - 0.5 or pref_wl > wl_max + 0.5:
                continue
            hit = _resolve_library_line(library, el, pref_wl)
            if hit is not None:
                _add(hit.wavelength_nm, hit.species or pref_sp)
            else:
                _add(pref_wl, pref_sp)

        # 2) Identify matches
        if identify_matches:
            for wl, species in identify_matches.get(el, []):
                if len(chosen) >= max_n:
                    break
                hit = _resolve_library_line(library, el, float(wl), tol_nm=0.15)
                if hit is not None:
                    _add(hit.wavelength_nm, hit.species or species)
                else:
                    _add(float(wl), species or el)

        # 3) Strong NIST fill (LIBS-diagnostic ranking)
        if len(chosen) < max_n:
            by_el = strong_library_lines(
                library,
                [el],
                wl_min=wl_min,
                wl_max=wl_max,
                max_per_element=max(max_n * 3, 12),
                libs_diagnostics=True,
            )
            for L in by_el.get(el, []):
                if len(chosen) >= max_n:
                    break
                _add(float(L.wavelength_nm), L.species or el)

        lines.extend(chosen)

    return flag_line_overlaps(lines, library, tol_nm=overlap_tol_nm)


def seed_lines_from_matches(
    matches_by_element: dict[str, list[tuple[float, str]]],
    library: list[LibraryLine],
    *,
    max_per_element: int = 4,
    overlap_tol_nm: float = 0.12,
    wl_min: float = 180.0,
    wl_max: float = 1022.0,
) -> list[DiagnosticLine]:
    """
    Build diagnostic lines from Identify matches, preferring calibrant λ.

    Thin wrapper around ``suggest_diagnostic_lines`` so Suggest lines and
    Identify seeding share one policy.
    """
    elements = list(matches_by_element.keys())
    return suggest_diagnostic_lines(
        library,
        elements,
        wl_min=wl_min,
        wl_max=wl_max,
        max_per_element=max_per_element,
        overlap_tol_nm=overlap_tol_nm,
        identify_matches=matches_by_element,
    )


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
                inten = extract_peak_intensity(
                    std.spectrum,
                    dline.wavelength_nm,
                    half_width_nm=cal.half_width_nm,
                    pad_nm=cal.baseline_pad_nm,
                    baseline_method=cal.baseline_method,
                    peak_model=cal.peak_model,
                    shift_tol_nm=cal.shift_tol_nm,
                    snip_iterations=cal.snip_iterations,
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
        if fit_response_slope(fit) < 0.0:
            # Keep the curve for QC plots so you can see inverted CRM points;
            # Quant will ignore rejected fits (do not add to ``skipped``).
            fit.rejected = "negative I→C slope"
        fits.append(fit)

    cal.fits = fits
    return fits


def fit_response_slope(fit: CurveFit) -> float:
    """
    Local dC/dI of the calibration polynomial at the mean CRM intensity.

    Negative means the curve is physically unusable for quantification.
    """
    coeffs = np.asarray(fit.coeffs, dtype=float)
    if coeffs.size == 0:
        return float("nan")
    xs = np.asarray(fit.intensities, dtype=float)
    xs = xs[np.isfinite(xs)]
    x0 = float(np.mean(xs)) if len(xs) else 0.0
    # Derivative of np.polyval(coeffs, x)
    if coeffs.size == 1:
        return 0.0
    dcoeffs = np.polyder(coeffs)
    return float(np.polyval(dcoeffs, x0))


def predict_from_fit(fit: CurveFit, intensity: float) -> float:
    """Evaluate I→C curve; concentrations below zero are floored to 0."""
    c, _ = predict_from_fit_ex(fit, intensity)
    return c


def predict_from_fit_ex(fit: CurveFit, intensity: float) -> tuple[float, bool]:
    """
    Evaluate I→C curve.

    Returns ``(concentration, below_calibration)``. When the linear/quadratic
    model yields C < 0 (intensity below the curve’s zero-crossing), concentration
    is floored to 0 and ``below_calibration`` is True.
    """
    c = float(np.polyval(fit.coeffs, intensity))
    if not np.isfinite(c):
        return 0.0, True
    if c < 0.0:
        return 0.0, True
    return c, False


def intensity_zero_crossing(fit: CurveFit) -> float | None:
    """Peak area where the linear I→C model crosses C = 0 (None if N/A)."""
    coeffs = np.asarray(fit.coeffs, dtype=float)
    if coeffs.size < 2 or fit.degree != 1:
        return None
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    if abs(slope) < 1e-18:
        return None
    i0 = -intercept / slope
    return float(i0) if np.isfinite(i0) else None


def _spectrum_matches_standard(unknown: Spectrum, std: StandardSample) -> bool:
    """True if ``unknown`` is the same file as a CRM standard."""
    try:
        up = unknown.meta.path.resolve()
        sp = std.path.resolve()
        if up == sp:
            return True
    except OSError:
        pass
    return unknown.meta.path.name == std.path.name


def intensity_for_prediction(
    cal: CalibrationSet,
    fit: CurveFit,
    unknown: Spectrum,
) -> tuple[float, bool]:
    """
    Peak intensity for I→C on ``unknown``.

    If ``unknown`` is one of the CRM spectra that built ``fit``, reuse the
    stored calibration intensity so Quant of a standard recovers the
    certificate point. Otherwise extract with current cal peak params.

    Returns ``(intensity, reused_from_crm)``.
    """
    for sid, stored in zip(fit.sample_ids, fit.intensities):
        if not np.isfinite(stored):
            continue
        for std in cal.standards:
            if std.sample_id != sid:
                continue
            if _spectrum_matches_standard(unknown, std):
                return float(stored), True
    inten = extract_peak_intensity(
        unknown,
        fit.wavelength_nm,
        half_width_nm=cal.half_width_nm,
        pad_nm=cal.baseline_pad_nm,
        baseline_method=cal.baseline_method,
        peak_model=cal.peak_model,
        shift_tol_nm=cal.shift_tol_nm,
        snip_iterations=cal.snip_iterations,
    )
    return float(inten), False


def predict_with_intensities(
    cal: CalibrationSet,
    unknown: Spectrum,
    *,
    elements: list[str] | None = None,
) -> tuple[list[ElementPrediction], dict[tuple[str, float], float]]:
    """
    Apply fitted curves to an unknown; return predictions and per-line intensities.

    Intensities are keyed by ``(element, wavelength_nm)`` matching ``cal.fits``.
    CRM spectra used to build the curves reuse their stored peak areas so
    Quant of a calibration file lands on the certificate point.
    """
    if not cal.fits:
        build_fits(cal)

    report = list(elements) if elements is not None else cal.active_elements()
    report_set = set(report)

    by_el: dict[str, list[tuple[float, float]]] = {}
    below_el: dict[str, bool] = {}
    intensities: dict[tuple[str, float], float] = {}
    for fit in usable_fits(cal):
        if fit.element not in report_set:
            continue
        inten, _reused = intensity_for_prediction(cal, fit, unknown)
        c, below = predict_from_fit_ex(fit, inten)
        by_el.setdefault(fit.element, []).append((fit.wavelength_nm, c))
        if below:
            below_el[fit.element] = True
        intensities[(fit.element, float(fit.wavelength_nm))] = float(inten)

    preds: list[ElementPrediction] = []
    for el in report:
        pairs = by_el.get(el, [])
        if not pairs:
            continue
        vals = [c for _, c in pairs]
        mean = float(np.mean(vals))
        if mean < 0.0:
            mean = 0.0
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        preds.append(
            ElementPrediction(
                element=el,
                concentration=mean,
                std=std,
                n_lines=len(vals),
                line_predictions=pairs,
                below_calibration=bool(below_el.get(el, False)),
            )
        )
    return preds, intensities


def predict_concentrations(
    cal: CalibrationSet,
    unknown: Spectrum,
    *,
    elements: list[str] | None = None,
) -> list[ElementPrediction]:
    """
    Apply fitted curves to an unknown; average enabled lines per element.

    ``elements`` limits the report (e.g. 2–3 of interest). Default:
    ``cal.active_elements()`` (checked quantify set). Negative curve
    extrapolations are reported as 0 (below calibration / LOD).
    """
    preds, _ = predict_with_intensities(cal, unknown, elements=elements)
    return preds


def quantify_spectrum(
    cal: CalibrationSet,
    unknown: Spectrum,
    *,
    index: int,
    elements: list[str] | None = None,
) -> QuantSpectrumResult:
    """Build a ``QuantSpectrumResult`` for one unknown spectrum."""
    preds, intensities = predict_with_intensities(cal, unknown, elements=elements)
    path = unknown.meta.path
    return QuantSpectrumResult(
        index=int(index),
        filename=path.name,
        spectrum_path=str(path),
        predictions=preds,
        line_intensities=intensities,
    )


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
        "baseline_method": cal.baseline_method,
        "snip_iterations": cal.snip_iterations,
        "peak_model": cal.peak_model,
        "shift_tol_nm": cal.shift_tol_nm,
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
                "rejected": f.rejected,
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
        half_width_nm=float(data.get("half_width_nm", 0.15)),
        baseline_pad_nm=float(data.get("baseline_pad_nm", 0.12)),
        baseline_method=(
            m
            if (m := str(data.get("baseline_method") or "snip").strip().lower())
            in BASELINE_METHODS
            else "snip"
        ),
        snip_iterations=int(data.get("snip_iterations", 40)),
        peak_model=_normalize_peak_model(str(data.get("peak_model") or "gaussian")),
        shift_tol_nm=float(data.get("shift_tol_nm", 0.15)),
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
                rejected=(str(r) if (r := f.get("rejected")) else None),
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
    """Load a spectrum and append it to the calibration set.

    Replicate shots of the same CRM are encouraged: each file is one point on
    the I→C curve. Give them the same concentration; unique ``sample_id`` values
    are auto-suffixed if the stem collides.
    """
    spectrum = load_spectrum(Path(txt_path), cfg_path)
    sid = sample_id or spectrum.meta.path.stem
    existing = {s.sample_id for s in cal.standards}
    if sid in existing:
        stem, i = sid, 2
        while f"{stem}_{i}" in existing:
            i += 1
        sid = f"{stem}_{i}"
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


def set_standard_concentrations(
    standards: list[StandardSample],
    element: str,
    value: float | None,
    *,
    indices: list[int] | None = None,
) -> int:
    """
    Set ``element`` concentration on selected standards (or all if indices is None).

    Returns how many standards were updated. Use the same value on replicate
    spectra of one CRM so each shot contributes an independent I at that C.
    """
    el = (element or "").strip()
    if not el:
        return 0
    if indices is None:
        targets = list(range(len(standards)))
    else:
        targets = [i for i in indices if 0 <= i < len(standards)]
    n = 0
    for i in targets:
        standards[i].concentrations[el] = None if value is None else float(value)
        n += 1
    return n


def concentration_level_summary(
    concentrations: list[float],
    *,
    tol: float = 1e-9,
) -> tuple[int, int]:
    """Return ``(n_points, n_unique_C_levels)`` for finite concentrations."""
    vals = [float(c) for c in concentrations if c is not None and np.isfinite(c)]
    if not vals:
        return 0, 0
    # Cluster nearly-equal C values (replicate shots share a level)
    levels: list[float] = []
    for v in sorted(vals):
        if not levels or abs(v - levels[-1]) > max(tol, 1e-6 * max(abs(v), 1.0)):
            levels.append(v)
    return len(vals), len(levels)

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
