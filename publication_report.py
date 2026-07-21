"""
Publication figures: for each identified element, plot its strongest matched lines
in separate zoomed windows (up to 5).

Example (CLI):
  .venv/bin/python publication_report.py docs/stone-9b.txt -o reports/stone-9b.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from identify_elements import (
    ElementHit,
    Match,
    Spectrum,
    find_spectrum_peaks,
    load_line_library,
    load_spectrum,
    match_peaks,
    score_elements,
)
from matplotlib_config import apply_matplotlib_config

ROOT = Path(__file__).resolve().parent
DEFAULT_LIBRARY = ROOT / "nist_lines" / "libs_line_library.csv"


def strongest_matches(hit: ElementHit, n: int = 5) -> list[Match]:
    """Return up to n matches ranked by observed peak intensity (then prominence)."""
    return sorted(
        hit.matches,
        key=lambda m: (m.peak.intensity, m.peak.prominence),
        reverse=True,
    )[:n]


def _window_slice(
    wavelength_nm: np.ndarray,
    intensity: np.ndarray,
    center_nm: float,
    half_width_nm: float,
) -> tuple[np.ndarray, np.ndarray]:
    lo = center_nm - half_width_nm
    hi = center_nm + half_width_nm
    mask = (wavelength_nm >= lo) & (wavelength_nm <= hi)
    return wavelength_nm[mask], intensity[mask]


def plot_element_line_panels(
    spectrum: Spectrum,
    hit: ElementHit,
    *,
    n_lines: int = 5,
    half_width_nm: float = 1.5,
    atmosphere: str | None = None,
    title_extra: str = "",
    fig: Figure | None = None,
) -> Figure:
    """
    One figure for an element: up to ``n_lines`` panels, each zoomed on a strong match.
    Unused panel slots are omitted (figure width scales with panel count).

    Pass an existing ``fig`` to redraw into an embedded GUI canvas.
    """
    apply_matplotlib_config()
    matches = strongest_matches(hit, n=n_lines)
    n = len(matches)

    if fig is None:
        if n == 0:
            fig = plt.figure(figsize=(8, 3))
        else:
            fig_w = max(3.2 * n, 6.0)
            fig = plt.figure(figsize=(fig_w, 3.6))
    else:
        fig.clear()

    if n == 0:
        fig.text(0.5, 0.5, f"{hit.element}: no matched lines", ha="center", va="center")
        fig.tight_layout()
        return fig

    axes = fig.subplots(1, n, sharey=False)
    if n == 1:
        axes = [axes]

    sample = spectrum.meta.path.stem
    atm = f" [{atmosphere}]" if atmosphere else ""
    fig.suptitle(
        f"{sample}{atm} — {hit.element}  "
        f"top {n} intense matched lines  "
        f"({hit.n_peaks} total, confidence {hit.confidence:.0f}%)"
        f"{title_extra}",
        fontsize=11,
        y=0.98,
    )

    wl_all = spectrum.wavelength_nm
    y_all = spectrum.intensity

    for ax, m, i in zip(axes, matches, range(1, n + 1)):
        center = m.peak.wavelength_nm
        x, y = _window_slice(wl_all, y_all, center, half_width_nm)
        if len(x) == 0:
            ax.set_title(f"({i}) no data near {center:.2f} nm", fontsize=9)
            continue

        ax.plot(x, y, color="#1a1a1a", lw=0.9)
        ax.axvline(m.peak.wavelength_nm, color="#c0392b", lw=1.1, alpha=0.85, label="observed")
        ax.axvline(m.line.wavelength_nm, color="#1e8449", lw=1.0, ls="--", alpha=0.9, label="NIST")
        ax.scatter(
            [m.peak.wavelength_nm],
            [m.peak.intensity],
            s=36,
            c="#c0392b",
            zorder=3,
            edgecolors="white",
            linewidths=0.5,
        )

        nist_i = (
            f", NIST I={m.line.intensity:.0f}"
            if m.line.intensity is not None
            else ""
        )
        ax.set_title(
            f"({i}) {m.line.species}  λ={m.peak.wavelength_nm:.3f} nm\n"
            f"I={m.peak.intensity:.0f}  Δλ={m.delta_nm:+.3f} nm{nist_i}",
            fontsize=9,
        )
        ax.set_xlabel("Wavelength (nm)")
        ax.set_xlim(center - half_width_nm, center + half_width_nm)
        # Local y-scale with a little headroom
        y_hi = float(np.max(y)) if len(y) else m.peak.intensity
        y_lo = float(np.min(y)) if len(y) else 0.0
        pad = 0.08 * (y_hi - y_lo + 1.0)
        ax.set_ylim(y_lo - pad, y_hi + pad)
        ax.tick_params(labelsize=8)

    axes[0].set_ylabel("Intensity (counts)")
    axes[0].legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    return fig


def export_element_report_pdf(
    spectrum: Spectrum,
    hits: list[ElementHit],
    out_path: Path,
    *,
    n_lines: int = 5,
    half_width_nm: float = 1.5,
    atmosphere: str | None = None,
    max_elements: int | None = None,
) -> Path:
    """
    Write a multipage PDF: one page per element with up to ``n_lines`` zoomed panels.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected = hits if max_elements is None else hits[:max_elements]
    if not selected:
        raise ValueError("No elements to export — run Find peaks + match first.")

    apply_matplotlib_config()
    with PdfPages(out_path) as pdf:
        # Cover / summary page
        fig_sum = _summary_figure(spectrum, selected, atmosphere=atmosphere)
        pdf.savefig(fig_sum, bbox_inches="tight")
        plt.close(fig_sum)

        for hit in selected:
            fig = plot_element_line_panels(
                spectrum,
                hit,
                n_lines=n_lines,
                half_width_nm=half_width_nm,
                atmosphere=atmosphere,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return out_path


def export_element_report_pngs(
    spectrum: Spectrum,
    hits: list[ElementHit],
    out_dir: Path,
    *,
    n_lines: int = 5,
    half_width_nm: float = 1.5,
    atmosphere: str | None = None,
    max_elements: int | None = None,
    dpi: int = 200,
) -> list[Path]:
    """Save one PNG per element into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = hits if max_elements is None else hits[:max_elements]
    paths: list[Path] = []
    for hit in selected:
        fig = plot_element_line_panels(
            spectrum,
            hit,
            n_lines=n_lines,
            half_width_nm=half_width_nm,
            atmosphere=atmosphere,
        )
        path = out_dir / f"{spectrum.meta.path.stem}_{hit.element}_lines.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def _summary_figure(
    spectrum: Spectrum,
    hits: list[ElementHit],
    *,
    atmosphere: str | None = None,
) -> Figure:
    apply_matplotlib_config()
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    m = spectrum.meta
    atm = atmosphere or "—"
    lines = [
        "LIBS element identification report",
        "",
        f"Sample:        {m.path.name}",
        f"Config:        {m.cfg_path.name if m.cfg_path else '—'}",
        f"Atmosphere:    {atm}",
    ]
    if m.laser_energy_mJ is not None:
        lines.append(f"Laser:         {m.laser_energy_mJ:g} mJ")
    if m.qs_delay_us is not None:
        lines.append(f"QS delay:      {m.qs_delay_us:g} µs")
    if m.integration_time_us is not None:
        delay = (
            f", delay {m.integration_delay_us:g} µs"
            if m.integration_delay_us is not None
            else ""
        )
        lines.append(f"Gate:          {m.integration_time_us:g} µs{delay}")
    if m.n_accumulations is not None:
        lines.append(f"Accumulations: {m.n_accumulations}")
    wl0 = float(spectrum.wavelength_nm.min())
    wl1 = float(spectrum.wavelength_nm.max())
    lines.append(f"Range:         {wl0:.2f}–{wl1:.2f} nm")
    lines += [
        "",
        "Ranked elements (confidence = ID strength, not concentration)",
        "",
        f"{'#':<4} {'El':<6} {'Peaks':>6} {'Conf %':>8}",
        "-" * 28,
    ]
    for i, hit in enumerate(hits, start=1):
        lines.append(
            f"{i:<4} {hit.element:<6} {hit.n_peaks:>6} {hit.confidence:>7.0f}%"
        )
    lines += [
        "",
        "Following pages: up to 5 strongest matched lines per element.",
        "Solid red = observed peak; dashed green = NIST wavelength.",
    ]
    ax.text(
        0.08,
        0.95,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LIBS publication line-panel report")
    parser.add_argument("spectrum", type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PDF path (default: reports/<stem>_lines.pdf)",
    )
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--tol-nm", type=float, default=0.12)
    parser.add_argument("--prominence-frac", type=float, default=0.015)
    parser.add_argument("--half-width-nm", type=float, default=1.5)
    parser.add_argument("--n-lines", type=int, default=5)
    parser.add_argument("--max-elements", type=int, default=None)
    parser.add_argument("--atmosphere", type=str, default=None)
    args = parser.parse_args()

    spectrum = load_spectrum(args.spectrum)
    library = load_line_library(args.library)
    peaks = find_spectrum_peaks(spectrum, min_prominence_frac=args.prominence_frac)
    support: dict[str, float] = {}
    primary: dict[str, bool] = {}
    primary_wl: dict[str, float] = {}
    matches = match_peaks(
        peaks,
        library,
        tol_nm=args.tol_nm,
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

    out = args.output
    if out is None:
        out = ROOT / "reports" / f"{spectrum.meta.path.stem}_lines.pdf"

    path = export_element_report_pdf(
        spectrum,
        hits,
        out,
        n_lines=args.n_lines,
        half_width_nm=args.half_width_nm,
        atmosphere=args.atmosphere,
        max_elements=args.max_elements,
    )
    print(f"Wrote {path}  ({min(len(hits), args.max_elements or len(hits))} element pages)")


if __name__ == "__main__":
    main()
