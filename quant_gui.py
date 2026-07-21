"""Quant results tab: CRM predictions, curve overlay, peak QC, C vs spectrum #."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from calibration import (
    CalibrationSet,
    CurveFit,
    QuantSpectrumResult,
    confidence_interval_95,
    extract_peak_intensity,
    peak_integration_view,
    predict_from_fit,
)
from identify_elements import Spectrum
from matplotlib_config import apply_matplotlib_config

ROOT = Path(__file__).resolve().parent


def _shrink_mpl_toolbar(toolbar: NavigationToolbar2QT) -> None:
    """Match Identify-tab matplotlib toolbar: icons ~50% with tighter spacing."""
    icon = toolbar.iconSize()
    toolbar.setIconSize(QSize(max(12, icon.width() // 2), max(12, icon.height() // 2)))
    toolbar.setStyleSheet(
        "QToolBar { spacing: 2px; padding: 1px; }"
        "QToolButton { padding: 1px; margin: 0px; }"
    )


class _PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, *, figsize: tuple[float, float] = (4.5, 3.2)) -> None:
        apply_matplotlib_config()
        self.fig = Figure(figsize=figsize, dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.fig.tight_layout()


def plot_quant_peak_panel(
    ax,
    spectrum: Spectrum,
    fit: CurveFit,
    *,
    half_width_nm: float,
    pad_nm: float,
    method: str,
    unit: str = "",
    peak_model: str = "gaussian",
    shift_tol_nm: float = 0.15,
) -> None:
    """Draw local baseline + fitted (or net-area) peak for one CRM line."""
    view = peak_integration_view(
        spectrum,
        fit.wavelength_nm,
        half_width_nm=half_width_nm,
        pad_nm=pad_nm,
        method=method,
        peak_model=peak_model,
        shift_tol_nm=shift_tol_nm,
    )
    inten = extract_peak_intensity(
        spectrum,
        fit.wavelength_nm,
        half_width_nm=half_width_nm,
        pad_nm=pad_nm,
        baseline_method=method,
        peak_model=peak_model,
        shift_tol_nm=shift_tol_nm,
    )
    c_line = predict_from_fit(fit, inten)
    unit_s = f" {unit}" if unit else ""

    if view is None or len(view.wavelength_nm) < 2:
        ax.text(
            0.5,
            0.5,
            f"No samples near {fit.wavelength_nm:.3f} nm",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#666",
        )
        ax.set_title(f"{fit.element} {fit.wavelength_nm:.3f} nm")
        return

    center = fit.wavelength_nm
    wl = view.wavelength_nm
    y = view.intensity
    base = view.baseline
    i_lo, i_hi = view.integrate_lo_nm, view.integrate_hi_nm
    core = (wl >= i_lo) & (wl <= i_hi)
    fit_ok = bool(view.fit_ok) and view.peak_model != "net_area"

    ax.axvspan(view.edge_lo_nm, view.edge_hi_left_nm, color="#e67e22", alpha=0.22, zorder=0)
    ax.axvspan(view.edge_lo_right_nm, view.edge_hi_nm, color="#e67e22", alpha=0.22, zorder=0)
    ax.plot(wl, y, color="#1a252f", lw=1.2, zorder=3)
    ax.plot(wl, base, color="#d35400", lw=1.15, ls="--", zorder=3)
    if fit_ok and len(view.fit_wavelength_nm) and len(view.fit_intensity):
        ax.plot(view.fit_wavelength_nm, view.fit_intensity, color="#1e8449", lw=1.6, zorder=4)
        ax.axvline(view.fitted_center_nm, color="#1e8449", lw=1.0, zorder=5)
    elif np.any(core):
        ax.fill_between(
            wl[core],
            base[core],
            y[core],
            where=(y[core] >= base[core]),
            color="#1a5276",
            alpha=0.38,
            zorder=2,
        )
    ax.axvline(center, color="#c0392b", lw=1.0, ls=":", zorder=4)

    yc = np.asarray(view.corrected, dtype=float)
    ymax_c = float(np.nanmax(yc)) if len(yc) else 0.0
    if ymax_c > 0 and np.count_nonzero(yc >= 0.06 * ymax_c) >= 2:
        idx = np.flatnonzero(yc >= 0.06 * ymax_c)
        p_lo, p_hi = float(wl[idx[0]]), float(wl[idx[-1]])
    else:
        p_lo, p_hi = i_lo, i_hi
    x0 = min(p_lo, i_lo, view.edge_hi_left_nm)
    x1 = max(p_hi, i_hi, view.edge_lo_right_nm)
    margin = max(0.012, 0.07 * max(x1 - x0, 1e-6))
    ax.set_xlim(x0 - margin, x1 + margin)
    in_view = (wl >= x0 - margin) & (wl <= x1 + margin)
    if np.any(in_view):
        y_lo = float(min(np.nanmin(y[in_view]), np.nanmin(base[in_view])))
        y_hi = float(max(np.nanmax(y[in_view]), np.nanmax(base[in_view])))
        pad_y = 0.10 * max(y_hi - y_lo, 1.0)
        ax.set_ylim(y_lo - 0.5 * pad_y, y_hi + pad_y)

    ax.set_xlabel("λ (nm)", fontsize=8)
    ax.set_ylabel("counts", fontsize=8)
    if fit_ok:
        ax.set_title(
            f"{fit.element} {center:.3f} nm · {view.peak_model} "
            f"Δλ={view.delta_nm:+.3f} · area={view.area:.4g} · C={c_line:.4g}{unit_s}",
            fontsize=8,
        )
    else:
        ax.set_title(
            f"{fit.element} {center:.3f} nm · area={view.area:.4g} · C={c_line:.4g}{unit_s}",
            fontsize=9,
        )
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.28)


def _fmt_num(v: float | None, *, digits: int = 4) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:.{digits}g}"


def _fmt_ci(lo: float | None, hi: float | None) -> str:
    if lo is None or hi is None:
        return "—"
    return f"{lo:.4g}–{hi:.4g}"


class QuantTab(QWidget):
    """Top-level Quant tab: results table, I→C overlay, peak QC, C vs spectrum #."""

    statusMessage = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._results: list[QuantSpectrumResult] = []
        self._cal: CalibrationSet | None = None
        self._unit: str = ""
        self._resolve_spectrum: Callable[[QuantSpectrumResult], Spectrum | None] | None = None
        self._updating_selectors = False

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        self.hint = QLabel(
            "Build curves on Calibrate, then Quant selected spectra on Identify."
        )
        self.hint.setStyleSheet("color: #555;")
        self.hint.setWordWrap(True)
        root.addWidget(self.hint)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Element"))
        self.combo_el = QComboBox()
        self.combo_el.setMinimumWidth(80)
        self.combo_el.currentIndexChanged.connect(self._on_element_changed)
        ctrl.addWidget(self.combo_el)

        ctrl.addWidget(QLabel("Spectrum"))
        self.combo_spec = QComboBox()
        self.combo_spec.setMinimumWidth(160)
        self.combo_spec.currentIndexChanged.connect(self._on_selection_changed)
        ctrl.addWidget(self.combo_spec, stretch=1)

        ctrl.addWidget(QLabel("Line"))
        self.combo_line = QComboBox()
        self.combo_line.setMinimumWidth(120)
        self.combo_line.currentIndexChanged.connect(self._on_selection_changed)
        ctrl.addWidget(self.combo_line)

        self.btn_export = QPushButton("Export CSV…")
        self.btn_export.clicked.connect(self.export_csv)
        ctrl.addWidget(self.btn_export)
        root.addLayout(ctrl)

        self.summary = QLabel("")
        self.summary.setStyleSheet("color: #333; font-size: 11px;")
        self.summary.setWordWrap(True)
        root.addWidget(self.summary)

        split = QSplitter(Qt.Orientation.Vertical)
        top = QSplitter(Qt.Orientation.Horizontal)

        # --- Results table ---
        table_wrap = QWidget()
        table_l = QVBoxLayout(table_wrap)
        table_l.setContentsMargins(0, 0, 0, 0)
        table_l.addWidget(QLabel("<b>Results</b>"))
        self.table = QTableWidget(0, 2)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        table_l.addWidget(self.table)
        top.addWidget(table_wrap)

        # --- Calibration curve ---
        curve_wrap = QWidget()
        curve_l = QVBoxLayout(curve_wrap)
        curve_l.setContentsMargins(0, 0, 0, 0)
        curve_l.addWidget(QLabel("<b>Calibration curve</b>"))
        self.curve_canvas = _PlotCanvas(figsize=(4.8, 3.4))
        self.curve_toolbar = NavigationToolbar2QT(self.curve_canvas, curve_wrap)
        _shrink_mpl_toolbar(self.curve_toolbar)
        curve_l.addWidget(self.curve_toolbar)
        curve_l.addWidget(self.curve_canvas, stretch=1)
        top.addWidget(curve_wrap)
        top.setStretchFactor(0, 2)
        top.setStretchFactor(1, 2)
        split.addWidget(top)

        bottom = QSplitter(Qt.Orientation.Horizontal)

        # --- Peak QC ---
        peak_wrap = QWidget()
        peak_l = QVBoxLayout(peak_wrap)
        peak_l.setContentsMargins(0, 0, 0, 0)
        peak_l.addWidget(QLabel("<b>Peak / background fit</b>"))
        self.peak_canvas = _PlotCanvas(figsize=(4.8, 3.4))
        self.peak_toolbar = NavigationToolbar2QT(self.peak_canvas, peak_wrap)
        _shrink_mpl_toolbar(self.peak_toolbar)
        peak_l.addWidget(self.peak_toolbar)
        peak_l.addWidget(self.peak_canvas, stretch=1)
        bottom.addWidget(peak_wrap)

        # --- Series plot ---
        series_wrap = QWidget()
        series_l = QVBoxLayout(series_wrap)
        series_l.setContentsMargins(0, 0, 0, 0)
        series_l.addWidget(QLabel("<b>Concentration vs spectrum #</b>"))
        self.series_canvas = _PlotCanvas(figsize=(4.8, 3.4))
        self.series_toolbar = NavigationToolbar2QT(self.series_canvas, series_wrap)
        _shrink_mpl_toolbar(self.series_toolbar)
        series_l.addWidget(self.series_toolbar)
        series_l.addWidget(self.series_canvas, stretch=1)
        bottom.addWidget(series_wrap)
        bottom.setStretchFactor(0, 1)
        bottom.setStretchFactor(1, 1)
        split.addWidget(bottom)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 2)
        root.addWidget(split, stretch=1)

        self._clear_plots_empty()

    # ------------------------------------------------------------------ API
    def set_spectrum_resolver(
        self, resolver: Callable[[QuantSpectrumResult], Spectrum | None]
    ) -> None:
        self._resolve_spectrum = resolver

    def clear_results(self) -> None:
        self._results = []
        self._cal = None
        self._unit = ""
        self._refresh_all()

    def set_results(
        self,
        results: list[QuantSpectrumResult],
        cal: CalibrationSet,
        *,
        unit: str = "",
    ) -> None:
        self._results = list(results)
        self._cal = cal
        self._unit = unit or cal.concentration_unit or ""
        self._refresh_all()
        if results:
            n_el = len({p.element for r in results for p in r.predictions})
            self.statusMessage.emit(
                f"Quant results: {len(results)} spectrum(a), {n_el} element(s)."
            )

    def results(self) -> list[QuantSpectrumResult]:
        return list(self._results)

    # --------------------------------------------------------------- refresh
    def _refresh_all(self) -> None:
        self._fill_selectors()
        self._fill_table()
        self._update_summary()
        self._redraw_plots()

    def _elements(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for r in self._results:
            for p in r.predictions:
                if p.element not in seen:
                    seen.add(p.element)
                    out.append(p.element)
        if not out and self._cal is not None:
            out = list(self._cal.active_elements())
        return out

    def _selected_element(self) -> str | None:
        data = self.combo_el.currentData()
        return str(data) if data else None

    def _selected_result(self) -> QuantSpectrumResult | None:
        idx = self.combo_spec.currentData()
        if idx is None or not self._results:
            return None
        i = int(idx)
        if 0 <= i < len(self._results):
            return self._results[i]
        return None

    def _selected_wavelength(self) -> float | None:
        data = self.combo_line.currentData()
        return float(data) if data is not None else None

    def _fit_for_selection(self) -> CurveFit | None:
        if self._cal is None:
            return None
        el = self._selected_element()
        wl = self._selected_wavelength()
        if el is None or wl is None:
            return None
        for f in self._cal.fits:
            if f.element == el and abs(float(f.wavelength_nm) - wl) < 1e-6:
                return f
        return None

    def _fill_selectors(self) -> None:
        self._updating_selectors = True
        prev_el = self._selected_element()
        prev_spec = self.combo_spec.currentData()
        prev_wl = self._selected_wavelength()

        self.combo_el.clear()
        for el in self._elements():
            self.combo_el.addItem(el, el)
        if prev_el:
            i = self.combo_el.findData(prev_el)
            if i >= 0:
                self.combo_el.setCurrentIndex(i)

        self.combo_spec.clear()
        for i, r in enumerate(self._results):
            self.combo_spec.addItem(f"{r.index}: {r.filename}", i)
        if prev_spec is not None:
            i = self.combo_spec.findData(prev_spec)
            if i >= 0:
                self.combo_spec.setCurrentIndex(i)

        self._fill_line_combo(prefer_wl=prev_wl)
        self._updating_selectors = False

        if self._results:
            unit = f" ({self._unit})" if self._unit else ""
            self.hint.setText(
                f"{len(self._results)} quantified spectrum(a). "
                f"Select element / spectrum / line to inspect peak fits and the I→C overlay{unit}."
            )
        else:
            self.hint.setText(
                "Build curves on Calibrate, then Quant selected spectra on Identify."
            )

    def _fill_line_combo(self, *, prefer_wl: float | None = None) -> None:
        self.combo_line.blockSignals(True)
        self.combo_line.clear()
        el = self._selected_element()
        if self._cal is not None and el:
            fits = [f for f in self._cal.fits if f.element == el]
            fits.sort(key=lambda f: (-f.r_squared, f.wavelength_nm))
            for f in fits:
                self.combo_line.addItem(
                    f"{f.wavelength_nm:.3f} nm  (R²={f.r_squared:.3f})",
                    float(f.wavelength_nm),
                )
        if prefer_wl is not None:
            for i in range(self.combo_line.count()):
                wl = self.combo_line.itemData(i)
                if wl is not None and abs(float(wl) - prefer_wl) < 1e-6:
                    self.combo_line.setCurrentIndex(i)
                    break
        self.combo_line.blockSignals(False)

    def _fill_table(self) -> None:
        el = self._selected_element()
        unit = self._unit
        if unit:
            headers = [
                "#",
                "File",
                f"C ({unit})",
                f"std ({unit})",
                f"95% CI ({unit})",
                "n_lines",
            ]
        else:
            headers = ["#", "File", "C", "std", "95% CI", "n_lines"]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(self._results))

        for i, r in enumerate(self._results):
            self.table.setItem(i, 0, QTableWidgetItem(str(r.index)))
            self.table.setItem(i, 1, QTableWidgetItem(r.filename))
            pred = r.prediction_for(el) if el else None
            if pred is None:
                for col in range(2, 6):
                    self.table.setItem(i, col, QTableWidgetItem("—"))
                continue
            ci = confidence_interval_95(pred.concentration, pred.std, pred.n_lines)
            self.table.setItem(i, 2, QTableWidgetItem(_fmt_num(pred.concentration)))
            self.table.setItem(i, 3, QTableWidgetItem(_fmt_num(pred.std)))
            lo, hi = (ci if ci else (None, None))
            self.table.setItem(i, 4, QTableWidgetItem(_fmt_ci(lo, hi)))
            self.table.setItem(i, 5, QTableWidgetItem(str(pred.n_lines)))

        # Keep table selection in sync with spectrum combo
        spec_i = self.combo_spec.currentData()
        if spec_i is not None and 0 <= int(spec_i) < self.table.rowCount():
            self.table.blockSignals(True)
            self.table.selectRow(int(spec_i))
            self.table.blockSignals(False)

    def _update_summary(self) -> None:
        el = self._selected_element()
        if not el or not self._results:
            self.summary.setText("")
            return
        vals: list[float] = []
        for r in self._results:
            p = r.prediction_for(el)
            if p is not None:
                vals.append(float(p.concentration))
        if not vals:
            self.summary.setText(f"{el}: no predictions in this Quant run.")
            return
        mean = float(np.mean(vals))
        unit = f" {self._unit}" if self._unit else ""
        if len(vals) >= 2:
            std = float(np.std(vals, ddof=1))
            ci = confidence_interval_95(mean, std, len(vals))
            ci_s = _fmt_ci(*(ci if ci else (None, None)))
            self.summary.setText(
                f"{el} across {len(vals)} spectra: "
                f"mean={mean:.4g}{unit} · std={std:.4g}{unit} · 95% CI={ci_s}{unit}"
            )
        else:
            pred = self._results[0].prediction_for(el)
            line_ci = None
            if pred is not None:
                line_ci = confidence_interval_95(
                    pred.concentration, pred.std, pred.n_lines
                )
            if line_ci:
                self.summary.setText(
                    f"{el}: C={mean:.4g}{unit} · line-to-line 95% CI="
                    f"{_fmt_ci(*line_ci)}{unit} (n_lines={pred.n_lines if pred else 0})"
                )
            else:
                self.summary.setText(f"{el}: C={mean:.4g}{unit}")

    # --------------------------------------------------------------- events
    def _on_element_changed(self, _index: int = 0) -> None:
        if self._updating_selectors:
            return
        self._updating_selectors = True
        self._fill_line_combo()
        self._updating_selectors = False
        self._fill_table()
        self._update_summary()
        self._redraw_plots()

    def _on_selection_changed(self, _index: int = 0) -> None:
        if self._updating_selectors:
            return
        spec_i = self.combo_spec.currentData()
        if spec_i is not None and 0 <= int(spec_i) < self.table.rowCount():
            self.table.blockSignals(True)
            self.table.selectRow(int(spec_i))
            self.table.blockSignals(False)
        self._redraw_plots()

    def _on_table_selection(self) -> None:
        if self._updating_selectors:
            return
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return
        row = rows[0].row()
        i = self.combo_spec.findData(row)
        if i >= 0 and self.combo_spec.currentIndex() != i:
            self._updating_selectors = True
            self.combo_spec.setCurrentIndex(i)
            self._updating_selectors = False
            self._redraw_plots()

    # ----------------------------------------------------------------- plots
    def _clear_plots_empty(self) -> None:
        for canvas, msg in (
            (self.curve_canvas, "No Quant results yet"),
            (self.peak_canvas, "No Quant results yet"),
            (self.series_canvas, "No Quant results yet"),
        ):
            ax = canvas.ax
            ax.clear()
            ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color="#666")
            ax.set_axis_off()
            canvas.fig.tight_layout()
            canvas.draw_idle()

    def _redraw_plots(self) -> None:
        if not self._results or self._cal is None:
            self._clear_plots_empty()
            return
        self._draw_curve()
        self._draw_peak()
        self._draw_series()

    def _draw_curve(self) -> None:
        ax = self.curve_canvas.ax
        ax.clear()
        ax.set_axis_on()
        fit = self._fit_for_selection()
        unit = self._unit or "C"
        if fit is None:
            ax.text(
                0.5,
                0.5,
                "No calibration fit for this element / line",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#666",
            )
            ax.set_axis_off()
            self.curve_canvas.fig.tight_layout()
            self.curve_canvas.draw_idle()
            return

        x = np.asarray(fit.intensities, dtype=float)
        y = np.asarray(fit.concentrations, dtype=float)
        ax.scatter(x, y, c="#1a5276", s=36, zorder=3, label="CRM")
        for sid, xi, yi in zip(fit.sample_ids, x, y):
            ax.annotate(
                str(sid),
                (xi, yi),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                color="#555",
            )
        if len(x):
            x_min = float(np.nanmin(x))
            x_max = float(np.nanmax(x))
        else:
            x_min, x_max = 0.0, 1.0

        result = self._selected_result()
        el = fit.element
        wl = float(fit.wavelength_nm)
        unknown_i = None
        unknown_c = None
        if result is not None:
            key = (el, wl)
            # tolerate float key mismatch
            for k, v in result.line_intensities.items():
                if k[0] == el and abs(float(k[1]) - wl) < 1e-6:
                    unknown_i = float(v)
                    break
            pred = result.prediction_for(el)
            if unknown_i is not None:
                unknown_c = predict_from_fit(fit, unknown_i)
            elif pred is not None:
                for lw, c in pred.line_predictions:
                    if abs(float(lw) - wl) < 1e-6:
                        unknown_c = float(c)
                        break

        xs = [x_min, x_max]
        if unknown_i is not None and np.isfinite(unknown_i):
            xs.append(unknown_i)
        pad = 0.05 * max(max(xs) - min(xs), 1e-9)
        x_line = np.linspace(min(xs) - pad, max(xs) + pad, 80)
        y_line = np.polyval(fit.coeffs, x_line)
        ax.plot(x_line, y_line, color="#c0392b", lw=1.4, label="I→C fit")

        if unknown_i is not None and unknown_c is not None and np.isfinite(unknown_i):
            ax.scatter(
                [unknown_i],
                [unknown_c],
                marker="*",
                s=160,
                c="#c0392b",
                zorder=5,
                label="Unknown",
            )
            mean_note = ""
            if result is not None:
                pred = result.prediction_for(el)
                if pred is not None and pred.n_lines > 1:
                    mean_note = f" · mean C={pred.concentration:.4g}"
            ax.annotate(
                f"C={unknown_c:.4g}{mean_note}",
                (unknown_i, unknown_c),
                textcoords="offset points",
                xytext=(8, -10),
                fontsize=8,
                color="#c0392b",
            )

        ax.set_xlabel("Peak area", fontsize=9)
        ax.set_ylabel(unit, fontsize=9)
        ax.set_title(
            f"{el} {wl:.3f} nm · R²={fit.r_squared:.3f} · n={fit.n_points}",
            fontsize=10,
        )
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.28)
        ax.tick_params(labelsize=8)
        self.curve_canvas.fig.tight_layout()
        self.curve_canvas.draw_idle()

    def _draw_peak(self) -> None:
        ax = self.peak_canvas.ax
        ax.clear()
        ax.set_axis_on()
        fit = self._fit_for_selection()
        result = self._selected_result()
        if fit is None or result is None or self._cal is None:
            ax.text(
                0.5,
                0.5,
                "Select a spectrum and line",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#666",
            )
            ax.set_axis_off()
            self.peak_canvas.fig.tight_layout()
            self.peak_canvas.draw_idle()
            return

        spectrum = None
        if self._resolve_spectrum is not None:
            spectrum = self._resolve_spectrum(result)
        if spectrum is None:
            ax.text(
                0.5,
                0.5,
                f"Spectrum not loaded:\n{result.filename}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#666",
            )
            ax.set_axis_off()
            self.peak_canvas.fig.tight_layout()
            self.peak_canvas.draw_idle()
            return

        plot_quant_peak_panel(
            ax,
            spectrum,
            fit,
            half_width_nm=self._cal.half_width_nm,
            pad_nm=self._cal.baseline_pad_nm,
            method=self._cal.baseline_method,
            unit=self._unit,
            peak_model=self._cal.peak_model,
            shift_tol_nm=self._cal.shift_tol_nm,
        )
        self.peak_canvas.fig.tight_layout()
        self.peak_canvas.draw_idle()

    def _draw_series(self) -> None:
        ax = self.series_canvas.ax
        ax.clear()
        ax.set_axis_on()
        el = self._selected_element()
        unit = self._unit or "C"
        if not el:
            ax.text(
                0.5,
                0.5,
                "No element selected",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#666",
            )
            ax.set_axis_off()
            self.series_canvas.fig.tight_layout()
            self.series_canvas.draw_idle()
            return

        xs: list[int] = []
        ys: list[float] = []
        yerr: list[float] = []
        for r in self._results:
            p = r.prediction_for(el)
            if p is None:
                continue
            xs.append(r.index)
            ys.append(float(p.concentration))
            yerr.append(float(p.std) if p.std is not None else 0.0)

        if not xs:
            ax.text(
                0.5,
                0.5,
                f"No {el} predictions",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#666",
            )
            ax.set_axis_off()
            self.series_canvas.fig.tight_layout()
            self.series_canvas.draw_idle()
            return

        ax.errorbar(
            xs,
            ys,
            yerr=yerr if any(e > 0 for e in yerr) else None,
            fmt="o-",
            color="#1a5276",
            ecolor="#7f8c8d",
            capsize=3,
            lw=1.2,
            ms=5,
            label=el,
        )
        if len(ys) >= 2:
            mean = float(np.mean(ys))
            std = float(np.std(ys, ddof=1))
            ax.axhline(mean, color="#c0392b", ls="--", lw=1.1, label=f"mean={mean:.4g}")
            ax.axhspan(mean - std, mean + std, color="#c0392b", alpha=0.12, label=f"±std")
        ax.set_xlabel("Spectrum #", fontsize=9)
        ax.set_ylabel(f"{el} ({unit})" if unit else el, fontsize=9)
        ax.set_title(f"{el} across Quant batch", fontsize=10)
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.28)
        ax.tick_params(labelsize=8)
        if len(xs) == 1:
            ax.set_xlim(xs[0] - 0.5, xs[0] + 0.5)
        self.series_canvas.fig.tight_layout()
        self.series_canvas.draw_idle()

    # ---------------------------------------------------------------- export
    def export_csv(self) -> None:
        if not self._results:
            QMessageBox.information(self, "Nothing to export", "Run Quant first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Quant results CSV",
            str(ROOT / "reports" / "quant_results.csv"),
            "CSV (*.csv);;All files (*)",
        )
        if not path:
            return
        elements = self._elements()
        unit = self._unit
        headers = ["index", "filename", "path"]
        for el in elements:
            suffix = f"_{unit}" if unit else ""
            headers.extend(
                [
                    f"{el}_C{suffix}",
                    f"{el}_std{suffix}",
                    f"{el}_ci95_lo{suffix}",
                    f"{el}_ci95_hi{suffix}",
                    f"{el}_n_lines",
                ]
            )
        with Path(path).open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for r in self._results:
                row: list[str] = [str(r.index), r.filename, r.spectrum_path]
                for el in elements:
                    p = r.prediction_for(el)
                    if p is None:
                        row.extend(["", "", "", "", ""])
                        continue
                    ci = confidence_interval_95(p.concentration, p.std, p.n_lines)
                    row.append(f"{p.concentration:g}")
                    row.append("" if p.std is None else f"{p.std:g}")
                    if ci:
                        row.append(f"{ci[0]:g}")
                        row.append(f"{ci[1]:g}")
                    else:
                        row.extend(["", ""])
                    row.append(str(p.n_lines))
                writer.writerow(row)
        self.statusMessage.emit(f"Wrote {path}")
