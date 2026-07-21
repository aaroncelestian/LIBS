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
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
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
    intensity_zero_crossing,
    peak_integration_view,
    predict_from_fit,
    usable_fits,
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
    snip_iterations: int = 40,
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
        snip_iterations=snip_iterations,
    )
    inten = extract_peak_intensity(
        spectrum,
        fit.wavelength_nm,
        half_width_nm=half_width_nm,
        pad_nm=pad_nm,
        baseline_method=method,
        peak_model=peak_model,
        shift_tol_nm=shift_tol_nm,
        snip_iterations=snip_iterations,
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
    """Top-level Quant tab: multi-select targets, results, peak QC, C series."""

    statusMessage = Signal(str)
    quantRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._results: list[QuantSpectrumResult] = []
        self._cal: CalibrationSet | None = None
        self._unit: str = ""
        self._resolve_spectrum: Callable[[QuantSpectrumResult], Spectrum | None] | None = None
        self._list_spectra: Callable[[], list[Spectrum]] | None = None
        self._updating_selectors = False
        self._focus_spec_path: str | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        self.hint = QLabel(
            "Build curves on Calibrate, check elements and spectra below, then Quant."
        )
        self.hint.setStyleSheet("color: #555;")
        self.hint.setWordWrap(True)
        root.addWidget(self.hint)

        sel = QHBoxLayout()
        sel.setSpacing(8)

        el_col = QVBoxLayout()
        el_hdr = QHBoxLayout()
        el_hdr.addWidget(QLabel("<b>Elements</b>"))
        btn_el_all = QPushButton("All")
        btn_el_all.setFixedWidth(40)
        btn_el_all.clicked.connect(lambda: self._check_all(self.list_el, True))
        btn_el_none = QPushButton("None")
        btn_el_none.setFixedWidth(48)
        btn_el_none.clicked.connect(lambda: self._check_all(self.list_el, False))
        el_hdr.addStretch(1)
        el_hdr.addWidget(btn_el_all)
        el_hdr.addWidget(btn_el_none)
        el_col.addLayout(el_hdr)
        self.list_el = QListWidget()
        self.list_el.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_el.setMinimumWidth(100)
        self.list_el.setMaximumHeight(110)
        self.list_el.itemChanged.connect(self._on_filter_changed)
        self.list_el.itemSelectionChanged.connect(self._on_element_focus_changed)
        el_col.addWidget(self.list_el)
        sel.addLayout(el_col, stretch=1)

        spec_col = QVBoxLayout()
        spec_hdr = QHBoxLayout()
        spec_hdr.addWidget(QLabel("<b>Spectra</b>"))
        btn_sp_all = QPushButton("All")
        btn_sp_all.setFixedWidth(40)
        btn_sp_all.clicked.connect(lambda: self._check_all(self.list_spec, True))
        btn_sp_none = QPushButton("None")
        btn_sp_none.setFixedWidth(48)
        btn_sp_none.clicked.connect(lambda: self._check_all(self.list_spec, False))
        spec_hdr.addStretch(1)
        spec_hdr.addWidget(btn_sp_all)
        spec_hdr.addWidget(btn_sp_none)
        spec_col.addLayout(spec_hdr)
        self.list_spec = QListWidget()
        self.list_spec.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_spec.setMinimumWidth(180)
        self.list_spec.setMaximumHeight(110)
        self.list_spec.itemChanged.connect(self._on_filter_changed)
        self.list_spec.itemSelectionChanged.connect(self._on_spectrum_focus_changed)
        spec_col.addWidget(self.list_spec)
        sel.addLayout(spec_col, stretch=3)

        right = QVBoxLayout()
        right.addWidget(QLabel("<b>Inspect line</b> (QC only)"))
        self.combo_line = QComboBox()
        self.combo_line.setMinimumWidth(140)
        self.combo_line.setToolTip(
            "Which fitted line to show in the calibration-curve and peak-QC plots.\n"
            "Quant concentration is the mean of ALL fitted lines for the element."
        )
        self.combo_line.currentIndexChanged.connect(self._on_line_changed)
        right.addWidget(self.combo_line)
        self.btn_quant = QPushButton("Quant")
        self.btn_quant.setToolTip(
            "Apply every fitted Calibrate CRM curve to checked spectra.\n"
            "Per-element C = mean of all fitted lines (not only the Inspect line)."
        )
        self.btn_quant.clicked.connect(self.quantRequested.emit)
        right.addWidget(self.btn_quant)
        self.btn_export = QPushButton("Export CSV…")
        self.btn_export.clicked.connect(self.export_csv)
        right.addWidget(self.btn_export)
        right.addStretch(1)
        sel.addLayout(right)
        root.addLayout(sel)

        self.summary = QLabel("")
        self.summary.setStyleSheet("color: #333; font-size: 11px;")
        self.summary.setWordWrap(True)
        root.addWidget(self.summary)

        self.line_breakdown = QLabel("")
        self.line_breakdown.setStyleSheet("color: #444; font-size: 11px;")
        self.line_breakdown.setWordWrap(True)
        root.addWidget(self.line_breakdown)

        split = QSplitter(Qt.Orientation.Vertical)
        top = QSplitter(Qt.Orientation.Horizontal)

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

    def set_spectra_provider(self, provider: Callable[[], list[Spectrum]]) -> None:
        """Provide loaded Identify-tab spectra for the Spectra checklist."""
        self._list_spectra = provider

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
        select_elements: list[str] | None = None,
    ) -> None:
        self._results = list(results)
        self._cal = cal
        self._unit = unit or cal.concentration_unit or ""
        self._refresh_all(prefer_elements=select_elements)
        if results:
            n_el = len({p.element for r in results for p in r.predictions})
            n_line = 0
            if results:
                # typical lines per element from first result predictions
                n_line = max(
                    (p.n_lines for r in results for p in r.predictions),
                    default=0,
                )
            self.statusMessage.emit(
                f"Quant results: {len(results)} spectrum(a), {n_el} element(s), "
                f"up to {n_line} line(s)/element (mean)."
            )

    def results(self) -> list[QuantSpectrumResult]:
        return list(self._results)

    def refresh_targets(self, cal: CalibrationSet | None, *, unit: str = "") -> None:
        """Refresh element/spectrum checklists from calibration + loaded spectra."""
        if cal is not None:
            self._cal = cal
            self._unit = unit or cal.concentration_unit or self._unit
        self._fill_selectors(prefer_elements=None, keep_checks=True)
        if not self._results:
            self._fill_table()
            self._update_summary()
            self._update_line_breakdown()
            self._redraw_plots()
        else:
            self._update_line_breakdown()

    def selected_elements(self) -> list[str]:
        """Checked elements (list order)."""
        return [str(x) for x in self._checked_data(self.list_el) if x]

    def selected_spectra_paths(self) -> list[str]:
        """Checked spectrum paths."""
        return [str(x) for x in self._checked_data(self.list_spec) if x]

    def selected_spectra_for_quant(self) -> list[Spectrum]:
        """Resolve checked spectra via the spectra provider."""
        if self._list_spectra is None:
            return []
        wanted = self.selected_spectra_paths()
        if not wanted:
            return []
        by_path = {str(s.meta.path): s for s in self._list_spectra()}
        out: list[Spectrum] = []
        for p in wanted:
            s = by_path.get(p)
            if s is not None:
                out.append(s)
        return out

    # --------------------------------------------------------------- helpers
    def _check_all(self, widget: QListWidget, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        widget.blockSignals(True)
        for i in range(widget.count()):
            widget.item(i).setCheckState(state)
        widget.blockSignals(False)
        if not self._updating_selectors:
            self._on_filter_changed()

    @staticmethod
    def _checked_data(widget: QListWidget) -> list:
        out = []
        for i in range(widget.count()):
            item = widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                out.append(item.data(Qt.ItemDataRole.UserRole))
        return out

    def _available_elements(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for r in self._results:
            for p in r.predictions:
                if p.element not in seen:
                    seen.add(p.element)
                    out.append(p.element)
        if self._cal is not None:
            for el in self._cal.active_elements():
                if el not in seen:
                    seen.add(el)
                    out.append(el)
            if not out:
                for f in usable_fits(self._cal):
                    if f.element not in seen:
                        seen.add(f.element)
                        out.append(f.element)
        return out

    def _checked_elements(self) -> list[str]:
        checked = [str(x) for x in self._checked_data(self.list_el) if x]
        return checked if checked else self._available_elements()

    def _focus_element(self) -> str | None:
        items = self.list_el.selectedItems()
        if items:
            data = items[0].data(Qt.ItemDataRole.UserRole)
            if data:
                return str(data)
        els = self._checked_elements()
        return els[0] if els else None

    def _visible_results(self) -> list[QuantSpectrumResult]:
        if not self._results:
            return []
        checked_paths = {str(x) for x in self._checked_data(self.list_spec) if x}
        if not checked_paths:
            return list(self._results)
        out = [r for r in self._results if r.spectrum_path in checked_paths]
        return out if out else list(self._results)

    def _focus_result(self) -> QuantSpectrumResult | None:
        visible = self._visible_results()
        if not visible:
            return None
        if self._focus_spec_path:
            for r in visible:
                if r.spectrum_path == self._focus_spec_path:
                    return r
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if rows:
            row = rows[0].row()
            if 0 <= row < len(visible):
                return visible[row]
        return visible[0]

    def _selected_wavelength(self) -> float | None:
        data = self.combo_line.currentData()
        return float(data) if data is not None else None

    def _fit_for_selection(self) -> CurveFit | None:
        if self._cal is None:
            return None
        el = self._focus_element()
        wl = self._selected_wavelength()
        if el is None or wl is None:
            return None
        for f in self._cal.fits:
            if f.element == el and abs(float(f.wavelength_nm) - wl) < 1e-6:
                return f
        return None

    # --------------------------------------------------------------- refresh
    def _refresh_all(self, *, prefer_elements: list[str] | None = None) -> None:
        self._fill_selectors(prefer_elements=prefer_elements, keep_checks=False)
        self._fill_table()
        self._update_summary()
        self._update_line_breakdown()
        self._redraw_plots()

    def _fill_selectors(
        self,
        *,
        prefer_elements: list[str] | None,
        keep_checks: bool,
    ) -> None:
        self._updating_selectors = True
        prev_el = set(self._checked_data(self.list_el)) if keep_checks else set()
        prev_spec = set(self._checked_data(self.list_spec)) if keep_checks else set()
        prev_wl = self._selected_wavelength()
        focus_el = self._focus_element()

        if prefer_elements:
            prev_el = set(prefer_elements)
        elif self._results and not keep_checks:
            prev_el = {p.element for r in self._results for p in r.predictions}

        self.list_el.blockSignals(True)
        self.list_el.clear()
        elements = self._available_elements()
        for el in elements:
            item = QListWidgetItem(el)
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEnabled
            )
            item.setData(Qt.ItemDataRole.UserRole, el)
            if not prev_el or el in prev_el:
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setCheckState(Qt.CheckState.Unchecked)
            self.list_el.addItem(item)
        if focus_el:
            for i in range(self.list_el.count()):
                if self.list_el.item(i).data(Qt.ItemDataRole.UserRole) == focus_el:
                    self.list_el.setCurrentRow(i)
                    break
        elif self.list_el.count():
            self.list_el.setCurrentRow(0)
        self.list_el.blockSignals(False)

        # Spectra: prefer loaded list; fall back to result rows
        self.list_spec.blockSignals(True)
        self.list_spec.clear()
        loaded: list[Spectrum] = self._list_spectra() if self._list_spectra else []
        if loaded:
            if self._results and not keep_checks:
                prev_spec = {r.spectrum_path for r in self._results}
            for i, spec in enumerate(loaded):
                path = str(spec.meta.path)
                label = f"{i + 1}: {spec.meta.path.name}"
                item = QListWidgetItem(label)
                item.setFlags(
                    item.flags()
                    | Qt.ItemFlag.ItemIsUserCheckable
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsEnabled
                )
                item.setData(Qt.ItemDataRole.UserRole, path)
                if not prev_spec or path in prev_spec:
                    item.setCheckState(Qt.CheckState.Checked)
                else:
                    item.setCheckState(Qt.CheckState.Unchecked)
                self.list_spec.addItem(item)
        else:
            for r in self._results:
                label = f"{r.index}: {r.filename}"
                item = QListWidgetItem(label)
                item.setFlags(
                    item.flags()
                    | Qt.ItemFlag.ItemIsUserCheckable
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsEnabled
                )
                item.setData(Qt.ItemDataRole.UserRole, r.spectrum_path)
                if not prev_spec or r.spectrum_path in prev_spec:
                    item.setCheckState(Qt.CheckState.Checked)
                else:
                    item.setCheckState(Qt.CheckState.Unchecked)
                self.list_spec.addItem(item)

        if self._focus_spec_path:
            for i in range(self.list_spec.count()):
                if self.list_spec.item(i).data(Qt.ItemDataRole.UserRole) == self._focus_spec_path:
                    self.list_spec.setCurrentRow(i)
                    break
        elif self.list_spec.count():
            self.list_spec.setCurrentRow(0)
            self._focus_spec_path = str(
                self.list_spec.item(0).data(Qt.ItemDataRole.UserRole) or ""
            ) or None
        self.list_spec.blockSignals(False)

        self._fill_line_combo(prefer_wl=prev_wl)
        self._updating_selectors = False

        n_el = self.list_el.count()
        n_sp = self.list_spec.count()
        if self._results:
            unit = f" ({self._unit})" if self._unit else ""
            self.hint.setText(
                f"{len(self._results)} quantified spectrum(a). "
                f"Check elements/spectra to filter; highlight one for peak QC{unit}."
            )
        elif n_el and n_sp:
            self.hint.setText(
                f"{n_el} calibrated element(s), {n_sp} loaded spectrum(a). "
                "Check what to quantify, then Quant."
            )
        elif n_sp and not n_el:
            self.hint.setText(
                "Spectra loaded, but no calibration curves yet. "
                "Build fits on the Calibrate tab first."
            )
        else:
            self.hint.setText(
                "Build curves on Calibrate, load spectra on Identify, "
                "then check elements and spectra here and Quant."
            )

    def _fill_line_combo(self, *, prefer_wl: float | None = None) -> None:
        self.combo_line.blockSignals(True)
        self.combo_line.clear()
        el = self._focus_element()
        if self._cal is not None and el:
            fits = [f for f in self._cal.fits if f.element == el]
            fits.sort(key=lambda f: (-f.r_squared, f.wavelength_nm))
            for f in fits:
                tag = " QC-only" if f.rejected else ""
                self.combo_line.addItem(
                    f"{f.wavelength_nm:.3f} nm  (R²={f.r_squared:.3f}){tag}",
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
        els = self._checked_elements()
        visible = self._visible_results()
        unit = self._unit
        headers = ["#", "File"]
        for el in els:
            if unit:
                headers.extend(
                    [f"{el} ({unit})", f"{el} std", f"{el} 95% CI", f"{el} n_lines"]
                )
            else:
                headers.extend([el, f"{el} std", f"{el} 95% CI", f"{el} n_lines"])
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(visible))

        for i, r in enumerate(visible):
            self.table.setItem(i, 0, QTableWidgetItem(str(r.index)))
            self.table.setItem(i, 1, QTableWidgetItem(r.filename))
            col = 2
            for el in els:
                pred = r.prediction_for(el)
                if pred is None:
                    for _ in range(4):
                        self.table.setItem(i, col, QTableWidgetItem("—"))
                        col += 1
                    continue
                ci = confidence_interval_95(pred.concentration, pred.std, pred.n_lines)
                c_txt = _fmt_num(pred.concentration)
                if pred.below_calibration:
                    c_txt = f"{c_txt}*"
                c_item = QTableWidgetItem(c_txt)
                tip_parts = [
                    f"Mean of {pred.n_lines} fitted line(s).",
                ]
                for wl, c in pred.line_predictions:
                    tip_parts.append(f"  {wl:.3f} nm → C={c:.4g}")
                if pred.below_calibration:
                    tip_parts.append(
                        "Peak area below I→C zero-crossing on ≥1 line "
                        "(raw C < 0 floored to 0)."
                    )
                c_item.setToolTip("\n".join(tip_parts))
                self.table.setItem(i, col, c_item)
                col += 1
                self.table.setItem(i, col, QTableWidgetItem(_fmt_num(pred.std)))
                col += 1
                lo, hi = (ci if ci else (None, None))
                self.table.setItem(i, col, QTableWidgetItem(_fmt_ci(lo, hi)))
                col += 1
                n_item = QTableWidgetItem(str(pred.n_lines))
                n_item.setToolTip("\n".join(tip_parts))
                self.table.setItem(i, col, n_item)
                col += 1

        # Sync table selection to focus spectrum
        self.table.blockSignals(True)
        self.table.clearSelection()
        if self._focus_spec_path:
            for i, r in enumerate(visible):
                if r.spectrum_path == self._focus_spec_path:
                    self.table.selectRow(i)
                    break
        elif visible:
            self.table.selectRow(0)
            self._focus_spec_path = visible[0].spectrum_path
        self.table.blockSignals(False)

    def _update_summary(self) -> None:
        els = self._checked_elements()
        visible = self._visible_results()
        if not els or not visible:
            self.summary.setText("")
            return
        parts: list[str] = []
        unit = f" {self._unit}" if self._unit else ""
        for el in els:
            preds = [r.prediction_for(el) for r in visible]
            preds = [p for p in preds if p is not None]
            if not preds:
                parts.append(f"{el}: —")
                continue
            vals = [float(p.concentration) for p in preds]
            n_lines = preds[0].n_lines
            # Flag if cal has more fitted lines than used (shouldn't) or fewer than enabled
            n_fit = 0
            n_enabled = 0
            if self._cal is not None:
                n_fit = sum(1 for f in usable_fits(self._cal) if f.element == el)
                n_enabled = sum(
                    1
                    for d in self._cal.diagnostic_lines
                    if d.element == el and d.enabled
                )
            mean = float(np.mean(vals))
            line_note = f" · {n_lines} line(s)"
            if n_enabled > n_fit > 0:
                line_note += f" of {n_enabled} enabled"
            if len(vals) >= 2:
                std = float(np.std(vals, ddof=1))
                ci = confidence_interval_95(mean, std, len(vals))
                ci_s = _fmt_ci(*(ci if ci else (None, None)))
                parts.append(
                    f"{el}: mean={mean:.4g}{unit} · std={std:.4g}{unit} · "
                    f"95% CI={ci_s}{line_note}"
                )
            else:
                note = ""
                if preds[0].below_calibration:
                    note = " · below calib→0"
                parts.append(f"{el}: C={mean:.4g}{unit} (n=1){line_note}{note}")
        self.summary.setText(
            f"{len(visible)} spectrum(a) · " + " · ".join(parts)
        )

    def _update_line_breakdown(self) -> None:
        """Per-line C for the focused spectrum + element (shows multi-line Quant)."""
        if not hasattr(self, "line_breakdown"):
            return
        result = self._focus_result()
        el = self._focus_element()
        if result is None or not el:
            self.line_breakdown.setText("")
            return
        pred = result.prediction_for(el)
        unit = f" {self._unit}" if self._unit else ""
        if pred is None or not pred.line_predictions:
            # Explain when cal has lines but prediction missing / single fit
            n_fit = 0
            n_en = 0
            if self._cal is not None:
                n_fit = sum(1 for f in usable_fits(self._cal) if f.element == el)
                n_en = sum(
                    1
                    for d in self._cal.diagnostic_lines
                    if d.element == el and d.enabled
                )
            if n_en or n_fit:
                self.line_breakdown.setText(
                    f"{el} on {result.filename}: no Quant lines "
                    f"(enabled diagnostics={n_en}, fitted curves={n_fit}). "
                    "Rebuild curves on Calibrate for enabled lines."
                )
            else:
                self.line_breakdown.setText("")
            return
        bits = []
        for wl, c in sorted(pred.line_predictions, key=lambda t: t[0]):
            inten = None
            for k, v in result.line_intensities.items():
                if k[0] == el and abs(float(k[1]) - float(wl)) < 1e-6:
                    inten = float(v)
                    break
            if inten is not None:
                bits.append(f"{wl:.3f} nm → C={c:.4g}{unit} (I={inten:.4g})")
            else:
                bits.append(f"{wl:.3f} nm → C={c:.4g}{unit}")
        mean = pred.concentration
        self.line_breakdown.setText(
            f"{el} on {result.filename}: mean C={mean:.4g}{unit} "
            f"from {pred.n_lines} line(s) — " + " · ".join(bits)
        )

    # --------------------------------------------------------------- events
    def _on_filter_changed(self, *_args) -> None:
        if self._updating_selectors:
            return
        self._fill_table()
        self._update_summary()
        self._update_line_breakdown()
        self._redraw_plots()

    def _on_element_focus_changed(self) -> None:
        if self._updating_selectors:
            return
        self._updating_selectors = True
        self._fill_line_combo()
        self._updating_selectors = False
        self._update_line_breakdown()
        self._redraw_plots()

    def _on_spectrum_focus_changed(self) -> None:
        if self._updating_selectors:
            return
        items = self.list_spec.selectedItems()
        if items:
            data = items[0].data(Qt.ItemDataRole.UserRole)
            self._focus_spec_path = str(data) if data else None
        self._fill_table()
        self._update_line_breakdown()
        self._redraw_plots()

    def _on_line_changed(self, _index: int = 0) -> None:
        if self._updating_selectors:
            return
        self._redraw_plots()

    def _on_table_selection(self) -> None:
        if self._updating_selectors:
            return
        visible = self._visible_results()
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows or not visible:
            return
        row = rows[0].row()
        if 0 <= row < len(visible):
            self._focus_spec_path = visible[row].spectrum_path
            self._updating_selectors = True
            for i in range(self.list_spec.count()):
                if self.list_spec.item(i).data(Qt.ItemDataRole.UserRole) == self._focus_spec_path:
                    self.list_spec.setCurrentRow(i)
                    break
            self._updating_selectors = False
            self._update_line_breakdown()
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

        result = self._focus_result()
        el = fit.element
        wl = float(fit.wavelength_nm)
        unknown_i = None
        unknown_c = None
        if result is not None:
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

        if unknown_c is not None and np.isfinite(unknown_c):
            unknown_c = max(0.0, float(unknown_c))

        xs = [x_min, x_max]
        if unknown_i is not None and np.isfinite(unknown_i):
            xs.append(unknown_i)
        pad = 0.05 * max(max(xs) - min(xs), 1e-9)
        x_line = np.linspace(min(xs) - pad, max(xs) + pad, 80)
        y_line = np.maximum(np.polyval(fit.coeffs, x_line), 0.0)
        ax.plot(x_line, y_line, color="#c0392b", lw=1.4, label="I→C fit")

        if unknown_i is not None and unknown_c is not None and np.isfinite(unknown_i):
            sample_name = result.filename if result is not None else "sample"
            below = False
            raw_c = float(np.polyval(fit.coeffs, unknown_i))
            if result is not None:
                pred = result.prediction_for(el)
                if pred is not None and pred.below_calibration:
                    below = True
            elif raw_c < 0:
                below = True
            ax.scatter(
                [unknown_i],
                [unknown_c],
                marker="s",
                s=48,
                c="#c0392b",
                zorder=5,
                label=sample_name,
            )
            mean_note = ""
            if result is not None:
                pred = result.prediction_for(el)
                if pred is not None and pred.n_lines > 1:
                    mean_c = max(0.0, float(pred.concentration))
                    mean_note = f" · mean C={mean_c:.4g}"
            lod_note = " (below calib → 0)" if below else ""
            ax.annotate(
                f"{sample_name}\nC={unknown_c:.4g}{mean_note}{lod_note}",
                (unknown_i, unknown_c),
                textcoords="offset points",
                xytext=(8, -10),
                fontsize=8,
                color="#c0392b",
            )
            i0 = intensity_zero_crossing(fit)
            if i0 is not None and np.isfinite(i0) and i0 > 0:
                ax.axvline(i0, color="#7f8c8d", ls=":", lw=1.0, zorder=2)
                ax.annotate(
                    "C=0",
                    (i0, 0.0),
                    textcoords="offset points",
                    xytext=(4, 8),
                    fontsize=7,
                    color="#7f8c8d",
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
        y_hi = float(np.nanmax(y)) if len(y) else 0.0
        y_hi = max(y_hi, float(np.nanmax(y_line)) if len(y_line) else 0.0)
        if unknown_c is not None and np.isfinite(unknown_c):
            y_hi = max(y_hi, float(unknown_c))
        ax.set_ylim(0.0, y_hi * 1.08 if y_hi > 0 else 1.0)
        self.curve_canvas.fig.tight_layout()
        self.curve_canvas.draw_idle()

    def _draw_peak(self) -> None:
        ax = self.peak_canvas.ax
        ax.clear()
        ax.set_axis_on()
        fit = self._fit_for_selection()
        result = self._focus_result()
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
            snip_iterations=self._cal.snip_iterations,
        )
        self.peak_canvas.fig.tight_layout()
        self.peak_canvas.draw_idle()

    def _draw_series(self) -> None:
        ax = self.series_canvas.ax
        ax.clear()
        ax.set_axis_on()
        els = self._checked_elements()
        visible = self._visible_results()
        unit = self._unit or "C"
        if not els:
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

        colors = ["#1a5276", "#c0392b", "#1e8449", "#8e44ad", "#d35400", "#16a085"]
        any_pts = False
        for i, el in enumerate(els):
            xs: list[int] = []
            ys: list[float] = []
            yerr: list[float] = []
            for r in visible:
                p = r.prediction_for(el)
                if p is None:
                    continue
                xs.append(r.index)
                ys.append(max(0.0, float(p.concentration)))
                yerr.append(float(p.std) if p.std is not None else 0.0)
            if not xs:
                continue
            any_pts = True
            color = colors[i % len(colors)]
            # Clip lower error bars so they do not extend below 0 wt%
            if any(e > 0 for e in yerr):
                yerr_lo = [min(e, y) for y, e in zip(ys, yerr)]
                err = [yerr_lo, yerr]
            else:
                err = None
            ax.errorbar(
                xs,
                ys,
                yerr=err,
                fmt="o-",
                color=color,
                ecolor="#7f8c8d",
                capsize=3,
                lw=1.2,
                ms=5,
                label=el,
            )

        if not any_pts:
            ax.text(
                0.5,
                0.5,
                "No predictions for checked elements",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#666",
            )
            ax.set_axis_off()
            self.series_canvas.fig.tight_layout()
            self.series_canvas.draw_idle()
            return

        ax.set_xlabel("Spectrum #", fontsize=9)
        ax.set_ylabel(unit, fontsize=9)
        title_els = ", ".join(els[:4]) + ("…" if len(els) > 4 else "")
        ax.set_title(f"{title_els} across Quant batch", fontsize=10)
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.28)
        ax.tick_params(labelsize=8)
        ax.set_ylim(bottom=0.0)
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
        elements = self._checked_elements()
        rows_out = self._visible_results()
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
            for r in rows_out:
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
