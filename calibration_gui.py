"""Calibration tab UI for CRM univariate LIBS calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QEvent, QPoint, QSize, QTimer, QUrl, Signal
from PySide6.QtGui import QCursor, QDragEnterEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from calibration import (
    CONCENTRATION_UNIT_CHOICES,
    DEFAULT_SUGGEST_LINES_PER_ELEMENT,
    CalibrationSet,
    CurveFit,
    ElementPrediction,
    add_standard_from_path,
    apply_concentrations,
    acquisition_mismatch_warnings,
    build_fits,
    concentration_level_summary,
    concentration_units_convertible,
    convert_calibration_concentrations,
    ensure_element_columns,
    flag_line_overlaps,
    fit_response_slope,
    load_calibration_set,
    load_concentrations_csv,
    normalize_concentration_unit,
    peak_integration_view,
    predict_concentrations,
    save_calibration_set,
    save_concentrations_csv,
    set_standard_concentrations,
    suggest_diagnostic_lines,
    usable_fits,
)
from identify_elements import ElementHit, LibraryLine, Spectrum
from matplotlib_config import apply_matplotlib_config

ROOT = Path(__file__).resolve().parent


class _LineHoverPreview(QFrame):
    """Frameless popup with a mini peak/baseline plot for one diagnostic line."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            parent,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setStyleSheet(
            "QFrame { background: #fafafa; border: 1px solid #888; border-radius: 4px; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        apply_matplotlib_config()
        self.fig = Figure(figsize=(3.6, 2.4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setFixedSize(360, 240)
        layout.addWidget(self.canvas)
        self._key: tuple[str, float] | None = None

    def show_line(
        self,
        *,
        spectrum: Spectrum,
        label: str,
        element: str,
        wavelength_nm: float,
        half_width_nm: float,
        pad_nm: float,
        method: str,
        peak_model: str,
        shift_tol_nm: float,
        snip_iterations: int,
    ) -> None:
        key = (element, round(float(wavelength_nm), 4))
        if self._key == key and self.isVisible():
            return
        self._key = key
        self.ax.clear()
        view = peak_integration_view(
            spectrum,
            wavelength_nm,
            half_width_nm=half_width_nm,
            pad_nm=pad_nm,
            method=method,
            peak_model=peak_model,
            shift_tol_nm=shift_tol_nm,
            snip_iterations=snip_iterations,
        )
        if view is None or len(view.wavelength_nm) < 2:
            self.ax.text(
                0.5,
                0.5,
                f"No samples near\n{element} {wavelength_nm:.3f} nm",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
                color="#666",
                fontsize=9,
            )
            self.ax.set_axis_off()
        else:
            CalibrationTab._draw_peak_integration(
                self.ax,
                view,
                title_prefix=f"{element} {wavelength_nm:.3f} · {label}",
                compact=True,
            )
        self.fig.tight_layout(pad=0.3)
        self.canvas.draw_idle()
        self.adjustSize()


class CurveCanvas(FigureCanvasQTAgg):
    def __init__(self) -> None:
        apply_matplotlib_config()
        # Compact figure; panels are rebuilt as a 2×N grid on redraw
        self.fig = Figure(figsize=(9.0, 6.2), dpi=100)
        super().__init__(self.fig)
        self.fig.tight_layout()


class CalibrationTab(QWidget):
    """CRM standards → diagnostic lines → I→C curves (Quant from Identify → Quant tab)."""

    statusMessage = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.cal = CalibrationSet()
        self.library: list[LibraryLine] = []
        self._unknown: Spectrum | None = None
        self._identify_hits: list[ElementHit] = []
        self._block_conc = False
        self._line_hover_popup: _LineHoverPreview | None = None
        self._line_hover_row: int = -1
        self._line_hover_timer = QTimer(self)
        self._line_hover_timer.setSingleShot(True)
        self._line_hover_timer.setInterval(280)
        self._line_hover_timer.timeout.connect(self._show_line_hover_preview)
        self._build_ui()
        self._enable_standards_drag_drop()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        self.sub_tabs = QTabWidget()
        root.addWidget(self.sub_tabs)

        # ======================== Data entry ========================
        data_page = QWidget()
        data_l = QHBoxLayout(data_page)
        data_l.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        data_l.addWidget(splitter)

        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)

        # --- Standards ---
        std_box = QGroupBox("Standards (CRM spectra)")
        std_l = QVBoxLayout(std_box)
        btn_row = QHBoxLayout()
        self.btn_add_std = QPushButton("Add spectrum…")
        self.btn_add_std.setToolTip(
            "Add one or more CRM spectra.\n"
            "Replicate shots of the same standard are encouraged — "
            "each file is a separate calibration point."
        )
        self.btn_add_std.clicked.connect(self._add_standards)
        self.btn_remove_std = QPushButton("Remove")
        self.btn_remove_std.setToolTip("Remove highlighted standard(s).")
        self.btn_remove_std.clicked.connect(self._remove_standard)
        self.btn_set_conc = QPushButton("Set C…")
        self.btn_set_conc.setToolTip(
            "Assign the same concentration to highlighted standards\n"
            "(e.g. all six 1500 ppm Pb replicates → Pb = 1500)."
        )
        self.btn_set_conc.clicked.connect(self._set_concentration_selected)
        btn_row.addWidget(self.btn_add_std)
        btn_row.addWidget(self.btn_remove_std)
        btn_row.addWidget(self.btn_set_conc)
        btn_row.addStretch(1)
        std_l.addLayout(btn_row)

        self.std_list = QListWidget()
        self.std_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.std_list.setToolTip(
            "CRM / standard spectra (multi-select with Shift/⌘).\n"
            "Replicates at the same C each become a point on the I→C curve — "
            "that scatter is empirical LIBS error.\n"
            "Drag-and-drop .txt files or a folder here to add standards."
        )
        self.std_list.currentRowChanged.connect(self._on_standard_selected)
        std_l.addWidget(self.std_list)

        self.std_meta = QLabel(
            "Select a standard for instrument details.\n"
            "Or drag-and-drop .txt spectra / a folder onto this panel."
        )
        self.std_meta.setWordWrap(True)
        self.std_meta.setStyleSheet("color: #444; font-size: 11px;")
        std_l.addWidget(self.std_meta)
        self.acq_warn_label = QLabel("")
        self.acq_warn_label.setWordWrap(True)
        self.acq_warn_label.setStyleSheet("color: #a04000; font-size: 11px;")
        self.acq_warn_label.setVisible(False)
        std_l.addWidget(self.acq_warn_label)
        self._std_box = std_box
        left_l.addWidget(std_box, stretch=2)

        # --- Elements ---
        el_box = QGroupBox("Elements (check = quantify)")
        el_l = QVBoxLayout(el_box)
        el_hint = QLabel(
            "CRM CSV may list many elements — check only those to fit, predict, and plot."
        )
        el_hint.setWordWrap(True)
        el_hint.setStyleSheet("color: #555; font-size: 11px;")
        el_l.addWidget(el_hint)
        el_btn = QHBoxLayout()
        self.btn_add_el = QPushButton("Add element…")
        self.btn_add_el.clicked.connect(self._add_element)
        self.btn_remove_el = QPushButton("Remove element")
        self.btn_remove_el.clicked.connect(self._remove_element)
        self.btn_seed_lines = QPushButton("Suggest lines")
        self.btn_seed_lines.setToolTip(
            "Suggest diagnostic lines for checked elements.\n"
            "Prefers good calibrants (Ca II IR, K 766/770, Na D, …),\n"
            "then Identify matches, then strong NIST lines.\n"
            "1 line is enough for a curve; 2–4 let you average / drop bad λ."
        )
        self.btn_seed_lines.clicked.connect(self._suggest_lines)
        self.spin_suggest_n = QSpinBox()
        self.spin_suggest_n.setRange(1, 8)
        self.spin_suggest_n.setValue(DEFAULT_SUGGEST_LINES_PER_ELEMENT)
        self.spin_suggest_n.setToolTip(
            "How many lines to suggest per checked element (default 4).\n"
            "Uncheck bad lines after reviewing hover previews."
        )
        self.btn_check_all = QPushButton("Check all")
        self.btn_check_all.clicked.connect(lambda: self._set_all_quantify(True))
        self.btn_check_none = QPushButton("Check none")
        self.btn_check_none.clicked.connect(lambda: self._set_all_quantify(False))
        el_btn.addWidget(self.btn_add_el)
        el_btn.addWidget(self.btn_remove_el)
        el_btn.addWidget(self.btn_seed_lines)
        el_btn.addWidget(QLabel("N"))
        el_btn.addWidget(self.spin_suggest_n)
        el_l.addLayout(el_btn)
        el_btn2 = QHBoxLayout()
        el_btn2.addWidget(self.btn_check_all)
        el_btn2.addWidget(self.btn_check_none)
        el_btn2.addStretch(1)
        el_l.addLayout(el_btn2)

        self.el_list = QListWidget()
        self.el_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.el_list.itemChanged.connect(self._on_element_check_changed)
        self.el_list.itemSelectionChanged.connect(self._on_element_selection_changed)
        el_l.addWidget(self.el_list)
        left_l.addWidget(el_box, stretch=1)

        # --- Concentrations ---
        conc_box = QGroupBox("Known concentrations")
        conc_l = QVBoxLayout(conc_box)
        conc_btn = QHBoxLayout()
        self.btn_import_csv = QPushButton("Import CSV…")
        self.btn_import_csv.clicked.connect(self._import_csv)
        self.btn_export_csv = QPushButton("Export CSV…")
        self.btn_export_csv.clicked.connect(self._export_csv)
        conc_btn.addWidget(self.btn_import_csv)
        conc_btn.addWidget(self.btn_export_csv)
        conc_btn.addStretch(1)
        conc_btn.addWidget(QLabel("Unit"))
        self.combo_unit = QComboBox()
        self.combo_unit.setEditable(True)
        self.combo_unit.addItems(list(CONCENTRATION_UNIT_CHOICES))
        self.combo_unit.setCurrentText("wt%")
        self.combo_unit.setMinimumWidth(100)
        self.combo_unit.setToolTip(
            "Unit for CRM concentrations and predictions.\n"
            "Switching between wt%, ppm, mg/kg, µg/g, and mass frac\n"
            "converts entered values automatically.\n"
            "at% / oxide wt% are labels only (no conversion)."
        )
        self.combo_unit.currentTextChanged.connect(self._on_unit_changed)
        conc_btn.addWidget(self.combo_unit)
        conc_l.addLayout(conc_btn)

        self.conc_table = QTableWidget(0, 1)
        self.conc_table.setToolTip(
            "Known concentrations in the selected unit.\n"
            "One row per spectrum — replicate shots of the same CRM should share "
            "the same C (use Set C… on the standards list).\n"
            "Drag-and-drop a concentrations CSV here to import."
        )
        self.conc_table.setHorizontalHeaderLabels(["standard_id"])
        self.conc_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.conc_table.horizontalHeader().setStretchLastSection(True)
        self.conc_table.cellChanged.connect(self._on_conc_cell_changed)
        conc_l.addWidget(self.conc_table)
        self.conc_box_hint = QLabel(
            "Enter certificate values in the selected unit "
            "(ppm = mg/kg = µg/g). CSV: standard_id, Element1, …"
        )
        self.conc_box_hint.setStyleSheet("color: #666; font-size: 11px;")
        conc_l.addWidget(self.conc_box_hint)
        left_l.addWidget(conc_box, stretch=2)

        splitter.addWidget(left)

        # --- Right of data entry: lines + fit params ---
        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)

        line_box = QGroupBox("Diagnostic lines")
        line_l = QVBoxLayout(line_box)
        line_hint = QLabel(
            "Hover a row to preview the peak on the QC spectrum. "
            "1 good line is enough; keep 2–4 and uncheck bad λ."
        )
        line_hint.setWordWrap(True)
        line_hint.setStyleSheet("color: #555; font-size: 11px;")
        line_l.addWidget(line_hint)
        self.line_table = QTableWidget(0, 5)
        self.line_table.setHorizontalHeaderLabels(
            ["On", "Element", "λ (nm)", "Species", "Overlap"]
        )
        self.line_table.horizontalHeader().setStretchLastSection(True)
        self.line_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.line_table.setMouseTracking(True)
        self.line_table.itemChanged.connect(self._on_line_item_changed)
        self.line_table.itemSelectionChanged.connect(self._on_line_selection_for_plot)
        self.line_table.cellEntered.connect(self._on_line_cell_entered)
        self.line_table.viewport().installEventFilter(self)
        line_l.addWidget(self.line_table)
        right_l.addWidget(line_box, stretch=3)

        params_box = QGroupBox("Fit parameters")
        params_outer = QHBoxLayout(params_box)
        form = QFormLayout()
        self.spin_half = QDoubleSpinBox()
        self.spin_half.setRange(0.05, 2.0)
        self.spin_half.setSingleStep(0.05)
        self.spin_half.setDecimals(2)
        self.spin_half.setValue(0.15)
        self.spin_half.setSuffix(" nm")
        self.spin_half.setToolTip(
            "Peak half-width around diagnostic λ.\n"
            "Net area = everything between the baseline edge anchors "
            "(half-width + inner pad), not a fitted Voigt profile."
        )
        self.spin_half.valueChanged.connect(self._on_integration_params_changed)
        form.addRow("Peak window ±", self.spin_half)

        self.spin_pad = QDoubleSpinBox()
        self.spin_pad.setRange(0.02, 3.0)
        self.spin_pad.setSingleStep(0.02)
        self.spin_pad.setDecimals(2)
        self.spin_pad.setValue(0.12)
        self.spin_pad.setSuffix(" nm")
        self.spin_pad.setToolTip(
            "Pad outside the peak window. Outer ~40% of each pad = continuum "
            "anchors; the inner pad is included in net area so shoulders are not cut off."
        )
        self.spin_pad.valueChanged.connect(self._on_integration_params_changed)
        form.addRow("Baseline pad", self.spin_pad)

        self.combo_baseline = QComboBox()
        self.combo_baseline.addItem("SNIP (peak-clipping)", "snip")
        self.combo_baseline.addItem("Linear (edge→edge)", "linear")
        self.combo_baseline.addItem("Flat (edge mean)", "flat")
        self.combo_baseline.setToolTip(
            "SNIP: RamanLab iterative peak-clipping (best for crowded LIBS).\n"
            "Linear: continuum tilted between left/right edge strips.\n"
            "Flat: constant level from both edge strips."
        )
        self.combo_baseline.currentIndexChanged.connect(self._on_integration_params_changed)
        form.addRow("Baseline", self.combo_baseline)

        self.spin_snip = QSpinBox()
        self.spin_snip.setRange(5, 100)
        self.spin_snip.setValue(40)
        self.spin_snip.setToolTip(
            "SNIP iterations (RamanLab default 40).\n"
            "Higher follows broader continuum; lower preserves narrower dips."
        )
        self.spin_snip.valueChanged.connect(self._on_integration_params_changed)
        form.addRow("SNIP iters", self.spin_snip)

        self.combo_peak_model = QComboBox()
        self.combo_peak_model.addItem("Gaussian fit", "gaussian")
        self.combo_peak_model.addItem("Voigt fit", "voigt")
        self.combo_peak_model.addItem("Net area", "net_area")
        self.combo_peak_model.setToolTip(
            "How peak intensity is measured for I→C.\n"
            "Gaussian/Voigt: fit the line and allow a local λ shift;\n"
            "area under the fitted profile is the intensity.\n"
            "Net area: trapezoid between edge anchors (no parametric fit)."
        )
        self.combo_peak_model.currentIndexChanged.connect(self._on_integration_params_changed)
        form.addRow("Peak model", self.combo_peak_model)

        self.spin_shift = QDoubleSpinBox()
        self.spin_shift.setRange(0.01, 0.50)
        self.spin_shift.setSingleStep(0.01)
        self.spin_shift.setDecimals(3)
        self.spin_shift.setValue(0.15)
        self.spin_shift.setSuffix(" nm")
        self.spin_shift.setToolTip(
            "Max |fitted − NIST| wavelength shift allowed for Gaussian/Voigt fits."
        )
        self.spin_shift.valueChanged.connect(self._on_integration_params_changed)
        form.addRow("Shift tol", self.spin_shift)

        self.combo_degree = QComboBox()
        self.combo_degree.addItem("Linear", 1)
        self.combo_degree.addItem("Quadratic", 2)
        form.addRow("I→C fit", self.combo_degree)

        self.combo_atm = QComboBox()
        self.combo_atm.addItems(["air", "argon", "unknown"])
        self.combo_atm.currentTextChanged.connect(self._on_atm_changed)
        form.addRow("Atmosphere", self.combo_atm)
        params_outer.addLayout(form)

        fit_btns = QVBoxLayout()
        self.btn_fit = QPushButton("Build calibration curves")
        self.btn_fit.setToolTip(
            "Fit I→C curves, then open the Curves & results tab. "
            "Quant unknowns from Identify → Quant tab."
        )
        self.btn_fit.clicked.connect(self._run_fit)
        self.btn_save = QPushButton("Save session…")
        self.btn_save.clicked.connect(self._save_session)
        self.btn_load = QPushButton("Load session…")
        self.btn_load.clicked.connect(self._load_session)
        self.btn_goto_plot = QPushButton("Go to curves…")
        self.btn_goto_plot.clicked.connect(self._show_plot_tab)
        fit_btns.addWidget(self.btn_fit)
        fit_btns.addWidget(self.btn_save)
        fit_btns.addWidget(self.btn_load)
        fit_btns.addWidget(self.btn_goto_plot)
        fit_btns.addStretch(1)
        params_outer.addLayout(fit_btns)
        right_l.addWidget(params_box)

        self.data_fit_label = QLabel("No fits yet — build curves, then review on Curves & results.")
        self.data_fit_label.setStyleSheet("color: #555;")
        self.data_fit_label.setWordWrap(True)
        right_l.addWidget(self.data_fit_label)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([420, 700])

        self.sub_tabs.addTab(data_page, "Data entry")

        # ======================== Curves & results ========================
        plot_page = QWidget()
        plot_l = QVBoxLayout(plot_page)
        plot_l.setContentsMargins(4, 4, 4, 4)

        plot_hint = QLabel(
            "Top row: peak fits for the selected element’s best lines (by R²). "
            "Bottom row: matching I→C curves. Green = fitted λ, red dotted = NIST. "
            "Uncheck a line below (or on Data entry) to exclude it from Quant, "
            "then rebuild if you re-enable lines. Negative-slope curves are still "
            "plotted (QC-only) but excluded from Quant."
        )
        plot_hint.setStyleSheet("color: #555; font-size: 11px;")
        plot_hint.setWordWrap(True)
        plot_l.addWidget(plot_hint)

        peak_ctrl = QHBoxLayout()
        peak_ctrl.addWidget(QLabel("QC spectrum"))
        self.combo_peak_spec = QComboBox()
        self.combo_peak_spec.setToolTip(
            "Spectrum used for the peak-fit panels.\n"
            "Uses Peak window / baseline / peak model from Fit parameters."
        )
        self.combo_peak_spec.currentIndexChanged.connect(self._redraw_plots)
        peak_ctrl.addWidget(self.combo_peak_spec, stretch=1)
        peak_ctrl.addWidget(QLabel("Show"))
        self.spin_n_panels = QSpinBox()
        self.spin_n_panels.setRange(1, 4)
        self.spin_n_panels.setValue(4)
        self.spin_n_panels.setToolTip("How many top-R² lines to show for the selected element.")
        self.spin_n_panels.valueChanged.connect(self._redraw_plots)
        peak_ctrl.addWidget(self.spin_n_panels)
        peak_ctrl.addWidget(QLabel("lines"))
        plot_l.addLayout(peak_ctrl)

        line_use = QHBoxLayout()
        line_use.addWidget(QLabel("Use lines"))
        self.list_curve_lines = QListWidget()
        self.list_curve_lines.setMaximumHeight(78)
        self.list_curve_lines.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.list_curve_lines.setToolTip(
            "Checked lines stay in Quant. Uncheck a bad line (poor peak / "
            "negative slope) to exclude it immediately."
        )
        self.list_curve_lines.itemChanged.connect(self._on_curve_line_check_changed)
        self.list_curve_lines.itemSelectionChanged.connect(self._on_curve_line_selected)
        line_use.addWidget(self.list_curve_lines, stretch=1)
        self.btn_exclude_line = QPushButton("Exclude selected")
        self.btn_exclude_line.setToolTip(
            "Disable the highlighted line and remove its I→C curve from Quant."
        )
        self.btn_exclude_line.clicked.connect(self._exclude_selected_curve_line)
        line_use.addWidget(self.btn_exclude_line)
        plot_l.addLayout(line_use)

        self.curve_canvas = CurveCanvas()
        self._panel_fits: list[CurveFit] = []
        self._panel_keys: list[tuple[str, float]] = []
        self._selected_panel_fit: CurveFit | None = None
        self.curve_canvas.mpl_connect("button_press_event", self._on_curve_canvas_click)
        self.curve_toolbar = NavigationToolbar2QT(self.curve_canvas, plot_page)
        icon = self.curve_toolbar.iconSize()
        self.curve_toolbar.setIconSize(
            QSize(max(12, icon.width() // 2), max(12, icon.height() // 2))
        )
        plot_l.addWidget(self.curve_toolbar)
        plot_l.addWidget(self.curve_canvas, stretch=1)

        self.fit_label = QLabel("No fits yet.")
        self.fit_label.setStyleSheet("color: #333;")
        self.fit_label.setWordWrap(True)
        plot_l.addWidget(self.fit_label)

        self.plot_page = plot_page
        self.sub_tabs.addTab(plot_page, "Curves & results")
        self._update_unit_labels()

    def _show_plot_tab(self) -> None:
        if hasattr(self, "sub_tabs") and hasattr(self, "plot_page"):
            self.sub_tabs.setCurrentWidget(self.plot_page)
            self._refresh_peak_spectrum_combo()
            self._redraw_plots()

    def _on_line_selection_for_plot(self) -> None:
        """Keep curve in sync; if already on plot tab, redraw immediately."""
        if (
            hasattr(self, "sub_tabs")
            and hasattr(self, "plot_page")
            and self.sub_tabs.currentWidget() is self.plot_page
        ):
            self._redraw_plots()

    def _on_integration_params_changed(self, *_args) -> None:
        self._sync_params_from_ui()
        if (
            hasattr(self, "sub_tabs")
            and hasattr(self, "plot_page")
            and self.sub_tabs.currentWidget() is self.plot_page
        ):
            self._redraw_plots()

    def _refresh_peak_spectrum_combo(self) -> None:
        if not hasattr(self, "combo_peak_spec"):
            return
        prev = self.combo_peak_spec.currentData()
        self.combo_peak_spec.blockSignals(True)
        self.combo_peak_spec.clear()
        for i, s in enumerate(self.cal.standards):
            self.combo_peak_spec.addItem(f"Standard: {s.sample_id}", ("std", i))
        if self._unknown is not None:
            self.combo_peak_spec.addItem(
                f"Identify: {self._unknown.meta.path.name}", ("unknown", None)
            )
        # Restore prior selection when possible
        if prev is not None:
            for i in range(self.combo_peak_spec.count()):
                if self.combo_peak_spec.itemData(i) == prev:
                    self.combo_peak_spec.setCurrentIndex(i)
                    break
        self.combo_peak_spec.blockSignals(False)

    def _peak_qc_spectrum(self) -> tuple[Spectrum | None, str]:
        """Return (spectrum, label) for the Peak QC combo selection."""
        if not hasattr(self, "combo_peak_spec") or self.combo_peak_spec.count() == 0:
            if self.cal.standards:
                s = self.cal.standards[0]
                return s.spectrum, s.sample_id
            return self._unknown, (
                self._unknown.meta.path.name if self._unknown is not None else ""
            )
        data = self.combo_peak_spec.currentData()
        if not data:
            return None, ""
        kind, idx = data
        if kind == "std" and idx is not None and 0 <= int(idx) < len(self.cal.standards):
            s = self.cal.standards[int(idx)]
            return s.spectrum, s.sample_id
        if kind == "unknown" and self._unknown is not None:
            return self._unknown, self._unknown.meta.path.name
        return None, ""

    def _selected_element(self) -> str | None:
        # Prefer highlighted element in the left list
        if hasattr(self, "el_list"):
            items = self.el_list.selectedItems()
            if items:
                return items[0].text().strip() or None
        el, _ = self._selected_diagnostic_wavelength()
        if el:
            return el
        if self.cal.fits:
            return self.cal.fits[0].element
        active = self.cal.active_elements()
        return active[0] if active else None

    def _on_element_selection_changed(self) -> None:
        self._fill_line_table()
        self._on_line_selection_for_plot()

    def _top_fits_for_element(self, element: str | None, n: int = 4) -> list[CurveFit]:
        """Best I→C fits for ``element`` by R² (descending), up to ``n``."""
        if not element:
            return []
        fits = [f for f in self.cal.fits if f.element == element]
        fits.sort(key=lambda f: (-float(f.r_squared), float(f.wavelength_nm)))
        return fits[: max(1, min(int(n), 4))]

    def _panels_for_element(
        self, element: str | None, n: int = 4
    ) -> list[tuple[str, float, CurveFit | None]]:
        """
        Columns for the Curves grid: fitted lines first (by R²), then other
        enabled diagnostic lines (peak QC only). Up to ``n`` columns.
        """
        if not element:
            return []
        n = max(1, min(int(n), 4))
        fits = [f for f in self.cal.fits if f.element == element]
        fits.sort(key=lambda f: (-float(f.r_squared), float(f.wavelength_nm)))
        panels: list[tuple[str, float, CurveFit | None]] = []
        seen: set[float] = set()
        for fit in fits:
            key = round(float(fit.wavelength_nm), 4)
            if key in seen:
                continue
            seen.add(key)
            panels.append((fit.element, float(fit.wavelength_nm), fit))
            if len(panels) >= n:
                return panels

        # Fill remaining slots with enabled diagnostics (even if not yet fitted)
        for d in self.cal.diagnostic_lines:
            if d.element != element or not d.enabled:
                continue
            key = round(float(d.wavelength_nm), 4)
            if key in seen:
                continue
            seen.add(key)
            panels.append((d.element, float(d.wavelength_nm), None))
            if len(panels) >= n:
                break
        return panels

    @staticmethod
    def _same_line(a_el: str, a_wl: float, b_el: str, b_wl: float) -> bool:
        return a_el == b_el and abs(float(a_wl) - float(b_wl)) < 1e-6

    def _refresh_curve_line_list(self, element: str | None) -> None:
        if not hasattr(self, "list_curve_lines"):
            return
        self.list_curve_lines.blockSignals(True)
        self.list_curve_lines.clear()
        if not element:
            self.list_curve_lines.blockSignals(False)
            return
        fits = [f for f in self.cal.fits if f.element == element]
        fits.sort(key=lambda f: (-float(f.r_squared), float(f.wavelength_nm)))
        # Also list disabled diagnostic lines for this element so they can be re-enabled
        fit_keys = {(f.element, round(float(f.wavelength_nm), 6)) for f in fits}
        for fit in fits:
            slope = fit_response_slope(fit)
            if fit.rejected:
                warn = "  ⚠ QC only"
            elif slope < 0:
                warn = "  ⚠ neg. slope"
            else:
                warn = ""
            item = QListWidgetItem(
                f"{fit.wavelength_nm:.3f} nm  R²={fit.r_squared:.3f}{warn}"
            )
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEnabled
            )
            item.setCheckState(Qt.CheckState.Checked)
            item.setData(
                Qt.ItemDataRole.UserRole,
                (fit.element, float(fit.wavelength_nm), True),
            )
            if fit.rejected or slope < 0:
                item.setForeground(Qt.GlobalColor.darkYellow)
                tip = fit.rejected or "Negative I→C slope"
                item.setToolTip(
                    f"{tip} — shown for QC, excluded from Quant. "
                    "Uncheck to drop, or pick another λ."
                )
            self.list_curve_lines.addItem(item)

        for d in self.cal.diagnostic_lines:
            if d.element != element:
                continue
            key = (d.element, round(float(d.wavelength_nm), 6))
            if key in fit_keys:
                continue
            if d.enabled:
                # Enabled but not fitted (skipped) — show unchecked? Still enabled in table
                item = QListWidgetItem(f"{d.wavelength_nm:.3f} nm  (no fit)")
                item.setFlags(
                    item.flags()
                    | Qt.ItemFlag.ItemIsUserCheckable
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsEnabled
                )
                item.setCheckState(Qt.CheckState.Checked)
                item.setData(
                    Qt.ItemDataRole.UserRole,
                    (d.element, float(d.wavelength_nm), False),
                )
                item.setToolTip("Enabled but no usable fit — rebuild or exclude.")
                self.list_curve_lines.addItem(item)
            else:
                item = QListWidgetItem(f"{d.wavelength_nm:.3f} nm  (excluded)")
                item.setFlags(
                    item.flags()
                    | Qt.ItemFlag.ItemIsUserCheckable
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsEnabled
                )
                item.setCheckState(Qt.CheckState.Unchecked)
                item.setData(
                    Qt.ItemDataRole.UserRole,
                    (d.element, float(d.wavelength_nm), False),
                )
                item.setToolTip("Excluded from Quant. Check and rebuild curves to restore.")
                self.list_curve_lines.addItem(item)

        # Restore selection for highlighted panel
        if self._selected_panel_fit is not None:
            sel = self._selected_panel_fit
            for i in range(self.list_curve_lines.count()):
                data = self.list_curve_lines.item(i).data(Qt.ItemDataRole.UserRole)
                if data and self._same_line(data[0], data[1], sel.element, sel.wavelength_nm):
                    self.list_curve_lines.setCurrentRow(i)
                    break
        self.list_curve_lines.blockSignals(False)

    def _set_diagnostic_enabled(self, element: str, wavelength_nm: float, enabled: bool) -> None:
        for d in self.cal.diagnostic_lines:
            if self._same_line(d.element, d.wavelength_nm, element, wavelength_nm):
                d.enabled = enabled
                break

    def _remove_fit(self, element: str, wavelength_nm: float) -> None:
        self.cal.fits = [
            f
            for f in self.cal.fits
            if not self._same_line(f.element, f.wavelength_nm, element, wavelength_nm)
        ]

    def _on_curve_line_check_changed(self, item: QListWidgetItem) -> None:
        data = item.data(Qt.ItemDataRole.UserRole)
        if not data:
            return
        el, wl, _had_fit = data
        enabled = item.checkState() == Qt.CheckState.Checked
        self._set_diagnostic_enabled(str(el), float(wl), enabled)
        if not enabled:
            self._remove_fit(str(el), float(wl))
            self.statusMessage.emit(f"Excluded {el} {float(wl):.3f} nm from Quant")
            self._fill_line_table()
            self._update_fit_summary_from_fits()
            self._redraw_plots()
        else:
            self._fill_line_table()
            self.statusMessage.emit(
                f"Re-enabled {el} {float(wl):.3f} nm — click Build curves to restore fit"
            )

    def _on_curve_line_selected(self) -> None:
        items = self.list_curve_lines.selectedItems()
        if not items:
            return
        data = items[0].data(Qt.ItemDataRole.UserRole)
        if not data:
            return
        el, wl, _ = data
        for f in self.cal.fits:
            if self._same_line(f.element, f.wavelength_nm, str(el), float(wl)):
                self._selected_panel_fit = f
                break

    def _exclude_selected_curve_line(self) -> None:
        items = self.list_curve_lines.selectedItems()
        if not items and self._selected_panel_fit is not None:
            fit = self._selected_panel_fit
            self._set_diagnostic_enabled(fit.element, fit.wavelength_nm, False)
            self._remove_fit(fit.element, fit.wavelength_nm)
            self.statusMessage.emit(
                f"Excluded {fit.element} {fit.wavelength_nm:.3f} nm from Quant"
            )
            self._selected_panel_fit = None
            self._fill_line_table()
            self._update_fit_summary_from_fits()
            self._redraw_plots()
            return
        if not items:
            QMessageBox.information(
                self,
                "Exclude line",
                "Click a curve panel or select a line in the list, then Exclude.",
            )
            return
        item = items[0]
        item.setCheckState(Qt.CheckState.Unchecked)

    def _on_curve_canvas_click(self, event) -> None:
        if event.inaxes is None or not self._panel_keys:
            return
        axes = list(self.curve_canvas.fig.axes)
        try:
            idx = axes.index(event.inaxes)
        except ValueError:
            return
        n = len(self._panel_keys)
        panel = idx % n if idx < 2 * n else None
        if panel is None or panel >= n:
            return
        el, wl = self._panel_keys[panel]
        # Prefer matching CurveFit when present
        self._selected_panel_fit = None
        for f in self.cal.fits:
            if self._same_line(f.element, f.wavelength_nm, el, wl):
                self._selected_panel_fit = f
                break
        if hasattr(self, "list_curve_lines"):
            self.list_curve_lines.blockSignals(True)
            for i in range(self.list_curve_lines.count()):
                data = self.list_curve_lines.item(i).data(Qt.ItemDataRole.UserRole)
                if data and self._same_line(data[0], data[1], el, wl):
                    self.list_curve_lines.setCurrentRow(i)
                    break
            self.list_curve_lines.blockSignals(False)
        self.statusMessage.emit(
            f"Selected {el} {wl:.3f} nm — Exclude selected to drop from Quant"
        )
        self._redraw_plots()

    def _update_fit_summary_from_fits(self) -> None:
        fits = self.cal.fits
        if not fits:
            self._set_fit_summary("No fits.")
            return
        parts = []
        for f in fits:
            tag = " QC-only" if f.rejected else ""
            parts.append(
                f"{f.element} {f.wavelength_nm:.3f} nm: "
                f"R²={f.r_squared:.3f} (n={f.n_points}){tag}"
            )
        summary = " · ".join(parts[:6]) + (" …" if len(parts) > 6 else "")
        self._set_fit_summary(summary)

    def _redraw_plots(self, *_args) -> None:
        if not hasattr(self, "curve_canvas"):
            return
        fig = self.curve_canvas.fig
        fig.clear()
        self._sync_params_from_ui()

        el = self._selected_element()
        n_want = int(self.spin_n_panels.value()) if hasattr(self, "spin_n_panels") else 4
        panels = self._panels_for_element(el, n_want)
        self._panel_fits = [p[2] for p in panels if p[2] is not None]
        self._panel_keys = [(p[0], p[1]) for p in panels]
        self._refresh_curve_line_list(el)
        spec, label = self._peak_qc_spectrum()

        if not panels:
            ax = fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "Build calibration curves, then select an element / line\n"
                "on Data entry to inspect the top fits.\n"
                "Uncheck bad lines in Use lines to exclude them from Quant.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#666",
            )
            ax.set_axis_off()
            fig.tight_layout(pad=0.4)
            self.curve_canvas.draw_idle()
            return

        n = len(panels)
        for i, (pel, pwl, fit) in enumerate(panels):
            ax_peak = fig.add_subplot(2, n, i + 1)
            ax_curve = fig.add_subplot(2, n, n + i + 1)
            self._draw_peak_panel_at(ax_peak, pel, pwl, spec, label)
            if fit is not None:
                self._draw_curve_for_fit(ax_curve, fit)
            else:
                ax_curve.text(
                    0.5,
                    0.5,
                    f"No I→C fit\n{pel} {pwl:.3f} nm\n(rebuild or exclude)",
                    ha="center",
                    va="center",
                    transform=ax_curve.transAxes,
                    color="#666",
                    fontsize=8,
                )
                ax_curve.set_axis_off()
            selected = False
            if self._selected_panel_fit is not None and fit is not None:
                selected = self._same_line(
                    fit.element,
                    fit.wavelength_nm,
                    self._selected_panel_fit.element,
                    self._selected_panel_fit.wavelength_nm,
                )
            elif (
                self._selected_panel_fit is None
                and hasattr(self, "list_curve_lines")
                and self.list_curve_lines.currentRow() == i
            ):
                selected = True
            if selected:
                for spine in ax_peak.spines.values():
                    spine.set_color("#c0392b")
                    spine.set_linewidth(1.6)
                for spine in ax_curve.spines.values():
                    spine.set_color("#c0392b")
                    spine.set_linewidth(1.6)

        n_fit = sum(1 for p in panels if p[2] is not None)
        n_quant = sum(1 for p in panels if p[2] is not None and p[2].usable)
        n_qc = n_fit - n_quant
        if n_qc:
            fit_txt = f"{n_quant} for Quant, {n_qc} QC-only"
        else:
            fit_txt = f"{n_fit} fitted"
        fig.suptitle(
            f"{el} — {n} line(s) ({fit_txt}) · QC: {label or '—'}",
            fontsize=11,
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97), pad=0.35, w_pad=0.6, h_pad=0.8)
        self.curve_canvas.draw_idle()

    def _draw_peak_panel_at(
        self,
        ax,
        element: str,
        wavelength_nm: float,
        spec: Spectrum | None,
        label: str,
    ) -> None:
        """Peak QC for any diagnostic λ (fitted or not)."""
        center = float(wavelength_nm)
        if spec is None:
            ax.text(
                0.5,
                0.5,
                "Add a standard\nto preview",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#666",
                fontsize=8,
            )
            ax.set_title(f"{element} {center:.3f}", fontsize=9)
            return
        view = peak_integration_view(
            spec,
            center,
            half_width_nm=self.cal.half_width_nm,
            pad_nm=self.cal.baseline_pad_nm,
            method=self.cal.baseline_method,
            peak_model=self.cal.peak_model,
            shift_tol_nm=self.cal.shift_tol_nm,
            snip_iterations=self.cal.snip_iterations,
        )
        if view is None or len(view.wavelength_nm) < 2:
            ax.text(
                0.5,
                0.5,
                f"No samples\nnear {center:.3f} nm",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#666",
                fontsize=8,
            )
            ax.set_title(f"{element} {center:.3f}", fontsize=9)
            return
        self._draw_peak_integration(
            ax,
            view,
            title_prefix=f"{element} {center:.3f}",
            compact=True,
        )

    def _draw_peak_panel_for_fit(
        self,
        ax,
        fit: CurveFit,
        spec: Spectrum | None,
        label: str,
    ) -> None:
        self._draw_peak_panel_at(
            ax, fit.element, float(fit.wavelength_nm), spec, label
        )

    def _draw_curve_for_fit(self, ax, fit: CurveFit) -> None:
        x = np.asarray(fit.intensities, dtype=float)
        y = np.asarray(fit.concentrations, dtype=float)
        ax.scatter(x, y, c="#1a5276", s=28, zorder=3, alpha=0.85)
        if len(x):
            x_line = np.linspace(float(np.nanmin(x)) * 0.95, float(np.nanmax(x)) * 1.05, 60)
            y_line = np.maximum(np.polyval(fit.coeffs, x_line), 0.0)
            style = {"color": "#c0392b", "lw": 1.3}
            if fit.rejected:
                style["ls"] = "--"
                style["alpha"] = 0.75
            ax.plot(x_line, y_line, **style)
        n_pts, n_levels = concentration_level_summary(list(fit.concentrations))
        slope = fit_response_slope(fit)
        if fit.rejected:
            slope_note = f"  ⚠ {fit.rejected} — QC only"
        elif slope < 0:
            slope_note = "  ⚠ neg. slope"
        else:
            slope_note = ""
        ax.set_xlabel("Peak area", fontsize=8)
        ax.set_ylabel(self._unit_label(), fontsize=8)
        level_txt = f"{n_levels} C level{'s' if n_levels != 1 else ''}"
        warn = bool(fit.rejected or slope < 0)
        ax.set_title(
            f"I→C  R²={fit.r_squared:.3f}  n={n_pts} ({level_txt}){slope_note}",
            fontsize=8 if fit.rejected else 9,
            color="#a04000" if warn else "#000000",
        )
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)
        y_hi = float(np.nanmax(y)) if len(y) else 0.0
        ax.set_ylim(0.0, y_hi * 1.08 if y_hi > 0 else 1.0)
        if fit.rejected:
            ax.text(
                0.98,
                0.02,
                "excluded from Quant",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7,
                color="#a04000",
                alpha=0.9,
            )

    def _redraw_peak_qc(self) -> None:
        """Backward-compatible alias — redraws the full Curves grid."""
        self._redraw_plots()

    def _selected_diagnostic_wavelength(self) -> tuple[str | None, float | None]:
        """Element + λ from line table selection, else current fit, else first enabled line."""
        fit = self._current_fit()
        if fit is not None:
            return fit.element, float(fit.wavelength_nm)
        rows = self.line_table.selectionModel().selectedRows() if hasattr(self, "line_table") else []
        if rows:
            item = self.line_table.item(rows[0].row(), 0)
            if item is not None:
                key = item.data(Qt.ItemDataRole.UserRole)
                if key:
                    el, wl = key
                    return str(el), float(wl)
        for d in self.cal.diagnostic_lines:
            if d.enabled:
                return d.element, float(d.wavelength_nm)
        return None, None

    @staticmethod
    def _draw_peak_integration(
        ax, view, *, title_prefix: str = "", compact: bool = False
    ) -> None:
        """Render peak QC with baseline, optional fit overlay, tight FOV."""
        center = view.center_nm
        wl = view.wavelength_nm
        y = view.intensity
        base = view.baseline
        i_lo = view.integrate_lo_nm
        i_hi = view.integrate_hi_nm
        core = (wl >= i_lo) & (wl <= i_hi)
        peak_model = getattr(view, "peak_model", "net_area") or "net_area"
        fit_ok = bool(getattr(view, "fit_ok", False)) and peak_model != "net_area"
        fs_title = 8 if compact else 10
        fs_lab = 7 if compact else 9
        lw = 1.0 if compact else 1.35

        ax.axvspan(
            view.edge_lo_nm,
            view.edge_hi_left_nm,
            color="#e67e22",
            alpha=0.22,
            zorder=0,
            label="Edge anchors" if not compact else None,
        )
        ax.axvspan(
            view.edge_lo_right_nm,
            view.edge_hi_nm,
            color="#e67e22",
            alpha=0.22,
            zorder=0,
        )
        ax.plot(wl, y, color="#1a252f", lw=lw, label="Raw" if not compact else None, zorder=3)
        ax.plot(
            wl,
            base,
            color="#d35400",
            lw=lw,
            ls="--",
            label="Baseline" if not compact else None,
            zorder=3,
        )

        if fit_ok and len(view.fit_wavelength_nm) and len(view.fit_intensity):
            ax.plot(
                view.fit_wavelength_nm,
                view.fit_intensity,
                color="#1e8449",
                lw=1.5 if compact else 1.8,
                label=f"{peak_model} fit" if not compact else None,
                zorder=4,
            )
            base_on_fit = np.interp(
                view.fit_wavelength_nm, wl, base, left=base[0], right=base[-1]
            )
            ax.fill_between(
                view.fit_wavelength_nm,
                base_on_fit,
                view.fit_intensity,
                where=(view.fit_intensity >= base_on_fit),
                color="#1e8449",
                alpha=0.22,
                zorder=2,
            )
            ax.axvline(
                view.fitted_center_nm,
                color="#1e8449",
                lw=1.0 if compact else 1.2,
                ls="-",
                label=(
                    None
                    if compact
                    else f"Fitted λ ({view.delta_nm:+.3f} nm)"
                ),
                zorder=5,
            )
        elif np.any(core):
            ax.fill_between(
                wl[core],
                base[core],
                y[core],
                where=(y[core] >= base[core]),
                color="#1a5276",
                alpha=0.38,
                label="Net area" if not compact else None,
                zorder=2,
            )

        ax.axvline(
            center,
            color="#c0392b",
            lw=1.0 if compact else 1.15,
            ls=":",
            label="NIST λ" if not compact else None,
            zorder=4,
        )
        if not compact:
            ax.axvline(i_lo, color="#2980b9", lw=0.9, ls="--", alpha=0.7, zorder=2)
            ax.axvline(i_hi, color="#2980b9", lw=0.9, ls="--", alpha=0.7, zorder=2)

        # Tight FOV: peak body (+ modest margin), always including net-area bounds
        yc = np.asarray(view.corrected, dtype=float)
        ymax_c = float(np.nanmax(yc)) if len(yc) else 0.0
        if ymax_c > 0:
            above = yc >= (0.06 * ymax_c)
            if np.count_nonzero(above) >= 2:
                idx = np.flatnonzero(above)
                p_lo = float(wl[idx[0]])
                p_hi = float(wl[idx[-1]])
            else:
                p_lo, p_hi = i_lo, i_hi
        else:
            p_lo, p_hi = i_lo, i_hi
        x0 = min(p_lo, i_lo, view.edge_hi_left_nm)
        x1 = max(p_hi, i_hi, view.edge_lo_right_nm)
        if fit_ok:
            x0 = min(x0, view.fitted_center_nm)
            x1 = max(x1, view.fitted_center_nm)
        margin = max(0.012, 0.07 * max(x1 - x0, 1e-6))
        x0 -= margin
        x1 += margin
        ax.set_xlim(x0, x1)

        in_view = (wl >= x0) & (wl <= x1)
        if np.any(in_view):
            y_lo = float(min(np.nanmin(y[in_view]), np.nanmin(base[in_view])))
            y_hi = float(max(np.nanmax(y[in_view]), np.nanmax(base[in_view])))
            if fit_ok and len(view.fit_intensity):
                y_hi = max(y_hi, float(np.nanmax(view.fit_intensity)))
            pad_y = 0.10 * max(y_hi - y_lo, 1.0)
            ax.set_ylim(y_lo - 0.5 * pad_y, y_hi + pad_y)

        ax.set_xlabel("λ (nm)" if compact else "Wavelength (nm)", fontsize=fs_lab)
        ax.set_ylabel("counts" if compact else "Intensity (counts)", fontsize=fs_lab)
        ax.tick_params(labelsize=6 if compact else 8)
        prefix = f"{title_prefix} · " if title_prefix else ""
        if fit_ok:
            if compact:
                ax.set_title(
                    f"{prefix}Δλ={view.delta_nm:+.3f}  A={view.area:.3g}",
                    fontsize=fs_title,
                )
            else:
                bl_lbl = view.method or "snip"
                ax.set_title(
                    f"{prefix}{peak_model}  area={view.area:.4g}  "
                    f"Δλ={view.delta_nm:+.3f} nm  FWHM={view.fwhm_nm:.3f} nm  "
                    f"({bl_lbl} baseline)",
                    fontsize=fs_title,
                )
        else:
            note = "fit→net" if peak_model != "net_area" else "net"
            ax.set_title(f"{prefix}{note} A={view.area:.3g}", fontsize=fs_title)
        if not compact:
            ax.legend(loc="upper right", fontsize=7, framealpha=0.92)
        ax.grid(True, alpha=0.28)

    # -------------------------------------------------------------- wiring
    def set_library(self, library: list[LibraryLine]) -> None:
        self.library = library

    def set_identify_context(
        self,
        spectrum: Spectrum | None,
        hits: list[ElementHit],
        atmosphere: str,
    ) -> None:
        """Called by main window when Identify data changes."""
        self._identify_hits = list(hits)
        if spectrum is not None:
            self._unknown = spectrum
            self._refresh_peak_spectrum_combo()
        if atmosphere in ("air", "argon", "unknown"):
            self.combo_atm.blockSignals(True)
            self.combo_atm.setCurrentText(atmosphere)
            self.combo_atm.blockSignals(False)
            self.cal.atmosphere = atmosphere

    def has_fits(self) -> bool:
        return bool(usable_fits(self.cal))

    def predict_for_spectrum(self, spectrum: Spectrum) -> list[ElementPrediction]:
        """Apply current CRM fits to one unknown spectrum."""
        self._sync_params_from_ui()
        if not usable_fits(self.cal):
            raise ValueError("No calibration curves — build fits on the Calibrate tab first.")
        return predict_concentrations(self.cal, spectrum)

    def active_quant_elements(self) -> list[str]:
        return list(self.cal.active_elements())

    def concentration_unit(self) -> str:
        self._sync_params_from_ui()
        return self.cal.concentration_unit or ""

    def _sync_params_from_ui(self) -> None:
        self.cal.half_width_nm = float(self.spin_half.value())
        self.cal.baseline_pad_nm = float(self.spin_pad.value())
        method = self.combo_baseline.currentData()
        self.cal.baseline_method = str(method or "snip")
        self.cal.snip_iterations = int(self.spin_snip.value())
        peak_model = self.combo_peak_model.currentData()
        self.cal.peak_model = str(peak_model or "gaussian")
        self.cal.shift_tol_nm = float(self.spin_shift.value())
        self.cal.fit_degree = int(self.combo_degree.currentData())
        self.cal.atmosphere = self.combo_atm.currentText()
        self.cal.concentration_unit = self.combo_unit.currentText().strip() or "wt%"
        if hasattr(self, "spin_shift"):
            self.spin_shift.setEnabled(self.cal.peak_model != "net_area")
        if hasattr(self, "spin_snip"):
            self.spin_snip.setEnabled(self.cal.baseline_method == "snip")

    def _on_atm_changed(self, atm: str) -> None:
        self.cal.atmosphere = atm
        for s in self.cal.standards:
            s.atmosphere = atm

    def _on_unit_changed(self, unit: str) -> None:
        new_unit = normalize_concentration_unit(unit)
        old_unit = normalize_concentration_unit(
            self.cal.concentration_unit or "wt%"
        )
        if new_unit.lower() == old_unit.lower():
            self.cal.concentration_unit = new_unit
            self._update_unit_labels()
            return

        converted = 0
        if concentration_units_convertible(old_unit, new_unit):
            converted = convert_calibration_concentrations(
                self.cal, old_unit, new_unit
            )
            # Existing fits were in the old unit — clear so user rebuilds
            if converted and self.cal.fits:
                self.cal.fits = []
                if hasattr(self, "fit_label"):
                    self.fit_label.setText(
                        "Unit changed — rebuild calibration curves."
                    )
                if hasattr(self, "data_fit_label"):
                    self.data_fit_label.setText(
                        "Unit changed — rebuild calibration curves."
                    )
        elif any(
            v is not None
            for s in self.cal.standards
            for v in s.concentrations.values()
        ):
            QMessageBox.information(
                self,
                "Unit label only",
                f"Cannot auto-convert between {old_unit} and {new_unit}.\n"
                "Entered numbers are left unchanged — use wt%, ppm, mg/kg, "
                "µg/g, or mass frac for automatic conversion.",
            )

        self.cal.concentration_unit = new_unit
        self._update_unit_labels()
        self._refresh_conc_table()
        if converted:
            self.statusMessage.emit(
                f"Converted {converted} concentration value(s) "
                f"{old_unit} → {new_unit}"
            )
        else:
            self.statusMessage.emit(f"Concentration unit: {new_unit}")

    def _unit_label(self) -> str:
        u = (self.cal.concentration_unit or self.combo_unit.currentText() or "wt%").strip()
        return normalize_concentration_unit(u) or "wt%"

    def _update_unit_labels(self) -> None:
        u = self._unit_label()
        if hasattr(self, "conc_box_hint"):
            if concentration_units_convertible(u, "ppm"):
                self.conc_box_hint.setText(
                    f"Values in {u} (ppm = mg/kg = µg/g; convertible to wt%). "
                    "CSV: standard_id, Element1, … (blank = skip)"
                )
            else:
                self.conc_box_hint.setText(
                    f"Values in {u} (label only — no auto-conversion). "
                    "CSV: standard_id, Element1, … (blank = skip)"
                )
    # ----------------------------------------------------------- standards
    def _add_standards(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add CRM / standard spectra",
            str(ROOT / "docs"),
            "Spectrum text (*.txt);;All files (*)",
        )
        if paths:
            self._add_standard_paths([Path(p) for p in paths])

    def _add_standard_paths(self, paths: list[Path]) -> None:
        if not paths:
            return
        before = len(self.cal.standards)
        added = 0
        for p in paths:
            try:
                add_standard_from_path(
                    self.cal,
                    Path(p),
                    atmosphere=self.combo_atm.currentText(),
                )
                added += 1
            except Exception as exc:
                QMessageBox.warning(self, "Load failed", f"{p}\n{exc}")
        if not added:
            return
        ensure_element_columns(self.cal, self.cal.elements)
        self._refresh_standards_list()
        self._refresh_conc_table()
        self._update_acquisition_warnings(popup=True)
        new_idxs = list(range(before, len(self.cal.standards)))
        self.statusMessage.emit(f"Standards: {len(self.cal.standards)} (+{added})")
        if len(new_idxs) >= 1:
            # Highlight newly added rows
            self.std_list.clearSelection()
            for i in new_idxs:
                item = self.std_list.item(i)
                if item is not None:
                    item.setSelected(True)
            if self.cal.elements:
                reply = QMessageBox.question(
                    self,
                    "Assign concentration?",
                    f"Assign the same concentration to the {len(new_idxs)} "
                    f"newly added spectrum{'a' if len(new_idxs) != 1 else ''}?\n\n"
                    "Use this for replicate shots of one CRM "
                    "(e.g. six files all at 1500 ppm Pb).",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes if len(new_idxs) > 1 else QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self._set_concentration_for_indices(new_idxs)

    def _remove_standard(self) -> None:
        rows = sorted(
            {self.std_list.row(it) for it in self.std_list.selectedItems()},
            reverse=True,
        )
        if not rows:
            row = self.std_list.currentRow()
            if row >= 0:
                rows = [row]
        if not rows:
            return
        for row in rows:
            if 0 <= row < len(self.cal.standards):
                del self.cal.standards[row]
        self._refresh_standards_list()
        self._refresh_conc_table()
        self._update_acquisition_warnings(popup=False)

    def _set_concentration_selected(self) -> None:
        rows = sorted({self.std_list.row(it) for it in self.std_list.selectedItems()})
        if not rows:
            QMessageBox.information(
                self,
                "Set concentration",
                "Highlight one or more standards in the list, then Set C…\n"
                "(Shift/⌘-click to multi-select replicates.)",
            )
            return
        self._set_concentration_for_indices(rows)

    def _set_concentration_for_indices(self, indices: list[int]) -> None:
        if not indices:
            return
        if not self.cal.elements:
            # Offer to add an element first
            el, ok = QInputDialog.getText(
                self,
                "Element",
                "Element symbol (e.g. Pb):",
            )
            if not ok or not el.strip():
                return
            el = el.strip()
            # Title-case common symbols
            if len(el) <= 2:
                el = el[0].upper() + (el[1:].lower() if len(el) > 1 else "")
            if el not in self.cal.elements:
                self.cal.elements.append(el)
                ensure_element_columns(self.cal, self.cal.elements)
                self._refresh_element_list()
        else:
            el, ok = QInputDialog.getItem(
                self,
                "Element",
                f"Set concentration for {len(indices)} standard(s):",
                self.cal.elements,
                0,
                False,
            )
            if not ok or not el:
                return
        unit = self._unit_label()
        value, ok = QInputDialog.getDouble(
            self,
            "Concentration",
            f"{el} concentration ({unit}) for {len(indices)} spectrum"
            f"{'a' if len(indices) != 1 else ''}:",
            1500.0 if unit.lower() in ("ppm", "mg/kg", "µg/g", "ug/g") else 0.15,
            -1e12,
            1e12,
            6,
        )
        if not ok:
            return
        n = set_standard_concentrations(
            self.cal.standards, el, float(value), indices=indices
        )
        self._refresh_conc_table()
        self.statusMessage.emit(
            f"Set {el}={value:g} {unit} on {n} standard"
            f"{'s' if n != 1 else ''} (replicates share C; each shot is a point)."
        )

    def _refresh_standards_list(self) -> None:
        self.std_list.clear()
        for s in self.cal.standards:
            cfg = s.spectrum.meta.cfg_path.name if s.spectrum.meta.cfg_path else "no cfg"
            self.std_list.addItem(f"{s.sample_id}  ({cfg})")
        if self.cal.standards:
            self.std_list.setCurrentRow(0)
        self._update_acquisition_warnings(popup=False)
        self._refresh_peak_spectrum_combo()

    def _update_acquisition_warnings(self, *, popup: bool = False) -> list[str]:
        """Surface mismatched laser / gate / integration settings across CRMs."""
        warns = acquisition_mismatch_warnings(self.cal.standards)
        if hasattr(self, "acq_warn_label"):
            if warns:
                self.acq_warn_label.setText(
                    "⚠ Acquisition mismatch — "
                    + "; ".join(warns[:2])
                    + ("…" if len(warns) > 2 else "")
                )
                self.acq_warn_label.setVisible(True)
                self.acq_warn_label.setToolTip("\n".join(warns))
            else:
                self.acq_warn_label.setText("")
                self.acq_warn_label.setToolTip("")
                self.acq_warn_label.setVisible(False)
        if popup and warns:
            QMessageBox.warning(
                self,
                "Acquisition settings differ",
                "CRM config files do not share identical acquisition parameters.\n"
                "Calibration is more reliable when laser energy, QS delay, gate, "
                "integration delay, and integration time match across standards.\n\n"
                + "\n".join(f"• {w}" for w in warns),
            )
        return warns

    def _on_standard_selected(self, row: int) -> None:
        if row < 0 or row >= len(self.cal.standards):
            self.std_meta.setText(
                "Select a standard for instrument details.\n"
                "Or drag-and-drop .txt spectra / a folder onto this panel."
            )
            return
        s = self.cal.standards[row]
        m = s.spectrum.meta
        parts = [f"<b>{s.sample_id}</b>", f"File: {m.path.name}"]

        if m.cfg_path:
            parts.append(f"Config: {m.cfg_path.name}")
        if m.laser_energy_mJ is not None:
            parts.append(f"Laser: {m.laser_energy_mJ:g} mJ")
        if m.qs_delay_us is not None:
            parts.append(f"QS delay: {m.qs_delay_us:g} µs")
        if m.integration_time_us is not None:
            delay = (
                f", delay {m.integration_delay_us:g} µs"
                if m.integration_delay_us is not None
                else ""
            )
            parts.append(f"Gate: {m.integration_time_us:g} µs{delay}")
        if m.n_accumulations is not None:
            parts.append(f"Accumulations: {m.n_accumulations}")
        parts.append(f"Atmosphere: {s.atmosphere}")
        self.std_meta.setText("<br>".join(parts))

    # ------------------------------------------------------------ elements
    def _add_element(self) -> None:
        text, ok = QInputDialog.getText(self, "Add element", "Element symbol (e.g. Fe):")
        if not ok:
            return
        el = text.strip()
        if not el:
            return
        # Title-case single letter / two-letter symbols
        if len(el) == 1:
            el = el.upper()
        elif len(el) >= 2:
            el = el[0].upper() + el[1:].lower()
        if el in self.cal.elements:
            return
        self.cal.elements.append(el)
        ensure_element_columns(self.cal, self.cal.elements)
        self._refresh_element_list()
        self._refresh_conc_table()

    def _remove_element(self) -> None:
        items = self.el_list.selectedItems()
        if not items:
            return
        remove = {it.text() for it in items}
        self.cal.elements = [e for e in self.cal.elements if e not in remove]
        self.cal.diagnostic_lines = [
            d for d in self.cal.diagnostic_lines if d.element not in remove
        ]
        ensure_element_columns(self.cal, self.cal.elements)
        self._refresh_element_list()
        self._refresh_conc_table()
        self._fill_line_table()

    def _refresh_element_list(self) -> None:
        prev_sel = {it.text() for it in self.el_list.selectedItems()}
        active = set(self.cal.active_elements())
        self.el_list.blockSignals(True)
        self.el_list.clear()
        for el in self.cal.elements:
            item = QListWidgetItem(el)
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEnabled
            )
            item.setCheckState(
                Qt.CheckState.Checked if el in active else Qt.CheckState.Unchecked
            )
            self.el_list.addItem(item)
            if el in prev_sel:
                item.setSelected(True)
        if not prev_sel and self.cal.elements:
            self.el_list.setCurrentRow(0)
        self.el_list.blockSignals(False)

    def _on_element_check_changed(self, item: QListWidgetItem) -> None:
        self._sync_quantify_from_list()
        self._fill_line_table()

    def _sync_quantify_from_list(self) -> None:
        checked: list[str] = []
        for i in range(self.el_list.count()):
            it = self.el_list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                checked.append(it.text())
        self.cal.quantify_elements = checked

    def _set_all_quantify(self, checked: bool) -> None:
        self.el_list.blockSignals(True)
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(self.el_list.count()):
            self.el_list.item(i).setCheckState(state)
        self.el_list.blockSignals(False)
        self._sync_quantify_from_list()
        self._fill_line_table()

    # ------------------------------------------------------ concentrations
    def _refresh_conc_table(self) -> None:
        self._block_conc = True
        els = self.cal.elements
        self.conc_table.setColumnCount(1 + len(els))
        self.conc_table.setHorizontalHeaderLabels(["standard_id", *els])
        self.conc_table.setRowCount(len(self.cal.standards))
        for i, s in enumerate(self.cal.standards):
            id_item = QTableWidgetItem(s.sample_id)
            self.conc_table.setItem(i, 0, id_item)
            for j, el in enumerate(els):
                v = s.concentrations.get(el)
                text = "" if v is None else f"{v:g}"
                self.conc_table.setItem(i, j + 1, QTableWidgetItem(text))
        self._block_conc = False

    def _on_conc_cell_changed(self, row: int, col: int) -> None:
        if self._block_conc:
            return
        if row < 0 or row >= len(self.cal.standards):
            return
        s = self.cal.standards[row]
        item = self.conc_table.item(row, col)
        if item is None:
            return
        text = item.text().strip()
        if col == 0:
            s.sample_id = text or s.sample_id
            self._refresh_standards_list()
            return
        el = self.cal.elements[col - 1]
        if text == "":
            s.concentrations[el] = None
        else:
            try:
                s.concentrations[el] = float(text)
            except ValueError:
                QMessageBox.warning(self, "Invalid number", f"Not a number: {text}")
                self._refresh_conc_table()

    def _import_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import concentrations CSV",
            str(ROOT / "docs"),
            "CSV (*.csv);;All files (*)",
        )
        if path:
            self._import_concentrations_path(Path(path))

    def _import_concentrations_path(self, path: Path) -> None:
        try:
            table = load_concentrations_csv(path)
        except Exception as exc:
            QMessageBox.critical(self, "CSV error", str(exc))
            return
        # Discover elements from CSV
        els: list[str] = []
        for concs in table.values():
            for el in concs:
                if el not in els:
                    els.append(el)
        for el in els:
            if el not in self.cal.elements:
                self.cal.elements.append(el)
        ensure_element_columns(self.cal, self.cal.elements)
        unmatched = apply_concentrations(self.cal.standards, table, match_by="sample_id")
        # Also try filename stems
        if unmatched:
            unmatched = apply_concentrations(
                self.cal.standards, {k: table[k] for k in unmatched}, match_by="stem"
            )
        self._refresh_element_list()
        self._refresh_conc_table()
        msg = f"Imported concentrations from {path.name}"
        if unmatched:
            msg += f"\nUnmatched rows: {', '.join(unmatched)}"
            QMessageBox.warning(self, "Import", msg)
        else:
            self.statusMessage.emit(msg)

    def _export_csv(self) -> None:
        if not self.cal.standards:
            QMessageBox.information(self, "Export", "No standards loaded.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export concentrations CSV",
            str(ROOT / "docs" / "concentrations.csv"),
            "CSV (*.csv)",
        )
        if not path:
            return
        save_concentrations_csv(Path(path), self.cal.standards, self.cal.elements)
        self.statusMessage.emit(f"Wrote {path}")

    # --------------------------------------------------------------- lines
    def _suggest_lines(self) -> None:
        self._sync_quantify_from_list()
        active = self.cal.active_elements()
        if not active:
            QMessageBox.information(
                self,
                "Lines",
                "Check at least one element to quantify, then suggest lines.",
            )
            return
        if not self.library:
            QMessageBox.warning(self, "Lines", "NIST library not loaded.")
            return

        n_want = (
            int(self.spin_suggest_n.value())
            if hasattr(self, "spin_suggest_n")
            else DEFAULT_SUGGEST_LINES_PER_ELEMENT
        )
        wl_min, wl_max = self._wl_range()

        # Prefer Identify matches when available (still ranked after preferred λ)
        match_map: dict[str, list[tuple[float, str]]] = {}
        for hit in self._identify_hits:
            if hit.element not in active:
                continue
            pairs = [
                (m.line.wavelength_nm, m.line.species)
                for m in sorted(
                    hit.matches,
                    key=lambda m: m.peak.prominence,
                    reverse=True,
                )
            ]
            match_map[hit.element] = pairs

        seeded = suggest_diagnostic_lines(
            self.library,
            active,
            wl_min=wl_min,
            wl_max=wl_max,
            max_per_element=n_want,
            overlap_tol_nm=self.cal.overlap_tol_nm,
            identify_matches=match_map or None,
        )
        keep_other = [
            d for d in self.cal.diagnostic_lines if d.element not in set(active)
        ]
        self.cal.diagnostic_lines = keep_other + seeded
        self._fill_line_table()
        self.statusMessage.emit(
            f"Suggested ≤{n_want} line(s)/element for {len(active)} element(s) "
            f"(preferred calibrants first — hover rows to preview peaks)."
        )

    def _wl_range(self) -> tuple[float, float]:
        wls: list[float] = []
        for s in self.cal.standards:
            wls.append(float(s.spectrum.wavelength_nm.min()))
            wls.append(float(s.spectrum.wavelength_nm.max()))
        if self._unknown is not None:
            wls.append(float(self._unknown.wavelength_nm.min()))
            wls.append(float(self._unknown.wavelength_nm.max()))
        if not wls:
            return 180.0, 1022.0
        return min(wls), max(wls)

    def _on_line_cell_entered(self, row: int, _column: int) -> None:
        self._line_hover_row = row
        self._line_hover_timer.start()

    def _hide_line_hover_preview(self) -> None:
        self._line_hover_timer.stop()
        if self._line_hover_popup is not None:
            self._line_hover_popup.hide()
            self._line_hover_popup._key = None

    def _show_line_hover_preview(self) -> None:
        row = self._line_hover_row
        if row < 0 or row >= self.line_table.rowCount():
            return
        item = self.line_table.item(row, 0)
        if item is None:
            return
        key = item.data(Qt.ItemDataRole.UserRole)
        if not key:
            return
        el, wl = key
        spec, label = self._peak_qc_spectrum()
        if spec is None:
            return
        self._sync_params_from_ui()
        if self._line_hover_popup is None:
            self._line_hover_popup = _LineHoverPreview(self)
        self._line_hover_popup.show_line(
            spectrum=spec,
            label=label or "QC",
            element=str(el),
            wavelength_nm=float(wl),
            half_width_nm=self.cal.half_width_nm,
            pad_nm=self.cal.baseline_pad_nm,
            method=self.cal.baseline_method,
            peak_model=self.cal.peak_model,
            shift_tol_nm=self.cal.shift_tol_nm,
            snip_iterations=self.cal.snip_iterations,
        )
        # Place near cursor, keep on screen
        pos = QCursor.pos() + QPoint(18, 14)
        self._line_hover_popup.move(pos)
        self._line_hover_popup.show()

    def _fill_line_table(self) -> None:
        self.line_table.blockSignals(True)
        wanted = set(self._selected_elements_for_lines())
        rows = [d for d in self.cal.diagnostic_lines if d.element in wanted]
        self.line_table.setRowCount(len(rows))
        for i, d in enumerate(rows):
            chk = QTableWidgetItem()
            chk.setFlags(
                Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
            )
            chk.setCheckState(
                Qt.CheckState.Checked if d.enabled else Qt.CheckState.Unchecked
            )
            chk.setData(Qt.ItemDataRole.UserRole, (d.element, d.wavelength_nm))
            self.line_table.setItem(i, 0, chk)
            self.line_table.setItem(i, 1, QTableWidgetItem(d.element))
            self.line_table.setItem(i, 2, QTableWidgetItem(f"{d.wavelength_nm:.3f}"))
            self.line_table.setItem(i, 3, QTableWidgetItem(d.species))
            warn = d.overlap_warning or ""
            warn_item = QTableWidgetItem(warn)
            if warn:
                warn_item.setToolTip(warn)
                warn_item.setForeground(Qt.GlobalColor.darkYellow)
            self.line_table.setItem(i, 4, warn_item)
            for col in (1, 2, 3, 4):
                it = self.line_table.item(i, col)
                if it:
                    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.line_table.blockSignals(False)
        self.line_table.resizeColumnsToContents()

    def _on_line_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != 0:
            return
        key = item.data(Qt.ItemDataRole.UserRole)
        if not key:
            return
        el, wl = key
        enabled = item.checkState() == Qt.CheckState.Checked
        for d in self.cal.diagnostic_lines:
            if d.element == el and abs(d.wavelength_nm - wl) < 1e-6:
                d.enabled = enabled
                break
        if not enabled:
            self._remove_fit(str(el), float(wl))
            self._update_fit_summary_from_fits()
            if (
                hasattr(self, "sub_tabs")
                and hasattr(self, "plot_page")
                and self.sub_tabs.currentWidget() is self.plot_page
            ):
                self._redraw_plots()
        elif (
            hasattr(self, "sub_tabs")
            and hasattr(self, "plot_page")
            and self.sub_tabs.currentWidget() is self.plot_page
        ):
            self._refresh_curve_line_list(self._selected_element())

    # --------------------------------------------------------- fit / plot
    def _selected_elements_for_lines(self) -> list[str]:
        """Lines table follows row selection; fall back to checked quantify set."""
        items = self.el_list.selectedItems()
        if items:
            return [it.text() for it in items]
        return self.cal.active_elements()

    def _set_fit_summary(self, text: str) -> None:
        if hasattr(self, "fit_label"):
            self.fit_label.setText(text)
        if hasattr(self, "data_fit_label"):
            self.data_fit_label.setText(text)

    def _run_fit(self) -> None:
        self._sync_quantify_from_list()
        if not self.cal.active_elements():
            QMessageBox.information(
                self,
                "Calibration",
                "Check at least one element to quantify.",
            )
            return
        if len(self.cal.standards) < 2:
            QMessageBox.information(
                self, "Calibration", "Add at least two standards with concentrations."
            )
            return
        if not any(
            d.enabled and d.element in set(self.cal.active_elements())
            for d in self.cal.diagnostic_lines
        ):
            QMessageBox.information(
                self,
                "Calibration",
                "Enable at least one diagnostic line for a checked element (Suggest lines).",
            )
            return
        self._sync_params_from_ui()
        acq_warns = self._update_acquisition_warnings(popup=False)
        if acq_warns:
            reply = QMessageBox.warning(
                self,
                "Acquisition settings differ",
                "CRM .cfg files do not have identical laser energy, QS delay, "
                "gate width, and/or integration delay settings.\n\n"
                + "\n".join(f"• {w}" for w in acq_warns)
                + "\n\nBuild curves anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        # Refresh overlap flags
        self.cal.diagnostic_lines = flag_line_overlaps(
            self.cal.diagnostic_lines,
            self.library,
            tol_nm=self.cal.overlap_tol_nm,
        )
        skipped: list[str] = []
        try:
            fits = build_fits(self.cal, skipped=skipped)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Calibration failed",
                f"Unexpected error while fitting:\n{exc}",
            )
            self._set_fit_summary("Fit error.")
            return
        self._fill_line_table()
        if not fits:
            detail = "\n".join(f"• {s}" for s in skipped[:12])
            more = f"\n… and {len(skipped) - 12} more" if len(skipped) > 12 else ""
            QMessageBox.warning(
                self,
                "Calibration",
                "No curves fitted.\n\n"
                "Common causes: missing concentrations, zero/identical peak areas "
                "across standards, or diagnostic λ outside the spectrum.\n\n"
                + (detail + more if detail else "Enable lines and check CRM values."),
            )
            self._set_fit_summary("No fits.")
            return
        parts = []
        for f in fits:
            tag = " QC-only" if f.rejected else ""
            parts.append(
                f"{f.element} {f.wavelength_nm:.3f} nm: "
                f"R²={f.r_squared:.3f} (n={f.n_points}){tag}"
            )
        summary = " · ".join(parts[:6]) + (" …" if len(parts) > 6 else "")
        self._set_fit_summary(summary)
        n_ok = sum(1 for f in fits if f.usable)
        n_rej = sum(1 for f in fits if f.rejected)
        msg = (
            f"Built {len(fits)} curve(s) for "
            f"{len(self.cal.active_elements())} checked element(s)"
        )
        if n_rej:
            msg += f" · {n_ok} for Quant, {n_rej} QC-only (neg. slope)"
        if skipped:
            msg += f" · failed {len(skipped)} line(s)"
            QMessageBox.information(
                self,
                "Some lines failed",
                msg
                + ":\n\n"
                + "\n".join(f"• {s}" for s in skipped[:10])
                + (f"\n… and {len(skipped) - 10} more" if len(skipped) > 10 else ""),
            )
        elif n_rej:
            rej_lines = [
                f"• {f.element} {f.wavelength_nm:.3f} nm: {f.rejected} — "
                "curve shown for QC, excluded from Quant"
                for f in fits
                if f.rejected
            ]
            QMessageBox.information(
                self,
                "Some curves QC-only",
                msg
                + ":\n\n"
                + "\n".join(rej_lines[:10])
                + (
                    f"\n… and {len(rej_lines) - 10} more"
                    if len(rej_lines) > 10
                    else ""
                )
                + "\n\nCheck CRM peak areas on those λ (interference / wrong line), "
                "or uncheck them in Use lines.",
            )
        self.statusMessage.emit(msg)
        # Select first fit line in table if possible
        if self.line_table.rowCount() > 0:
            self.line_table.selectRow(0)
        self._show_plot_tab()

    def _current_fit(self) -> CurveFit | None:
        rows = self.line_table.selectionModel().selectedRows() if self.line_table.selectionModel() else []
        if rows:
            item = self.line_table.item(rows[0].row(), 0)
            if item is not None:
                key = item.data(Qt.ItemDataRole.UserRole)
                if key:
                    el, wl = key
                    for f in self.cal.fits:
                        if f.element == el and abs(f.wavelength_nm - wl) < 1e-6:
                            return f
        return self.cal.fits[0] if self.cal.fits else None

    # ---------------------------------------------------------- session
    def _save_session(self) -> None:
        self._sync_quantify_from_list()
        self._sync_params_from_ui()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save calibration session",
            str(ROOT / "docs" / "calibration.json"),
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            save_calibration_set(self.cal, Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.statusMessage.emit(f"Saved {path}")

    def _load_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load calibration session",
            str(ROOT / "docs"),
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            self.cal = load_calibration_set(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        self.spin_half.setValue(self.cal.half_width_nm)
        self.spin_pad.setValue(self.cal.baseline_pad_nm)
        bidx = self.combo_baseline.findData(self.cal.baseline_method or "snip")
        if bidx >= 0:
            self.combo_baseline.setCurrentIndex(bidx)
        self.spin_snip.setValue(int(self.cal.snip_iterations or 40))
        self.spin_snip.setEnabled((self.cal.baseline_method or "snip") == "snip")
        pidx = self.combo_peak_model.findData(self.cal.peak_model or "gaussian")
        if pidx >= 0:
            self.combo_peak_model.setCurrentIndex(pidx)
        self.spin_shift.setValue(self.cal.shift_tol_nm)
        self.spin_shift.setEnabled((self.cal.peak_model or "gaussian") != "net_area")
        idx = self.combo_degree.findData(self.cal.fit_degree)
        if idx >= 0:
            self.combo_degree.setCurrentIndex(idx)
        if self.cal.atmosphere in ("air", "argon", "unknown"):
            self.combo_atm.setCurrentText(self.cal.atmosphere)
        unit = self.cal.concentration_unit or "wt%"
        if self.combo_unit.findText(unit) < 0:
            self.combo_unit.addItem(unit)
        self.combo_unit.setCurrentText(unit)
        self._update_unit_labels()
        self._refresh_standards_list()
        self._refresh_element_list()
        self._refresh_conc_table()
        self._fill_line_table()
        self._update_acquisition_warnings(popup=False)
        self._refresh_peak_spectrum_combo()
        self._redraw_plots()
        self.statusMessage.emit(f"Loaded {path}")

    # ---------------------------------------------------------- drag & drop
    def _enable_standards_drag_drop(self) -> None:
        """Accept CRM .txt / folder drops and concentration CSV drops."""
        self.setAcceptDrops(True)
        targets: list[QWidget] = [self]
        for w in (
            getattr(self, "_std_box", None),
            getattr(self, "std_list", None),
            getattr(self, "conc_table", None),
            getattr(self, "curve_canvas", None),
        ):
            if isinstance(w, QWidget):
                targets.append(w)
        for w in targets:
            w.setAcceptDrops(True)
            w.installEventFilter(self)

    @staticmethod
    def _paths_from_urls(urls: list[QUrl]) -> tuple[list[Path], list[Path]]:
        """Resolve dropped URLs → (spectrum .txt paths, concentration .csv paths)."""
        txts: list[Path] = []
        csvs: list[Path] = []
        seen: set[Path] = set()

        def _add_txt(path: Path) -> None:
            key = path.resolve()
            if key in seen:
                return
            seen.add(key)
            txts.append(path)

        for url in urls:
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.is_file():
                suf = path.suffix.lower()
                if suf == ".txt":
                    _add_txt(path)
                elif suf == ".csv":
                    key = path.resolve()
                    if key not in seen:
                        seen.add(key)
                        csvs.append(path)
            elif path.is_dir():
                for child in sorted(path.iterdir()):
                    if child.is_file() and child.suffix.lower() == ".txt":
                        _add_txt(child)
        return txts, csvs

    def _drop_paths_from_mime(self, mime) -> tuple[list[Path], list[Path]]:
        if mime is None or not mime.hasUrls():
            return [], []
        return self._paths_from_urls(list(mime.urls()))

    def eventFilter(self, watched, event):  # noqa: N802 — Qt API
        et = event.type()
        if (
            hasattr(self, "line_table")
            and watched is self.line_table.viewport()
            and et in (QEvent.Type.Leave, QEvent.Type.HoverLeave)
        ):
            self._hide_line_hover_preview()
        if et in (QEvent.Type.DragEnter, QEvent.Type.DragMove):
            if isinstance(event, (QDragEnterEvent, QDragMoveEvent)):
                txts, csvs = self._drop_paths_from_mime(event.mimeData())
                if txts or csvs:
                    event.acceptProposedAction()
                    return True
                event.ignore()
                return True
        if et == QEvent.Type.Drop and isinstance(event, QDropEvent):
            txts, csvs = self._drop_paths_from_mime(event.mimeData())
            if txts or csvs:
                event.acceptProposedAction()
                self._handle_dropped_paths(txts, csvs)
                return True
            event.ignore()
            return True
        return super().eventFilter(watched, event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        txts, csvs = self._drop_paths_from_mime(event.mimeData())
        if txts or csvs:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        txts, csvs = self._drop_paths_from_mime(event.mimeData())
        if txts or csvs:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        txts, csvs = self._drop_paths_from_mime(event.mimeData())
        if not txts and not csvs:
            event.ignore()
            return
        event.acceptProposedAction()
        self._handle_dropped_paths(txts, csvs)

    def _handle_dropped_paths(self, txts: list[Path], csvs: list[Path]) -> None:
        if txts:
            self._add_standard_paths(txts)
        for csv_path in csvs:
            self._import_concentrations_path(csv_path)
