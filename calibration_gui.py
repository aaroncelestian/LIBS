"""Calibration tab UI for CRM univariate LIBS calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QEvent, QSize, QUrl, Signal
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from calibration import (
    CalibrationSet,
    CurveFit,
    ElementPrediction,
    add_standard_from_path,
    apply_concentrations,
    acquisition_mismatch_warnings,
    build_fits,
    ensure_element_columns,
    flag_line_overlaps,
    load_calibration_set,
    load_concentrations_csv,
    predict_concentrations,
    save_calibration_set,
    save_concentrations_csv,
    save_predictions_csv,
    seed_lines_from_matches,
    suggest_diagnostic_lines,
)
from identify_elements import ElementHit, LibraryLine, Spectrum
from matplotlib_config import apply_matplotlib_config

ROOT = Path(__file__).resolve().parent


class CurveCanvas(FigureCanvasQTAgg):
    def __init__(self) -> None:
        apply_matplotlib_config()
        self.fig = Figure(figsize=(5.5, 4.0), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.fig.tight_layout()


class CalibrationTab(QWidget):
    """CRM standards → diagnostic lines → I→C curves → apply to unknown."""

    statusMessage = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.cal = CalibrationSet()
        self.library: list[LibraryLine] = []
        self._unknown: Spectrum | None = None
        self._identify_hits: list[ElementHit] = []
        self._last_preds: list[ElementPrediction] = []
        self._block_conc = False
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
        self.btn_add_std.clicked.connect(self._add_standards)
        self.btn_remove_std = QPushButton("Remove")
        self.btn_remove_std.clicked.connect(self._remove_standard)
        btn_row.addWidget(self.btn_add_std)
        btn_row.addWidget(self.btn_remove_std)
        btn_row.addStretch(1)
        std_l.addLayout(btn_row)

        self.std_list = QListWidget()
        self.std_list.setToolTip(
            "CRM / standard spectra.\n"
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
            "Suggest diagnostic lines for checked elements "
            "(from Identify matches if available, else NIST)."
        )
        self.btn_seed_lines.clicked.connect(self._suggest_lines)
        self.btn_check_all = QPushButton("Check all")
        self.btn_check_all.clicked.connect(lambda: self._set_all_quantify(True))
        self.btn_check_none = QPushButton("Check none")
        self.btn_check_none.clicked.connect(lambda: self._set_all_quantify(False))
        el_btn.addWidget(self.btn_add_el)
        el_btn.addWidget(self.btn_remove_el)
        el_btn.addWidget(self.btn_seed_lines)
        el_l.addLayout(el_btn)
        el_btn2 = QHBoxLayout()
        el_btn2.addWidget(self.btn_check_all)
        el_btn2.addWidget(self.btn_check_none)
        el_btn2.addStretch(1)
        el_l.addLayout(el_btn2)

        self.el_list = QListWidget()
        self.el_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.el_list.itemChanged.connect(self._on_element_check_changed)
        self.el_list.itemSelectionChanged.connect(self._fill_line_table)
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
        conc_l.addLayout(conc_btn)

        self.conc_table = QTableWidget(0, 1)
        self.conc_table.setToolTip(
            "Known concentrations.\n"
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
            "Values in wt%. CSV: standard_id, Element1, … (blank = skip)"
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
        self.line_table = QTableWidget(0, 5)
        self.line_table.setHorizontalHeaderLabels(
            ["On", "Element", "λ (nm)", "Species", "Overlap"]
        )
        self.line_table.horizontalHeader().setStretchLastSection(True)
        self.line_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.line_table.itemChanged.connect(self._on_line_item_changed)
        self.line_table.itemSelectionChanged.connect(self._on_line_selection_for_plot)
        line_l.addWidget(self.line_table)
        right_l.addWidget(line_box, stretch=3)

        params_box = QGroupBox("Fit parameters")
        params_outer = QHBoxLayout(params_box)
        form = QFormLayout()
        self.spin_half = QDoubleSpinBox()
        self.spin_half.setRange(0.05, 2.0)
        self.spin_half.setSingleStep(0.05)
        self.spin_half.setDecimals(2)
        self.spin_half.setValue(0.20)
        self.spin_half.setSuffix(" nm")
        self.spin_half.setToolTip("Integration half-width around diagnostic λ")
        form.addRow("Integrate ±", self.spin_half)

        self.spin_pad = QDoubleSpinBox()
        self.spin_pad.setRange(0.05, 3.0)
        self.spin_pad.setSingleStep(0.05)
        self.spin_pad.setDecimals(2)
        self.spin_pad.setValue(0.40)
        self.spin_pad.setSuffix(" nm")
        self.spin_pad.setToolTip("Extra pad outside core for local baseline wings")
        form.addRow("Baseline pad", self.spin_pad)

        self.combo_degree = QComboBox()
        self.combo_degree.addItem("Linear", 1)
        self.combo_degree.addItem("Quadratic", 2)
        form.addRow("Fit", self.combo_degree)

        self.combo_atm = QComboBox()
        self.combo_atm.addItems(["air", "argon", "unknown"])
        self.combo_atm.currentTextChanged.connect(self._on_atm_changed)
        form.addRow("Atmosphere", self.combo_atm)

        self.combo_unit = QComboBox()
        self.combo_unit.setEditable(True)
        self.combo_unit.addItems(
            ["wt%", "ppm", "µg/g", "mg/kg", "at%", "oxide wt%", "mass frac"]
        )
        self.combo_unit.setCurrentText("wt%")
        self.combo_unit.setToolTip(
            "Unit for CRM concentrations and predictions.\n"
            "The fit does not convert units — use one unit for the whole session."
        )
        self.combo_unit.currentTextChanged.connect(self._on_unit_changed)
        form.addRow("Conc. unit", self.combo_unit)
        params_outer.addLayout(form)

        fit_btns = QVBoxLayout()
        self.btn_fit = QPushButton("Build calibration curves")
        self.btn_fit.setToolTip("Fit I→C curves, then open the Curves & results tab.")
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
            "Select a diagnostic line on Data entry to show its calibration curve. "
            "Predict unknowns below."
        )
        plot_hint.setStyleSheet("color: #555; font-size: 11px;")
        plot_hint.setWordWrap(True)
        plot_l.addWidget(plot_hint)

        self.curve_canvas = CurveCanvas()
        self.curve_toolbar = NavigationToolbar2QT(self.curve_canvas, plot_page)
        icon = self.curve_toolbar.iconSize()
        self.curve_toolbar.setIconSize(
            QSize(max(12, icon.width() // 2), max(12, icon.height() // 2))
        )
        plot_l.addWidget(self.curve_toolbar)
        plot_l.addWidget(self.curve_canvas, stretch=3)

        self.fit_label = QLabel("No fits yet.")
        self.fit_label.setStyleSheet("color: #333;")
        self.fit_label.setWordWrap(True)
        plot_l.addWidget(self.fit_label)

        apply_box = QGroupBox("Apply to unknown")
        apply_l = QVBoxLayout(apply_box)
        apply_btns = QHBoxLayout()
        self.btn_use_identify = QPushButton("Use Identify spectrum")
        self.btn_use_identify.clicked.connect(self._use_identify_spectrum)
        self.btn_browse_unknown = QPushButton("Browse unknown…")
        self.btn_browse_unknown.clicked.connect(self._browse_unknown)
        self.btn_predict = QPushButton("Predict concentrations")
        self.btn_predict.clicked.connect(self._predict)
        self.btn_export_pred = QPushButton("Export predictions…")
        self.btn_export_pred.clicked.connect(self._export_predictions)
        apply_btns.addWidget(self.btn_use_identify)
        apply_btns.addWidget(self.btn_browse_unknown)
        apply_btns.addWidget(self.btn_predict)
        apply_btns.addWidget(self.btn_export_pred)
        apply_l.addLayout(apply_btns)
        self.unknown_label = QLabel("Unknown: (none)")
        apply_l.addWidget(self.unknown_label)
        self.pred_table = QTableWidget(0, 4)
        self.pred_table.setHorizontalHeaderLabels(
            ["Element", "Concentration (wt%)", "± std", "# lines"]
        )
        self.pred_table.horizontalHeader().setStretchLastSection(True)
        self.pred_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        apply_l.addWidget(self.pred_table)
        plot_l.addWidget(apply_box, stretch=2)

        self.plot_page = plot_page
        self.sub_tabs.addTab(plot_page, "Curves & results")

    def _show_plot_tab(self) -> None:
        if hasattr(self, "sub_tabs") and hasattr(self, "plot_page"):
            self.sub_tabs.setCurrentWidget(self.plot_page)
            self._redraw_curve()

    def _on_line_selection_for_plot(self) -> None:
        """Keep curve in sync; if already on plot tab, redraw immediately."""
        if (
            hasattr(self, "sub_tabs")
            and hasattr(self, "plot_page")
            and self.sub_tabs.currentWidget() is self.plot_page
        ):
            self._redraw_curve()

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
            self.unknown_label.setText(f"Unknown: {spectrum.meta.path.name} (Identify)")
        if atmosphere in ("air", "argon", "unknown"):
            self.combo_atm.blockSignals(True)
            self.combo_atm.setCurrentText(atmosphere)
            self.combo_atm.blockSignals(False)
            self.cal.atmosphere = atmosphere

    def has_fits(self) -> bool:
        return bool(self.cal.fits)

    def predict_for_spectrum(self, spectrum: Spectrum) -> list[ElementPrediction]:
        """Apply current CRM fits to one unknown spectrum."""
        self._sync_params_from_ui()
        if not self.cal.fits:
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
        self.cal.fit_degree = int(self.combo_degree.currentData())
        self.cal.atmosphere = self.combo_atm.currentText()
        self.cal.concentration_unit = self.combo_unit.currentText().strip() or "wt%"

    def _on_atm_changed(self, atm: str) -> None:
        self.cal.atmosphere = atm
        for s in self.cal.standards:
            s.atmosphere = atm

    def _on_unit_changed(self, unit: str) -> None:
        self.cal.concentration_unit = (unit or "").strip() or "wt%"
        self._update_unit_labels()

    def _unit_label(self) -> str:
        u = (self.cal.concentration_unit or self.combo_unit.currentText() or "wt%").strip()
        return u or "wt%"

    def _update_unit_labels(self) -> None:
        u = self._unit_label()
        if hasattr(self, "pred_table"):
            self.pred_table.setHorizontalHeaderLabels(
                ["Element", f"Concentration ({u})", "± std", "# lines"]
            )
        if hasattr(self, "conc_box_hint"):
            self.conc_box_hint.setText(
                f"Values in {u}. CSV: standard_id, Element1, … (blank = skip)"
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
        self.statusMessage.emit(f"Standards: {len(self.cal.standards)} (+{added})")

    def _remove_standard(self) -> None:
        row = self.std_list.currentRow()
        if row < 0 or row >= len(self.cal.standards):
            return
        del self.cal.standards[row]
        self._refresh_standards_list()
        self._refresh_conc_table()
        self._update_acquisition_warnings(popup=False)

    def _refresh_standards_list(self) -> None:
        self.std_list.clear()
        for s in self.cal.standards:
            cfg = s.spectrum.meta.cfg_path.name if s.spectrum.meta.cfg_path else "no cfg"
            self.std_list.addItem(f"{s.sample_id}  ({cfg})")
        if self.cal.standards:
            self.std_list.setCurrentRow(0)
        self._update_acquisition_warnings(popup=False)

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

        # Prefer Identify matches when available
        match_map: dict[str, list[tuple[float, str]]] = {}
        for hit in self._identify_hits:
            if hit.element not in active:
                continue
            pairs = [
                (m.peak.wavelength_nm, m.line.species)
                for m in sorted(
                    hit.matches,
                    key=lambda m: m.peak.prominence,
                    reverse=True,
                )
            ]
            match_map[hit.element] = pairs

        if match_map:
            seeded = seed_lines_from_matches(
                match_map,
                self.library,
                max_per_element=8,
                overlap_tol_nm=self.cal.overlap_tol_nm,
            )
            # Fill missing active elements from NIST
            have = {d.element for d in seeded}
            missing = [e for e in active if e not in have]
            if missing:
                wl_min, wl_max = self._wl_range()
                extra = suggest_diagnostic_lines(
                    self.library,
                    missing,
                    wl_min=wl_min,
                    wl_max=wl_max,
                    max_per_element=8,
                    overlap_tol_nm=self.cal.overlap_tol_nm,
                )
                seeded.extend(extra)
            # Keep lines for unchecked elements that already exist
            keep_other = [
                d for d in self.cal.diagnostic_lines if d.element not in set(active)
            ]
            self.cal.diagnostic_lines = keep_other + seeded
        else:
            wl_min, wl_max = self._wl_range()
            seeded = suggest_diagnostic_lines(
                self.library,
                active,
                wl_min=wl_min,
                wl_max=wl_max,
                max_per_element=8,
                overlap_tol_nm=self.cal.overlap_tol_nm,
            )
            keep_other = [
                d for d in self.cal.diagnostic_lines if d.element not in set(active)
            ]
            self.cal.diagnostic_lines = keep_other + seeded
        self._fill_line_table()
        self.statusMessage.emit(
            f"Suggested lines for {len(active)} checked element(s) "
            f"(overlap warnings are soft — review before fitting)."
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
        parts = [
            f"{f.element} {f.wavelength_nm:.3f} nm: R²={f.r_squared:.3f} (n={f.n_points})"
            for f in fits
        ]
        summary = " · ".join(parts[:6]) + (" …" if len(parts) > 6 else "")
        self._set_fit_summary(summary)
        msg = (
            f"Built {len(fits)} curve(s) for "
            f"{len(self.cal.active_elements())} checked element(s)"
        )
        if skipped:
            msg += f" · skipped {len(skipped)} line(s)"
            QMessageBox.information(
                self,
                "Some lines skipped",
                msg
                + ":\n\n"
                + "\n".join(f"• {s}" for s in skipped[:10])
                + (f"\n… and {len(skipped) - 10} more" if len(skipped) > 10 else ""),
            )
        self.statusMessage.emit(msg)
        # Select first fit line in table if possible
        if self.line_table.rowCount() > 0:
            self.line_table.selectRow(0)
        self._show_plot_tab()

    def _predict(self) -> None:
        self._sync_quantify_from_list()
        if not self.cal.active_elements():
            QMessageBox.information(
                self,
                "Predict",
                "Check the elements you want to quantify (left panel).",
            )
            return
        if self._unknown is None:
            QMessageBox.information(self, "Predict", "Load an unknown spectrum first.")
            return
        if not self.cal.fits:
            self._run_fit()
        if not self.cal.fits:
            return
        self._sync_params_from_ui()
        preds = predict_concentrations(self.cal, self._unknown)
        self._last_preds = preds
        self.pred_table.setRowCount(len(preds))
        for i, p in enumerate(preds):
            self.pred_table.setItem(i, 0, QTableWidgetItem(p.element))
            self.pred_table.setItem(i, 1, QTableWidgetItem(f"{p.concentration:g}"))
            std_txt = "—" if p.std is None else f"{p.std:g}"
            self.pred_table.setItem(i, 2, QTableWidgetItem(std_txt))
            self.pred_table.setItem(i, 3, QTableWidgetItem(str(p.n_lines)))
        self.pred_table.resizeColumnsToContents()
        self._plot_predictions(preds)
        self.statusMessage.emit(
            f"Predicted {len(preds)} checked element(s)"
        )

    def _plot_predictions(self, preds: list[ElementPrediction]) -> None:
        """Bar chart of selected quantified elements (with multi-line ±std)."""
        ax = self.curve_canvas.ax
        ax.clear()
        if not preds:
            ax.text(
                0.5,
                0.5,
                "No predictions for checked elements",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            self.curve_canvas.draw_idle()
            return
        names = [p.element for p in preds]
        vals = [p.concentration for p in preds]
        errs = [0.0 if p.std is None else p.std for p in preds]
        x = np.arange(len(names))
        ax.bar(x, vals, color="#1a5276", alpha=0.85, zorder=2)
        if any(e > 0 for e in errs):
            ax.errorbar(
                x,
                vals,
                yerr=errs,
                fmt="none",
                ecolor="#922b21",
                capsize=4,
                zorder=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel(f"Concentration ({self._unit_label()})")
        ax.set_title("Quantified elements (checked subset)")
        self.curve_canvas.fig.tight_layout()
        self.curve_canvas.draw_idle()

    def _export_predictions(self) -> None:
        if not self._last_preds:
            QMessageBox.information(
                self, "Export", "Run Predict concentrations first."
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export predictions CSV",
            str(ROOT / "docs" / "predictions.csv"),
            "CSV (*.csv)",
        )
        if not path:
            return
        self._sync_params_from_ui()
        save_predictions_csv(
            Path(path), self._last_preds, unit=self.cal.concentration_unit
        )
        self.statusMessage.emit(f"Wrote {path}")

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

    def _redraw_curve(self) -> None:
        ax = self.curve_canvas.ax
        ax.clear()
        fit = self._current_fit()
        if fit is None:
            ax.text(0.5, 0.5, "Build calibration curves", ha="center", va="center", transform=ax.transAxes)
            self.curve_canvas.draw_idle()
            return
        x = np.asarray(fit.intensities, dtype=float)
        y = np.asarray(fit.concentrations, dtype=float)
        ax.scatter(x, y, c="#1a5276", s=40, zorder=3, label="Standards")
        for sid, xi, yi in zip(fit.sample_ids, x, y):
            ax.annotate(sid, (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=8)
        if len(x):
            x_line = np.linspace(float(x.min()) * 0.95, float(x.max()) * 1.05, 80)
            y_line = np.polyval(fit.coeffs, x_line)
            ax.plot(x_line, y_line, color="#c0392b", lw=1.5, label=f"Fit R²={fit.r_squared:.3f}")
        ax.set_xlabel("Net peak area (counts·nm)")
        ax.set_ylabel(f"Concentration ({self._unit_label()})")
        ax.set_title(f"{fit.element} — {fit.wavelength_nm:.3f} nm")
        ax.legend(loc="best", fontsize=8)
        self.curve_canvas.fig.tight_layout()
        self.curve_canvas.draw_idle()

    # ----------------------------------------------------------- unknown
    def _use_identify_spectrum(self) -> None:
        if self._unknown is None:
            QMessageBox.information(
                self,
                "Unknown",
                "No Identify spectrum loaded yet. Open a spectrum on the Identify tab.",
            )
            return
        self.unknown_label.setText(f"Unknown: {self._unknown.meta.path.name} (Identify)")
        self.statusMessage.emit(f"Using Identify spectrum: {self._unknown.meta.path.name}")

    def _browse_unknown(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open unknown spectrum",
            str(ROOT / "docs"),
            "Spectrum text (*.txt);;All files (*)",
        )
        if not path:
            return
        from identify_elements import load_spectrum

        try:
            self._unknown = load_spectrum(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            return
        self.unknown_label.setText(f"Unknown: {self._unknown.meta.path.name}")

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
        self._redraw_curve()
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
