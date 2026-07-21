#!/usr/bin/env python3
"""
LIBS Spectrum Explorer — interactive GUI for plotting and element matching.

Features:
  - Load one or many spectra (Open, Add, or drag-and-drop files/folders)
  - Single / Waterfall / Working display modes
  - Prev/Next navigation; sum / mean multi-spot shots on the same sample
  - Quant button: CRM quantification of checked/highlighted spectra (Batch tab)
  - Plot full spectrum with detected peaks
  - Run NIST search/match; ranked element list
  - Multi-select elements to preview NIST stick spectra
  - Browse NIST tab: periodic table + catalog lines (click view / double-click pin);
    Match auto-pins Overlay (default top 5) on the plot
  - Top-5 intense matched-line preview window (double-click element / Ctrl+L)
  - Click near a peak to list candidate elements/lines
  - Manually add weak peaks (Shift/right-click) into the match
  - Export publication report (≤5 strongest lines per element)
  - Atmosphere tag (air / argon)
  - Calibrate tab: CRM univariate standards calibration (I → C)

Launch:
  .venv/bin/python libs_gui.py
  .venv/bin/python libs_gui.py path/to/spectrum.txt

Note: on macOS, PySide6's cocoa plugin can be invisible to Qt if the
dylibs have the UF_HIDDEN flag (common after iCloud / Finder copies).
libs_gui clears that when possible, and otherwise mirrors Qt plugins to
~/Library/Caches/LIBS-PySide6-Qt before QApplication starts.
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Set MPLCONFIGDIR before any matplotlib import (font cache after path moves).
import matplotlib_config  # noqa: F401


def _configure_qt_plugins() -> None:
    """
    Point Qt at a usable platforms/ directory before QApplication starts.

    Qt discovers plugins via QDir. On macOS, files with the UF_HIDDEN flag
    (and some iCloud / CloudDocs layouts) are omitted from that listing even
    though they exist on disk — then QApplication aborts with
    "Could not find the Qt platform plugin cocoa". Clear UF_HIDDEN when we
    can; otherwise mirror plugins/libs to ~/Library/Caches/LIBS-PySide6-Qt.
    """
    try:
        import PySide6
    except ImportError:
        return

    src_qt = Path(PySide6.__file__).resolve().parent / "Qt"
    src_plugins = src_qt / "plugins"
    src_platforms = src_plugins / "platforms"
    cocoa_name = "libqcocoa.dylib"
    if not src_platforms.is_dir():
        return

    def _qdir_sees_cocoa(platforms: Path) -> bool:
        try:
            from PySide6.QtCore import QDir

            names = list(QDir(str(platforms)).entryList(["*.dylib"]))
            return cocoa_name in names
        except Exception:
            return False

    def _clear_uf_hidden(root: Path) -> None:
        """Best-effort: make Qt plugin dylibs visible to QDir on macOS."""
        if sys.platform != "darwin" or not root.is_dir():
            return
        try:
            import subprocess

            subprocess.run(
                ["chflags", "-R", "nohidden", str(root)],
                check=False,
                capture_output=True,
            )
        except Exception:
            pass

    if not _qdir_sees_cocoa(src_platforms):
        _clear_uf_hidden(src_plugins)
        # Also clear on the cocoa dylib itself in case -R was blocked mid-tree.
        cocoa = src_platforms / cocoa_name
        if cocoa.exists():
            _clear_uf_hidden(cocoa)

    plugins = src_plugins
    platforms = src_platforms

    if not _qdir_sees_cocoa(src_platforms):
        cache = Path.home() / "Library" / "Caches" / "LIBS-PySide6-Qt"
        cache_plugins = cache / "plugins"
        cache_platforms = cache_plugins / "platforms"
        stamp = cache / ".pyside_version"
        need_sync = True
        if (cache_platforms / cocoa_name).exists() and stamp.exists():
            if stamp.read_text(encoding="utf-8").strip() == PySide6.__version__:
                need_sync = False
        if need_sync:
            import shutil

            cache.mkdir(parents=True, exist_ok=True)
            for name in ("plugins", "lib"):
                src = src_qt / name
                dst = cache / name
                if not src.is_dir():
                    continue
                if dst.exists():
                    shutil.rmtree(dst)
                print(
                    f"Copying PySide6 Qt {name}/ to\n  {dst}\n"
                    "(Qt could not see plugins in the venv; often UF_HIDDEN "
                    "or iCloud. Cache copy is used instead.)",
                    file=sys.stderr,
                )
                shutil.copytree(src, dst)
            # Ensure the mirror is not hidden either.
            _clear_uf_hidden(cache_plugins)
            stamp.write_text(PySide6.__version__ + "\n", encoding="utf-8")
        plugins = cache_plugins
        platforms = cache_platforms

    if platforms.is_dir():
        os.environ["QT_PLUGIN_PATH"] = str(plugins)
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platforms)

    if sys.platform == "darwin" and not (platforms / cocoa_name).exists():
        print(
            "PySide6 cocoa plugin missing.\n"
            f"  Python: {sys.executable}\n"
            f"  Looked in: {platforms}\n"
            "Use the project venv:\n"
            "  .venv/bin/python libs_gui.py",
            file=sys.stderr,
        )


_configure_qt_plugins()

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QEvent, QSize, QItemSelectionModel, QPoint, QRectF, QUrl, Signal
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QIcon,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QPolygon,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)


from calibration_gui import CalibrationTab
from identify_elements import (
    ElementHit,
    LibraryLine,
    Peak,
    Spectrum,
    SpectrumMeta,
    browse_library_lines,
    candidates_near_wavelength,
    combine_spectra,
    elements_in_wavelength_range,
    find_spectrum_peaks,
    load_line_library,
    load_spectrum,
    make_peak_at_wavelength,
    match_peaks,
    merge_peaks,
    nearest_peak,
    score_elements,
    strong_library_lines,
    write_spectrum,
)
from matplotlib_config import apply_matplotlib_config
from publication_report import (
    export_element_report_pdf,
    export_element_report_pngs,
    plot_element_line_panels,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_LIBRARY = ROOT / "nist_lines" / "libs_line_library.csv"

# Distinct colors for multi-element NIST stick previews
ELEMENT_COLORS = (
    "#1e8449",
    "#8e44ad",
    "#d35400",
    "#2980b9",
    "#c0392b",
    "#16a085",
    "#7d3c98",
    "#b9770e",
)

# Compact periodic table: (symbol, row, column) — 18 cols; La/Ac series on rows 8–9
_PERIODIC_LAYOUT: tuple[tuple[str, int, int], ...] = (
    ("H", 0, 0), ("He", 0, 17),
    ("Li", 1, 0), ("Be", 1, 1),
    ("B", 1, 12), ("C", 1, 13), ("N", 1, 14), ("O", 1, 15), ("F", 1, 16), ("Ne", 1, 17),
    ("Na", 2, 0), ("Mg", 2, 1),
    ("Al", 2, 12), ("Si", 2, 13), ("P", 2, 14), ("S", 2, 15), ("Cl", 2, 16), ("Ar", 2, 17),
    ("K", 3, 0), ("Ca", 3, 1),
    ("Sc", 3, 2), ("Ti", 3, 3), ("V", 3, 4), ("Cr", 3, 5), ("Mn", 3, 6), ("Fe", 3, 7),
    ("Co", 3, 8), ("Ni", 3, 9), ("Cu", 3, 10), ("Zn", 3, 11),
    ("Ga", 3, 12), ("Ge", 3, 13), ("As", 3, 14), ("Se", 3, 15), ("Br", 3, 16), ("Kr", 3, 17),
    ("Rb", 4, 0), ("Sr", 4, 1),
    ("Y", 4, 2), ("Zr", 4, 3), ("Nb", 4, 4), ("Mo", 4, 5), ("Tc", 4, 6), ("Ru", 4, 7),
    ("Rh", 4, 8), ("Pd", 4, 9), ("Ag", 4, 10), ("Cd", 4, 11),
    ("In", 4, 12), ("Sn", 4, 13), ("Sb", 4, 14), ("Te", 4, 15), ("I", 4, 16), ("Xe", 4, 17),
    ("Cs", 5, 0), ("Ba", 5, 1), ("La", 5, 2),
    ("Hf", 5, 3), ("Ta", 5, 4), ("W", 5, 5), ("Re", 5, 6), ("Os", 5, 7), ("Ir", 5, 8),
    ("Pt", 5, 9), ("Au", 5, 10), ("Hg", 5, 11),
    ("Tl", 5, 12), ("Pb", 5, 13), ("Bi", 5, 14), ("Po", 5, 15), ("At", 5, 16), ("Rn", 5, 17),
    ("Fr", 6, 0), ("Ra", 6, 1), ("Ac", 6, 2),
    ("Rf", 6, 3), ("Db", 6, 4), ("Sg", 6, 5), ("Bh", 6, 6), ("Hs", 6, 7), ("Mt", 6, 8),
    ("Ds", 6, 9), ("Rg", 6, 10), ("Cn", 6, 11),
    ("Nh", 6, 12), ("Fl", 6, 13), ("Mc", 6, 14), ("Lv", 6, 15), ("Ts", 6, 16), ("Og", 6, 17),
    ("Ce", 8, 3), ("Pr", 8, 4), ("Nd", 8, 5), ("Pm", 8, 6), ("Sm", 8, 7), ("Eu", 8, 8),
    ("Gd", 8, 9), ("Tb", 8, 10), ("Dy", 8, 11), ("Ho", 8, 12), ("Er", 8, 13), ("Tm", 8, 14),
    ("Yb", 8, 15), ("Lu", 8, 16),
    ("Th", 9, 3), ("Pa", 9, 4), ("U", 9, 5), ("Np", 9, 6), ("Pu", 9, 7), ("Am", 9, 8),
    ("Cm", 9, 9), ("Bk", 9, 10), ("Cf", 9, 11), ("Es", 9, 12), ("Fm", 9, 13), ("Md", 9, 14),
    ("No", 9, 15), ("Lr", 9, 16),
)


class _ElementCell(QPushButton):
    """Tiny periodic-table cell: click = view, double-click = pin toggle."""

    def __init__(self, symbol: str, parent: QWidget | None = None) -> None:
        super().__init__(symbol, parent)
        self.symbol = symbol
        self._available = False
        self._viewing = False
        self._pinned = False
        self.setFixedSize(18, 16)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._apply_style()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        parent = self.parent()
        if isinstance(parent, PeriodicTableWidget) and self._available:
            parent.elementPinToggled.emit(self.symbol)
        super().mouseDoubleClickEvent(event)

    def set_available(self, available: bool) -> None:
        self._available = available
        self.setEnabled(available)
        self._apply_style()

    def set_viewing(self, viewing: bool) -> None:
        self._viewing = viewing
        self._apply_style()

    def set_pinned(self, pinned: bool) -> None:
        self._pinned = pinned
        self._apply_style()

    def _apply_style(self) -> None:
        if not self._available:
            bg, fg, border = "#f0f0f0", "#bbb", "#ddd"
        elif self._viewing and self._pinned:
            bg, fg, border = "#d5f5e3", "#145a32", "#1e8449"
        elif self._pinned:
            bg, fg, border = "#fcf3cf", "#7d6608", "#b9770e"
        elif self._viewing:
            bg, fg, border = "#d6eaf8", "#1a5276", "#2980b9"
        else:
            bg, fg, border = "#fafafa", "#333", "#c8c8c8"
        self.setStyleSheet(
            f"QPushButton {{"
            f"  background:{bg}; color:{fg}; border:1px solid {border};"
            f"  border-radius:2px; font-size:8px; font-weight:600; padding:0;"
            f"}}"
            f"QPushButton:hover:!disabled {{ border-color:#555; }}"
            f"QPushButton:disabled {{ color:{fg}; }}"
        )


class PeriodicTableWidget(QWidget):
    """
    Mini periodic table for Browse NIST.

    Single-click → view element lines. Double-click → pin/unpin for stick overlay.
    After Match, only the Overlay combo (default top 5) are auto-pinned.
    """

    elementViewed = Signal(str)
    elementPinToggled = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cells: dict[str, _ElementCell] = {}
        self._viewing: str | None = None
        self._pinned: set[str] = set()

        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(1)
        grid.setVerticalSpacing(1)

        for symbol, row, col in _PERIODIC_LAYOUT:
            cell = _ElementCell(symbol, self)
            cell.clicked.connect(lambda checked=False, s=symbol: self._on_cell_clicked(s))
            cell.setToolTip(
                f"{symbol}\nClick to view NIST lines\n"
                "Double-click to pin / unpin on the plot "
                "(syncs with matched list; Match pins Overlay only)"
            )
            cell.set_available(False)
            grid.addWidget(cell, row, col)
            self._cells[symbol] = cell

        # Spacer row between main block and f-block
        spacer = QWidget()
        spacer.setFixedHeight(4)
        grid.addWidget(spacer, 7, 0, 1, 18)

        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)

    def _on_cell_clicked(self, symbol: str) -> None:
        cell = self._cells.get(symbol)
        if cell is None or not cell._available:
            return
        self.elementViewed.emit(symbol)

    def set_available(self, symbols: set[str] | list[str]) -> None:
        avail = set(symbols)
        for symbol, cell in self._cells.items():
            cell.set_available(symbol in avail)

    def set_viewing(self, symbol: str | None) -> None:
        if self._viewing and self._viewing in self._cells:
            self._cells[self._viewing].set_viewing(False)
        self._viewing = symbol or None
        if self._viewing and self._viewing in self._cells:
            self._cells[self._viewing].set_viewing(True)

    def set_pinned(self, symbols: set[str] | list[str]) -> None:
        new_pinned = set(symbols)
        for symbol in self._pinned - new_pinned:
            if symbol in self._cells:
                self._cells[symbol].set_pinned(False)
        for symbol in new_pinned - self._pinned:
            if symbol in self._cells:
                self._cells[symbol].set_pinned(True)
        self._pinned = new_pinned

    def clear_selection(self) -> None:
        self.set_viewing(None)


@dataclass
class SpectrumMatchCache:
    auto_peaks: list[Peak] = field(default_factory=list)
    manual_peaks: list[Peak] = field(default_factory=list)
    peaks: list[Peak] = field(default_factory=list)
    hits: list[ElementHit] = field(default_factory=list)


@dataclass
class BulkQuantRow:
    index: int  # 1-based spectrum number
    filename: str
    concentrations: dict[str, float]  # element -> C


class BatchPlotCanvas(FigureCanvasQTAgg):
    def __init__(self) -> None:
        apply_matplotlib_config()
        self.fig = Figure(figsize=(4.5, 3.0), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.fig.tight_layout()


def _paint_icon(size: int, paint) -> QIcon:
    """Build a toolbar icon via a paint(painter, size) callback."""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    paint(painter, size)
    painter.end()
    return QIcon(pm)


def _icon_open(size: int = 64) -> QIcon:
    def paint(p: QPainter, s: int) -> None:
        p.setPen(QPen(QColor("#2c3e50"), max(2, s // 16)))
        p.setBrush(QBrush(QColor("#f5c542")))
        p.drawRoundedRect(int(s * 0.18), int(s * 0.28), int(s * 0.64), int(s * 0.48), 4, 4)
        p.setBrush(QBrush(QColor("#e67e22")))
        p.drawRoundedRect(int(s * 0.18), int(s * 0.22), int(s * 0.34), int(s * 0.14), 3, 3)

    return _paint_icon(size, paint)


def _icon_match(size: int = 64) -> QIcon:
    def paint(p: QPainter, s: int) -> None:
        pen = QPen(QColor("#1a1a1a"), max(2, s // 18))
        p.setPen(pen)
        p.drawLine(int(s * 0.12), int(s * 0.72), int(s * 0.88), int(s * 0.72))
        p.setPen(QPen(QColor("#c0392b"), max(2, s // 16)))
        for x, h in ((0.28, 0.45), (0.48, 0.62), (0.68, 0.38)):
            p.drawLine(int(s * x), int(s * 0.72), int(s * x), int(s * (0.72 - h)))
        p.setBrush(QBrush(QColor("#1e8449")))
        p.setPen(Qt.PenStyle.NoPen)
        for x, h in ((0.28, 0.45), (0.48, 0.62), (0.68, 0.38)):
            p.drawEllipse(int(s * x) - s // 14, int(s * (0.72 - h)) - s // 14, s // 7, s // 7)

    return _paint_icon(size, paint)


def _icon_add_peak(size: int = 64) -> QIcon:
    def paint(p: QPainter, s: int) -> None:
        p.setPen(QPen(QColor("#2980b9"), max(2, s // 16)))
        p.drawLine(int(s * 0.2), int(s * 0.7), int(s * 0.8), int(s * 0.7))
        p.drawLine(int(s * 0.5), int(s * 0.7), int(s * 0.5), int(s * 0.28))
        p.setBrush(QBrush(QColor("#2980b9")))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRect(int(s * 0.42), int(s * 0.22), int(s * 0.16), int(s * 0.16))
        p.setPen(QPen(QColor("#27ae60"), max(3, s // 12)))
        p.drawLine(int(s * 0.72), int(s * 0.22), int(s * 0.72), int(s * 0.42))
        p.drawLine(int(s * 0.62), int(s * 0.32), int(s * 0.82), int(s * 0.32))

    return _paint_icon(size, paint)


def _icon_remove_peak(size: int = 64) -> QIcon:
    def paint(p: QPainter, s: int) -> None:
        p.setPen(QPen(QColor("#2980b9"), max(2, s // 16)))
        p.drawLine(int(s * 0.2), int(s * 0.7), int(s * 0.8), int(s * 0.7))
        p.drawLine(int(s * 0.5), int(s * 0.7), int(s * 0.5), int(s * 0.28))
        p.setBrush(QBrush(QColor("#2980b9")))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRect(int(s * 0.42), int(s * 0.22), int(s * 0.16), int(s * 0.16))
        p.setPen(QPen(QColor("#c0392b"), max(3, s // 12)))
        p.drawLine(int(s * 0.62), int(s * 0.28), int(s * 0.82), int(s * 0.28))

    return _paint_icon(size, paint)


def _icon_preview(size: int = 64) -> QIcon:
    def paint(p: QPainter, s: int) -> None:
        p.setPen(QPen(QColor("#34495e"), max(2, s // 18)))
        for i, x0 in enumerate((0.12, 0.38, 0.64)):
            r = QRectF(s * x0, s * 0.18, s * 0.22, s * 0.64)
            p.setBrush(QBrush(QColor("#ecf0f1")))
            p.drawRoundedRect(r, 3, 3)
            p.setPen(QPen(QColor("#c0392b"), max(2, s // 20)))
            cx = s * (x0 + 0.11)
            p.drawLine(int(cx), int(s * 0.28), int(cx), int(s * 0.7))
            p.setPen(QPen(QColor("#34495e"), max(2, s // 18)))

    return _paint_icon(size, paint)


def _icon_report(size: int = 64) -> QIcon:
    def paint(p: QPainter, s: int) -> None:
        p.setPen(QPen(QColor("#2c3e50"), max(2, s // 18)))
        p.setBrush(QBrush(QColor("#fdfefe")))
        p.drawRoundedRect(int(s * 0.22), int(s * 0.12), int(s * 0.52), int(s * 0.76), 4, 4)
        p.setPen(QPen(QColor("#7f8c8d"), max(2, s // 16)))
        for y in (0.30, 0.42, 0.54, 0.66):
            p.drawLine(int(s * 0.32), int(s * y), int(s * 0.64), int(s * y))

    return _paint_icon(size, paint)


def _icon_clear(size: int = 64) -> QIcon:
    def paint(p: QPainter, s: int) -> None:
        p.setPen(QPen(QColor("#7f8c8d"), max(3, s // 12)))
        p.drawEllipse(int(s * 0.18), int(s * 0.18), int(s * 0.64), int(s * 0.64))
        p.drawLine(int(s * 0.30), int(s * 0.30), int(s * 0.70), int(s * 0.70))

    return _paint_icon(size, paint)


def _icon_reload(size: int = 64) -> QIcon:
    def paint(p: QPainter, s: int) -> None:
        pen = QPen(QColor("#2980b9"), max(3, s // 12))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        r = QRectF(s * 0.2, s * 0.2, s * 0.52, s * 0.52)
        p.drawArc(r, 40 * 16, 250 * 16)
        p.setBrush(QBrush(QColor("#2980b9")))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPolygon(
            QPolygon(
                [
                    QPoint(int(s * 0.68), int(s * 0.18)),
                    QPoint(int(s * 0.88), int(s * 0.28)),
                    QPoint(int(s * 0.70), int(s * 0.40)),
                ]
            )
        )

    return _paint_icon(size, paint)

ATMOSPHERE_NOTE = (
    "Argon atmosphere notes:\n\n"
    "• Usually lowers air-related N/O/CN noise and continuum scatter\n"
    "• Elemental line wavelengths stay the same; relative intensities can change\n"
    "• Ar I/II emission lines may appear — treat Ar as a possible match, "
    "not necessarily a sample element\n"
    "• When you add standards calibrations, record atmosphere (air vs Ar) "
    "as a calibration condition"
)

SCORE_HELP = (
    "Confidence (0–100%) estimates how convincing the ID is — "
    "not concentration or abundance.\n\n"
    "It increases when:\n"
    "• More peaks match the element (≥2 required to list; H allowed with 1)\n"
    "• Wavelength matches are tight (small Δλ)\n"
    "• Those peaks are strong in your spectrum\n\n"
    "Fe/V can still look confident because their NIST lists are dense — "
    "confirm with the green markers on diagnostic lines."
)

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self) -> None:
        apply_matplotlib_config()
        self.fig = Figure(figsize=(8, 5.4), dpi=100)
        gs = self.fig.add_gridspec(2, 1, height_ratios=[3.4, 1.0], hspace=0.06)
        self.ax = self.fig.add_subplot(gs[0])
        self.ax_sticks = self.fig.add_subplot(gs[1], sharex=self.ax)
        self.ax.tick_params(labelbottom=False)
        self.ax_sticks.set_ylabel("NIST\nsticks", fontsize=8)
        self.ax_sticks.set_ylim(0, 1.08)
        self.ax_sticks.set_yticks([])
        self.ax_sticks.tick_params(labelsize=8)
        super().__init__(self.fig)


class LinePreviewWindow(QDialog):
    """Non-modal window: top 5 most intense matched lines for one element."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setModal(False)
        self.setWindowTitle("Matched line preview")
        self.resize(1100, 420)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Window ±"))
        self.spin_half = QDoubleSpinBox()
        self.spin_half.setRange(0.3, 10.0)
        self.spin_half.setSingleStep(0.1)
        self.spin_half.setDecimals(2)
        self.spin_half.setValue(1.50)
        self.spin_half.setSuffix(" nm")
        self.spin_half.valueChanged.connect(self._redraw_current)
        controls.addWidget(self.spin_half)
        controls.addStretch(1)
        hint = QLabel("Solid red = observed peak · dashed green = NIST λ")
        hint.setStyleSheet("color: #555;")
        controls.addWidget(hint)
        layout.addLayout(controls)

        apply_matplotlib_config()
        self.fig = Figure(figsize=(11, 3.6), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.canvas)

        self._spectrum: Spectrum | None = None
        self._hit: ElementHit | None = None
        self._atmosphere: str | None = None

    def update_preview(
        self,
        spectrum: Spectrum,
        hit: ElementHit,
        *,
        atmosphere: str | None = None,
    ) -> None:
        self._spectrum = spectrum
        self._hit = hit
        self._atmosphere = atmosphere
        self.setWindowTitle(
            f"Matched line preview — {hit.element} "
            f"({hit.n_peaks} peaks, {hit.confidence:.0f}% conf)"
        )
        self._redraw_current()
        if not self.isVisible():
            self.show()
        self.raise_()
        self.activateWindow()

    def _redraw_current(self) -> None:
        if self._spectrum is None or self._hit is None:
            self.fig.clear()
            self.fig.text(0.5, 0.5, "Select a matched element", ha="center", va="center")
            self.canvas.draw_idle()
            return
        plot_element_line_panels(
            self._spectrum,
            self._hit,
            n_lines=5,
            half_width_nm=float(self.spin_half.value()),
            atmosphere=self._atmosphere,
            fig=self.fig,
        )
        self.canvas.draw_idle()

    def clear_preview(self) -> None:
        self._spectrum = None
        self._hit = None
        self._atmosphere = None
        self.setWindowTitle("Matched line preview")
        self.fig.clear()
        self.fig.text(0.5, 0.5, "Select a matched element", ha="center", va="center")
        self.canvas.draw_idle()


class ReportExportDialog(QDialog):
    """Options for multipage PDF / PNG publication figures."""

    def __init__(
        self,
        parent: QWidget | None,
        *,
        n_elements: int,
        has_selection: bool,
        default_stem: str,
        spectrum_label: str,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export publication report")
        self._stem = default_stem

        form = QFormLayout(self)

        src = QLabel(spectrum_label)
        src.setWordWrap(True)
        src.setStyleSheet("font-weight: 600;")
        src.setToolTip(
            "Reports always use the active Working spectrum and its current\n"
            "match results (not every loaded file).\n"
            "Switch with Prev/Next, View, or double-click in the Spectra list."
        )
        form.addRow("Spectrum", src)

        self.combo_scope = QComboBox()
        self.combo_scope.addItem(f"All ranked elements ({n_elements})", "all")
        top_n = min(10, n_elements)
        self.combo_scope.addItem(f"Top {top_n} by confidence", f"top:{top_n}")
        if n_elements >= 20:
            self.combo_scope.addItem("Top 20 by confidence", "top:20")
        sel_label = "Selected element(s)"
        self.combo_scope.addItem(sel_label, "selected")
        if not has_selection:
            # still listed but user gets a clear empty warning if chosen without selection
            idx = self.combo_scope.findData("selected")
            self.combo_scope.model().item(idx).setEnabled(False)
        form.addRow("Elements", self.combo_scope)

        self.spin_lines = QSpinBox()
        self.spin_lines.setRange(1, 5)
        self.spin_lines.setValue(5)
        form.addRow("Lines per element", self.spin_lines)

        self.spin_half = QDoubleSpinBox()
        self.spin_half.setRange(0.3, 10.0)
        self.spin_half.setSingleStep(0.1)
        self.spin_half.setDecimals(2)
        self.spin_half.setValue(1.50)
        self.spin_half.setSuffix(" nm")
        form.addRow("Window ± half-width", self.spin_half)

        self.combo_fmt = QComboBox()
        self.combo_fmt.addItem("PDF (multipage, one element per page)", "pdf")
        self.combo_fmt.addItem("PNG folder (one figure per element)", "png")
        form.addRow("Format", self.combo_fmt)

        hint = QLabel(
            "Uses the Working spectrum shown above (and its match cache).\n"
            "Each page shows up to 5 strongest matched lines in separate zoomed panels.\n"
            "Solid red = observed peak; dashed green = NIST λ.\n"
            "Default file name stem: "
            f"{default_stem}_lines.pdf"
        )
        hint.setStyleSheet("color: #555;")
        hint.setWordWrap(True)
        form.addRow(hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def options(self) -> dict:
        return {
            "scope": self.combo_scope.currentData(),
            "n_lines": int(self.spin_lines.value()),
            "half_width_nm": float(self.spin_half.value()),
            "format": self.combo_fmt.currentData(),
            "stem": self._stem,
        }


class LibsExplorerWindow(QMainWindow):
    def __init__(self, spectrum_path: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("LIBS Spectrum Explorer")
        self.resize(1280, 800)

        self.spectrum: Spectrum | None = None
        self.loaded_spectra: list[Spectrum] = []
        self._working_label: str | None = None
        self._spectrum_index: int = 0
        self._display_mode: str = "single"  # single | waterfall | working
        self._waterfall_offset_frac: float = 0.15  # vertical gap as fraction of max intensity
        self._match_cache: dict[str, SpectrumMatchCache] = {}
        self._bulk_quant_results: list[BulkQuantRow] = []
        self.library: list[LibraryLine] = []
        self.auto_peaks: list[Peak] = []
        self.manual_peaks: list[Peak] = []
        self.peaks: list[Peak] = []  # auto + manual, used for matching
        self.hits: list[ElementHit] = []
        self._selected_wl: float | None = None
        self._selected_elements: list[str] = []
        self._primary_element: str | None = None
        self._browse_element: str | None = None
        self._pinned_browse_elements: list[str] = []
        self._line_preview: LinePreviewWindow | None = None
        self._browse_lines: list[LibraryLine] = []

        self._build_actions()
        self._build_toolbar()
        self.setStatusBar(QStatusBar())
        self._build_central()
        self.statusBar().showMessage(
            "Open or drag-and-drop spectrum .txt files (or a folder) to begin."
        )
        self._enable_spectrum_drag_drop()

        self._load_library()
        if spectrum_path is not None and spectrum_path.exists():
            self.open_spectra([spectrum_path], replace=True)

    # ----------------------------------------------------------------- menus
    def _build_actions(self) -> None:
        self.act_open = QAction(_icon_open(), "Open spectra…", self)
        self.act_open.setShortcut(QKeySequence.Open)
        self.act_open.setToolTip(
            "Open spectra…\n"
            "Load one or more .txt spectra (optional matching .cfg).\n"
            "Multi-select for multi-spot LIBS on the same sample.\n"
            "You can also drag-and-drop .txt files or a folder onto the window."
        )
        self.act_open.triggered.connect(self.browse_spectra)

        self.act_add_spectra = QAction("Add spectra…", self)
        self.act_add_spectra.setToolTip(
            "Add spectra…\nAppend more .txt files to the loaded list without clearing."
        )
        self.act_add_spectra.triggered.connect(self.browse_add_spectra)

        self.act_mean_spectra = QAction("Mean checked spectra", self)
        self.act_mean_spectra.setToolTip(
            "Mean of checked spectra\n"
            "Average multi-spot shots (recommended when focus/sampling volume varies)."
        )
        self.act_mean_spectra.triggered.connect(lambda: self.combine_checked_spectra("mean"))

        self.act_sum_spectra = QAction("Sum checked spectra", self)
        self.act_sum_spectra.setToolTip(
            "Sum of checked spectra\n"
            "Add intensities (scales with N; useful to boost weak lines)."
        )
        self.act_sum_spectra.triggered.connect(lambda: self.combine_checked_spectra("sum"))

        self.act_export_sum = QAction("Export sum of checked…", self)
        self.act_export_sum.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.act_export_sum.setToolTip(
            "Export sum of checked… (Ctrl+Shift+S)\n"
            "Write the intensity sum of checked spectra to a .txt file\n"
            "(same format as loaded spectra)."
        )
        self.act_export_sum.triggered.connect(lambda: self.export_combined_spectra("sum"))

        self.act_export_spectrum = QAction("Export working spectrum…", self)
        self.act_export_spectrum.setShortcut(QKeySequence("Ctrl+E"))
        self.act_export_spectrum.setToolTip(
            "Export working spectrum… (Ctrl+E)\n"
            "Save the current Working spectrum (single, Mean, or Sum) as .txt."
        )
        self.act_export_spectrum.triggered.connect(self.export_working_spectrum)

        self.act_bulk_match = QAction("Bulk match checked…", self)
        self.act_bulk_match.setToolTip(
            "Bulk match checked\nRun Find peaks + match on each checked spectrum."
        )
        self.act_bulk_match.triggered.connect(self.bulk_match_checked)

        self.act_bulk_quant = QAction("Quant selected…", self)
        self.act_bulk_quant.setToolTip(
            "Quant selected\nApply Calibrate-tab CRM curves to checked "
            "(or highlighted) spectra."
        )
        self.act_bulk_quant.triggered.connect(self.quant_selected_spectra)

        self.act_match = QAction(_icon_match(), "Find peaks + match", self)
        self.act_match.setToolTip(
            "Find peaks + match\nDetect peaks and match them to the NIST line library.\n"
            "Pins Overlay elements on the plot (default: top 5 by confidence)."
        )
        self.act_match.triggered.connect(self.run_match)

        self.act_add_peak = QAction(_icon_add_peak(), "Add peak", self)
        self.act_add_peak.setShortcut(QKeySequence("A"))
        self.act_add_peak.setToolTip(
            "Add peak at selection (A)\n"
            "Add a manual peak at the blue line / last click.\n"
            "Also: Shift+click or right-click on the plot."
        )
        self.act_add_peak.triggered.connect(self.add_peak_at_selection)

        self.act_remove_manual = QAction(_icon_remove_peak(), "Remove manual peak", self)
        self.act_remove_manual.setShortcut(QKeySequence.Delete)
        self.act_remove_manual.setToolTip(
            "Remove nearest manual peak (Delete)\n"
            "Removes the manual peak closest to the selection / click."
        )
        self.act_remove_manual.triggered.connect(self.remove_nearest_manual_peak)

        self.act_clear_manual = QAction("Clear manual peaks", self)
        self.act_clear_manual.setToolTip("Clear all manually added peaks.")
        self.act_clear_manual.triggered.connect(self.clear_manual_peaks)

        self.act_clear = QAction(_icon_clear(), "Clear selection", self)
        self.act_clear.setToolTip("Clear selection\nClear element selection and click marker.")
        self.act_clear.triggered.connect(self.clear_selection)

        self.act_reload_lib = QAction(_icon_reload(), "Reload NIST library", self)
        self.act_reload_lib.setToolTip("Reload NIST library\nRe-read libs_line_library.csv from disk.")
        self.act_reload_lib.triggered.connect(self._load_library)

        self.act_calib = QAction("Go to Calibrate tab", self)
        self.act_calib.triggered.connect(self._show_calibrate_tab)
        self.act_atm = QAction("Atmosphere notes…", self)
        self.act_atm.triggered.connect(
            lambda: QMessageBox.information(self, "Argon atmosphere", ATMOSPHERE_NOTE)
        )

        self.act_report = QAction(_icon_report(), "Export report…", self)
        self.act_report.setShortcut(QKeySequence("Ctrl+R"))
        self.act_report.setToolTip(
            "Export publication report… (Ctrl+R)\n"
            "PDF/PNG figures for the active Working spectrum only\n"
            "(use Prev/Next or View to choose which loaded file)."
        )
        self.act_report.triggered.connect(self.export_publication_report)

        self.act_line_preview = QAction(_icon_preview(), "Line preview", self)
        self.act_line_preview.setShortcut(QKeySequence("Ctrl+L"))
        self.act_line_preview.setToolTip(
            "Top-5 line preview (Ctrl+L)\n"
            "Zoomed panels for the 5 most intense matched lines of the selected element.\n"
            "Also: double-click an element in the ranking table."
        )
        self.act_line_preview.triggered.connect(self.show_line_preview)

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.act_open)
        file_menu.addAction(self.act_add_spectra)
        file_menu.addSeparator()
        file_menu.addAction(self.act_mean_spectra)
        file_menu.addAction(self.act_sum_spectra)
        file_menu.addAction(self.act_export_sum)
        file_menu.addAction(self.act_export_spectrum)
        file_menu.addSeparator()
        file_menu.addAction(self.act_bulk_match)
        file_menu.addAction(self.act_bulk_quant)
        file_menu.addSeparator()
        file_menu.addAction(self.act_reload_lib)
        file_menu.addSeparator()
        file_menu.addAction(self.act_report)
        file_menu.addSeparator()
        file_menu.addAction("Quit", self.close)

        analysis = self.menuBar().addMenu("&Analysis")
        analysis.addAction(self.act_match)
        analysis.addSeparator()
        analysis.addAction(self.act_bulk_match)
        analysis.addAction(self.act_bulk_quant)
        analysis.addSeparator()
        analysis.addAction(self.act_add_peak)
        analysis.addAction(self.act_remove_manual)
        analysis.addAction(self.act_clear_manual)
        analysis.addSeparator()
        analysis.addAction(self.act_line_preview)
        analysis.addAction(self.act_report)
        analysis.addSeparator()
        analysis.addAction(self.act_clear)

        calib_menu = self.menuBar().addMenu("&Calibrate")
        calib_menu.addAction(self.act_calib)
        calib_menu.addAction(self.act_atm)

    def _show_calibrate_tab(self) -> None:
        if hasattr(self, "tabs"):
            self.tabs.setCurrentWidget(self.calibrate_tab)
            self._sync_calibrate_context()

    @staticmethod
    def _toolbar_group_label(text: str) -> QLabel:
        lab = QLabel(text)
        lab.setStyleSheet("color: #666; font-size: 11px; padding: 0 2px 0 6px;")
        return lab

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main")
        tb.setObjectName("mainToolBar")
        tb.setMovable(False)
        tb.setIconSize(QSize(22, 22))
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        tb.setStyleSheet(
            "QToolBar#mainToolBar { spacing: 4px; padding: 2px 4px; }"
            "QToolBar#mainToolBar QToolButton { padding: 3px; margin: 0px; }"
            "QToolBar#mainToolBar QToolButton:hover { background: #e8eef5; border-radius: 4px; }"
        )
        self.addToolBar(tb)

        # File
        tb.addWidget(self._toolbar_group_label("File"))
        tb.addAction(self.act_open)
        tb.addAction(self.act_reload_lib)

        tb.addSeparator()

        # Peaks / match
        tb.addWidget(self._toolbar_group_label("Peaks"))
        tb.addAction(self.act_match)
        tb.addAction(self.act_add_peak)
        tb.addAction(self.act_remove_manual)
        tb.addAction(self.act_clear)

        tb.addSeparator()

        # View / export
        tb.addWidget(self._toolbar_group_label("View"))
        tb.addAction(self.act_line_preview)
        tb.addAction(self.act_report)

        tb.addSeparator()

        # Match parameters (compact)
        tb.addWidget(self._toolbar_group_label("Params"))

        tol_lab = QLabel("Tol")
        tol_lab.setToolTip("Wavelength match tolerance (nm)")
        tol_lab.setStyleSheet("color: #444; padding-left: 2px;")
        tb.addWidget(tol_lab)
        self.spin_tol = QDoubleSpinBox()
        self.spin_tol.setRange(0.02, 0.50)
        self.spin_tol.setSingleStep(0.01)
        self.spin_tol.setDecimals(3)
        self.spin_tol.setValue(0.12)
        self.spin_tol.setSuffix(" nm")
        self.spin_tol.setToolTip("Wavelength match tolerance (nm)")
        self.spin_tol.setMaximumWidth(92)
        tb.addWidget(self.spin_tol)

        prom_lab = QLabel("Prom")
        prom_lab.setToolTip("Peak prominence as a fraction of max intensity")
        prom_lab.setStyleSheet("color: #444; padding-left: 4px;")
        tb.addWidget(prom_lab)
        self.spin_prom = QDoubleSpinBox()
        self.spin_prom.setRange(0.001, 0.20)
        self.spin_prom.setSingleStep(0.001)
        self.spin_prom.setDecimals(3)
        self.spin_prom.setValue(0.015)
        self.spin_prom.setToolTip("Peak prominence fraction (of max intensity)")
        self.spin_prom.setMaximumWidth(78)
        tb.addWidget(self.spin_prom)

        atm_lab = QLabel("Atm")
        atm_lab.setToolTip("Measurement atmosphere tag (air / argon / unknown)")
        atm_lab.setStyleSheet("color: #444; padding-left: 4px;")
        tb.addWidget(atm_lab)
        self.combo_atm = QComboBox()
        self.combo_atm.addItems(["air", "argon", "unknown"])
        self.combo_atm.setToolTip("Atmosphere tag — recorded with plots/reports")
        self.combo_atm.currentTextChanged.connect(self._on_atmosphere_change)
        self.combo_atm.setMaximumWidth(96)
        tb.addWidget(self.combo_atm)

        ov_lab = QLabel("Overlay")
        ov_lab.setToolTip(
            "Which ranked elements are pinned on the plot after Match.\n"
            "Ranking table still lists all hits. Double-click to add/remove."
        )
        ov_lab.setStyleSheet("color: #444; padding-left: 4px;")
        tb.addWidget(ov_lab)
        self.combo_overlay = QComboBox()
        self.combo_overlay.addItem("Top 5", "top:5")
        self.combo_overlay.addItem("None", "none")
        self.combo_overlay.addItem("All ranked", "all")
        self.combo_overlay.setCurrentIndex(0)
        self.combo_overlay.setToolTip(
            "Plot overlays after Match (default: top 5 by confidence).\n"
            "Changing this re-applies pins from the current ranking — no re-match.\n"
            "Double-click periodic table / ranking to add or remove; Unpin clears."
        )
        self.combo_overlay.setMaximumWidth(110)
        self.combo_overlay.currentIndexChanged.connect(self._on_overlay_mode_changed)
        tb.addWidget(self.combo_overlay)

    def _build_central(self) -> None:
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # ---- Identify tab (existing spectrum explorer) ----
        identify = QWidget()
        identify_layout = QVBoxLayout(identify)
        identify_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        identify_layout.addWidget(splitter)

        # ---- Left: spectra file management ----
        spectra_pane = QWidget()
        spectra_layout = QVBoxLayout(spectra_pane)
        spectra_layout.setContentsMargins(6, 6, 6, 6)

        spectra_hdr = QHBoxLayout()
        spectra_hdr.addWidget(QLabel("<b>Spectra</b>"))
        spectra_hdr.addStretch(1)
        self.spectra_count_label = QLabel("")
        self.spectra_count_label.setStyleSheet("color: #666;")
        spectra_hdr.addWidget(self.spectra_count_label)
        spectra_layout.addLayout(spectra_hdr)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode"))
        self.combo_display_mode = QComboBox()
        self.combo_display_mode.addItem("Single", "single")
        self.combo_display_mode.addItem("Waterfall", "waterfall")
        self.combo_display_mode.addItem("Working only", "working")
        self.combo_display_mode.setToolTip(
            "Single: one file with Prev/Next\n"
            "Waterfall: offset traces for checked files (adjust Offset %)\n"
            "Working only: active spectrum (or Mean/Sum result)"
        )
        self.combo_display_mode.currentIndexChanged.connect(self._on_display_mode_changed)
        mode_row.addWidget(self.combo_display_mode, stretch=1)
        spectra_layout.addLayout(mode_row)

        wf_row = QHBoxLayout()
        wf_row.addWidget(QLabel("Offset"))
        self.spin_waterfall_offset = QDoubleSpinBox()
        self.spin_waterfall_offset.setRange(0.0, 300.0)
        self.spin_waterfall_offset.setSingleStep(5.0)
        self.spin_waterfall_offset.setDecimals(0)
        self.spin_waterfall_offset.setSuffix(" %")
        self.spin_waterfall_offset.setValue(self._waterfall_offset_frac * 100.0)
        self.spin_waterfall_offset.setToolTip(
            "Waterfall vertical gap between traces, as a percent of the\n"
            "maximum intensity among the plotted spectra.\n"
            "0% = overlaid; larger values spread the stack apart."
        )
        self.spin_waterfall_offset.valueChanged.connect(self._on_waterfall_offset_changed)
        wf_row.addWidget(self.spin_waterfall_offset, stretch=1)
        spectra_layout.addLayout(wf_row)
        self._update_waterfall_offset_enabled()

        nav_row = QHBoxLayout()
        self.btn_prev_spec = QPushButton("◀")
        self.btn_prev_spec.setFixedWidth(32)
        self.btn_prev_spec.setToolTip("Previous spectrum (Single mode)")
        self.btn_prev_spec.clicked.connect(self._prev_spectrum)
        nav_row.addWidget(self.btn_prev_spec)
        self.spectrum_index_label = QLabel("— / —")
        self.spectrum_index_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spectrum_index_label.setToolTip("Current spectrum index in Single mode")
        nav_row.addWidget(self.spectrum_index_label, stretch=1)
        self.btn_next_spec = QPushButton("▶")
        self.btn_next_spec.setFixedWidth(32)
        self.btn_next_spec.setToolTip("Next spectrum (Single mode)")
        self.btn_next_spec.clicked.connect(self._next_spectrum)
        nav_row.addWidget(self.btn_next_spec)
        spectra_layout.addLayout(nav_row)

        self.spectra_list = QListWidget()
        self.spectra_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.spectra_list.setToolTip(
            "Check spectra for Mean/Sum, Bulk match, and Quant.\n"
            "Double-click to view one shot (Single mode).\n"
            "Drag-and-drop .txt files or a folder here."
        )
        self.spectra_list.itemDoubleClicked.connect(self._on_spectrum_item_activated)
        self.spectra_list.itemChanged.connect(self._on_spectrum_item_changed)
        spectra_layout.addWidget(self.spectra_list, stretch=1)

        sel_row = QHBoxLayout()
        btn_sel_all = QPushButton("Select all")
        btn_sel_all.setToolTip("Check all spectra in the list.")
        btn_sel_all.clicked.connect(self.select_all_spectra)
        sel_row.addWidget(btn_sel_all)
        btn_sel_none = QPushButton("Deselect all")
        btn_sel_none.setToolTip("Uncheck all spectra in the list.")
        btn_sel_none.clicked.connect(self.deselect_all_spectra)
        sel_row.addWidget(btn_sel_none)
        spectra_layout.addLayout(sel_row)

        # 2×2 so labels stay readable in the narrow left pane
        spec_grid = QGridLayout()
        spec_grid.setSpacing(4)
        btn_mean = QPushButton("Mean")
        btn_mean.setToolTip(
            "Average checked spectra (recommended for multi-spot / focus variation)."
        )
        btn_mean.clicked.connect(lambda: self.combine_checked_spectra("mean"))
        spec_grid.addWidget(btn_mean, 0, 0)

        btn_sum = QPushButton("Sum")
        btn_sum.setToolTip("Sum checked spectra (scales with N).")
        btn_sum.clicked.connect(lambda: self.combine_checked_spectra("sum"))
        spec_grid.addWidget(btn_sum, 0, 1)

        btn_view = QPushButton("View")
        btn_view.setToolTip("View highlighted list item in Single mode.")
        btn_view.clicked.connect(self.activate_highlighted_spectrum)
        spec_grid.addWidget(btn_view, 1, 0)

        btn_rm = QPushButton("Remove")
        btn_rm.setToolTip("Remove highlighted spectra from the list.")
        btn_rm.clicked.connect(self.remove_highlighted_spectra)
        spec_grid.addWidget(btn_rm, 1, 1)
        spectra_layout.addLayout(spec_grid)

        export_btns = QHBoxLayout()
        btn_export_sum = QPushButton("Export sum…")
        btn_export_sum.setToolTip(
            "Sum checked spectra and save as a .txt file\n"
            "(wavelength TAB intensity; reloadable in this app)."
        )
        btn_export_sum.clicked.connect(lambda: self.export_combined_spectra("sum"))
        export_btns.addWidget(btn_export_sum)
        btn_export_work = QPushButton("Export working…")
        btn_export_work.setToolTip(
            "Save the current Working spectrum as .txt\n"
            "(after Sum/Mean, or the active single file)."
        )
        btn_export_work.clicked.connect(self.export_working_spectrum)
        export_btns.addWidget(btn_export_work)
        spectra_layout.addLayout(export_btns)

        bulk_btns = QHBoxLayout()
        btn_bmatch = QPushButton("Match")
        btn_bmatch.setToolTip("Find peaks + match each checked spectrum.")
        btn_bmatch.clicked.connect(self.bulk_match_checked)
        bulk_btns.addWidget(btn_bmatch)
        btn_quant = QPushButton("Quant")
        btn_quant.setToolTip(
            "Quantify checked spectra with Calibrate-tab CRM curves.\n"
            "If none are checked, uses the highlighted list selection."
        )
        btn_quant.clicked.connect(self.quant_selected_spectra)
        bulk_btns.addWidget(btn_quant)
        spectra_layout.addLayout(bulk_btns)

        self.working_label = QLabel("Working: —")
        self.working_label.setStyleSheet("color: #333; font-size: 11px;")
        self.working_label.setWordWrap(True)
        spectra_layout.addWidget(self.working_label)

        splitter.addWidget(spectra_pane)

        # ---- Center: plot pane ----
        plot_wrap = QWidget()
        plot_layout = QVBoxLayout(plot_wrap)
        plot_layout.setContentsMargins(4, 4, 4, 4)
        self.canvas = MplCanvas()
        self.toolbar = NavigationToolbar2QT(self.canvas, plot_wrap)
        # Matplotlib's Qt toolbar defaults are oversized; shrink icons ~50%.
        icon = self.toolbar.iconSize()
        self.toolbar.setIconSize(
            QSize(max(12, icon.width() // 2), max(12, icon.height() // 2))
        )
        self.toolbar.setStyleSheet(
            "QToolBar { spacing: 2px; padding: 1px; }"
            "QToolButton { padding: 1px; margin: 0px; }"
        )
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.ax = self.canvas.ax
        self.ax_sticks = self.canvas.ax_sticks
        self._suspend_xlim_cb = False
        self.ax.callbacks.connect("xlim_changed", self._on_xlim_changed)
        self.ax.tick_params(labelbottom=False)
        self.ax.set_xlabel("")
        self._format_intensity_axis(ylabel="Intensity (counts)")
        self.ax.set_title("No spectrum loaded")
        self.ax_sticks.set_xlabel("Wavelength (nm)")
        self.ax_sticks.text(
            0.5,
            0.5,
            "Select element(s) to preview NIST LIBS lines",
            transform=self.ax_sticks.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color="#777",
        )
        splitter.addWidget(plot_wrap)

        # ---- Right: ranking / results / measurement ----
        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(6, 6, 6, 6)

        rank_row = QHBoxLayout()
        rank_row.addWidget(QLabel("<b>Element ranking</b>"))
        score_btn = QLabel('<a href="#">What is Confidence?</a>')
        score_btn.setOpenExternalLinks(False)
        score_btn.linkActivated.connect(
            lambda _=None: QMessageBox.information(self, "What Confidence means", SCORE_HELP)
        )
        rank_row.addStretch(1)
        rank_row.addWidget(score_btn)
        side_layout.addLayout(rank_row)

        self.elem_table = QTableWidget(0, 3)
        self.elem_table.setHorizontalHeaderLabels(["Element", "#peaks", "Conf %"])
        hdr = self.elem_table.horizontalHeaderItem(2)
        if hdr is not None:
            hdr.setToolTip(SCORE_HELP)
        self.elem_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.elem_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.elem_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.elem_table.verticalHeader().setVisible(False)
        self.elem_table.horizontalHeader().setStretchLastSection(True)
        self.elem_table.setToolTip(
            "Select one or more elements to preview NIST sticks under the spectrum.\n"
            "Cmd/Ctrl-click or Shift-click to multi-select.\n"
            "Double-click an element for a top-5 matched-line preview window."
        )
        self.elem_table.itemSelectionChanged.connect(self._on_element_select)
        self.elem_table.cellDoubleClicked.connect(self._on_element_double_click)
        side_layout.addWidget(self.elem_table, stretch=2)

        self.side_tabs = QTabWidget()
        side_layout.addWidget(self.side_tabs, stretch=3)

        # ---- Results tab (click candidates / matched lines) ----
        results_page = QWidget()
        results_layout = QVBoxLayout(results_page)
        results_layout.setContentsMargins(2, 4, 2, 2)
        self.cand_label = QLabel("<b>Peak candidates (click plot)</b>")
        results_layout.addWidget(self.cand_label)
        self.cand_table = QTableWidget(0, 5)
        self.cand_table.setHorizontalHeaderLabels(["Species", "λ NIST", "Δλ", "NIST I", "Aki"])
        self.cand_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.cand_table.verticalHeader().setVisible(False)
        self.cand_table.horizontalHeader().setStretchLastSection(True)
        results_layout.addWidget(self.cand_table)
        self.side_tabs.addTab(results_page, "Results")

        # ---- Browse NIST tab (any element, keeps match results) ----
        browse_page = QWidget()
        browse_layout = QVBoxLayout(browse_page)
        browse_layout.setContentsMargins(2, 4, 2, 2)

        self.periodic_table = PeriodicTableWidget()
        self.periodic_table.setToolTip(
            "Click an element to view its NIST lines.\n"
            "Double-click to pin / unpin on the plot (syncs with matched list).\n"
            "After Match, only Overlay (default top 5) are pinned — add more here."
        )
        self.periodic_table.elementViewed.connect(self._on_browse_element_viewed)
        self.periodic_table.elementPinToggled.connect(self._on_browse_element_pin_toggled)
        browse_layout.addWidget(self.periodic_table, alignment=Qt.AlignmentFlag.AlignHCenter)

        pt_hint = QLabel("Click = view · double-click = pin on plot (Overlay starts at top 5)")
        pt_hint.setStyleSheet("color: #666; font-size: 11px;")
        pt_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        browse_layout.addWidget(pt_hint)

        browse_filters = QHBoxLayout()
        browse_filters.addWidget(QLabel("El"))
        self.combo_browse_el = QComboBox()
        self.combo_browse_el.setEditable(True)
        self.combo_browse_el.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.combo_browse_el.setMinimumWidth(72)
        self.combo_browse_el.setToolTip(
            "Browse NIST lines for any element in the spectrum range.\n"
            "Does not clear Find peaks + match results.\n"
            "Or click the periodic table above."
        )
        self.combo_browse_el.currentTextChanged.connect(self._on_browse_filters_changed)
        browse_filters.addWidget(self.combo_browse_el)

        browse_filters.addWidget(QLabel("Ion"))
        self.combo_browse_ion = QComboBox()
        self.combo_browse_ion.addItem("All", None)
        self.combo_browse_ion.addItem("I", 1)
        self.combo_browse_ion.addItem("II", 2)
        self.combo_browse_ion.addItem("III", 3)
        self.combo_browse_ion.setToolTip("Ion stage filter")
        self.combo_browse_ion.currentIndexChanged.connect(self._on_browse_filters_changed)
        browse_filters.addWidget(self.combo_browse_ion)

        self.btn_clear_pins = QPushButton("Unpin")
        self.btn_clear_pins.setToolTip(
            "Clear all plot overlays (periodic pins + ranking selection).\n"
            "Match only auto-pins Overlay (default top 5); use this to clear them."
        )
        self.btn_clear_pins.setMaximumWidth(52)
        self.btn_clear_pins.clicked.connect(self._clear_browse_pins)
        browse_filters.addWidget(self.btn_clear_pins)
        browse_layout.addLayout(browse_filters)

        browse_opts = QHBoxLayout()
        self.combo_browse_scope = QComboBox()
        self.combo_browse_scope.addItem("Visible λ window", "visible")
        self.combo_browse_scope.addItem("Full spectrum", "full")
        self.combo_browse_scope.setToolTip(
            "Visible = current zoomed wavelength range on the plot.\n"
            "Full = entire loaded spectrum range."
        )
        self.combo_browse_scope.currentIndexChanged.connect(self._on_browse_filters_changed)
        browse_opts.addWidget(self.combo_browse_scope, stretch=1)

        browse_opts.addWidget(QLabel("Max"))
        self.spin_browse_max = QSpinBox()
        self.spin_browse_max.setRange(10, 300)
        self.spin_browse_max.setValue(60)
        self.spin_browse_max.setToolTip("Maximum NIST lines to list")
        self.spin_browse_max.valueChanged.connect(self._on_browse_filters_changed)
        browse_opts.addWidget(self.spin_browse_max)
        browse_layout.addLayout(browse_opts)

        self.browse_label = QLabel("Open a spectrum to browse NIST lines.")
        self.browse_label.setStyleSheet("color: #555;")
        self.browse_label.setWordWrap(True)
        browse_layout.addWidget(self.browse_label)

        self.browse_table = QTableWidget(0, 4)
        self.browse_table.setHorizontalHeaderLabels(["Species", "λ NIST", "NIST I", "Aki"])
        self.browse_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.browse_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.browse_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.browse_table.verticalHeader().setVisible(False)
        self.browse_table.horizontalHeader().setStretchLastSection(True)
        self.browse_table.setToolTip("Click a row to mark that wavelength on the spectrum.")
        self.browse_table.itemSelectionChanged.connect(self._on_browse_line_select)
        browse_layout.addWidget(self.browse_table)

        self.side_tabs.addTab(browse_page, "Browse NIST")

        # ---- Batch tab (bulk quant results + C vs spectrum #) ----
        batch_page = QWidget()
        batch_layout = QVBoxLayout(batch_page)
        batch_layout.setContentsMargins(2, 4, 2, 2)
        self.batch_label = QLabel("Run Quant on selected spectra to fill this table and plot.")
        self.batch_label.setStyleSheet("color: #555;")
        self.batch_label.setWordWrap(True)
        batch_layout.addWidget(self.batch_label)

        batch_plot_row = QHBoxLayout()
        batch_plot_row.addWidget(QLabel("Plot"))
        self.combo_batch_el = QComboBox()
        self.combo_batch_el.setToolTip("Element(s) to show on concentration vs spectrum #")
        self.combo_batch_el.currentIndexChanged.connect(self._refresh_batch_plot)
        batch_plot_row.addWidget(self.combo_batch_el, stretch=1)
        btn_csv = QPushButton("CSV…")
        btn_csv.setToolTip("Export quant table to CSV")
        btn_csv.clicked.connect(self.export_bulk_quant_csv)
        batch_plot_row.addWidget(btn_csv)
        batch_layout.addLayout(batch_plot_row)

        self.batch_canvas = BatchPlotCanvas()
        batch_layout.addWidget(self.batch_canvas, stretch=2)

        self.batch_table = QTableWidget(0, 2)
        self.batch_table.setHorizontalHeaderLabels(["#", "File"])
        self.batch_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.batch_table.verticalHeader().setVisible(False)
        self.batch_table.horizontalHeader().setStretchLastSection(True)
        batch_layout.addWidget(self.batch_table, stretch=2)
        self.side_tabs.addTab(batch_page, "Batch")

        self.side_tabs.currentChanged.connect(self._on_side_tab_changed)

        side_layout.addWidget(QLabel("<b>Measurement</b>"))
        self.meta_text = QTextEdit()
        self.meta_text.setReadOnly(True)
        self.meta_text.setMaximumHeight(100)
        self.meta_text.setPlainText("No file loaded.")
        side_layout.addWidget(self.meta_text)

        splitter.addWidget(side)
        spectra_pane.setMinimumWidth(240)
        side.setMinimumWidth(360)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 2)
        # Spectra | plot | results
        splitter.setSizes([280, 780, 420])
        self._spectra_pane = spectra_pane
        self._identify_splitter = splitter

        self.tabs.addTab(identify, "Identify")

        # ---- Calibrate tab ----
        self.calibrate_tab = CalibrationTab()
        self.calibrate_tab.statusMessage.connect(self.statusBar().showMessage)
        self.tabs.addTab(self.calibrate_tab, "Calibrate")
        self.tabs.currentChanged.connect(self._on_tab_changed)

    # --------------------------------------------------------------- data
    def _load_library(self) -> None:
        if not DEFAULT_LIBRARY.exists():
            self.library = []
            self.calibrate_tab.set_library([])
            self.statusBar().showMessage(f"NIST library missing: {DEFAULT_LIBRARY}")
            return
        self.library = load_line_library(DEFAULT_LIBRARY)
        self.calibrate_tab.set_library(self.library)
        self._populate_browse_elements()
        self._refresh_browse_table()
        self.statusBar().showMessage(f"Loaded NIST library: {len(self.library)} lines")

    def _sync_calibrate_context(self) -> None:
        self.calibrate_tab.set_identify_context(
            self.spectrum,
            self.hits,
            self.combo_atm.currentText(),
        )

    def _on_tab_changed(self, index: int) -> None:
        if index == self.tabs.indexOf(self.calibrate_tab):
            self._sync_calibrate_context()

    def browse_spectra(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open LIBS spectra",
            str(ROOT / "docs"),
            "Spectrum text (*.txt);;All files (*)",
        )
        if paths:
            self.open_spectra([Path(p) for p in paths], replace=True)

    def browse_add_spectra(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add LIBS spectra",
            str(ROOT / "docs"),
            "Spectrum text (*.txt);;All files (*)",
        )
        if paths:
            self.open_spectra([Path(p) for p in paths], replace=False)

    # ---------------------------------------------------------- drag & drop
    def _enable_spectrum_drag_drop(self) -> None:
        """Accept spectrum file/folder drops on the window and main panes."""
        self.setAcceptDrops(True)
        targets: list[QWidget] = [self]
        central = self.centralWidget()
        if central is not None:
            targets.append(central)
        for name in ("canvas", "spectra_list", "tabs", "meta_text"):
            w = getattr(self, name, None)
            if isinstance(w, QWidget):
                targets.append(w)
        pane = getattr(self, "_spectra_pane", None)
        if isinstance(pane, QWidget):
            targets.append(pane)
        for w in targets:
            w.setAcceptDrops(True)
            w.installEventFilter(self)

    @staticmethod
    def _spectrum_paths_from_urls(urls: list[QUrl]) -> list[Path]:
        """Resolve dropped URLs to .txt spectrum paths (files + folder contents)."""
        found: list[Path] = []
        seen: set[Path] = set()

        def _add(path: Path) -> None:
            key = path.resolve()
            if key in seen:
                return
            seen.add(key)
            found.append(path)

        for url in urls:
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.is_file() and path.suffix.lower() == ".txt":
                _add(path)
            elif path.is_dir():
                # Immediate .txt children (sorted); skip nested trees to stay predictable
                for child in sorted(path.iterdir()):
                    if child.is_file() and child.suffix.lower() == ".txt":
                        _add(child)
        return found

    def _spectrum_paths_from_mime(self, mime) -> list[Path]:
        if mime is None or not mime.hasUrls():
            return []
        return self._spectrum_paths_from_urls(list(mime.urls()))

    def eventFilter(self, watched, event):  # noqa: N802 — Qt API
        et = event.type()
        if et in (QEvent.Type.DragEnter, QEvent.Type.DragMove):
            if isinstance(event, (QDragEnterEvent, QDragMoveEvent)):
                if self._spectrum_paths_from_mime(event.mimeData()):
                    event.acceptProposedAction()
                    return True
                event.ignore()
                return True
        if et == QEvent.Type.Drop and isinstance(event, QDropEvent):
            paths = self._spectrum_paths_from_mime(event.mimeData())
            if paths:
                event.acceptProposedAction()
                self._handle_dropped_spectra(paths)
                return True
            event.ignore()
            return True
        return super().eventFilter(watched, event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self._spectrum_paths_from_mime(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if self._spectrum_paths_from_mime(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = self._spectrum_paths_from_mime(event.mimeData())
        if not paths:
            event.ignore()
            return
        event.acceptProposedAction()
        self._handle_dropped_spectra(paths)

    def _handle_dropped_spectra(self, paths: list[Path]) -> None:
        if not paths:
            return
        replace = True
        if self.loaded_spectra:
            box = QMessageBox(self)
            box.setWindowTitle("Load dropped spectra")
            n = len(paths)
            box.setText(
                f"Load {n} spectrum file{'s' if n != 1 else ''}?"
            )
            box.setInformativeText(
                "Replace the current list, or add these files to it?"
            )
            btn_replace = box.addButton("Replace", QMessageBox.ButtonRole.AcceptRole)
            btn_add = box.addButton("Add", QMessageBox.ButtonRole.ActionRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.exec()
            clicked = box.clickedButton()
            if clicked is btn_replace:
                replace = True
            elif clicked is btn_add:
                replace = False
            else:
                return
        self.open_spectra(paths, replace=replace)

    def open_spectrum(self, path: Path) -> None:
        """Convenience wrapper used by CLI / single-file callers."""
        self.open_spectra([path], replace=True)

    def open_spectra(self, paths: list[Path], *, replace: bool = True) -> None:
        loaded: list[Spectrum] = []
        errors: list[str] = []
        for path in paths:
            try:
                loaded.append(load_spectrum(path))
            except Exception as exc:
                errors.append(f"{Path(path).name}: {exc}")
        if errors:
            QMessageBox.warning(
                self,
                "Load warning",
                "Some files could not be loaded:\n\n" + "\n".join(errors[:12]),
            )
        if not loaded:
            if paths:
                QMessageBox.critical(self, "Load error", "No spectra were loaded.")
            return

        if replace:
            self.loaded_spectra = loaded
        else:
            # Avoid duplicate paths
            existing = {s.meta.path.resolve() for s in self.loaded_spectra}
            for spec in loaded:
                key = spec.meta.path.resolve()
                if key in existing:
                    continue
                self.loaded_spectra.append(spec)
                existing.add(key)

        self._fill_spectra_list(check_all=True)
        self._match_cache.clear()
        self._bulk_quant_results = []
        self._refresh_batch_results_ui()

        self._spectrum_index = 0
        self._set_display_mode("single", redraw=False)
        self._activate_loaded_at_index(0, restore_cache=False, reset_view=True)

        n = len(self.loaded_spectra)
        if n == 1:
            self.statusBar().showMessage(
                f"Loaded {self.loaded_spectra[0].meta.path.name} — "
                "Find peaks + match, or Shift/right-click to add weak peaks"
            )
        else:
            self.statusBar().showMessage(
                f"Loaded {n} spectra — use Prev/Next, Waterfall, Mean/Sum, "
                "or Bulk match/quant on checked files."
            )

    def _fill_spectra_list(self, *, check_all: bool = False) -> None:
        if not hasattr(self, "spectra_list"):
            return
        checked_names: set[str] = set()
        if not check_all:
            for i in range(self.spectra_list.count()):
                item = self.spectra_list.item(i)
                if item is None:
                    continue
                # Strip match suffix for name matching
                raw = item.data(Qt.ItemDataRole.UserRole)
                key = Path(raw).name if raw else item.text().split("  ·")[0]
                if item.checkState() == Qt.CheckState.Checked:
                    checked_names.add(key)

        self.spectra_list.blockSignals(True)
        self.spectra_list.clear()
        for i, spec in enumerate(self.loaded_spectra):
            name = spec.meta.path.name
            matched = name in self._match_cache
            label = f"{i + 1}. {name}" + ("  · matched" if matched else "")
            item = QListWidgetItem(label)
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEnabled
            )
            if check_all or name in checked_names or not checked_names:
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setCheckState(Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, str(spec.meta.path))
            tip = name
            if matched:
                tip += f"\nCached match: {len(self._match_cache[name].hits)} elements"
            item.setToolTip(tip)
            self.spectra_list.addItem(item)
        self.spectra_list.blockSignals(False)

        n = len(self.loaded_spectra)
        self.spectra_count_label.setText(f"{n} file{'s' if n != 1 else ''}")
        self._update_spectrum_nav_label()

    def _checked_spectra(self) -> list[Spectrum]:
        if not hasattr(self, "spectra_list"):
            return list(self.loaded_spectra)
        by_path = {str(s.meta.path.resolve()): s for s in self.loaded_spectra}
        by_name = {s.meta.path.name: s for s in self.loaded_spectra}
        out: list[Spectrum] = []
        for i in range(self.spectra_list.count()):
            item = self.spectra_list.item(i)
            if item is None or item.checkState() != Qt.CheckState.Checked:
                continue
            raw = item.data(Qt.ItemDataRole.UserRole)
            spec = None
            if raw:
                try:
                    spec = by_path.get(str(Path(raw).resolve()))
                except Exception:
                    spec = None
            if spec is None:
                name = item.text().split("  ·")[0]
                if ". " in name:
                    name = name.split(". ", 1)[-1]
                spec = by_name.get(name)
            if spec is not None:
                out.append(spec)
        return out

    def _set_display_mode(self, mode: str, *, redraw: bool = True) -> None:
        if mode not in ("single", "waterfall", "working"):
            mode = "single"
        self._display_mode = mode
        if hasattr(self, "combo_display_mode"):
            idx = self.combo_display_mode.findData(mode)
            if idx >= 0:
                self.combo_display_mode.blockSignals(True)
                self.combo_display_mode.setCurrentIndex(idx)
                self.combo_display_mode.blockSignals(False)
        single = mode == "single"
        if hasattr(self, "btn_prev_spec"):
            self.btn_prev_spec.setEnabled(single and len(self.loaded_spectra) > 1)
            self.btn_next_spec.setEnabled(single and len(self.loaded_spectra) > 1)
        self._update_spectrum_nav_label()
        self._update_waterfall_offset_enabled()
        if redraw:
            self._redraw(reset_view=False, preserve_ylim=False)

    def _update_waterfall_offset_enabled(self) -> None:
        if hasattr(self, "spin_waterfall_offset"):
            self.spin_waterfall_offset.setEnabled(self._display_mode == "waterfall")

    def _on_waterfall_offset_changed(self, value: float) -> None:
        self._waterfall_offset_frac = max(0.0, float(value) / 100.0)
        if self._display_mode != "waterfall":
            return
        # Keep wavelength zoom; re-autoscale Y so the new stack fits
        saved_xlim = None
        if hasattr(self, "ax"):
            saved_xlim = self.ax.get_xlim()
        self._redraw(reset_view=True)
        if saved_xlim is not None:
            self.ax.set_xlim(saved_xlim)
            self.canvas.draw_idle()

    def _on_display_mode_changed(self, _index: int = 0) -> None:
        mode = self.combo_display_mode.currentData()
        self._display_mode = str(mode or "single")
        if self._display_mode == "single" and self.loaded_spectra:
            self._activate_loaded_at_index(
                self._spectrum_index, restore_cache=True, reset_view=False
            )
        else:
            self._set_display_mode(self._display_mode, redraw=True)

    def _update_spectrum_nav_label(self) -> None:
        if not hasattr(self, "spectrum_index_label"):
            return
        n = len(self.loaded_spectra)
        if n == 0:
            self.spectrum_index_label.setText("— / —")
        else:
            self.spectrum_index_label.setText(f"{self._spectrum_index + 1} / {n}")

    def _prev_spectrum(self) -> None:
        if not self.loaded_spectra or self._spectrum_index <= 0:
            return
        self._activate_loaded_at_index(
            self._spectrum_index - 1, restore_cache=True, reset_view=False
        )

    def _next_spectrum(self) -> None:
        if not self.loaded_spectra or self._spectrum_index >= len(self.loaded_spectra) - 1:
            return
        self._activate_loaded_at_index(
            self._spectrum_index + 1, restore_cache=True, reset_view=False
        )

    def _activate_loaded_at_index(
        self,
        index: int,
        *,
        restore_cache: bool = True,
        reset_view: bool = False,
    ) -> None:
        if not self.loaded_spectra:
            return
        index = max(0, min(index, len(self.loaded_spectra) - 1))
        self._spectrum_index = index
        self._display_mode = "single"
        if hasattr(self, "combo_display_mode"):
            self._set_display_mode("single", redraw=False)
        spec = self.loaded_spectra[index]
        self._set_active_spectrum(
            spec,
            label=spec.meta.path.name,
            reset_view=reset_view,
            restore_cache=restore_cache,
        )
        self._update_spectrum_nav_label()
        # Highlight list row
        if hasattr(self, "spectra_list") and index < self.spectra_list.count():
            self.spectra_list.blockSignals(True)
            self.spectra_list.setCurrentRow(index)
            self.spectra_list.blockSignals(False)

    def combine_checked_spectra(self, mode: str = "mean", *, reset_view: bool = False) -> None:
        checked = self._checked_spectra()
        if not checked:
            QMessageBox.information(
                self,
                "No spectra checked",
                "Check one or more spectra in the list, then Mean or Sum.",
            )
            return
        if len(checked) == 1:
            # Jump to that file in Single mode
            try:
                idx = self.loaded_spectra.index(checked[0])
            except ValueError:
                idx = 0
            self._activate_loaded_at_index(idx, restore_cache=True, reset_view=reset_view)
            self.statusBar().showMessage(f"Working spectrum: {checked[0].meta.path.name}")
            return
        try:
            combined = combine_spectra(checked, mode=mode)
        except Exception as exc:
            QMessageBox.critical(self, "Combine failed", str(exc))
            return
        combined = self._register_combined_spectrum(
            combined, activate=True, reset_view=reset_view
        )
        names = ", ".join(s.meta.path.stem for s in checked[:6])
        more = f" +{len(checked) - 6}" if len(checked) > 6 else ""
        self.statusBar().showMessage(
            f"Added {combined.meta.path.name} to list ({names}{more})"
        )

    def _unique_spectrum_path(self, preferred: Path) -> Path:
        """Pick a list name that does not collide with already-loaded spectra."""
        preferred = Path(preferred)
        if not preferred.suffix:
            preferred = preferred.with_suffix(".txt")
        existing_names = {s.meta.path.name for s in self.loaded_spectra}
        if preferred.name not in existing_names:
            return preferred
        stem, suffix = preferred.stem, preferred.suffix
        i = 2
        while f"{stem}_{i}{suffix}" in existing_names:
            i += 1
        return preferred.with_name(f"{stem}_{i}{suffix}")

    def _register_combined_spectrum(
        self,
        spectrum: Spectrum,
        *,
        path: Path | None = None,
        activate: bool = True,
        reset_view: bool = False,
    ) -> Spectrum:
        """Append (or replace) a Mean/Sum spectrum in the Spectra list."""
        if path is not None:
            spectrum.meta.path = Path(path)
        else:
            spectrum.meta.path = self._unique_spectrum_path(spectrum.meta.path)

        replace_idx: int | None = None
        try:
            key = spectrum.meta.path.resolve()
        except Exception:
            key = None
        for i, existing in enumerate(self.loaded_spectra):
            if existing.meta.path.name == spectrum.meta.path.name:
                replace_idx = i
                break
            if key is not None:
                try:
                    if existing.meta.path.resolve() == key:
                        replace_idx = i
                        break
                except Exception:
                    pass

        if replace_idx is not None:
            self.loaded_spectra[replace_idx] = spectrum
            idx = replace_idx
        else:
            self.loaded_spectra.append(spectrum)
            idx = len(self.loaded_spectra) - 1

        self._fill_spectra_list(check_all=False)
        if hasattr(self, "spectra_list") and idx < self.spectra_list.count():
            item = self.spectra_list.item(idx)
            if item is not None:
                self.spectra_list.blockSignals(True)
                item.setCheckState(Qt.CheckState.Checked)
                self.spectra_list.blockSignals(False)

        if activate:
            self._activate_loaded_at_index(
                idx, restore_cache=False, reset_view=reset_view
            )
        return spectrum

    def export_working_spectrum(self) -> None:
        """Save the current Working spectrum as a reloadable .txt file."""
        if self.spectrum is None:
            QMessageBox.information(self, "No spectrum", "Open or Sum/Mean spectra first.")
            return
        stem = Path(self._working_label or self.spectrum.meta.path.stem).stem
        stem = stem.replace(" ", "_")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export working spectrum",
            str(ROOT / "docs" / f"{stem}.txt"),
            "Spectrum text (*.txt);;All files (*)",
        )
        if not path:
            return
        try:
            out = write_spectrum(Path(path), self.spectrum)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        msg = f"Wrote spectrum: {out}"
        self.statusBar().showMessage(msg)
        QMessageBox.information(self, "Spectrum exported", msg)

    def export_combined_spectra(self, mode: str = "sum") -> None:
        """Combine checked spectra (sum/mean) and write a .txt file."""
        checked = self._checked_spectra()
        if not checked:
            QMessageBox.information(
                self,
                "No spectra checked",
                "Check one or more spectra in the list, then Export sum…",
            )
            return
        mode_l = mode.strip().lower()
        try:
            if len(checked) == 1:
                src = checked[0]
                combined = Spectrum(
                    wavelength_nm=np.asarray(src.wavelength_nm, dtype=float).copy(),
                    intensity=np.asarray(src.intensity, dtype=float).copy(),
                    meta=SpectrumMeta(
                        path=Path(f"{mode_l}_of_1.txt"),
                        cfg_path=src.meta.cfg_path,
                        n_conditioning_shots=src.meta.n_conditioning_shots,
                        n_accumulations=src.meta.n_accumulations,
                        laser_energy_mJ=src.meta.laser_energy_mJ,
                        qs_delay_us=src.meta.qs_delay_us,
                        integration_time_us=src.meta.integration_time_us,
                        integration_delay_us=src.meta.integration_delay_us,
                        wavelength_ranges=list(src.meta.wavelength_ranges or []),
                    ),
                )
            else:
                combined = combine_spectra(checked, mode=mode)
        except Exception as exc:
            QMessageBox.critical(self, "Combine failed", str(exc))
            return

        default_stem = f"{mode_l}_of_{len(checked)}"
        path, _ = QFileDialog.getSaveFileName(
            self,
            f"Export {mode_l} of checked spectra",
            str(ROOT / "docs" / f"{default_stem}.txt"),
            "Spectrum text (*.txt);;All files (*)",
        )
        if not path:
            return
        try:
            out = write_spectrum(Path(path), combined)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return

        self._register_combined_spectrum(combined, path=out, activate=True)
        msg = f"Wrote {mode_l} spectrum ({len(checked)} files): {out}"
        self.statusBar().showMessage(msg)
        QMessageBox.information(self, "Spectrum exported", msg)

    def activate_highlighted_spectrum(self) -> None:
        row = self.spectra_list.currentRow()
        if row < 0:
            QMessageBox.information(self, "Nothing selected", "Highlight a spectrum in the list.")
            return
        self._activate_loaded_at_index(row, restore_cache=True, reset_view=False)
        self.statusBar().showMessage(
            f"Working spectrum: {self.loaded_spectra[row].meta.path.name}"
        )

    def _on_spectrum_item_activated(self, item: QListWidgetItem) -> None:
        row = self.spectra_list.row(item)
        if row < 0:
            return
        self._activate_loaded_at_index(row, restore_cache=True, reset_view=False)

    def _on_spectrum_item_changed(self, _item: QListWidgetItem) -> None:
        if self._display_mode == "waterfall":
            self._redraw(reset_view=False, preserve_ylim=False)

    def _set_all_spectra_checked(self, checked: bool) -> None:
        if not hasattr(self, "spectra_list"):
            return
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self.spectra_list.blockSignals(True)
        for i in range(self.spectra_list.count()):
            item = self.spectra_list.item(i)
            if item is not None:
                item.setCheckState(state)
        self.spectra_list.blockSignals(False)
        if self._display_mode == "waterfall":
            self._redraw(reset_view=False, preserve_ylim=False)
        n = self.spectra_list.count()
        self.statusBar().showMessage(
            f"{'Checked' if checked else 'Unchecked'} all {n} spectrum"
            f"{'a' if n != 1 else ''}."
        )

    def select_all_spectra(self) -> None:
        self._set_all_spectra_checked(True)

    def deselect_all_spectra(self) -> None:
        self._set_all_spectra_checked(False)

    def remove_highlighted_spectra(self) -> None:
        items = self.spectra_list.selectedItems()
        if not items:
            QMessageBox.information(self, "Nothing selected", "Highlight spectra to remove.")
            return
        remove_paths = set()
        for it in items:
            raw = it.data(Qt.ItemDataRole.UserRole)
            if raw:
                remove_paths.add(str(Path(raw).resolve()))
        self.loaded_spectra = [
            s for s in self.loaded_spectra if str(s.meta.path.resolve()) not in remove_paths
        ]
        # Drop caches for removed
        keep_names = {s.meta.path.name for s in self.loaded_spectra}
        self._match_cache = {k: v for k, v in self._match_cache.items() if k in keep_names}
        self._fill_spectra_list(check_all=False)
        if not self.loaded_spectra:
            self.spectrum = None
            self._working_label = None
            self._spectrum_index = 0
            self.auto_peaks = []
            self.manual_peaks = []
            self.peaks = []
            self.hits = []
            self._update_working_label()
            self._fill_element_table()
            self._clear_table(self.cand_table)
            self._redraw(reset_view=True)
            self._sync_calibrate_context()
            self.statusBar().showMessage("All spectra removed.")
            return
        self._spectrum_index = min(self._spectrum_index, len(self.loaded_spectra) - 1)
        self._activate_loaded_at_index(
            self._spectrum_index, restore_cache=True, reset_view=False
        )

    def _cache_key_for_spectrum(self, spectrum: Spectrum | None = None) -> str | None:
        spec = spectrum if spectrum is not None else self.spectrum
        if spec is None:
            return None
        # Only cache real loaded files, not synthetic combined spectra
        name = spec.meta.path.name
        if any(s.meta.path.name == name for s in self.loaded_spectra):
            return name
        return None

    def _save_match_cache(self) -> None:
        key = self._cache_key_for_spectrum()
        if key is None:
            return
        self._match_cache[key] = SpectrumMatchCache(
            auto_peaks=list(self.auto_peaks),
            manual_peaks=list(self.manual_peaks),
            peaks=list(self.peaks),
            hits=list(self.hits),
        )
        self._fill_spectra_list(check_all=False)

    def _restore_match_cache(self, key: str) -> bool:
        cached = self._match_cache.get(key)
        if cached is None:
            return False
        self.auto_peaks = list(cached.auto_peaks)
        self.manual_peaks = list(cached.manual_peaks)
        self.peaks = list(cached.peaks)
        self.hits = list(cached.hits)
        self._pin_matched_elements()
        return True

    def _set_active_spectrum(
        self,
        spectrum: Spectrum,
        *,
        label: str | None = None,
        reset_view: bool = False,
        restore_cache: bool = False,
    ) -> None:
        self.spectrum = spectrum
        self._working_label = label or spectrum.meta.path.name
        self._selected_wl = None
        self._selected_elements = []
        self._primary_element = None
        self._pinned_browse_elements = []

        restored = False
        if restore_cache:
            key = self._cache_key_for_spectrum(spectrum)
            if key is not None:
                restored = self._restore_match_cache(key)
        if not restored:
            self.auto_peaks = []
            self.manual_peaks = []
            self.peaks = []
            self.hits = []
        self._sync_periodic_pins()

        self._update_working_label()
        self._update_meta_panel()
        self._fill_element_table()
        self._clear_table(self.cand_table)
        self.cand_label.setText("<b>Peak candidates (click plot)</b>")
        self._populate_browse_elements()
        self._refresh_browse_table()
        # New spectrum selection: keep wavelength zoom, fit intensity to this file
        self._redraw(reset_view=reset_view, preserve_ylim=False)
        self._sync_calibrate_context()

    def _update_working_label(self) -> None:
        if not hasattr(self, "working_label"):
            return
        if self.spectrum is None:
            self.working_label.setText("Working: —")
            return
        mode = self._display_mode
        self.working_label.setText(
            f"Working: {self._working_label or self.spectrum.meta.path.name}  [{mode}]"
        )

    def _update_meta_panel(self) -> None:
        if self.spectrum is None:
            return
        m = self.spectrum.meta
        lines = [f"Working: {self._working_label or m.path.name}"]
        if self.loaded_spectra:
            lines.append(f"Loaded files: {len(self.loaded_spectra)}")
        lines.append(f"File: {m.path.name}")
        if m.cfg_path:
            lines.append(f"Config: {m.cfg_path.name}")
        if m.laser_energy_mJ is not None:
            lines.append(f"Laser: {m.laser_energy_mJ:g} mJ")
        if m.qs_delay_us is not None:
            lines.append(f"QS delay: {m.qs_delay_us:g} µs")
        if m.integration_time_us is not None:
            delay = (
                f", delay {m.integration_delay_us:g} µs"
                if m.integration_delay_us is not None
                else ""
            )
            lines.append(f"Gate: {m.integration_time_us:g} µs{delay}")
        if m.n_accumulations is not None:
            lines.append(f"Accumulations: {m.n_accumulations}")
        # Combined-source note stored in wavelength_ranges tail
        if m.wavelength_ranges:
            for note in m.wavelength_ranges:
                if isinstance(note, str) and note.startswith("combined:"):
                    parts = note.split(":", 2)
                    if len(parts) == 3:
                        lines.append(f"Combined ({parts[1]}): {parts[2]}")
        lines.append(f"Atmosphere tag: {self.combo_atm.currentText()}")
        wl0 = float(self.spectrum.wavelength_nm.min())
        wl1 = float(self.spectrum.wavelength_nm.max())
        lines.append(f"Range: {wl0:.2f}–{wl1:.2f} nm  ({len(self.spectrum.wavelength_nm)} pts)")
        self.meta_text.setPlainText("\n".join(lines))

    # ------------------------------------------------------------ analysis
    def run_match(self) -> None:
        if self.spectrum is None:
            QMessageBox.information(self, "No spectrum", "Open a spectrum first.")
            return
        if not self.library:
            QMessageBox.warning(self, "No library", f"NIST library not found at\n{DEFAULT_LIBRARY}")
            return

        tol = float(self.spin_tol.value())
        prom = float(self.spin_prom.value())
        self.auto_peaks = find_spectrum_peaks(self.spectrum, min_prominence_frac=prom)
        # Drop manuals that now coincide with an auto peak
        self.manual_peaks = [
            mp
            for mp in self.manual_peaks
            if not any(abs(mp.wavelength_nm - ap.wavelength_nm) < 0.05 for ap in self.auto_peaks)
        ]
        self.peaks = merge_peaks(self.auto_peaks, self.manual_peaks)
        self._rematch_peaks(tol=tol, clear_selection=True)
        self._save_match_cache()
        n_man = len(self.manual_peaks)
        man_note = f" + {n_man} manual" if n_man else ""
        n_pin = len(self._selected_elements)
        ov = self._overlay_mode_label()
        self.statusBar().showMessage(
            f"Found {len(self.auto_peaks)} auto peaks{man_note} "
            f"→ {len(self.hits)} elements ranked; pinned {n_pin} on plot "
            f"(Overlay={ov}). Double-click to add/remove; Unpin clears.  "
            f"(atm={self.combo_atm.currentText()})"
        )
        self._sync_calibrate_context()

    def _rematch_peaks(self, *, tol: float | None = None, clear_selection: bool = False) -> None:
        """Re-run NIST matching on current auto+manual peaks."""
        if self.spectrum is None:
            return
        self.peaks = merge_peaks(self.auto_peaks, self.manual_peaks)
        if not self.library:
            self.hits = []
            self._fill_element_table()
            self._redraw(reset_view=False)
            return
        if tol is None:
            tol = float(self.spin_tol.value())
        if not self.peaks:
            self.hits = []
        else:
            support: dict[str, float] = {}
            primary: dict[str, bool] = {}
            primary_wl: dict[str, float] = {}
            matches = match_peaks(
                self.peaks,
                self.library,
                tol_nm=tol,
                diagnostic_support_out=support,
                primary_diagnostic_out=primary,
                primary_wavelength_out=primary_wl,
            )
            self.hits = score_elements(
                matches,
                min_peaks=2,
                diagnostic_support=support,
                primary_diagnostic=primary,
                primary_wavelength=primary_wl,
            )
        if clear_selection:
            # Auto-pin matched elements on the periodic table and select them
            self._pin_matched_elements()
            self._selected_wl = None
            self._clear_table(self.cand_table)
            self.cand_label.setText("<b>Peak candidates (click plot)</b>")
        elif self._selected_elements:
            keep = {h.element for h in self.hits}
            # Keep manually pinned elements that aren't in hits yet
            pinned_extra = [
                e for e in self._pinned_browse_elements if e not in keep
            ]
            kept = [e for e in self._selected_elements if e in keep]
            self._selected_elements = kept + [e for e in pinned_extra if e not in kept]
            self._pinned_browse_elements = list(self._selected_elements)
            if self._primary_element not in self._selected_elements:
                self._primary_element = (
                    self._selected_elements[0] if self._selected_elements else None
                )
            self._sync_periodic_pins()
            self._fill_element_table()
            self._redraw(reset_view=False)
            self._save_match_cache()
            return
        self._fill_element_table()
        self._redraw(reset_view=False)
        self._save_match_cache()

    def bulk_match_checked(self) -> None:
        """Find peaks + NIST match on each checked spectrum; cache hits per file."""
        checked = self._checked_spectra()
        if not checked:
            QMessageBox.information(
                self,
                "No spectra checked",
                "Check one or more spectra in the list, then Bulk match.",
            )
            return
        if not self.library:
            QMessageBox.warning(self, "No library", f"NIST library not found at\n{DEFAULT_LIBRARY}")
            return

        tol = float(self.spin_tol.value())
        prom = float(self.spin_prom.value())
        # Preserve current working view
        prev_spec = self.spectrum
        prev_label = self._working_label
        prev_mode = self._display_mode
        prev_index = self._spectrum_index

        n_ok = 0
        for i, spec in enumerate(checked):
            self.statusBar().showMessage(
                f"Bulk match {i + 1}/{len(checked)}: {spec.meta.path.name}…"
            )
            QApplication.processEvents()
            auto = find_spectrum_peaks(spec, min_prominence_frac=prom)
            peaks = merge_peaks(auto, [])
            if peaks and self.library:
                support: dict[str, float] = {}
                primary: dict[str, bool] = {}
                primary_wl: dict[str, float] = {}
                matches = match_peaks(
                    peaks,
                    self.library,
                    tol_nm=tol,
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
            else:
                hits = []
            key = spec.meta.path.name
            self._match_cache[key] = SpectrumMatchCache(
                auto_peaks=list(auto),
                manual_peaks=[],
                peaks=list(peaks),
                hits=list(hits),
            )
            n_ok += 1

        self._fill_spectra_list(check_all=False)

        # Restore previous working spectrum / cache
        if prev_spec is not None:
            self.spectrum = prev_spec
            self._working_label = prev_label
            self._spectrum_index = prev_index
            self._display_mode = prev_mode
            if hasattr(self, "combo_display_mode"):
                self._set_display_mode(prev_mode, redraw=False)
            key = self._cache_key_for_spectrum(prev_spec)
            if key and not self._restore_match_cache(key):
                # Combined working spectrum — keep whatever was active
                pass
            self._update_working_label()
            self._update_meta_panel()
            self._fill_element_table()
            self._redraw(reset_view=False)
            self._sync_calibrate_context()

        self.statusBar().showMessage(
            f"Bulk match done — {n_ok} spectrum{'a' if n_ok != 1 else ''} cached. "
            "Use Single + Prev/Next to review."
        )

    def _highlighted_spectra(self) -> list[Spectrum]:
        """Spectra highlighted in the list (row selection), not necessarily checked."""
        if not hasattr(self, "spectra_list"):
            return []
        by_path = {str(s.meta.path.resolve()): s for s in self.loaded_spectra}
        by_name = {s.meta.path.name: s for s in self.loaded_spectra}
        out: list[Spectrum] = []
        for item in self.spectra_list.selectedItems():
            raw = item.data(Qt.ItemDataRole.UserRole)
            spec = None
            if raw:
                try:
                    spec = by_path.get(str(Path(raw).resolve()))
                except Exception:
                    spec = None
            if spec is None:
                name = item.text().split("  ·")[0]
                if ". " in name:
                    name = name.split(". ", 1)[-1]
                spec = by_name.get(name)
            if spec is not None:
                out.append(spec)
        return out

    def _spectra_for_quant(self) -> list[Spectrum]:
        """Checked spectra if any; otherwise highlighted list selection."""
        checked = self._checked_spectra()
        if checked:
            return checked
        return self._highlighted_spectra()

    def quant_selected_spectra(self) -> None:
        """Apply Calibrate-tab CRM fits to each selected spectrum."""
        targets = self._spectra_for_quant()
        if not targets:
            QMessageBox.information(
                self,
                "No spectra selected",
                "Check one or more spectra in the list (or highlight rows), then Quant.",
            )
            return
        if not self.calibrate_tab.has_fits():
            QMessageBox.information(
                self,
                "No calibration curves",
                "Build calibration fits on the Calibrate tab first "
                "(Add standards → concentrations → Build curves), then Quant.",
            )
            return

        rows: list[BulkQuantRow] = []
        errors: list[str] = []
        for i, spec in enumerate(targets):
            self.statusBar().showMessage(
                f"Quant {i + 1}/{len(targets)}: {spec.meta.path.name}…"
            )
            QApplication.processEvents()
            try:
                preds = self.calibrate_tab.predict_for_spectrum(spec)
            except Exception as exc:
                errors.append(f"{spec.meta.path.name}: {exc}")
                continue
            try:
                idx = self.loaded_spectra.index(spec) + 1
            except ValueError:
                idx = i + 1
            conc = {p.element: float(p.concentration) for p in preds}
            rows.append(
                BulkQuantRow(index=idx, filename=spec.meta.path.name, concentrations=conc)
            )

        self._bulk_quant_results = rows
        self._refresh_batch_results_ui()
        if hasattr(self, "side_tabs"):
            for ti in range(self.side_tabs.count()):
                if self.side_tabs.tabText(ti) == "Batch":
                    self.side_tabs.setCurrentIndex(ti)
                    break

        msg = f"Quant done — {len(rows)} spectrum{'a' if len(rows) != 1 else ''}."
        if errors:
            msg += f" {len(errors)} failed."
            QMessageBox.warning(
                self,
                "Quant partial",
                msg + "\n\n" + "\n".join(errors[:8]),
            )
        self.statusBar().showMessage(msg)

    def bulk_quant_checked(self) -> None:
        """Backward-compatible alias for Quant."""
        self.quant_selected_spectra()

    def _refresh_batch_results_ui(self) -> None:
        if not hasattr(self, "batch_table"):
            return
        rows = self._bulk_quant_results
        elements: list[str] = []
        seen: set[str] = set()
        for r in rows:
            for el in r.concentrations:
                if el not in seen:
                    seen.add(el)
                    elements.append(el)
        if not elements and hasattr(self, "calibrate_tab"):
            elements = list(self.calibrate_tab.active_quant_elements())

        headers = ["#", "File"] + elements
        self.batch_table.setColumnCount(len(headers))
        self.batch_table.setHorizontalHeaderLabels(headers)
        self.batch_table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            self.batch_table.setItem(i, 0, QTableWidgetItem(str(r.index)))
            self.batch_table.setItem(i, 1, QTableWidgetItem(r.filename))
            for j, el in enumerate(elements):
                c = r.concentrations.get(el)
                text = f"{c:.4g}" if c is not None else ""
                self.batch_table.setItem(i, 2 + j, QTableWidgetItem(text))
        self.batch_table.resizeColumnsToContents()

        if hasattr(self, "batch_label"):
            if rows:
                unit = ""
                if hasattr(self, "calibrate_tab"):
                    unit = self.calibrate_tab.concentration_unit()
                unit_txt = f" ({unit})" if unit else ""
                self.batch_label.setText(
                    f"{len(rows)} spectra quantified · {len(elements)} element(s)"
                    f"{unit_txt}. "
                    "Plot concentration vs spectrum # below."
                )
            else:
                self.batch_label.setText("Run Quant on selected spectra to fill this table and plot.")

        if hasattr(self, "combo_batch_el"):
            self.combo_batch_el.blockSignals(True)
            self.combo_batch_el.clear()
            if elements:
                self.combo_batch_el.addItem("All (≤4)", "__all__")
                for el in elements:
                    self.combo_batch_el.addItem(el, el)
            self.combo_batch_el.blockSignals(False)

        self._refresh_batch_plot()

    def _refresh_batch_plot(self, _index: int = 0) -> None:
        if not hasattr(self, "batch_canvas"):
            return
        ax = self.batch_canvas.ax
        ax.clear()
        rows = self._bulk_quant_results
        if not rows:
            ax.text(
                0.5,
                0.5,
                "No bulk quant results",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#777",
            )
            self.batch_canvas.fig.tight_layout()
            self.batch_canvas.draw_idle()
            return

        elements: list[str] = []
        seen: set[str] = set()
        for r in rows:
            for el in r.concentrations:
                if el not in seen:
                    seen.add(el)
                    elements.append(el)

        choice = self.combo_batch_el.currentData() if hasattr(self, "combo_batch_el") else None
        if choice == "__all__" or choice is None:
            plot_els = elements[:4]
        else:
            plot_els = [str(choice)]

        xs = [r.index for r in rows]
        for i, el in enumerate(plot_els):
            ys = [r.concentrations.get(el, np.nan) for r in rows]
            color = ELEMENT_COLORS[i % len(ELEMENT_COLORS)]
            ax.plot(xs, ys, "o-", color=color, lw=1.2, ms=5, label=el)

        ax.set_xlabel("Spectrum #")
        unit = ""
        if hasattr(self, "calibrate_tab"):
            unit = self.calibrate_tab.concentration_unit()
        ax.set_ylabel(f"Concentration ({unit})" if unit else "Concentration")
        ax.set_title("C vs spectrum index")
        ax.grid(True, alpha=0.3)
        if plot_els:
            ax.legend(fontsize=8, framealpha=0.9)
        if xs:
            ax.set_xticks(xs)
        self.batch_canvas.fig.tight_layout()
        self.batch_canvas.draw_idle()

    def export_bulk_quant_csv(self) -> None:
        rows = self._bulk_quant_results
        if not rows:
            QMessageBox.information(self, "Nothing to export", "Run Quant first.")
            return
        elements: list[str] = []
        seen: set[str] = set()
        for r in rows:
            for el in r.concentrations:
                if el not in seen:
                    seen.add(el)
                    elements.append(el)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export quant CSV",
            str(ROOT / "reports" / "bulk_quant.csv"),
            "CSV (*.csv);;All files (*)",
        )
        if not path:
            return
        unit = ""
        if hasattr(self, "calibrate_tab"):
            unit = self.calibrate_tab.concentration_unit()
        headers = ["index", "filename"] + [
            f"{el}_{unit}" if unit else el for el in elements
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(headers)
            for r in rows:
                w.writerow(
                    [
                        r.index,
                        r.filename,
                        *[
                            f"{r.concentrations[el]:.6g}" if el in r.concentrations else ""
                            for el in elements
                        ],
                    ]
                )
        self.statusBar().showMessage(f"Exported quant → {path}")

    def add_peak_at_selection(self) -> None:
        """Add a manual peak at the blue selection line (or warn if none)."""
        if self.spectrum is None:
            QMessageBox.information(self, "No spectrum", "Open a spectrum first.")
            return
        if self._selected_wl is None:
            QMessageBox.information(
                self,
                "No selection",
                "Click the plot first to place the blue selection line, "
                "or Shift/right-click a wavelength to add a peak there.",
            )
            return
        self._add_manual_peak(self._selected_wl)

    def _add_manual_peak(self, wavelength_nm: float) -> None:
        if self.spectrum is None:
            return
        peak = make_peak_at_wavelength(self.spectrum, wavelength_nm)

        # Already covered by an existing peak?
        existing = nearest_peak(self.peaks, peak.wavelength_nm, max_dist_nm=0.05)
        if existing is not None:
            kind = "manual" if existing.manual else "auto"
            self._selected_wl = existing.wavelength_nm
            self.show_candidates_at(existing.wavelength_nm)
            self.statusBar().showMessage(
                f"Peak already present at {existing.wavelength_nm:.3f} nm ({kind}) — not duplicated"
            )
            return

        self.manual_peaks.append(peak)
        self.manual_peaks.sort(key=lambda p: p.wavelength_nm)
        self._selected_wl = peak.wavelength_nm
        self._rematch_peaks(clear_selection=False)
        self.show_candidates_at(peak.wavelength_nm)
        self.statusBar().showMessage(
            f"Added manual peak at {peak.wavelength_nm:.3f} nm "
            f"(I={peak.intensity:.0f}) — {len(self.manual_peaks)} manual total"
        )

    def remove_nearest_manual_peak(self) -> None:
        if not self.manual_peaks:
            self.statusBar().showMessage("No manual peaks to remove.")
            return
        ref = self._selected_wl
        if ref is None and self.spectrum is not None:
            # fall back to plot center of current view
            ref = float(sum(self.ax.get_xlim()) / 2.0)
        if ref is None:
            return
        nearest = min(self.manual_peaks, key=lambda p: abs(p.wavelength_nm - ref))
        self.manual_peaks = [p for p in self.manual_peaks if p is not nearest]
        self._selected_wl = nearest.wavelength_nm
        self._rematch_peaks(clear_selection=False)
        self.statusBar().showMessage(
            f"Removed manual peak at {nearest.wavelength_nm:.3f} nm "
            f"({len(self.manual_peaks)} remaining)"
        )

    def clear_manual_peaks(self) -> None:
        if not self.manual_peaks:
            self.statusBar().showMessage("No manual peaks to clear.")
            return
        n = len(self.manual_peaks)
        self.manual_peaks = []
        self._rematch_peaks(clear_selection=False)
        self.statusBar().showMessage(f"Cleared {n} manual peak(s).")

    def _fill_element_table(self) -> None:
        prev = set(self._selected_elements)
        primary = self._primary_element
        self.elem_table.blockSignals(True)
        self._clear_table(self.elem_table)
        self.elem_table.setRowCount(len(self.hits[:40]))
        for i, hit in enumerate(self.hits[:40]):
            self.elem_table.setItem(i, 0, QTableWidgetItem(hit.element))
            self.elem_table.setItem(i, 1, QTableWidgetItem(str(hit.n_peaks)))
            self.elem_table.setItem(i, 2, QTableWidgetItem(f"{hit.confidence:.0f}"))
        self.elem_table.resizeColumnsToContents()
        sm = self.elem_table.selectionModel()
        if sm is not None and prev:
            flags = (
                QItemSelectionModel.SelectionFlag.Select
                | QItemSelectionModel.SelectionFlag.Rows
            )
            for i, hit in enumerate(self.hits[:40]):
                if hit.element in prev:
                    idx = self.elem_table.model().index(i, 0)
                    sm.select(idx, flags)
                    if hit.element == primary:
                        self.elem_table.setCurrentIndex(idx)
        self.elem_table.blockSignals(False)
        if self._primary_element:
            hit = next((h for h in self.hits if h.element == self._primary_element), None)
            if hit is not None:
                self._fill_matched_lines_table(hit)
            else:
                self.cand_label.setText("<b>Peak candidates (click plot)</b>")
        elif not self._selected_elements:
            self.cand_label.setText("<b>Peak candidates (click plot)</b>")

    def _element_color(self, element: str) -> str:
        stick_els = self._stick_elements()
        if not stick_els:
            return ELEMENT_COLORS[0]
        try:
            idx = stick_els.index(element)
        except ValueError:
            idx = abs(hash(element)) % len(ELEMENT_COLORS)
        return ELEMENT_COLORS[idx % len(ELEMENT_COLORS)]

    def _on_element_select(self) -> None:
        rows = self.elem_table.selectionModel().selectedRows()
        if not rows or not self.hits:
            self._selected_elements = []
            self._primary_element = None
            # Keep only pins that are not in the ranking table (manual pin-only)
            ranked = {h.element for h in self.hits}
            self._pinned_browse_elements = [
                e for e in self._pinned_browse_elements if e not in ranked
            ]
            self._sync_periodic_pins()
            self.cand_label.setText("<b>Peak candidates (click plot)</b>")
            if self._line_preview is not None and self._line_preview.isVisible():
                self._line_preview.clear_preview()
            self._redraw(reset_view=False)
            return

        elements: list[str] = []
        for idx in sorted(rows, key=lambda r: r.row()):
            item = self.elem_table.item(idx.row(), 0)
            if item is not None:
                elements.append(item.text())
        # Preserve pinned elements that aren't in the ranking hits (e.g. manual Pb pin)
        ranked = {h.element for h in self.hits}
        extras = [
            e for e in self._pinned_browse_elements
            if e not in ranked and e not in elements
        ]
        self._selected_elements = elements + extras
        # Sync pins ↔ matched selection
        self._pinned_browse_elements = list(self._selected_elements)
        self._sync_periodic_pins()

        current = self.elem_table.currentRow()
        if current >= 0:
            cur_item = self.elem_table.item(current, 0)
            primary = cur_item.text() if cur_item is not None else elements[0]
        else:
            primary = elements[0]
        if primary not in elements:
            primary = elements[0]
        self._primary_element = primary

        hit = next((h for h in self.hits if h.element == primary), None)
        if hit is None:
            self._redraw(reset_view=False)
            return

        # Keep click-selection marker only if it belongs to any selected element
        if self._selected_wl is not None:
            matched_wls = {
                round(m.peak.wavelength_nm, 4)
                for h in self.hits
                if h.element in self._selected_elements
                for m in h.matches
            }
            if round(self._selected_wl, 4) not in matched_wls:
                self._selected_wl = None

        self._fill_matched_lines_table(hit)
        self._redraw(reset_view=False)
        self._update_line_preview(hit)
        n_sticks = sum(
            1
            for lines in strong_library_lines(
                self.library,
                self._stick_elements(),
                wl_min=float(self.spectrum.wavelength_nm.min()) if self.spectrum else 0.0,
                wl_max=float(self.spectrum.wavelength_nm.max()) if self.spectrum else 0.0,
            ).values()
            for _ in lines
        )
        sel_txt = ", ".join(self._selected_elements)
        self.statusBar().showMessage(
            f"Preview {sel_txt}: {n_sticks} NIST sticks  |  "
            f"{primary} matches highlighted ({hit.n_peaks} peaks, {hit.confidence:.0f}%)"
        )

    def _on_element_double_click(self, row: int, _column: int) -> None:
        item = self.elem_table.item(row, 0)
        if item is None:
            return
        hit = next((h for h in self.hits if h.element == item.text()), None)
        if hit is None or self.spectrum is None:
            return
        self._primary_element = hit.element
        if hit.element not in self._selected_elements:
            self._selected_elements = [hit.element]
        self.show_line_preview()

    def _ensure_line_preview(self) -> LinePreviewWindow:
        if self._line_preview is None:
            self._line_preview = LinePreviewWindow(self)
        return self._line_preview

    def show_line_preview(self) -> None:
        """Open/raise the top-5 matched-line preview for the primary selected element."""
        if self.spectrum is None:
            QMessageBox.information(self, "No spectrum", "Open a spectrum first.")
            return
        el = self._primary_element
        if el is None and self._selected_elements:
            el = self._selected_elements[0]
        if el is None:
            QMessageBox.information(
                self,
                "No element selected",
                "Select a matched element in the ranking table first "
                "(or double-click a row).",
            )
            return
        hit = next((h for h in self.hits if h.element == el), None)
        if hit is None:
            QMessageBox.information(
                self,
                "No matches",
                "Run Find peaks + match first, then select an element.",
            )
            return
        self._update_line_preview(hit, force_show=True)

    def _update_line_preview(self, hit: ElementHit, *, force_show: bool = False) -> None:
        if self.spectrum is None:
            return
        preview = self._ensure_line_preview()
        if force_show or preview.isVisible():
            preview.update_preview(
                self.spectrum,
                hit,
                atmosphere=self.combo_atm.currentText(),
            )

    def _sync_periodic_pins(self) -> None:
        if hasattr(self, "periodic_table"):
            self.periodic_table.set_pinned(self._pinned_browse_elements)

    def _overlay_mode(self) -> str:
        """Current Overlay combo mode: ``top:N``, ``none``, or ``all``."""
        if hasattr(self, "combo_overlay"):
            data = self.combo_overlay.currentData()
            if isinstance(data, str) and data:
                return data
        return "top:5"

    def _overlay_mode_label(self) -> str:
        if hasattr(self, "combo_overlay"):
            return self.combo_overlay.currentText()
        return "Top 5"

    def _elements_for_overlay_mode(self, mode: str | None = None) -> list[str]:
        """Ranked element symbols to pin for the given Overlay mode."""
        ranked = [h.element for h in self.hits if h.element]
        mode = mode or self._overlay_mode()
        if mode == "none":
            return []
        if mode == "all":
            return ranked[:40]
        n = 5
        if mode.startswith("top:"):
            try:
                n = max(0, int(mode.split(":", 1)[1]))
            except ValueError:
                n = 5
        return ranked[:n]

    def _pin_matched_elements(self) -> None:
        """Pin ranked elements for the plot per the Overlay combo (default top 5)."""
        ranked = [h.element for h in self.hits if h.element]
        els = self._elements_for_overlay_mode()
        self._selected_elements = list(els)
        self._pinned_browse_elements = list(els)
        # Keep a primary for the lines table even when Overlay is None
        self._primary_element = els[0] if els else (ranked[0] if ranked else None)
        self._sync_periodic_pins()

    def _on_overlay_mode_changed(self, *_args) -> None:
        """Re-apply pins from current ranking when Overlay combo changes."""
        if not self.hits:
            return
        self._pin_matched_elements()
        self._fill_element_table()
        # Show primary element's matched lines when available
        if self._primary_element:
            hit = next(
                (h for h in self.hits if h.element == self._primary_element), None
            )
            if hit is not None:
                self._fill_matched_lines_table(hit)
        self._redraw(reset_view=False)
        n_pin = len(self._selected_elements)
        self.statusBar().showMessage(
            f"Overlay={self._overlay_mode_label()}: pinned {n_pin} of "
            f"{len(self.hits)} ranked. Double-click to add/remove; Unpin clears."
        )

    def _set_element_selected(self, symbol: str, *, selected: bool) -> None:
        """Add/remove an element from the matched selection + pin set."""
        symbol = (symbol or "").strip()
        if not symbol:
            return
        if selected:
            if symbol not in self._selected_elements:
                self._selected_elements.append(symbol)
            if symbol not in self._pinned_browse_elements:
                self._pinned_browse_elements.append(symbol)
            self._primary_element = symbol
        else:
            self._selected_elements = [e for e in self._selected_elements if e != symbol]
            self._pinned_browse_elements = [
                e for e in self._pinned_browse_elements if e != symbol
            ]
            if self._primary_element == symbol:
                self._primary_element = (
                    self._selected_elements[0] if self._selected_elements else None
                )
        self._sync_periodic_pins()

    def _stick_elements(self) -> list[str]:
        """Elements shown in the NIST stick panel (matched/pinned + viewing)."""
        els: list[str] = []
        for el in self._selected_elements:
            if el and el not in els:
                els.append(el)
        for el in self._pinned_browse_elements:
            if el and el not in els:
                els.append(el)
        if self._browse_element and self._browse_element not in els:
            els.append(self._browse_element)
        return els

    def _populate_browse_elements(self) -> None:
        if not hasattr(self, "combo_browse_el"):
            return
        prev = self.combo_browse_el.currentText().strip()
        self.combo_browse_el.blockSignals(True)
        self.combo_browse_el.clear()
        self.combo_browse_el.addItem("")  # blank = pick an element
        available: list[str] = []
        if self.library and self.spectrum is not None:
            wl0 = float(self.spectrum.wavelength_nm.min())
            wl1 = float(self.spectrum.wavelength_nm.max())
            available = elements_in_wavelength_range(self.library, wl_min=wl0, wl_max=wl1)
        elif self.library:
            available = sorted(
                {L.element for L in self.library if L.element},
                key=lambda e: (len(e), e),
            )
        for el in available:
            self.combo_browse_el.addItem(el)
        idx = self.combo_browse_el.findText(prev)
        if idx >= 0:
            self.combo_browse_el.setCurrentIndex(idx)
        self.combo_browse_el.blockSignals(False)
        if hasattr(self, "periodic_table"):
            self.periodic_table.set_available(available)
            self.periodic_table.set_viewing(self._browse_element)
            self.periodic_table.set_pinned(self._pinned_browse_elements)

    def _browse_wavelength_window(self) -> tuple[float, float]:
        if self.spectrum is None:
            return 0.0, 0.0
        scope = self.combo_browse_scope.currentData()
        if scope == "visible":
            xlim = self.ax.get_xlim()
            return float(xlim[0]), float(xlim[1])
        return (
            float(self.spectrum.wavelength_nm.min()),
            float(self.spectrum.wavelength_nm.max()),
        )

    def _on_side_tab_changed(self, index: int) -> None:
        if index == 1:  # Browse NIST
            self._populate_browse_elements()
            self._refresh_browse_table()

    def _set_browse_element(self, el: str | None, *, sync_combo: bool = True) -> None:
        """Set the currently viewed Browse NIST element and refresh table/sticks."""
        el = (el or "").strip() or None
        self._browse_element = el
        if sync_combo and hasattr(self, "combo_browse_el"):
            self.combo_browse_el.blockSignals(True)
            idx = self.combo_browse_el.findText(el or "")
            if idx >= 0:
                self.combo_browse_el.setCurrentIndex(idx)
            else:
                self.combo_browse_el.setEditText(el or "")
            self.combo_browse_el.blockSignals(False)
        if hasattr(self, "periodic_table"):
            self.periodic_table.set_viewing(el)
        self._refresh_browse_table()
        self._redraw(reset_view=False)

    def _on_browse_element_viewed(self, symbol: str) -> None:
        self._set_browse_element(symbol)
        self.statusBar().showMessage(f"Browse NIST: viewing {symbol}")

    def _on_browse_element_pin_toggled(self, symbol: str) -> None:
        if symbol in self._pinned_browse_elements:
            self._set_element_selected(symbol, selected=False)
            msg = f"Unpinned {symbol} (removed from matched list)"
        else:
            self._set_element_selected(symbol, selected=True)
            msg = f"Pinned {symbol} (added to matched list)"
        # Refresh ranking-table selection to match pins
        self._fill_element_table()
        # Also view the pinned element so the line table matches
        self._set_browse_element(symbol)
        hit = next((h for h in self.hits if h.element == symbol), None)
        if hit is not None and symbol in self._selected_elements:
            self._fill_matched_lines_table(hit)
        self._redraw(reset_view=False)
        self.statusBar().showMessage(msg)

    def _clear_browse_pins(self) -> None:
        if not self._pinned_browse_elements and not self._selected_elements:
            self.statusBar().showMessage("No pinned elements.")
            return
        n = len(self._pinned_browse_elements) or len(self._selected_elements)
        self._pinned_browse_elements = []
        self._selected_elements = []
        self._primary_element = None
        self._sync_periodic_pins()
        self._fill_element_table()
        self.elem_table.clearSelection()
        self._redraw(reset_view=False)
        self.statusBar().showMessage(f"Cleared {n} pinned / matched element(s).")

    def _on_browse_filters_changed(self, *_args) -> None:
        el = self.combo_browse_el.currentText().strip()
        self._set_browse_element(el or None, sync_combo=False)

    def _refresh_browse_table(self) -> None:
        if not hasattr(self, "browse_table"):
            return
        self._clear_table(self.browse_table)
        self._browse_lines = []
        if self.spectrum is None:
            self.browse_label.setText("Open a spectrum to browse NIST lines.")
            return
        if not self.library:
            self.browse_label.setText("NIST library is not loaded.")
            return

        el = self.combo_browse_el.currentText().strip() or None
        ion = self.combo_browse_ion.currentData()
        wl0, wl1 = self._browse_wavelength_window()
        if wl1 < wl0:
            wl0, wl1 = wl1, wl0

        if not el:
            self.browse_label.setText(
                f"Click the periodic table (or El) to list NIST lines in "
                f"{wl0:.1f}–{wl1:.1f} nm · double-click to pin"
            )
            return

        lines = browse_library_lines(
            self.library,
            element=el,
            ion_stage=ion,
            wl_min=wl0,
            wl_max=wl1,
            max_lines=int(self.spin_browse_max.value()),
        )
        self._browse_lines = lines
        ion_txt = self.combo_browse_ion.currentText()
        pin_txt = " · pinned" if el in self._pinned_browse_elements else ""
        self.browse_label.setText(
            f"{el}"
            + (f" {ion_txt}" if ion is not None else "")
            + f": {len(lines)} lines in {wl0:.1f}–{wl1:.1f} nm"
            + pin_txt
            + " · click a row to mark λ"
        )
        self.browse_table.setRowCount(len(lines))
        for i, line in enumerate(lines):
            nist_i = f"{line.intensity:.0f}" if line.intensity is not None else ""
            aki = f"{line.aki:.2e}" if line.aki is not None else ""
            vals = (
                line.species,
                f"{line.wavelength_nm:.3f}",
                nist_i,
                aki,
            )
            for j, v in enumerate(vals):
                self.browse_table.setItem(i, j, QTableWidgetItem(v))
        self.browse_table.resizeColumnsToContents()

    def _on_browse_line_select(self) -> None:
        rows = self.browse_table.selectionModel().selectedRows()
        if not rows or not self._browse_lines:
            return
        idx = rows[0].row()
        if idx < 0 or idx >= len(self._browse_lines):
            return
        line = self._browse_lines[idx]
        self._selected_wl = line.wavelength_nm
        self._redraw(reset_view=False)
        self.statusBar().showMessage(
            f"Browse mark: {line.species}  λ={line.wavelength_nm:.3f} nm"
            + (f"  NIST I={line.intensity:.0f}" if line.intensity is not None else "")
        )

    def _fill_matched_lines_table(self, hit: ElementHit) -> None:
        self.side_tabs.setCurrentIndex(0)
        self.cand_label.setText(
            f"<b>Matched lines — {hit.element}</b> "
            f"<span style='color:#666;font-weight:normal'>(primary selection)</span>"
        )
        self._clear_table(self.cand_table)
        ms = sorted(hit.matches, key=lambda x: x.peak.prominence, reverse=True)[:30]
        self.cand_table.setRowCount(len(ms))
        for i, m in enumerate(ms):
            nist_i = f"{m.line.intensity:.0f}" if m.line.intensity is not None else ""
            aki = f"{m.line.aki:.2e}" if m.line.aki is not None else ""
            vals = (
                m.line.species,
                f"{m.line.wavelength_nm:.3f}",
                f"{m.delta_nm:+.3f}",
                nist_i,
                aki,
            )
            for j, v in enumerate(vals):
                self.cand_table.setItem(i, j, QTableWidgetItem(v))
        self.cand_table.resizeColumnsToContents()

    # --------------------------------------------------------------- plot
    def _format_intensity_axis(self, *, ylabel: str = "Intensity (counts)") -> None:
        """Compact scientific Y ticks so the axis label is not clipped."""
        self.ax.set_ylabel(ylabel)
        self.ax.ticklabel_format(
            axis="y",
            style="sci",
            scilimits=(0, 0),
            useMathText=True,
        )
        self.ax.yaxis.get_offset_text().set_fontsize(9)

    def _finish_canvas_layout(self) -> None:
        """tight_layout, then keep left margin for ylabel + sci offset."""
        self.canvas.fig.tight_layout(pad=0.35)
        # tight_layout can still pinch the left edge when the left dock is narrow
        self.canvas.fig.subplots_adjust(left=max(0.12, self.canvas.fig.subplotpars.left))

    def _on_xlim_changed(self, _ax) -> None:
        """Refresh NIST sticks to the visible wavelength window after zoom/pan."""
        if self._suspend_xlim_cb or self.spectrum is None:
            return
        if self._display_mode == "waterfall":
            return
        if not self._stick_elements():
            return
        self._suspend_xlim_cb = True
        try:
            xlim = self.ax.get_xlim()
            self.ax_sticks.clear()
            self.ax_sticks.set_ylabel("NIST\nsticks", fontsize=8)
            self.ax_sticks.set_ylim(0, 1.08)
            self.ax_sticks.set_yticks([])
            self.ax_sticks.tick_params(labelsize=8)
            self.ax_sticks.set_xlabel("Wavelength (nm)")
            self._draw_nist_sticks(xlim)
            self.ax_sticks.set_xlim(xlim)
            if (
                hasattr(self, "combo_browse_scope")
                and self.combo_browse_scope.currentData() == "visible"
                and self.side_tabs.currentIndex() == 1
            ):
                self._refresh_browse_table()
            self.canvas.draw_idle()
        finally:
            self._suspend_xlim_cb = False

    def _redraw(self, *, reset_view: bool = False, preserve_ylim: bool = True) -> None:
        saved_xlim = saved_ylim = None
        if not reset_view and self.spectrum is not None and hasattr(self, "ax"):
            saved_xlim = self.ax.get_xlim()
            if preserve_ylim:
                saved_ylim = self.ax.get_ylim()

        self._suspend_xlim_cb = True
        try:
            self._redraw_body(saved_xlim=saved_xlim, saved_ylim=saved_ylim, reset_view=reset_view)
        finally:
            self._suspend_xlim_cb = False

    def _autoscale_ylim(self) -> None:
        """Fit Y to plotted intensity in the current wavelength window."""
        xlim = self.ax.get_xlim()
        y_lo: float | None = None
        y_hi: float | None = None
        for line in self.ax.get_lines():
            x = np.asarray(line.get_xdata(), dtype=float)
            y = np.asarray(line.get_ydata(), dtype=float)
            if x.size == 0 or y.size == 0:
                continue
            mask = (x >= xlim[0]) & (x <= xlim[1]) & np.isfinite(y)
            if not np.any(mask):
                continue
            yy = y[mask]
            lo = float(np.min(yy))
            hi = float(np.max(yy))
            y_lo = lo if y_lo is None else min(y_lo, lo)
            y_hi = hi if y_hi is None else max(y_hi, hi)
        if y_lo is None or y_hi is None:
            self.ax.relim()
            self.ax.autoscale_view(scalex=False, scaley=True)
            return
        span = y_hi - y_lo
        pad = (span * 0.05) if span > 0 else max(abs(y_hi) * 0.05, 1.0)
        self.ax.set_ylim(y_lo - pad * 0.2, y_hi + pad)

    @staticmethod
    def _apply_axis_limits(
        ax,
        *,
        saved_xlim: tuple[float, float] | None,
        saved_ylim: tuple[float, float] | None,
        reset_view: bool,
        full_xlim: tuple[float, float] | None = None,
        autoscale_y=None,
    ) -> None:
        """Restore zoom independently for X and Y; optionally autoscale Y."""
        if saved_xlim is not None and not reset_view:
            ax.set_xlim(saved_xlim)
        elif full_xlim is not None:
            ax.set_xlim(full_xlim)
        if saved_ylim is not None and not reset_view:
            ax.set_ylim(saved_ylim)
        elif autoscale_y is not None:
            autoscale_y()

    def _redraw_body(
        self,
        *,
        saved_xlim: tuple[float, float] | None,
        saved_ylim: tuple[float, float] | None,
        reset_view: bool,
    ) -> None:
        self.ax.clear()
        self.ax_sticks.clear()
        self.ax.tick_params(labelbottom=False)
        self.ax_sticks.set_ylabel("NIST\nsticks", fontsize=8)
        self.ax_sticks.set_ylim(0, 1.08)
        self.ax_sticks.set_yticks([])
        self.ax_sticks.tick_params(labelsize=8)

        if self.spectrum is None:
            self.ax.set_title("No spectrum loaded")
            self.ax_sticks.text(
                0.5,
                0.5,
                "Select element(s) to preview NIST LIBS lines",
                transform=self.ax_sticks.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="#777",
            )
            self.canvas.draw_idle()
            return

        if self._display_mode == "waterfall":
            self._redraw_waterfall(
                saved_xlim=saved_xlim,
                saved_ylim=saved_ylim,
                reset_view=reset_view,
            )
            return

        wl = self.spectrum.wavelength_nm
        y = self.spectrum.intensity

        working_name = self._working_label or self.spectrum.meta.path.stem
        self.ax.plot(wl, y, color="#1a1a1a", lw=0.8, label=working_name, zorder=2)

        if self.peaks:
            auto = [p for p in self.peaks if not p.manual]
            manual = [p for p in self.peaks if p.manual]
            if auto:
                self.ax.scatter(
                    [p.wavelength_nm for p in auto],
                    [p.intensity for p in auto],
                    s=14,
                    c="#c0392b",
                    zorder=3,
                    alpha=0.55,
                    label="auto peaks",
                )
            if manual:
                self.ax.scatter(
                    [p.wavelength_nm for p in manual],
                    [p.intensity for p in manual],
                    s=48,
                    c="#2980b9",
                    marker="s",
                    zorder=4,
                    edgecolors="white",
                    linewidths=0.7,
                    label="manual peaks",
                )

        selected_hits = [
            h for h in self.hits if h.element in self._selected_elements
        ]
        primary_hit = next(
            (h for h in selected_hits if h.element == self._primary_element),
            selected_hits[0] if selected_hits else None,
        )

        # Highlight matched peaks for every selected element
        for hit in selected_hits:
            color = self._element_color(hit.element)
            for m in hit.matches:
                self.ax.axvline(
                    m.peak.wavelength_nm,
                    color=color,
                    lw=1.0,
                    alpha=0.35,
                    zorder=2,
                )
            self.ax.scatter(
                [m.peak.wavelength_nm for m in hit.matches],
                [m.peak.intensity for m in hit.matches],
                s=55 if hit is primary_hit else 36,
                c=color,
                marker="D",
                zorder=4,
                edgecolors="white",
                linewidths=0.6,
                label=f"{hit.element} matches",
            )

        if primary_hit is not None:
            top_ms = sorted(
                primary_hit.matches, key=lambda m: m.peak.prominence, reverse=True
            )[:8]
            for m in top_ms:
                self.ax.annotate(
                    f"{m.line.species}\n{m.peak.wavelength_nm:.1f}",
                    (m.peak.wavelength_nm, m.peak.intensity),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=7,
                    color=self._element_color(primary_hit.element),
                )

        # NIST stick preview for selected elements (current x-range)
        self._draw_nist_sticks(saved_xlim)

        if self._selected_wl is not None:
            self.ax.axvline(self._selected_wl, color="#2471a3", lw=1.4, alpha=0.9, zorder=5)
            self.ax_sticks.axvline(
                self._selected_wl, color="#2471a3", lw=1.2, alpha=0.85, zorder=5
            )

        title = self._working_label or self.spectrum.meta.path.stem
        if selected_hits:
            bits = ", ".join(
                f"{h.element} {h.confidence:.0f}%" for h in selected_hits[:4]
            )
            more = f" +{len(selected_hits) - 4}" if len(selected_hits) > 4 else ""
            self.ax.set_title(
                f"{title}  [{self.combo_atm.currentText()}]  —  {bits}{more}"
            )
        else:
            self.ax.set_title(f"{title}  [{self.combo_atm.currentText()}]")
        self.ax.set_xlabel("")
        self.ax.tick_params(labelbottom=False)
        self._format_intensity_axis(ylabel="Intensity (counts)")
        self.ax_sticks.set_xlabel("Wavelength (nm)")

        self._apply_axis_limits(
            self.ax,
            saved_xlim=saved_xlim,
            saved_ylim=saved_ylim,
            reset_view=reset_view,
            full_xlim=(float(wl.min()), float(wl.max())),
            autoscale_y=self._autoscale_ylim,
        )

        if self.hits and not selected_hits:
            top = ", ".join(f"{h.element} {h.confidence:.0f}%" for h in self.hits[:6])
            man = f"  |  {len(self.manual_peaks)} manual" if self.manual_peaks else ""
            self.ax.text(
                0.01,
                0.98,
                f"Top: {top}{man}",
                transform=self.ax.transAxes,
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#ccc"),
            )
        if selected_hits or self.manual_peaks:
            self.ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        self._finish_canvas_layout()
        # Restore again after layout, which can nudge limits
        self._apply_axis_limits(
            self.ax,
            saved_xlim=saved_xlim,
            saved_ylim=saved_ylim,
            reset_view=reset_view,
            full_xlim=(float(wl.min()), float(wl.max())),
            autoscale_y=self._autoscale_ylim,
        )
        self.canvas.draw_idle()

    def _redraw_waterfall(
        self,
        *,
        saved_xlim: tuple[float, float] | None,
        saved_ylim: tuple[float, float] | None,
        reset_view: bool,
    ) -> None:
        """Vertically offset checked spectra (no peak/stick overlays)."""
        checked = self._checked_spectra()
        if not checked:
            checked = list(self.loaded_spectra)
        if not checked and self.spectrum is not None:
            checked = [self.spectrum]
        if not checked:
            self.ax.set_title("Waterfall — no spectra")
            self.ax_sticks.text(
                0.5,
                0.5,
                "Check spectra in the list",
                transform=self.ax_sticks.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="#777",
            )
            self.canvas.draw_idle()
            return

        max_i = max(float(np.nanmax(s.intensity)) for s in checked)
        frac = max(0.0, float(self._waterfall_offset_frac))
        offset_step = (max_i * frac) if max_i > 0 else 0.0
        active_path = None
        if (
            self.spectrum is not None
            and not (self._working_label or "").startswith(("Mean of", "Sum of"))
        ):
            active_path = self.spectrum.meta.path

        wl_min = min(float(s.wavelength_nm.min()) for s in checked)
        wl_max = max(float(s.wavelength_nm.max()) for s in checked)

        for i, spec in enumerate(checked):
            try:
                idx = self.loaded_spectra.index(spec) + 1
            except ValueError:
                idx = i + 1
            color = ELEMENT_COLORS[i % len(ELEMENT_COLORS)]
            lw = 1.5 if active_path is not None and spec.meta.path == active_path else 0.9
            self.ax.plot(
                spec.wavelength_nm,
                spec.intensity + i * offset_step,
                color=color,
                lw=lw,
                label=f"{idx}. {spec.meta.path.name}",
                zorder=2,
            )

        self.ax_sticks.text(
            0.5,
            0.5,
            "Sticks hidden in Waterfall mode",
            transform=self.ax_sticks.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color="#777",
        )

        pct = int(round(frac * 100))
        self.ax.set_title(
            f"Waterfall — {len(checked)} checked · offset {pct}%  "
            f"[{self.combo_atm.currentText()}]"
        )
        self.ax.set_xlabel("")
        self.ax.tick_params(labelbottom=False)
        self._format_intensity_axis(ylabel="Intensity + offset")
        self.ax_sticks.set_xlabel("Wavelength (nm)")
        self.ax.grid(True, alpha=0.25)
        if len(checked) <= 16:
            self.ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

        self._apply_axis_limits(
            self.ax,
            saved_xlim=saved_xlim,
            saved_ylim=saved_ylim,
            reset_view=reset_view,
            full_xlim=(wl_min, wl_max),
            autoscale_y=self._autoscale_ylim,
        )

        self._finish_canvas_layout()
        self._apply_axis_limits(
            self.ax,
            saved_xlim=saved_xlim,
            saved_ylim=saved_ylim,
            reset_view=reset_view,
            full_xlim=(wl_min, wl_max),
            autoscale_y=self._autoscale_ylim,
        )
        self.canvas.draw_idle()

    def _draw_nist_sticks(self, xlim: tuple[float, float] | None) -> None:
        """
        Stick panel for ranked selection and/or Browse NIST element.

        • Thick tall sticks = matched peaks, height ∝ observed intensity
        • Thin faint sticks = other NIST catalog lines in view, height ∝ log(NIST I)
        """
        if self.spectrum is None or self._display_mode == "waterfall":
            return
        stick_els = self._stick_elements()
        if not stick_els or not self.library:
            self.ax_sticks.text(
                0.5,
                0.5,
                "Select ranked element(s) or Browse NIST",
                transform=self.ax_sticks.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="#777",
            )
            return

        wl_min = float(xlim[0]) if xlim is not None else float(self.spectrum.wavelength_nm.min())
        wl_max = float(xlim[1]) if xlim is not None else float(self.spectrum.wavelength_nm.max())
        pad = 0.02 * (wl_max - wl_min + 1.0)
        lo = wl_min - pad
        hi = wl_max + pad

        matches_in_view: dict[str, list] = {el: [] for el in stick_els}
        for hit in self.hits:
            if hit.element not in matches_in_view:
                continue
            for m in hit.matches:
                if lo <= m.peak.wavelength_nm <= hi:
                    matches_in_view[hit.element].append(m)

        preview = strong_library_lines(
            self.library,
            stick_els,
            wl_min=lo,
            wl_max=hi,
            max_per_element=60,
        )

        any_sticks = False
        for el in stick_els:
            color = self._element_color(el)
            matched = matches_in_view.get(el, [])
            matched_nist_wl = {round(m.line.wavelength_nm, 3) for m in matched}

            catalog = [
                L
                for L in preview.get(el, [])
                if round(L.wavelength_nm, 3) not in matched_nist_wl
            ]
            if catalog:
                any_sticks = True
                log_s = [
                    float(
                        np.log10(
                            max(L.intensity or 0.0, (L.aki or 0.0) / 1e7, 1.0)
                        )
                    )
                    for L in catalog
                ]
                smax = max(log_s)
                smin = min(log_s)
                span = max(smax - smin, 0.5)
                for L, ls in zip(catalog, log_s):
                    h = 0.06 + 0.38 * ((ls - smin) / span)
                    self.ax_sticks.vlines(
                        L.wavelength_nm,
                        0,
                        h,
                        colors=color,
                        lw=0.6,
                        alpha=0.35,
                        zorder=2,
                    )

            if matched:
                any_sticks = True
                imax = max(m.peak.intensity for m in matched) or 1.0
                for m in matched:
                    h = 0.30 + 0.70 * (m.peak.intensity / imax)
                    self.ax_sticks.vlines(
                        m.peak.wavelength_nm,
                        0,
                        h,
                        colors=color,
                        lw=2.0,
                        alpha=0.95,
                        zorder=4,
                    )

            tags: list[str] = []
            if el in self._pinned_browse_elements and el not in self._selected_elements:
                tags.append("pin")
            elif el == self._browse_element and el not in self._selected_elements:
                tags.append("browse")
            label = el if not tags else f"{el} ({', '.join(tags)})"
            self.ax_sticks.plot([], [], color=color, lw=2, label=label)

        if not any_sticks:
            self.ax_sticks.text(
                0.5,
                0.5,
                "No NIST lines in this wavelength window",
                transform=self.ax_sticks.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="#777",
            )
        else:
            self.ax_sticks.legend(
                loc="upper right",
                fontsize=7,
                framealpha=0.9,
                ncols=min(4, len(stick_els)),
            )
            self.ax_sticks.text(
                0.01,
                0.92,
                "thick = matched (height ∝ observed I) · thin = NIST catalog (log I)",
                transform=self.ax_sticks.transAxes,
                fontsize=7,
                color="#555",
                va="top",
            )

    def _on_click(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None or self.spectrum is None:
            return
        # Skip while zoom/pan tools are active
        mode = getattr(self.toolbar, "mode", "")
        if mode:
            return

        wl = float(event.xdata)
        try:
            from matplotlib.backend_bases import MouseButton

            right = event.button == MouseButton.RIGHT
            left = event.button == MouseButton.LEFT
        except Exception:
            right = event.button == 3
            left = event.button == 1

        shift_held = False
        gui = getattr(event, "guiEvent", None)
        if gui is not None and hasattr(gui, "modifiers"):
            shift_held = bool(gui.modifiers() & Qt.KeyboardModifier.ShiftModifier)

        if right or (left and shift_held):
            self._add_manual_peak(wl)
            return
        if left:
            self.show_candidates_at(wl)

    def show_candidates_at(self, wavelength_nm: float) -> None:
        if not self.library:
            QMessageBox.warning(self, "No library", "NIST library is not loaded.")
            return

        tol = float(self.spin_tol.value())
        peak = (
            nearest_peak(self.peaks, wavelength_nm, max_dist_nm=max(1.0, 5 * tol))
            if self.peaks
            else None
        )
        query_wl = peak.wavelength_nm if peak is not None else wavelength_nm
        self._selected_wl = query_wl

        cands = candidates_near_wavelength(query_wl, self.library, tol_nm=tol, max_results=40)
        self.side_tabs.setCurrentIndex(0)
        self.cand_label.setText("<b>Peak candidates (click plot)</b>")
        self._clear_table(self.cand_table)
        self.cand_table.setRowCount(len(cands))
        for i, c in enumerate(cands):
            nist_i = f"{c.line.intensity:.0f}" if c.line.intensity is not None else ""
            aki = f"{c.line.aki:.2e}" if c.line.aki is not None else ""
            vals = (
                c.line.species,
                f"{c.line.wavelength_nm:.3f}",
                f"{c.delta_nm:+.3f}",
                nist_i,
                aki,
            )
            for j, v in enumerate(vals):
                self.cand_table.setItem(i, j, QTableWidgetItem(v))
        self.cand_table.resizeColumnsToContents()
        self._redraw(reset_view=False)

        if peak is not None:
            tag = "manual" if peak.manual else "auto"
            self.statusBar().showMessage(
                f"Peak {peak.wavelength_nm:.3f} nm [{tag}] (I={peak.intensity:.0f}) — "
                f"{len(cands)} candidates within ±{tol:.2f} nm"
            )
        else:
            self.statusBar().showMessage(
                f"λ = {query_wl:.3f} nm — {len(cands)} candidates within ±{tol:.2f} nm "
                f"(no peak nearby; Shift/right-click to add)"
            )

    def clear_selection(self) -> None:
        self._selected_wl = None
        self._selected_elements = []
        self._primary_element = None
        self._pinned_browse_elements = []
        self._sync_periodic_pins()
        self.elem_table.clearSelection()
        self._clear_table(self.cand_table)
        self.cand_label.setText("<b>Peak candidates (click plot)</b>")
        self._redraw(reset_view=False)
        self.statusBar().showMessage("Selection cleared.")

    def export_publication_report(self) -> None:
        if self.spectrum is None:
            QMessageBox.information(self, "No spectrum", "Open a spectrum first.")
            return
        if not self.hits:
            QMessageBox.information(
                self,
                "No matches",
                "Run Find peaks + match first so the report has ranked elements.",
            )
            return

        dlg = ReportExportDialog(
            self,
            n_elements=len(self.hits),
            has_selection=bool(self._selected_elements),
            default_stem=self.spectrum.meta.path.stem,
            spectrum_label=self._working_label or self.spectrum.meta.path.name,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        opts = dlg.options()
        hits = self._hits_for_report(opts["scope"])
        if not hits:
            QMessageBox.warning(self, "Nothing to export", "No elements match the chosen scope.")
            return

        atm = self.combo_atm.currentText()
        try:
            if opts["format"] == "pdf":
                path, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save publication report PDF",
                    str(ROOT / "reports" / f"{opts['stem']}_lines.pdf"),
                    "PDF (*.pdf)",
                )
                if not path:
                    return
                out = export_element_report_pdf(
                    self.spectrum,
                    hits,
                    Path(path),
                    n_lines=opts["n_lines"],
                    half_width_nm=opts["half_width_nm"],
                    atmosphere=atm,
                )
                msg = f"Wrote PDF report: {out}"
            else:
                path = QFileDialog.getExistingDirectory(
                    self,
                    "Choose folder for PNG figures",
                    str(ROOT / "reports"),
                )
                if not path:
                    return
                paths = export_element_report_pngs(
                    self.spectrum,
                    hits,
                    Path(path),
                    n_lines=opts["n_lines"],
                    half_width_nm=opts["half_width_nm"],
                    atmosphere=atm,
                )
                msg = f"Wrote {len(paths)} PNG figure(s) to {path}"
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return

        self.statusBar().showMessage(msg)
        QMessageBox.information(self, "Report exported", msg)

    def _hits_for_report(self, scope: str) -> list[ElementHit]:
        if scope == "selected":
            if not self._selected_elements:
                return []
            wanted = set(self._selected_elements)
            return [h for h in self.hits if h.element in wanted]
        if scope.startswith("top"):
            n = int(scope.split(":", 1)[1])
            return self.hits[:n]
        return list(self.hits)

    @staticmethod
    def _clear_table(table: QTableWidget) -> None:
        table.setRowCount(0)

    def _on_atmosphere_change(self, atm: str) -> None:
        if self.spectrum is not None:
            self._update_meta_panel()
            self._redraw(reset_view=False)
        self._sync_calibrate_context()
        if atm == "argon":
            self.statusBar().showMessage(
                "Tagged argon — Ar lines may appear; air N/O noise usually lower. "
                "Matching still uses the same NIST library."
            )
        else:
            self.statusBar().showMessage(f"Atmosphere tagged as {atm}.")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="LIBS Spectrum Explorer GUI")
    parser.add_argument("spectrum", nargs="?", type=Path, default=None)
    args, qt_args = parser.parse_known_args()

    venv_python = ROOT / ".venv" / "bin" / "python"
    if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
        print(
            f"Warning: not using project venv ({sys.executable}).\n"
            f"If Qt fails, run:  .venv/bin/python libs_gui.py",
            file=sys.stderr,
        )

    app = QApplication([sys.argv[0], *qt_args])
    app.setApplicationName("LIBS Spectrum Explorer")
    win = LibsExplorerWindow(spectrum_path=args.spectrum)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
