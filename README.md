# LIBS Spectrum Explorer

Interactive tools for Laser-Induced Breakdown Spectroscopy (LIBS): load spectra, detect peaks, match emission lines against the NIST Atomic Spectra Database, export publication figures, and build CRM univariate calibrations.

## Features

- Load spectrum (`.txt`) with optional instrument `.cfg`
- Peak detection and NIST ASD line matching
- Ranked element list with candidate lines per peak
- Interactive GUI (click peaks, add weak lines manually)
- Multi-file load: Single / Waterfall / Working modes, Mean/Sum, bulk match & CRM bulk quant
- Publication PDF report (strongest matched lines per element)
- Atmosphere tag (air / argon)
- **Calibrate** tab: CRM standards → diagnostic lines → intensity→concentration curves → apply to unknowns

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## NIST line library

Download emission lines (H–U, ions I–III, ~180–1022 nm) into `nist_lines/`:

```bash
python download_nist_lines.py
```

Cite [NIST ASD](https://physics.nist.gov/PhysRefData/ASD/lines_form.html) in any resulting work.

## Usage

### GUI

```bash
python libs_gui.py
python libs_gui.py path/to/spectrum.txt
```

Tabs:

- **Identify** — load spectrum(s), find peaks, match NIST lines, export reports
- **Calibrate** — CRM univariate calibration (see below)

### Multi-spot / multi-file spectra

Open several `.txt` files at once (or **Add spectra** / drag-and-drop). The left **Spectra** panel supports:

| Mode | Behavior |
|------|----------|
| **Single** | One file at a time; **Prev / Next** (or list double-click). Match results are cached per file. |
| **Waterfall** | Checked spectra with vertical offsets. Use **Offset %** (gap as % of max intensity; default 15%). |
| **Working only** | Active spectrum only (a single file, or Mean/Sum combine). |

- **Mean / Sum** — combine checked rows, add the result to the Spectra list (e.g. `sum_of_11.txt`), and make it active. **Export sum…** writes that result to disk and lists the saved file.
- **Bulk match** — Find peaks + NIST match on each checked file; review with Single + Prev/Next (`· matched` in the list).
- **Bulk quant** — applies **existing Calibrate-tab CRM fits** to each checked spectrum (does not rebuild curves). Results appear on the Identify **Batch** side-tab: table, concentration vs spectrum # plot, and CSV export.

Opening multiple files stays on **Single** (first spectrum); it does **not** auto-mean.

### CLI identification

```bash
python identify_elements.py path/to/spectrum.txt
python identify_elements.py path/to/spectrum.txt --plot
```

### Publication report

```bash
python publication_report.py path/to/spectrum.txt -o reports/out.pdf
```

## Calibration (CRM univariate)

On the **Calibrate** tab:

1. **Add standards** — CRM spectra (`.txt`); matching `.cfg` is picked up automatically when present
2. **Add elements** — import may bring many CRM columns; **check only** the ones you want to fit / predict / plot (**Check all** / **Check none** helpers)
3. **Enter concentrations** — edit the table by hand, or **Import CSV** / **Export CSV**
4. **Suggest lines** — seeds diagnostic lines for checked elements from Identify matches (when available) or strong NIST lines; soft **overlap** warnings flag nearby lines from other elements
5. **Build calibration curves** — local baseline subtraction, net peak area integration, linear (or quadratic) I→C fit per enabled line
6. **Apply to unknown** — use the Identify spectrum or browse another; predictions, bar plot, and **Export predictions** cover only the checked subset

### Concentrations CSV format

```text
standard_id,Fe,Ca,Si
CRM1,12.5,8.1,22.0
CRM2,5.0,,18.4
```

- First column is `standard_id` (must match the standard’s id, or the spectrum filename stem)
- Blank cells mean that element is not calibrated for that standard
- Units: set **Conc. unit** on the Calibrate tab (wt%, ppm, …). The fit does not convert units — enter CRM values and read predictions in that same unit.

### Session files

**Save session** / **Load session** writes a JSON file with spectrum paths, concentrations, selected lines, and fit coefficients so you can reload a calibration later.

### Method notes

- Signal: local continuum subtract around each diagnostic λ, then net peak **area**
- Overlaps: warned, not auto-rejected
- Multi-line: average enabled lines that have a valid fit
- Not included (yet): CF-LIBS / fundamental parameters, Voigt deconvolution, hard overlap rejection

Match acquisition conditions (gate, laser energy, atmosphere) across standards and unknowns; the `.cfg` meta panel surfaces those parameters.

## Project layout

| Path | Description |
|------|-------------|
| `libs_gui.py` | Interactive PySide6 spectrum explorer (Identify + Calibrate tabs) |
| `calibration.py` | CRM calibration math, CSV/JSON I/O |
| `calibration_gui.py` | Calibrate tab UI |
| `identify_elements.py` | Peak finding and NIST matching |
| `publication_report.py` | PDF figures of strongest matched lines |
| `download_nist_lines.py` | Fetch / build the NIST line library |
| `matplotlib_config.py` | Shared matplotlib styling |
| `nist_lines/` | Downloaded line library (CSV) |
| `docs/` | Local spectra and scratch data (not tracked) |
| `reports/` | Generated report output |

## Requirements

- Python 3.10+
- See `requirements.txt`: `numpy`, `scipy`, `matplotlib`, `PySide6`, `requests`
