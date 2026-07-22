# LIBS Spectrum Explorer

Interactive tools for Laser-Induced Breakdown Spectroscopy (LIBS): load spectra, detect peaks, match emission lines against the NIST Atomic Spectra Database, export publication figures, and build CRM univariate calibrations.

## Features

- Load spectrum (`.txt`) with optional instrument `.cfg`
- Peak detection and NIST ASD line matching
- Ranked element list with candidate lines per peak
- Interactive GUI (click peaks, add weak lines manually)
- Multi-file load: Single / Waterfall / Working modes, Mean/Sum, bulk match & CRM Quant
- Word (.docx) identification report (element values + two strongest-line spectra)
- Atmosphere tag (air / argon)
- **Calibrate** tab: CRM standards → diagnostic lines → intensity→concentration curves
- **Quant** tab: apply CRM fits to unknowns — results table (std / 95% CI), unknown on I→C curve, peak QC, C vs spectrum #

## Setup

Easiest — one command (creates `.venv`, installs deps; prompts to download NIST lines only if missing):

```bash
python3 install.py
```

Options: `python3 install.py --skip-nist` · `python3 install.py --force-nist` · `python3 install.py --venv .venv` · `python3 install.py --online`

### Offline / air-gapped Windows lab PCs

Those machines cannot reach PyPI. Bundle wheels on a connected PC first:

```bash
# Match the lab Python (Anaconda log showed 3.12) and Windows x64:
python download_wheels.py --platform win_amd64 --python 3.12
```

Copy the **entire** LIBS folder (including `wheels/` and `nist_lines/`) to the lab PC, then:

```powershell
cd C:\LIBS\LIBS
python install.py
```

`install.py` detects `wheels/` and installs with `--no-index` (no internet). NIST is already shipped under `nist_lines/` when present.

**Windows note:** if you see `getaddrinfo failed` with an empty `wheels/` folder, the PC is offline — run `download_wheels.py` elsewhere and re-copy.

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## NIST line library

`install.py` fetches this automatically. To download (or refresh) by hand:

```bash
python download_nist_lines.py
```

Emission lines (H–U, ions I–III, ~180–1022 nm) go into `nist_lines/`. Cite [NIST ASD](https://physics.nist.gov/PhysRefData/ASD/lines_form.html) in any resulting work.

## Usage

### GUI

```bash
python libs_gui.py
python libs_gui.py path/to/spectrum.txt
```

Tabs:

- **Identify** — load spectrum(s), find peaks, match NIST lines, export reports
- **Calibrate** — CRM univariate calibration (see below)
- **Quant** — apply calibration curves to unknowns; inspect fits and concentration trends

### Multi-spot / multi-file spectra

Open several `.txt` files at once (or **Add spectra** / drag-and-drop). The left **Spectra** panel supports:

| Mode | Behavior |
|------|----------|
| **Single** | One file at a time; **Prev / Next** (or list double-click). Match results are cached per file. |
| **Waterfall** | Checked spectra with vertical offsets. Use **Offset %** (gap as % of max intensity; default 15%). |
| **Working only** | Active spectrum only (a single file, or Mean/Sum combine). |

- **Mean / Sum** — combine checked rows, add the result to the Spectra list (e.g. `sum_of_11.txt`), and make it active. **Export sum…** writes that result to disk and lists the saved file.
- **Match** — Find peaks + NIST match on each **checked** file (or highlighted rows if none checked); review with Single + Prev/Next (`· matched` in the list). Waterfall mode switches to Single so rankings/peaks are visible.
- **Quant** — applies **existing Calibrate-tab CRM fits** to each checked spectrum (does not rebuild curves). Results open on the **Quant** tab.

Opening multiple files stays on **Single** (first spectrum); it does **not** auto-mean.

### CLI identification

```bash
python identify_elements.py path/to/spectrum.txt
python identify_elements.py path/to/spectrum.txt --plot
```

### Identification report

```bash
python publication_report.py path/to/spectrum.txt -o reports/out.docx
```

Exports a Word document: table of element values, then spectra of the two most intense matched lines per element.
## Calibration (CRM univariate)

On the **Calibrate** tab:

1. **Add standards** — CRM spectra (`.txt`); matching `.cfg` is picked up automatically when present. **Replicate shots** of the same CRM (e.g. six 1500 ppm Pb files) are encouraged: each spectrum is a separate I→C point at the same C, so vertical scatter at that concentration is empirical LIBS variability. After adding files, use **Set C…** (or the prompt) to assign one concentration to all selected replicates.
2. **Add elements** — import may bring many CRM columns; **check only** the ones you want to fit / predict / plot (**Check all** / **Check none** helpers)
3. **Enter concentrations** — edit the table by hand, or **Import CSV** / **Export CSV**
4. **Suggest lines** — offers up to **N** lines/element (default 4): preferred calibrants first (Ca II IR, K 766/770, Na D, …), then Identify matches, then strong NIST. **1 good line is enough** for an I→C curve; 2–4 support multi-line means and dropping bad λ. Soft **overlap** warnings flag nearby lines from other elements. **Hover** a diagnostic-line row to preview the peak on the QC spectrum.
5. **Build calibration curves** — local **SNIP** baseline (RamanLab peak-clipping; linear/flat also available) + Gaussian/Voigt (or net-area) with λ shift; linear/quadratic I→C fit per enabled line. **Negative-slope** curves are still plotted for QC but marked **QC-only** and excluded from Quant. On **Curves & results**, uncheck a line under **Use lines** (or **Exclude selected**) to drop a bad λ from Quant without waiting for rebuild; re-enable and rebuild to restore.
6. **Quant unknowns** — on the **Quant** tab, check elements and spectra then **Quant**; or use **Quant** on Identify for checked spectra. Calibrate **Curves & results** remains CRM QC only (peak fits + I→C for standards)

Measured peaks often sit 0.02–0.15 nm off NIST rest wavelengths (spectrometer calibration/drift, resolution, blends). Peak fitting finds the observed center within **Shift tol** and uses the **fitted area** as intensity. Identify matching already softens Δλ for strong lines; rebuild Calibrate curves after changing peak model or shift tol.

### Quant tab

Check **Elements** and **Spectra** (multi-select), then **Quant** on this tab — or use **Quant** on Identify for checked spectra. Requires Calibrate curves first.

1. **Results table** — concentrations for checked elements (std + 95% CI per cell group)
2. **Calibration curve** — CRM points + I→C for the highlighted element/line; unknown as a star
3. **Peak / background fit** — local baseline and fit for the highlighted spectrum and line
4. **Concentration vs spectrum #** — checked elements across the Quant batch

Series summary appears above the plots. **Export CSV…** writes checked spectra/elements.

### Concentrations CSV format

```text
standard_id,Fe,Ca,Si
CRM1,12.5,8.1,22.0
CRM2,5.0,,18.4
```

- First column is `standard_id` (must match the standard’s id, or the spectrum filename stem)
- Blank cells mean that element is not calibrated for that standard
- Units: set **Unit** on **Known concentrations** (wt%, ppm, mg/kg, µg/g, mass frac, …). Switching among mass units converts entered CRM values automatically; predictions use the same unit. `at%` / `oxide wt%` are labels only (no conversion).

### Session files

**Save session** / **Load session** writes a JSON file with spectrum paths, concentrations, selected lines, and fit coefficients so you can reload a calibration later.

### Method notes

- Signal: **SNIP** continuum (or edge linear/flat), then **Gaussian/Voigt fitted area** (default) with λ shift tolerance, or trapezoid **net area**
- Overlaps: warned, not auto-rejected
- Multi-line: average enabled lines that have a valid fit
- Not included (yet): CF-LIBS / fundamental parameters, global λ recalibration, hard overlap rejection

Match acquisition conditions (gate, laser energy, atmosphere) across standards and unknowns; the `.cfg` meta panel surfaces those parameters.

## Project layout

| Path | Description |
|------|-------------|
| `libs_gui.py` | Interactive PySide6 spectrum explorer (Identify + Calibrate + Quant) |
| `calibration.py` | CRM calibration math, CSV/JSON I/O, quant predictions |
| `calibration_gui.py` | Calibrate tab UI |
| `quant_gui.py` | Quant tab UI (results, curve overlay, peak QC, series plot) |
| `identify_elements.py` | Peak finding and NIST matching |
| `publication_report.py` | Word (.docx) ID report (values + 2 strongest lines) |
| `download_nist_lines.py` | Fetch / build the NIST line library |
| `matplotlib_config.py` | Shared matplotlib styling |
| `nist_lines/` | Downloaded line library (CSV) |
| `docs/` | Local spectra and scratch data (not tracked) |
| `reports/` | Generated report output |

## Requirements

- Python 3.10+
- See `requirements.txt`: `numpy`, `scipy`, `matplotlib`, `PySide6`, `requests`
