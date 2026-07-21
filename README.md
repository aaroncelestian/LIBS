# LIBS Spectrum Explorer

Interactive tools for Laser-Induced Breakdown Spectroscopy (LIBS): load spectra, detect peaks, match emission lines against the NIST Atomic Spectra Database, export publication figures, and build CRM univariate calibrations.

## Features

- Load spectrum (`.txt`) with optional instrument `.cfg`
- Peak detection and NIST ASD line matching
- Ranked element list with candidate lines per peak
- Interactive GUI (click peaks, add weak lines manually)
- Multi-file load: Single / Waterfall / Working modes, Mean/Sum, bulk match & CRM Quant
- Publication PDF report (strongest matched lines per element)
- Atmosphere tag (air / argon)
- **Calibrate** tab: CRM standards → diagnostic lines → intensity→concentration curves
- **Quant** tab: apply CRM fits to unknowns — results table (std / 95% CI), unknown on I→C curve, peak QC, C vs spectrum #

## Setup

Easiest — one command (creates `.venv`, installs deps; prompts to download NIST lines only if missing):

```bash
python3 install.py
```

Options: `python3 install.py --skip-nist` · `python3 install.py --force-nist` · `python3 install.py --venv .venv`

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
- **Bulk match** — Find peaks + NIST match on each checked file; review with Single + Prev/Next (`· matched` in the list).
- **Quant** — applies **existing Calibrate-tab CRM fits** to each checked spectrum (does not rebuild curves). Results open on the **Quant** tab.

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

1. **Add standards** — CRM spectra (`.txt`); matching `.cfg` is picked up automatically when present. **Replicate shots** of the same CRM (e.g. six 1500 ppm Pb files) are encouraged: each spectrum is a separate I→C point at the same C, so vertical scatter at that concentration is empirical LIBS variability. After adding files, use **Set C…** (or the prompt) to assign one concentration to all selected replicates.
2. **Add elements** — import may bring many CRM columns; **check only** the ones you want to fit / predict / plot (**Check all** / **Check none** helpers)
3. **Enter concentrations** — edit the table by hand, or **Import CSV** / **Export CSV**
4. **Suggest lines** — seeds diagnostic lines for checked elements from Identify matches (when available) or strong NIST lines; soft **overlap** warnings flag nearby lines from other elements
5. **Build calibration curves** — local baseline from narrow edge strips; **Gaussian** (default) or **Voigt** peak fit with allowed λ shift (or net-area fallback); linear/quadratic I→C fit per enabled line
6. **Quant unknowns from Identify** — use **Quant** on Identify (checked spectra); results open on the **Quant** tab. Calibrate **Curves & results** remains CRM QC only (peak fits + I→C for standards)

Measured peaks often sit 0.02–0.15 nm off NIST rest wavelengths (spectrometer calibration/drift, resolution, blends). Peak fitting finds the observed center within **Shift tol** and uses the **fitted area** as intensity. Identify matching already softens Δλ for strong lines; rebuild Calibrate curves after changing peak model or shift tol.

### Quant tab

After **Quant** on Identify (with Calibrate curves built):

1. **Results table** — mean concentration per element, line-to-line **std**, **95% CI** (`mean ± t·std/√n`), and `n_lines`
2. **Calibration curve** — CRM points + I→C fit for the selected element/line; unknown intensity→C overlaid as a star
3. **Peak / background fit** — local baseline and Gaussian/Voigt (or net-area) window for the selected spectrum and line
4. **Concentration vs spectrum #** — selected element across the Quant batch (time series / line scans), with line-to-line error bars and a series mean±std band when n≥2

Series summary (mean / std / 95% CI across spectra) appears above the plots. **Export CSV…** writes per-spectrum mean, std, CI, and n_lines.

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

- Signal: local continuum subtract, then **Gaussian/Voigt fitted area** (default) with λ shift tolerance, or trapezoid **net area**
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
| `publication_report.py` | PDF figures of strongest matched lines |
| `download_nist_lines.py` | Fetch / build the NIST line library |
| `matplotlib_config.py` | Shared matplotlib styling |
| `nist_lines/` | Downloaded line library (CSV) |
| `docs/` | Local spectra and scratch data (not tracked) |
| `reports/` | Generated report output |

## Requirements

- Python 3.10+
- See `requirements.txt`: `numpy`, `scipy`, `matplotlib`, `PySide6`, `requests`
