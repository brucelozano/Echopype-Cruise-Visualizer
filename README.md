# Echopype Cruise Visualization

Process EK80 `.raw` files into daily MVBS echogram HTML exports for cruise or mission review.

## What This Repo Does

- Reads EK80 `.raw` files (one day at a time in batch mode)
- Calibrates to `Sv`
- Computes `MVBS`
- Exports interactive HTML echograms (one file per channel/day)
- Optionally builds a timeline viewer to step through days

This workflow is EK80-general (not tied to one deployment type). Script names still use `lander` for backward compatibility.

Main scripts:

- `run_daily_lander_batch.py` - recommended production workflow
- `ek80_chunked_echogram.py` - single-run processing/debugging
- `analyze_mvbs_sidecars.py` - post-process `.mvbs.nc` sidecars for analysis
- `view_mvbs_sidecars.py` - live interactive viewer for `.mvbs.nc` sidecars
- `build_lander_timeline_viewer.py` - local timeline HTML viewer

## Install

- Python 3.10+

```cmd
python -m pip install -r requirements.txt
```

Command syntax note:

- Windows Command Prompt (`cmd.exe`) line continuation uses `^`
- PowerShell line continuation uses backtick `` ` `` (must be the final character on the line)

## Filename Requirement

Datetime filtering and daily batching expect filenames containing:

- `DYYYYMMDD-THHMMSS`

Example:

- `...D20250828-T230017.raw`

## Recommended Workflow

1. Run one daily batch export.
2. Confirm output HTMLs look correct.
3. Optionally rerun with different bins or color limits.
4. Build timeline viewer.

## Core Batch Command (Default-Style)

Command Prompt (`cmd.exe`):

```cmd
python run_daily_lander_batch.py ^
  --raw-dir "D:\Cruise\EK80\Raw" ^
  --output-dir ".\outputs\daily" ^
  --skip-existing
```

PowerShell:

```powershell
python run_daily_lander_batch.py `
  --raw-dir "D:\Cruise\EK80\Raw" `
  --output-dir ".\outputs\daily" `
  --skip-existing
```

## High-Detail Batch Command (Typical Pattern)

This is the recommended command to generate outputs for all channels/frequencies at 1m/1s binning (fill in input/output paths):

Command Prompt (`cmd.exe`):

```cmd
python run_daily_lander_batch.py ^
  --raw-dir "D:\Cruise\EK80\Raw" ^
  --range-meter-bin 1 ^
  --ping-time-bin 1s ^
  --vmin -85 ^
  --vmax -55 ^
  --output-dir ".\outputs\daily_1m_1s_rerun" ^
  --chunk-size 8 ^
  --transducer-facing auto
```

PowerShell:

```powershell
python run_daily_lander_batch.py `
  --raw-dir "D:\Cruise\EK80\Raw" `
  --range-meter-bin 1 `
  --ping-time-bin 1s `
  --vmin -85 `
  --vmax -55 `
  --output-dir ".\outputs\daily_1m_1s_rerun" `
  --chunk-size 8 `
  --transducer-facing auto
```

What those flags control:

- `--range-meter-bin 1` -> finer vertical binning
- `--ping-time-bin 1s` -> finer temporal binning
- `--vmin/--vmax` -> echogram color scale
- `--raw-dir` -> input data location
- `--output-dir` -> where per-day/channel HTML files are written
- `--chunk-size` -> speed/memory lever
- `--transducer-facing` -> forces/auto infers transducer facing direction

Output filename prefix behavior:

- By default, batch exports infer prefix from raw filenames before `DYYYYMMDD-THHMMSS`
  (for example, `DSB2_-D20250819-T150600.raw` -> prefix `DSB2`)
- If inference fails, prefix falls back to `lander`
- Prefix casing from raw filenames is preserved
- Override manually with `--output-prefix`

## Transducer Metadata and Orientation

Each run logs a per-day transducer metadata check to the day log file:

- Samples representative files per day (first/last file in that day)
- Logs transducer name/serial, transceiver serial, nominal frequencies, and `beam_direction_z`
- Checks consistency across sampled files and warns when metadata changes

Vertical plot orientation modes:

- `--transducer-facing auto` (default) -> use metadata inference when possible; fallback to down-looking
- `--transducer-facing down` -> force depth-style orientation (`invert_yaxis=True`)
- `--transducer-facing up` -> force up-looking orientation (`invert_yaxis=False`)

Notes:

- Many datasets do not populate `beam_direction_z`; in that case auto mode logs a warning and defaults to down-looking.
- For batch runs, pass the same option through `run_daily_lander_batch.py` using `--transducer-facing`.

## Hide Duty-Cycle Gaps (Experimental)

Use `--hide-na-gaps` to collapse all-NaN ping bins in display.  
Sv/MVBS values are not interpolated or modified.

Use a dedicated output folder for this mode:

Command Prompt (`cmd.exe`):

```cmd
python run_daily_lander_batch.py ^
  --raw-dir "D:\Cruise\EK80\Raw" ^
  --hide-na-gaps ^
  --output-dir ".\outputs\daily_hide_na" ^
  --skip-existing
```

PowerShell:

```powershell
python run_daily_lander_batch.py `
  --raw-dir "D:\Cruise\EK80\Raw" `
  --hide-na-gaps `
  --output-dir ".\outputs\daily_hide_na" `
  --skip-existing
```

## Single-Run Script (Optional)

Useful for quick checks before full batch runs:

Command Prompt (`cmd.exe`):

```cmd
python ek80_chunked_echogram.py ^
  --raw-dir "D:\Cruise\EK80\Raw" ^
  --start-datetime 2025-08-28 ^
  --duration-days 1 ^
  --ui-mode static ^
  --save-html ".\outputs\one_day_check.html"
```

PowerShell:

```powershell
python ek80_chunked_echogram.py `
  --raw-dir "D:\Cruise\EK80\Raw" `
  --start-datetime 2025-08-28 `
  --duration-days 1 `
  --ui-mode static `
  --save-html ".\outputs\one_day_check.html"
```

## Plot-Data Sidecar Export

When an HTML export is saved, the scripts now also export plot-consistent MVBS sidecar data by default.

Default behavior:

- `--export-plot-data netcdf` (default) writes `.mvbs.nc` sidecars
- Sidecar base name matches HTML base name, for example:
  - `DSB2_20250819__es70_70khz.html`
  - `DSB2_20250819__es70_70khz.mvbs.nc`

Useful flags:

- `--export-plot-data none|netcdf|csv|both`
- `--plot-data-dir <path>` (optional; defaults to HTML directory)
- `--plot-data-csv-compression none|gzip` (used when CSV is requested)

Examples:

- Disable sidecars entirely:

```powershell
python run_daily_lander_batch.py `
  --raw-dir "D:\Cruise\EK80\Raw" `
  --output-dir ".\outputs\daily" `
  --export-plot-data none
```

- Write both NetCDF and compressed CSV sidecars:

```powershell
python run_daily_lander_batch.py `
  --raw-dir "D:\Cruise\EK80\Raw" `
  --output-dir ".\outputs\daily" `
  --export-plot-data both `
  --plot-data-csv-compression gzip
```

## Post-Process MVBS Sidecars

Analysis now runs as a separate post-processing step on exported `.mvbs.nc` files.
This avoids rerunning raw processing when tuning analysis windows.

Run scalar analysis over all sidecars in an output directory:

```powershell
python analyze_mvbs_sidecars.py `
  --input-dir ".\outputs\daily_1m_1s_analysis" `
  --analysis-mode scalar `
  --analysis-depth-min 20 `
  --analysis-depth-max 80 `
  --output-jsonl ".\outputs\daily_1m_1s_analysis\analysis_scalar.jsonl"
```

Run mean-vs-time analysis for one channel and export profile CSVs:

```powershell
python analyze_mvbs_sidecars.py `
  --input-dir ".\outputs\daily_1m_1s_analysis" `
  --analysis-mode mean-vs-time `
  --channel-filter "70 kHz" `
  --analysis-depth-min 20 `
  --analysis-depth-max 80 `
  --profile-csv-dir ".\outputs\daily_1m_1s_analysis\profiles" `
  --output-jsonl ".\outputs\daily_1m_1s_analysis\analysis_mean_vs_time.jsonl"
```

Common analysis flags:

- `--analysis-mode mean-vs-depth|mean-vs-time|scalar` (required)
- `--channel <exact channel>` or `--channel-filter <substring>` (optional)
- `--analysis-time-start <datetime>` / `--analysis-time-end <datetime>` (optional)
- `--analysis-depth-min <float>` / `--analysis-depth-max <float>` (optional)
- `--output-jsonl <path.jsonl>` (optional)
- `--profile-csv-dir <dir>` (optional for profile modes)

## Live Viewer from Sidecars

Use `view_mvbs_sidecars.py` to open an interactive Panel app directly from
`.mvbs.nc` sidecars. This provides real-time colormap and dB-range tuning plus
pan/zoom interactions without reprocessing `.raw` files.
Default colormap for this viewer is `viridis` to match batch export defaults.

Example (single day):

```powershell
python view_mvbs_sidecars.py `
  --input-dir ".\outputs\daily_1m_1s_netcdf_test" `
  --date 20250820 `
  --vmin -85 `
  --vmax -55
```

Optional filters and export controls:

- `--channel-filter "70 kHz"`
- `--hide-na-gaps`
- `--save-html ".\outputs\mvbs_sidecar_view.html"`
- `--export-plot-data none|netcdf|csv|both`

## Interpretation and Export Limits

- Post-process analysis uses exported `MVBS` sidecars (binned Sv), not full-resolution ping-by-ping `Sv`.
- Mean-Sv calculations use linear-domain averaging internally before converting back to dB.
- For small/short ROIs, use finer MVBS bins (`--range-meter-bin`, `--ping-time-bin`) to reduce binning bias.
- Existing exported echogram HTML files are rasterized visualization artifacts; they do not contain enough numeric MVBS data to reliably recompute new ROI statistics post hoc.
- Use exported `.mvbs.nc` sidecars for repeatable post-hoc analysis without rerunning raw processing.

## Timeline Viewer

Build viewer:

Command Prompt (`cmd.exe`):

```cmd
python build_lander_timeline_viewer.py ^
  --input-dir ".\outputs\daily"
```

PowerShell:

```powershell
python build_lander_timeline_viewer.py `
  --input-dir ".\outputs\daily"
```

Serve locally (recommended for browser compatibility):

```cmd
python -m http.server 8000 --directory ".\outputs\daily"
```

Open:

- `http://localhost:8000/<prefix>_timeline_viewer.html` (single-prefix directory)
- `http://localhost:8000/timeline_viewer.html` (mixed-prefix directory)

## Path Handling (Portable)

Prefer explicit CLI paths for sharing scripts across machines.

Optional env vars:

- `EK80_RAW_DIR`
- `EK80_TIMELINE_INPUT_DIR`
- `EK80_OUTPUT_DIR` (timeline fallback)

PowerShell example:

```powershell
$env:EK80_RAW_DIR = "D:\Cruise\EK80\Raw"
python run_daily_lander_batch.py --skip-existing
```

## HTML Resource Mode

Both processing scripts support:

- `--html-resources inline` (default) -> larger files, works offline
- `--html-resources cdn` -> smaller files, needs internet access for Bokeh assets

## Memory and Performance Tips

`--chunk-size` is the main speed/memory lever.

| Machine RAM | Suggested start for `--chunk-size` |
| --- | --- |
| 16 GB | 2 to 5 |
| 32 GB | 8 to 12 |
| 64 GB+ | 15 to 25 |

Guidelines:

- Tune on one day first, then scale to full date ranges.
- Fine bins (`1m`, `1s`) increase processing load and output size.
- Batch-by-day is safest for memory and fault isolation.

## Notes

- For `--channel`, use the exact channel string printed by the script.
- Hide-NA and normal exports should use separate output directories when `--skip-existing` is enabled.
- `ek80_chunked_echogram.html` is a transient preview file from Bokeh `show()`; your important artifacts are `outputs/.../<prefix>_*.html`.
- Timeline viewer discovers files named `<prefix>_YYYYMMDD.html` or `<prefix>_YYYYMMDD__<channel>.html`.
