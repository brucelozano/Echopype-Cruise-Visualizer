# Echopype Cruise Visualization

Process EK80 `.raw` files into daily MVBS echogram HTML exports for cruise or mission review.

## What This Repo Does

- Reads EK80 `.raw` files (one day at a time in batch mode)
- Calibrates to `Sv`
- Computes `MVBS`
- Exports interactive HTML echograms (one file per channel/day)
- Optionally builds a timeline viewer to step through days

This workflow is EK80-general (not tied to one deployment type).

Main scripts:

- `run_daily_batch.py` - recommended production workflow
- `ek80_chunked_echogram.py` - single-run processing/debugging
- `analyze_mvbs_outputs.py` - post-process `.mvbs.nc` outputs for analysis
- `view_mvbs_outputs.py` - live interactive viewer for `.mvbs.nc` outputs
- `build_cruise_timeline_viewer.py` - local timeline HTML viewer



## Install

- Python 3.10+

```cmd
python -m pip install -r requirements.txt
```

Command syntax note:

- Windows Command Prompt (`cmd.exe`) line continuation uses `^`
- PowerShell line continuation uses backtick ``` (must be the final character on the line)



## Filename Requirement

Datetime filtering and daily batching expect filenames containing:

- `DYYYYMMDD-THHMMSS`

Example:

- `...D20250828-T230017.raw`



## Recommended Workflow

1. Run one daily batch export (NetCDF outputs by default).
2. Inspect NetCDF outputs with `view_mvbs_outputs.py`.
3. Run post-process analysis with `analyze_mvbs_outputs.py`.
4. Optionally generate ready-to-open HTML snapshots from NetCDF outputs (standalone or as a post-batch step).



## Core Batch Command (Default NetCDF-First)

Command Prompt (`cmd.exe`):

```cmd
python run_daily_batch.py ^
  --raw-dir "D:\Cruise\EK80\Raw" ^
  --output-dir ".\outputs\daily" ^
  --skip-existing
```

PowerShell:

```powershell
python run_daily_batch.py `
  --raw-dir "D:\Cruise\EK80\Raw" `
  --output-dir ".\outputs\daily" `
  --skip-existing
```

Default output behavior in batch mode:

- `--output-type netcdf` (default): NetCDF outputs only (`.mvbs.nc` by default), no HTML files
- `--output-type both`: save HTML + NetCDF outputs
- `--output-type html`: save HTML only

Default display settings (batch + viewer/headless export):

- `--cmap viridis`
- `--vmin -90`
- `--vmax -55`
- `--hide-na-gaps` enabled (use `--no-hide-na-gaps` to disable)
- `--flip-vertical` enabled (use `--no-flip-vertical` to disable)



## High-Detail Batch Command (Typical Pattern)

This is the recommended command to generate outputs for all channels/frequencies at 1m/1s binning (fill in input/output paths):

Command Prompt (`cmd.exe`):

```cmd
python run_daily_batch.py ^
  --raw-dir "D:\Cruise\EK80\Raw" ^
  --range-meter-bin 1 ^
  --ping-time-bin 1s ^
  --vmin -90 ^
  --vmax -55 ^
  --output-dir ".\outputs\daily_1m_1s_rerun" ^
  --chunk-size 8 ^
  --transducer-facing auto
```

PowerShell:

```powershell
python run_daily_batch.py `
  --raw-dir "D:\Cruise\EK80\Raw" `
  --range-meter-bin 1 `
  --ping-time-bin 1s `
  --vmin -90 `
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
- `--output-dir` -> where per-day/channel outputs are written
- `--chunk-size` -> speed/memory lever
- `--transducer-facing` -> forces/auto infers transducer facing direction
- `--output-type` -> choose `netcdf`, `both`, or `html`

Output filename prefix behavior:

- By default, batch exports infer prefix from raw filenames before `DYYYYMMDD-THHMMSS`
(for example, `DSB2_-D20250819-T150600.raw` -> prefix `DSB2`)
- If inference fails, prefix falls back to `cruise`
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
- `--flip-vertical` -> flip the rendered y-axis after orientation is resolved

Notes:

- Many datasets do not populate `beam_direction_z`; in that case auto mode logs a warning and defaults to down-looking.
- For batch runs, pass the same options through `run_daily_batch.py` using
`--transducer-facing` and optionally `--flip-vertical`.



## Hide Duty-Cycle Gaps (Experimental)

Hide-NA rendering is enabled by default.  
Use `--hide-na-gaps` explicitly if you want, or `--no-hide-na-gaps` to disable.  
Sv/MVBS values are not interpolated or modified.

Use a dedicated output folder for this mode:

Command Prompt (`cmd.exe`):

```cmd
python run_daily_batch.py ^
  --raw-dir "D:\Cruise\EK80\Raw" ^
  --hide-na-gaps ^
  --output-dir ".\outputs\daily_hide_na" ^
  --skip-existing
```

PowerShell:

```powershell
python run_daily_batch.py `
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



## MVBS Data Output Export

The scripts can export plot-consistent MVBS data outputs (`.mvbs.nc` / `.mvbs.csv`).
In batch mode, NetCDF outputs are the default primary output type.

Default behavior:

- `run_daily_batch.py` defaults to `--output-type netcdf`
- `--data-output-format netcdf` (default) writes `.mvbs.nc` outputs
- Output base name matches the HTML (or would-be HTML) base name, for example:
  - `DSB2_20250819__es70_70khz.html`
  - `DSB2_20250819__es70_70khz.mvbs.nc`

Useful flags:

- `--output-type netcdf|both|html` (batch wrapper)
- `--data-output-format none|netcdf|csv|both`
- `--data-output-dir <path>` (optional; defaults to output directory)
- `--data-csv-compression none|gzip` (used when CSV is requested)

Examples:

- NetCDF-only daily outputs (default behavior shown explicitly):

```powershell
python run_daily_batch.py `
  --raw-dir "D:\Cruise\EK80\Raw" `
  --output-dir ".\outputs\daily_netcdf" `
  --output-type netcdf `
  --data-output-format netcdf
```

- Keep both HTML and NetCDF outputs:

```powershell
python run_daily_batch.py `
  --raw-dir "D:\Cruise\EK80\Raw" `
  --output-dir ".\outputs\daily_both" `
  --output-type both `
  --data-output-format netcdf
```

- Disable data outputs entirely:

```powershell
python run_daily_batch.py `
  --raw-dir "D:\Cruise\EK80\Raw" `
  --output-dir ".\outputs\daily" `
  --output-type html `
  --data-output-format none
```

- Write both NetCDF and compressed CSV outputs:

```powershell
python run_daily_batch.py `
  --raw-dir "D:\Cruise\EK80\Raw" `
  --output-dir ".\outputs\daily" `
  --data-output-format both `
  --data-csv-compression gzip
```



## Post-Process MVBS NetCDF Outputs

Analysis now runs as a separate post-processing step on exported `.mvbs.nc` files.
This avoids rerunning raw processing when tuning analysis windows.

Run scalar analysis over all outputs in an output directory:

```powershell
python analyze_mvbs_outputs.py `
  --input-dir ".\outputs\daily_1m_1s_analysis" `
  --analysis-mode scalar `
  --analysis-depth-min 20 `
  --analysis-depth-max 80 `
  --output-jsonl ".\outputs\daily_1m_1s_analysis\analysis_scalar.jsonl"
```

Run mean-vs-time analysis for one channel and export profile CSVs:

```powershell
python analyze_mvbs_outputs.py `
  --input-dir ".\outputs\daily_1m_1s_analysis" `
  --analysis-mode mean-vs-time `
  --channel-filter "ES70-7CD" `
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



## Live Viewer from NetCDF Outputs

Use `view_mvbs_outputs.py` to open an interactive Panel app directly from
`.mvbs.nc` outputs. This provides real-time colormap and dB-range tuning plus
pan/zoom interactions without reprocessing `.raw` files.
Viewer defaults: `--cmap viridis`, `--vmin -90`, `--vmax -55`,
`--hide-na-gaps` enabled, and `--flip-vertical` enabled.

Example (single day):

```powershell
python view_mvbs_outputs.py `
  --input-dir ".\outputs\daily_1m_1s_netcdf_test" `
  --date 20250820 `
  --vmin -90 `
  --vmax -55
```

Example (three-day stitched window from NetCDF outputs):

```powershell
python view_mvbs_outputs.py `
  --input-dir ".\outputs\daily_1m_1s_netcdf_test" `
  --date 20250819 `
  --window-days 3 `
  --flip-vertical `
  --vmin -90 `
  --vmax -55
```

Optional filters and export controls:

- `--channel-filter "ES70-7CD"`
- `--no-hide-na-gaps` (disable default gap hiding)
- `--window-days 3` (requires `--date`; stitches consecutive dates)
- `--no-flip-vertical` (disable default vertical flip)
- `--save-html ".\outputs\mvbs_netcdf_view.html"`
- `--data-output-format none|netcdf|csv|both`



## Headless HTML Export from NetCDF Outputs

You can generate static HTML snapshots directly from `.mvbs.nc` outputs without opening
the live Panel app. This is useful for producing ready-to-view daily pages after
NetCDF-first processing.

Standalone headless export:

```powershell
python view_mvbs_outputs.py `
  --input-dir ".\outputs\daily_1m_1s_netcdf" `
  --glob-pattern "*.mvbs.nc" `
  --export-html-dir ".\outputs\daily_1m_1s_netcdf\viewer_html" `
  --cmap viridis `
  --vmin -90 `
  --vmax -55 `
  --hide-na-gaps `
  --flip-vertical
```

What this does:

- Writes one HTML file per matched `.mvbs.nc` file (date/channel naming is preserved)
- Applies display settings (`--cmap`, `--vmin`, `--vmax`, `--hide-na-gaps`, `--flip-vertical`)
- Creates an index page by default (`index.html`) in `--export-html-dir`

Useful headless flags:

- `--export-html-index-name <name>.html` (rename index page)
- `--export-html-no-index` (skip index page creation)
- `--channel-filter "<substring>"` (export only selected channels)



## Post-Batch Auto HTML Export

To keep NetCDF as the primary pipeline output while still getting ready-to-open HTML
snapshots automatically, add these flags to `run_daily_batch.py`:

```powershell
python run_daily_batch.py `
  --raw-dir "D:\Cruise\EK80\Raw" `
  --output-dir ".\outputs\daily_1m_1s_netcdf" `
  --output-type netcdf `
  --data-output-format netcdf `
  --viewer-export-html `
  --viewer-export-dir ".\outputs\daily_1m_1s_netcdf\viewer_html" `
  --vmin -90 `
  --vmax -55 `
  --cmap viridis `
  --hide-na-gaps `
  --flip-vertical
```

Notes:

- `--viewer-export-html` runs after daily NetCDF files are produced.
- It calls `view_mvbs_outputs.py` in headless mode (no browser server).
- Use `--viewer-export-no-index` to disable index generation.
- Default glob pattern for this step is `<output-prefix>_*.mvbs.nc` (override with `--viewer-export-glob-pattern`).
- To disable the new default render behavior, add `--no-hide-na-gaps` and/or `--no-flip-vertical`.



## Interpretation and Export Limits

- Post-process analysis uses exported `MVBS` outputs (binned Sv), not full-resolution ping-by-ping `Sv`.
- Mean-Sv calculations use linear-domain averaging internally before converting back to dB.
- For small/short ROIs, use finer MVBS bins (`--range-meter-bin`, `--ping-time-bin`) to reduce binning bias.
- Existing exported echogram HTML files are rasterized visualization outputs; they do not contain enough numeric MVBS data to reliably recompute new ROI statistics post hoc.
- Use exported `.mvbs.nc` outputs for repeatable post-hoc analysis without rerunning raw processing.



## Timeline Viewer

Build viewer:

Command Prompt (`cmd.exe`):

```cmd
python build_cruise_timeline_viewer.py ^
  --input-dir ".\outputs\daily"
```

PowerShell:

```powershell
python build_cruise_timeline_viewer.py `
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
python run_daily_batch.py --skip-existing
```



## HTML Resource Mode

Both processing scripts support:

- `--html-resources inline` (default) -> larger files, works offline
- `--html-resources cdn` -> smaller files, needs internet access for Bokeh assets



## Memory and Performance Tips

`--chunk-size` is the main speed/memory lever.


| Machine RAM | Suggested start for `--chunk-size` |
| ----------- | ---------------------------------- |
| 16 GB       | 2 to 5                             |
| 32 GB       | 8 to 12                            |
| 64 GB+      | 15 to 25                           |


Guidelines:

- Tune on one day first, then scale to full date ranges.
- Fine bins (`1m`, `1s`) increase processing load and output size.
- Batch-by-day is safest for memory and fault isolation.



## Notes

- For `--channel`, use the exact channel string printed by the script.
- Hide-NA and normal exports should use separate output directories when `--skip-existing` is enabled.
- `ek80_chunked_echogram.html` is a transient preview file from Bokeh `show()`; your primary outputs are `.mvbs.nc` files (and optional HTML exports when `--output-type` includes HTML).
- Timeline viewer discovers files named `<prefix>_YYYYMMDD.html` or `<prefix>_YYYYMMDD__<channel>.html`.

