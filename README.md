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
- `build_lander_timeline_viewer.py` - local timeline HTML viewer

## Install

- Python 3.10+

```bash
python -m pip install -r requirements.txt
```

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

```bash
python run_daily_lander_batch.py \
  --raw-dir "/path/to/ek80_raw" \
  --output-dir "./outputs/daily" \
  --skip-existing
```

## High-Detail Batch Command (Typical Pattern)

This is the recommended command to generate outputs for all channels/frequencies at 1m/1s binning (fill in input/output paths):

```bash
python run_daily_lander_batch.py \
  --raw-dir "/path/to/ek80_raw" \
  --range-meter-bin 1 \
  --ping-time-bin 1s \
  --vmin -85 \
  --vmax -55 \
  --output-dir "./outputs/daily_1m_1s_rerun"
```

What those flags control:

- `--range-meter-bin 1` -> finer vertical binning
- `--ping-time-bin 1s` -> finer temporal binning
- `--vmin/--vmax` -> echogram color scale
- `--raw-dir` -> input data location
- `--output-dir` -> where per-day/channel HTML files are written

Output filename prefix behavior:

- By default, batch exports infer prefix from raw filenames before `DYYYYMMDD-THHMMSS`
  (for example, `DSB2_-D20250819-T150600.raw` -> prefix `DSB2`)
- If inference fails, prefix falls back to `lander`
- Prefix casing from raw filenames is preserved
- Override manually with `--output-prefix`

## Hide Duty-Cycle Gaps (Experimental)

Use `--hide-na-gaps` to collapse all-NaN ping bins in display.  
Sv/MVBS values are not interpolated or modified.

Use a dedicated output folder for this mode:

```bash
python run_daily_lander_batch.py \
  --raw-dir "/path/to/ek80_raw" \
  --hide-na-gaps \
  --output-dir "./outputs/daily_hide_na" \
  --skip-existing
```

## Single-Run Script (Optional)

Useful for quick checks before full batch runs:

```bash
python ek80_chunked_echogram.py \
  --raw-dir "/path/to/ek80_raw" \
  --start-datetime 2025-08-28 \
  --duration-days 1 \
  --ui-mode static \
  --save-html "./outputs/one_day_check.html"
```

## Timeline Viewer

Build viewer:

```bash
python build_lander_timeline_viewer.py \
  --input-dir "./outputs/daily"
```

Serve locally (recommended for browser compatibility):

```bash
python -m http.server 8000 --directory "./outputs/daily"
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
