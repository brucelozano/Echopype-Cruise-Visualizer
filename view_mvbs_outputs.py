"""Viewer/export utility for exported MVBS NetCDF outputs.

This script can either:
- serve a live interactive Panel app from `.mvbs.nc` outputs, or
- run a headless static HTML export pass from `.mvbs.nc` outputs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import dask
import holoviews as hv
import numpy as np
import xarray as xr

from channel_naming import infer_channel_frequency_khz
from ek80_chunked_echogram import (
    build_plot_dataarray,
    configure_logging,
    create_echogram_plot,
    create_panel_layout,
    describe_x_axis_mode,
    normalize_color_limits,
    prepare_display_dataarray,
    save_bokeh_plot_html,
)

DATE_TOKEN_PATTERN = re.compile(r"(?<!\d)(20\d{6})(?!\d)")


@dataclass
class HeadlessExportRecord:
    """Metadata for one headless HTML export."""

    source_output_path: Path
    html_output_path: Path
    channel_name: str
    frequency_khz: int | None
    output_date: dt.date | None
    x_axis_note: str


def _path_from_env(var_name: str) -> Path | None:
    """Return a Path from an environment variable when set."""
    import os

    value = os.getenv(var_name)
    if not value:
        return None
    return Path(value).expanduser()


def _normalize_date_filter(value: str | None) -> str | None:
    """Normalize optional date filter to YYYYMMDD digits."""
    if value is None:
        return None
    digits = "".join(ch for ch in value if ch.isdigit())
    if not digits:
        return None
    if len(digits) != 8:
        raise ValueError(
            f"Date filter '{value}' is invalid. Use YYYYMMDD or YYYY-MM-DD."
        )
    return digits


def _iter_netcdf_outputs(input_dir: Path, glob_pattern: str) -> Iterable[Path]:
    """Yield MVBS NetCDF output files recursively by glob pattern."""
    yield from sorted(input_dir.rglob(glob_pattern))


def _extract_output_date(path: Path) -> dt.date | None:
    """Parse YYYYMMDD token from MVBS NetCDF output filename."""
    match = DATE_TOKEN_PATTERN.search(path.name)
    if match is None:
        return None
    try:
        return dt.datetime.strptime(match.group(1), "%Y%m%d").date()
    except ValueError:
        return None


def _filter_outputs_by_window(
    files: list[Path],
    start_token: str | None,
    window_days: int,
) -> tuple[list[Path], dt.date | None, dt.date | None, int]:
    """Filter MVBS outputs by an optional date window."""
    if start_token is None:
        return files, None, None, 0

    window_start = dt.datetime.strptime(start_token, "%Y%m%d").date()
    window_end_exclusive = window_start + dt.timedelta(days=window_days)
    filtered: list[Path] = []
    fallback_single_day: list[Path] = []
    missing_date_tokens = 0

    for path in files:
        output_date = _extract_output_date(path)
        if output_date is None:
            missing_date_tokens += 1
            if window_days == 1 and start_token in path.name:
                fallback_single_day.append(path)
            continue
        if window_start <= output_date < window_end_exclusive:
            filtered.append(path)

    if window_days == 1 and not filtered and fallback_single_day:
        filtered = fallback_single_day

    return filtered, window_start, window_end_exclusive, missing_date_tokens


def _matches_channel_filter(channel_name: str, channel_filter: str | None) -> bool:
    """Return True when channel passes the optional substring filter."""
    if channel_filter is None:
        return True
    return channel_filter.lower() in channel_name.lower()


def _strip_mvbs_netcdf_suffix(filename: str) -> str:
    """Return filename stem without the trailing `.mvbs.nc` token."""
    if filename.lower().endswith(".mvbs.nc"):
        return filename[:-8]
    return Path(filename).stem


def _derive_export_html_path(
    source_output_path: Path,
    input_dir: Path,
    export_root_dir: Path,
) -> Path:
    """Build output HTML path for one NetCDF source file."""
    try:
        relative_parent = source_output_path.relative_to(input_dir).parent
    except ValueError:
        relative_parent = Path()
    base_name = _strip_mvbs_netcdf_suffix(source_output_path.name)
    return (export_root_dir / relative_parent / f"{base_name}.html").resolve()


def _headless_record_sort_key(record: HeadlessExportRecord) -> tuple[str, int, str, str]:
    """Sort records by date, frequency, channel, then filename."""
    date_key = record.output_date.isoformat() if record.output_date is not None else "9999-99-99"
    freq_key = record.frequency_khz if record.frequency_khz is not None else 1_000_000
    return (date_key, freq_key, record.channel_name.lower(), record.html_output_path.name.lower())


def _write_headless_index_html(
    records: list[HeadlessExportRecord],
    export_root_dir: Path,
    input_dir: Path,
    cmap: str,
    vmin: float,
    vmax: float,
    plot_theme: str,
    hide_na_gaps: bool,
    flip_vertical: bool,
    transducer_facing: str,
    index_name: str,
) -> Path:
    """Write index HTML linking all headless-exported pages."""
    safe_index_name = index_name.strip() or "index.html"
    if not safe_index_name.lower().endswith(".html"):
        safe_index_name = f"{safe_index_name}.html"
    index_path = (export_root_dir / safe_index_name).resolve()
    index_path.parent.mkdir(parents=True, exist_ok=True)

    background = "#000000" if plot_theme == "dark" else "#ffffff"
    foreground = "#f3f4f6" if plot_theme == "dark" else "#111827"
    muted = "#9ca3af" if plot_theme == "dark" else "#4b5563"
    link_color = "#93c5fd" if plot_theme == "dark" else "#1d4ed8"

    sorted_records = sorted(records, key=_headless_record_sort_key)
    list_items: list[str] = []
    for record in sorted_records:
        try:
            html_href = record.html_output_path.relative_to(export_root_dir).as_posix()
        except ValueError:
            html_href = record.html_output_path.as_uri()
        date_text = record.output_date.isoformat() if record.output_date is not None else "unknown-date"
        freq_text = (
            f"{record.frequency_khz} kHz" if record.frequency_khz is not None else "frequency unknown"
        )
        source_display = html.escape(str(record.source_output_path))
        entry_label = html.escape(record.html_output_path.stem)
        list_items.append(
            "<li>"
            f"<a href=\"{html.escape(html_href)}\">{entry_label}</a>"
            f" <span class=\"meta\">({html.escape(date_text)} | {html.escape(freq_text)})</span><br>"
            f"<span class=\"source\">{source_display}</span>"
            "</li>"
        )

    generated_at = dt.datetime.now().replace(microsecond=0).isoformat(sep=" ")
    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MVBS Viewer Export Index</title>
  <style>
    :root {{
      --bg: {background};
      --fg: {foreground};
      --muted: {muted};
      --link: {link_color};
    }}
    html, body {{
      margin: 0;
      padding: 0;
      background: var(--bg);
      color: var(--fg);
      font-family: Segoe UI, Arial, sans-serif;
      line-height: 1.4;
    }}
    main {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 20px;
    }}
    a {{
      color: var(--link);
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    .meta {{
      color: var(--muted);
      font-size: 0.95rem;
    }}
    .source {{
      color: var(--muted);
      font-family: Consolas, Menlo, monospace;
      font-size: 0.85rem;
      word-break: break-all;
    }}
    ul {{
      padding-left: 22px;
    }}
    li {{
      margin-bottom: 12px;
    }}
    code {{
      font-family: Consolas, Menlo, monospace;
    }}
  </style>
</head>
<body>
  <main>
    <h1>MVBS Viewer Export Index</h1>
    <p>Generated at: <code>{html.escape(generated_at)}</code></p>
    <p>Input directory: <code>{html.escape(str(input_dir))}</code></p>
    <p>
      Render settings:
      <code>cmap={html.escape(cmap)}</code>,
      <code>vmin={vmin}</code>,
      <code>vmax={vmax}</code>,
      <code>plot_theme={html.escape(plot_theme)}</code>,
      <code>transducer_facing={html.escape(transducer_facing)}</code>,
      <code>hide_na_gaps={'true' if hide_na_gaps else 'false'}</code>,
      <code>flip_vertical={'true' if flip_vertical else 'false'}</code>
    </p>
    <p>Total HTML files: <strong>{len(sorted_records)}</strong></p>
    <ul>
      {"".join(list_items)}
    </ul>
  </main>
</body>
</html>
"""
    index_path.write_text(html_text, encoding="utf-8")
    return index_path


def _run_headless_html_exports(
    *,
    files: list[Path],
    input_dir: Path,
    export_html_dir: Path,
    channel_filter: str | None,
    vmin: float,
    vmax: float,
    cmap: str,
    width: int,
    height: int,
    plot_theme: str,
    plot_sizing: str,
    hide_na_gaps: bool,
    transducer_facing: str,
    flip_vertical: bool,
    html_resources: str,
) -> list[HeadlessExportRecord]:
    """Render one static HTML file per matched MVBS NetCDF output."""
    plot_vmin, plot_vmax = normalize_color_limits(vmin, vmax)
    export_root_dir = export_html_dir.expanduser().resolve()
    export_root_dir.mkdir(parents=True, exist_ok=True)

    records: list[HeadlessExportRecord] = []
    filtered_out_count = 0
    for output_path in files:
        ds = xr.open_dataset(output_path, chunks={})
        try:
            channel_name = _infer_output_channel(ds)
            if not _matches_channel_filter(channel_name, channel_filter):
                filtered_out_count += 1
                continue

            normalized_ds = _normalize_output_dataset(ds=ds, channel_name=channel_name)
            sv_da, y_coord, selected_channel = build_plot_dataarray(
                normalized_ds,
                channel_name,
            )
            sv_da, x_coord, x_label, removed_count = prepare_display_dataarray(
                sv_da,
                hide_na_gaps=hide_na_gaps,
            )
            x_axis_note = describe_x_axis_mode(
                x_coord=x_coord,
                hide_na_requested=hide_na_gaps,
                removed_count=removed_count,
            )
            effective_facing, facing_source = _resolve_transducer_facing(
                requested_facing=transducer_facing,
                output_attrs=[dict(ds.attrs)],
            )
            channel_freq = infer_channel_frequency_khz(selected_channel)
            title = f"EK80 MVBS Echogram | {selected_channel}"
            if channel_freq is not None:
                title = f"{title} ({channel_freq} kHz)"
            plot_obj = create_echogram_plot(
                sv_da=sv_da,
                y_coord=y_coord,
                x_coord=x_coord,
                x_label=x_label,
                vmin=plot_vmin,
                vmax=plot_vmax,
                cmap=cmap,
                width=width,
                height=height,
                title=title,
                plot_theme=plot_theme,
                plot_sizing=plot_sizing,
                transducer_facing=effective_facing,
                flip_vertical=flip_vertical,
            )
            bokeh_plot = hv.render(plot_obj, backend="bokeh")
            html_output_path = _derive_export_html_path(
                source_output_path=output_path,
                input_dir=input_dir,
                export_root_dir=export_root_dir,
            )
            saved_path = save_bokeh_plot_html(
                bokeh_plot,
                output_path=html_output_path,
                title=title,
                plot_theme=plot_theme,
                html_resources=html_resources,
            )
            record = HeadlessExportRecord(
                source_output_path=output_path.resolve(),
                html_output_path=saved_path,
                channel_name=selected_channel,
                frequency_khz=channel_freq,
                output_date=_extract_output_date(output_path),
                x_axis_note=x_axis_note,
            )
            records.append(record)
            print(
                f"Exported HTML: {saved_path} "
                f"(channel={selected_channel}, facing={effective_facing}, source={facing_source})"
            )
        finally:
            ds.close()

    if not records:
        raise RuntimeError(
            "No MVBS outputs remained after filtering for headless HTML export. "
            "Adjust --glob-pattern and/or --channel-filter."
        )
    if filtered_out_count > 0:
        print(f"Skipped {filtered_out_count} file(s) due to --channel-filter.")
    return records

def _infer_output_channel(ds: xr.Dataset) -> str:
    """Infer channel label from MVBS output metadata/coordinates."""
    selected_attr = ds.attrs.get("selected_channel")
    if selected_attr:
        return str(selected_attr)
    if "channel" in ds.coords:
        values = np.asarray(ds["channel"].values)
        if values.size == 1:
            return str(values.item())
    return "unknown_channel"


def _normalize_output_dataset(ds: xr.Dataset, channel_name: str) -> xr.Dataset:
    """Ensure MVBS output dataset has a proper channel dimension."""
    normalized = ds
    if "channel" in normalized.coords and "channel" not in normalized.dims:
        normalized = normalized.drop_vars("channel")
    if "channel" not in normalized.dims:
        normalized = normalized.expand_dims(channel=[channel_name])
    else:
        normalized = normalized.assign_coords(channel=[channel_name])
    return normalized


def _combine_output_groups(grouped: dict[str, list[xr.Dataset]]) -> xr.Dataset:
    """Combine grouped MVBS output datasets into one channel-aware MVBS dataset."""
    per_channel: list[xr.Dataset] = []
    for channel_name, datasets in sorted(grouped.items(), key=lambda pair: pair[0]):
        if len(datasets) == 1:
            channel_ds = datasets[0]
        else:
            channel_ds = xr.concat(
                datasets,
                dim="ping_time",
                data_vars="minimal",
                coords="minimal",
                compat="override",
                combine_attrs="override",
                join="outer",
            )
        if "ping_time" in channel_ds.coords:
            channel_ds = channel_ds.sortby("ping_time")
        if "channel" not in channel_ds.dims:
            channel_ds = channel_ds.expand_dims(channel=[channel_name])
        per_channel.append(channel_ds)

    if len(per_channel) == 1:
        ds_mvbs = per_channel[0]
    else:
        ds_mvbs = xr.concat(
            per_channel,
            dim="channel",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            combine_attrs="override",
            join="outer",
        )
        if "ping_time" in ds_mvbs.coords:
            ds_mvbs = ds_mvbs.sortby("ping_time")
    return ds_mvbs


def _resolve_transducer_facing(
    requested_facing: str,
    output_attrs: list[dict[str, object]],
) -> tuple[str, str]:
    """Resolve plotting transducer-facing mode from MVBS output metadata."""
    if requested_facing in {"down", "up"}:
        return requested_facing, "cli_override"

    inferred = sorted(
        {
            str(attrs.get("transducer_facing"))
            for attrs in output_attrs
            if str(attrs.get("transducer_facing")) in {"down", "up"}
        }
    )
    if len(inferred) == 1:
        return inferred[0], "output_metadata_auto"
    if len(inferred) > 1:
        return "down", "output_metadata_conflict_default_down"
    return "down", "output_metadata_unavailable_default_down"


def _resolve_ping_time_bin(output_attrs: list[dict[str, object]]) -> str:
    """Resolve ping-time bin text from MVBS output metadata."""
    for attrs in output_attrs:
        value = attrs.get("mvbs_ping_time_bin")
        if value:
            return str(value)
    return "unknown"


def _resolve_range_meter_bin(output_attrs: list[dict[str, object]]) -> float:
    """Resolve range-meter bin value from MVBS output metadata."""
    for attrs in output_attrs:
        value = attrs.get("mvbs_range_meter_bin")
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            return number
    return float("nan")


def _channel_sort_key(channel_name: str) -> tuple[int, str]:
    """Sort channel labels by inferred frequency, then lexicographically."""
    freq = infer_channel_frequency_khz(channel_name)
    if freq is None:
        return (1_000_000, channel_name.lower())
    return (int(freq), channel_name.lower())


def _select_initial_channel(channels: list[str]) -> str:
    """Choose default channel with lowest inferred frequency first."""
    if not channels:
        raise ValueError("No channels available to select.")
    return sorted(channels, key=_channel_sort_key)[0]


def _load_grouped_outputs(
    files: list[Path],
    channel_filter: str | None,
) -> tuple[xr.Dataset, list[str], list[dict[str, object]], int]:
    """Load files, filter channels, and combine into one channel-aware MVBS dataset."""
    grouped: dict[str, list[xr.Dataset]] = {}
    output_attrs: list[dict[str, object]] = []
    for output_path in files:
        ds = xr.open_dataset(output_path, chunks={})
        channel_name = _infer_output_channel(ds)
        if channel_filter and channel_filter.lower() not in channel_name.lower():
            ds.close()
            continue
        grouped.setdefault(channel_name, []).append(
            _normalize_output_dataset(ds=ds, channel_name=channel_name)
        )
        output_attrs.append(dict(ds.attrs))

    if not grouped:
        raise RuntimeError(
            "No MVBS outputs matched the provided filters. "
            "Adjust --date and/or --channel-filter."
        )

    ds_mvbs = _combine_output_groups(grouped)
    channels = sorted(ds_mvbs["channel"].astype(str).values.tolist(), key=_channel_sort_key)
    loaded_files = sum(len(items) for items in grouped.values())
    return ds_mvbs, channels, output_attrs, loaded_files


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "View exported MVBS NetCDF outputs in a live Panel app "
            "or generate headless HTML exports."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_path_from_env("EK80_OUTPUT_DIR"),
        help=(
            "Directory containing .mvbs.nc outputs. "
            "Required unless EK80_OUTPUT_DIR is set."
        ),
    )
    parser.add_argument(
        "--glob-pattern",
        type=str,
        default="*.mvbs.nc",
        help="Recursive glob pattern for MVBS output files.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help=(
            "Optional date filter (YYYYMMDD or YYYY-MM-DD). "
            "Used as the start date for --window-days stitching."
        ),
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=1,
        help=(
            "Number of consecutive days to stitch starting at --date "
            "(for example, 3 for a three-day stitched view)."
        ),
    )
    parser.add_argument(
        "--channel-filter",
        type=str,
        default=None,
        help="Optional channel substring filter (case-insensitive).",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=-90.0,
        help="Lower color limit in dB.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=-55.0,
        help="Upper color limit in dB.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Colormap for echogram rendering.",
    )
    parser.add_argument(
        "--plot-theme",
        type=str,
        choices=["dark", "light"],
        default="dark",
        help="Plot theme style for the viewer and exports.",
    )
    parser.add_argument(
        "--plot-sizing",
        type=str,
        choices=["responsive", "fixed"],
        default="responsive",
        help="Use responsive browser-fill sizing or fixed pixel dimensions.",
    )
    parser.add_argument(
        "--hide-na-gaps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Display-only option: drop all-NaN ping bins and plot a dense x-axis "
            "to hide duty-cycle gaps while retaining datetime tick labels. "
            "Enabled by default; use --no-hide-na-gaps to disable."
        ),
    )
    parser.add_argument(
        "--transducer-facing",
        type=str,
        choices=["auto", "down", "up"],
        default="auto",
        help=(
            "Vertical orientation mode for plotting. "
            "'auto' infers from output metadata when available."
        ),
    )
    parser.add_argument(
        "--flip-vertical",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Flip rendered echograms vertically after orientation is resolved. "
            "Enabled by default; use --no-flip-vertical to disable."
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1400,
        help="Plot width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=500,
        help="Plot height in pixels.",
    )
    parser.add_argument(
        "--panel-port",
        type=int,
        default=0,
        help="Port for Panel app (0 = auto-select).",
    )
    parser.add_argument(
        "--panel-no-browser",
        action="store_true",
        help="Do not auto-open browser.",
    )
    parser.add_argument(
        "--save-html",
        type=Path,
        default=None,
        help=(
            "Default path used by the Panel 'Export Current View to HTML' button. "
            "Defaults to <input-dir>/mvbs_netcdf_view.html."
        ),
    )
    parser.add_argument(
        "--export-html-dir",
        type=Path,
        default=None,
        help=(
            "Headless mode: write static HTML files from matched `.mvbs.nc` outputs "
            "to this directory, then exit without starting the live Panel server."
        ),
    )
    parser.add_argument(
        "--export-html-index-name",
        type=str,
        default="index.html",
        help=(
            "Filename for the headless export index page "
            "(default: index.html, written under --export-html-dir)."
        ),
    )
    parser.add_argument(
        "--export-html-no-index",
        action="store_true",
        help="In --export-html-dir mode, skip writing the index HTML page.",
    )
    parser.add_argument(
        "--html-resources",
        type=str,
        choices=["inline", "cdn"],
        default="inline",
        help=(
            "Bokeh resource mode for exported HTML. "
            "'inline' embeds JS/CSS for offline portability; 'cdn' keeps files smaller."
        ),
    )
    parser.add_argument(
        "--data-output-format",
        dest="data_output_format",
        type=str,
        choices=["none", "netcdf", "csv", "both"],
        default="none",
        help=(
            "Optional MVBS data output format for Panel HTML exports. "
            "Defaults to 'none' for NetCDF-view workflows."
        ),
    )
    parser.add_argument(
        "--data-output-dir",
        dest="data_output_dir",
        type=Path,
        default=None,
        help="Optional MVBS data output directory for Panel HTML exports.",
    )
    parser.add_argument(
        "--data-csv-compression",
        dest="data_csv_compression",
        type=str,
        choices=["none", "gzip"],
        default="gzip",
        help="CSV compression mode when --data-output-format includes csv.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args()


def main() -> None:
    """Load MVBS NetCDF outputs for live viewing or headless HTML export."""
    args = parse_args()
    configure_logging(args.log_level)
    if args.input_dir is None:
        raise ValueError(
            "Missing MVBS NetCDF output directory. Pass --input-dir or set EK80_OUTPUT_DIR."
        )
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    if args.window_days < 1:
        raise ValueError("--window-days must be >= 1.")

    date_filter = _normalize_date_filter(args.date)
    if args.window_days > 1 and date_filter is None:
        raise ValueError("--window-days > 1 requires --date as the stitch start date.")

    files = list(_iter_netcdf_outputs(args.input_dir, args.glob_pattern))
    files, window_start, window_end_exclusive, skipped_missing_dates = _filter_outputs_by_window(
        files=files,
        start_token=date_filter,
        window_days=args.window_days,
    )
    if not files:
        raise RuntimeError(
            f"No MVBS outputs found in {args.input_dir} matching pattern '{args.glob_pattern}'."
        )

    dask.config.set(scheduler="synchronous")
    hv.extension("bokeh")

    if window_start is not None and window_end_exclusive is not None:
        window_end_inclusive = window_end_exclusive - dt.timedelta(days=1)
        print(
            "Date-window stitch: "
            f"{window_start.isoformat()} -> {window_end_inclusive.isoformat()} "
            f"({args.window_days} day(s))."
        )
        if skipped_missing_dates > 0:
            print(
                "Outputs skipped for date-window filtering (no parseable YYYYMMDD token): "
                f"{skipped_missing_dates}"
            )

    if args.export_html_dir is not None:
        export_records = _run_headless_html_exports(
            files=files,
            input_dir=args.input_dir,
            export_html_dir=args.export_html_dir,
            channel_filter=args.channel_filter,
            vmin=args.vmin,
            vmax=args.vmax,
            cmap=args.cmap,
            width=args.width,
            height=args.height,
            plot_theme=args.plot_theme,
            plot_sizing=args.plot_sizing,
            hide_na_gaps=args.hide_na_gaps,
            transducer_facing=args.transducer_facing,
            flip_vertical=args.flip_vertical,
            html_resources=args.html_resources,
        )
        print(
            f"Headless HTML export complete: {len(export_records)} file(s) "
            f"written to {args.export_html_dir.expanduser().resolve()}."
        )
        if not args.export_html_no_index:
            index_path = _write_headless_index_html(
                records=export_records,
                export_root_dir=args.export_html_dir.expanduser().resolve(),
                input_dir=args.input_dir,
                cmap=args.cmap,
                vmin=args.vmin,
                vmax=args.vmax,
                plot_theme=args.plot_theme,
                hide_na_gaps=args.hide_na_gaps,
                flip_vertical=args.flip_vertical,
                transducer_facing=args.transducer_facing,
                index_name=args.export_html_index_name,
            )
            print(f"Headless export index: {index_path}")
        return

    import panel as pn

    plot_vmin, plot_vmax = normalize_color_limits(args.vmin, args.vmax)
    ds_mvbs, channels, output_attrs, loaded_files = _load_grouped_outputs(
        files=files,
        channel_filter=args.channel_filter,
    )
    initial_channel = _select_initial_channel(channels)

    transducer_facing, facing_source = _resolve_transducer_facing(
        requested_facing=args.transducer_facing,
        output_attrs=output_attrs,
    )
    ping_time_bin = _resolve_ping_time_bin(output_attrs)
    range_meter_bin = _resolve_range_meter_bin(output_attrs)
    print(f"Loaded {loaded_files} output file(s) across {len(channels)} channel(s).")
    print(f"Transducer facing for plotting: {transducer_facing} ({facing_source})")
    print(f"Vertical flip for plotting: {'enabled' if args.flip_vertical else 'disabled'}")
    print("Available channel names:")
    for channel in channels:
        print(f" - {channel}")

    if "ping_time" in ds_mvbs.coords:
        print(
            f"Time range: {str(ds_mvbs['ping_time'].min().values)} -> "
            f"{str(ds_mvbs['ping_time'].max().values)}"
        )

    default_export_path = args.save_html or (args.input_dir / "mvbs_netcdf_view.html")
    app = create_panel_layout(
        ds_mvbs=ds_mvbs,
        channels=channels,
        initial_channel=initial_channel,
        initial_cmap=args.cmap,
        initial_vmin=plot_vmin,
        initial_vmax=plot_vmax,
        width=args.width,
        height=args.height,
        default_export_path=default_export_path,
        plot_theme=args.plot_theme,
        plot_sizing=args.plot_sizing,
        hide_na_gaps=args.hide_na_gaps,
        html_resources=args.html_resources,
        transducer_facing=transducer_facing,
        flip_vertical=args.flip_vertical,
        ping_time_bin=ping_time_bin,
        range_meter_bin=range_meter_bin,
        data_output_format=args.data_output_format,
        data_output_dir=args.data_output_dir,
        data_csv_compression=args.data_csv_compression,
    )
    pn.serve(
        app,
        title="MVBS NetCDF Viewer",
        port=args.panel_port,
        show=not args.panel_no_browser,
    )


if __name__ == "__main__":
    main()
