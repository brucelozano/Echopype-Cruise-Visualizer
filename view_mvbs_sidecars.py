"""Live viewer for exported MVBS NetCDF sidecars.

This script loads `.mvbs.nc` sidecars (produced during HTML export) and serves
an interactive Panel app with the same plot controls used in the processing UI.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import dask
import holoviews as hv
import numpy as np
import xarray as xr

from ek80_chunked_echogram import (
    configure_logging,
    create_panel_layout,
    normalize_color_limits,
)


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


def _iter_sidecars(input_dir: Path, glob_pattern: str) -> Iterable[Path]:
    """Yield sidecar files recursively by glob pattern."""
    yield from sorted(input_dir.rglob(glob_pattern))


def _infer_sidecar_channel(ds: xr.Dataset) -> str:
    """Infer channel label from sidecar metadata/coordinates."""
    selected_attr = ds.attrs.get("selected_channel")
    if selected_attr:
        return str(selected_attr)
    if "channel" in ds.coords:
        values = np.asarray(ds["channel"].values)
        if values.size == 1:
            return str(values.item())
    return "unknown_channel"


def _normalize_sidecar_dataset(ds: xr.Dataset, channel_name: str) -> xr.Dataset:
    """Ensure sidecar dataset has a proper channel dimension."""
    normalized = ds
    if "channel" in normalized.coords and "channel" not in normalized.dims:
        normalized = normalized.drop_vars("channel")
    if "channel" not in normalized.dims:
        normalized = normalized.expand_dims(channel=[channel_name])
    else:
        normalized = normalized.assign_coords(channel=[channel_name])
    return normalized


def _combine_sidecar_groups(grouped: dict[str, list[xr.Dataset]]) -> xr.Dataset:
    """Combine grouped sidecar datasets into one channel-aware MVBS dataset."""
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
    sidecar_attrs: list[dict[str, object]],
) -> tuple[str, str]:
    """Resolve plotting transducer-facing mode from sidecar metadata."""
    if requested_facing in {"down", "up"}:
        return requested_facing, "cli_override"

    inferred = sorted(
        {
            str(attrs.get("transducer_facing"))
            for attrs in sidecar_attrs
            if str(attrs.get("transducer_facing")) in {"down", "up"}
        }
    )
    if len(inferred) == 1:
        return inferred[0], "sidecar_metadata_auto"
    if len(inferred) > 1:
        return "down", "sidecar_metadata_conflict_default_down"
    return "down", "sidecar_metadata_unavailable_default_down"


def _resolve_ping_time_bin(sidecar_attrs: list[dict[str, object]]) -> str:
    """Resolve ping-time bin text from sidecar metadata."""
    for attrs in sidecar_attrs:
        value = attrs.get("mvbs_ping_time_bin")
        if value:
            return str(value)
    return "unknown"


def _resolve_range_meter_bin(sidecar_attrs: list[dict[str, object]]) -> float:
    """Resolve range-meter bin value from sidecar metadata."""
    for attrs in sidecar_attrs:
        value = attrs.get("mvbs_range_meter_bin")
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            return number
    return float("nan")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="View exported MVBS NetCDF sidecars in a live Panel app."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_path_from_env("EK80_OUTPUT_DIR"),
        help=(
            "Directory containing .mvbs.nc sidecars. "
            "Required unless EK80_OUTPUT_DIR is set."
        ),
    )
    parser.add_argument(
        "--glob-pattern",
        type=str,
        default="*.mvbs.nc",
        help="Recursive glob pattern for sidecar files.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Optional date filter (YYYYMMDD or YYYY-MM-DD) applied to filename.",
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
        default=-30.0,
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
        action="store_true",
        help=(
            "Display-only option: drop all-NaN ping bins and plot a dense x-axis "
            "to hide duty-cycle gaps while retaining datetime tick labels."
        ),
    )
    parser.add_argument(
        "--transducer-facing",
        type=str,
        choices=["auto", "down", "up"],
        default="auto",
        help=(
            "Vertical orientation mode for plotting. "
            "'auto' infers from sidecar metadata when available."
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
            "Defaults to <input-dir>/mvbs_sidecar_view.html."
        ),
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
        "--export-plot-data",
        type=str,
        choices=["none", "netcdf", "csv", "both"],
        default="none",
        help=(
            "Optional sidecar export mode for Panel HTML exports. "
            "Defaults to 'none' for sidecar-view workflows."
        ),
    )
    parser.add_argument(
        "--plot-data-dir",
        type=Path,
        default=None,
        help="Optional sidecar output directory for Panel HTML exports.",
    )
    parser.add_argument(
        "--plot-data-csv-compression",
        type=str,
        choices=["none", "gzip"],
        default="gzip",
        help="CSV compression mode when --export-plot-data includes csv.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args()


def main() -> None:
    """Load sidecars and serve an interactive MVBS viewer."""
    args = parse_args()
    configure_logging(args.log_level)
    if args.input_dir is None:
        raise ValueError(
            "Missing sidecar input directory. Pass --input-dir or set EK80_OUTPUT_DIR."
        )
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    date_filter = _normalize_date_filter(args.date)
    files = list(_iter_sidecars(args.input_dir, args.glob_pattern))
    if date_filter is not None:
        files = [path for path in files if date_filter in path.name]
    if not files:
        raise RuntimeError(
            f"No sidecars found in {args.input_dir} matching pattern '{args.glob_pattern}'."
        )

    grouped: dict[str, list[xr.Dataset]] = {}
    sidecar_attrs: list[dict[str, object]] = []
    for sidecar_path in files:
        ds = xr.open_dataset(sidecar_path, chunks={})
        channel_name = _infer_sidecar_channel(ds)
        if args.channel_filter and args.channel_filter.lower() not in channel_name.lower():
            ds.close()
            continue
        grouped.setdefault(channel_name, []).append(
            _normalize_sidecar_dataset(ds=ds, channel_name=channel_name)
        )
        sidecar_attrs.append(dict(ds.attrs))

    if not grouped:
        raise RuntimeError(
            "No sidecars matched the provided filters. "
            "Adjust --date and/or --channel-filter."
        )

    dask.config.set(scheduler="synchronous")
    hv.extension("bokeh")

    ds_mvbs = _combine_sidecar_groups(grouped)
    channels = ds_mvbs["channel"].astype(str).values.tolist()
    initial_channel = channels[0]

    transducer_facing, facing_source = _resolve_transducer_facing(
        requested_facing=args.transducer_facing,
        sidecar_attrs=sidecar_attrs,
    )
    ping_time_bin = _resolve_ping_time_bin(sidecar_attrs)
    range_meter_bin = _resolve_range_meter_bin(sidecar_attrs)
    print(
        f"Loaded {sum(len(items) for items in grouped.values())} sidecar file(s) "
        f"across {len(channels)} channel(s)."
    )
    print(f"Transducer facing for plotting: {transducer_facing} ({facing_source})")
    print("Available channel names:")
    for channel in channels:
        print(f" - {channel}")

    if "ping_time" in ds_mvbs.coords:
        print(
            f"Time range: {str(ds_mvbs['ping_time'].min().values)} -> "
            f"{str(ds_mvbs['ping_time'].max().values)}"
        )

    import panel as pn

    plot_vmin, plot_vmax = normalize_color_limits(args.vmin, args.vmax)
    default_export_path = args.save_html or (args.input_dir / "mvbs_sidecar_view.html")
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
        ping_time_bin=ping_time_bin,
        range_meter_bin=range_meter_bin,
        export_plot_data=args.export_plot_data,
        plot_data_dir=args.plot_data_dir,
        plot_data_csv_compression=args.plot_data_csv_compression,
    )
    pn.serve(
        app,
        title="MVBS Sidecar Viewer",
        port=args.panel_port,
        show=not args.panel_no_browser,
    )


if __name__ == "__main__":
    main()
