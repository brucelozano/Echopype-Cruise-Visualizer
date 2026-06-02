"""Chunked EK80 test visualization using in-memory MVBS aggregation.

This script processes the first N EK80 `.raw` files from a directory, computes
MVBS in small chunks, concatenates only the reduced products, and renders either
a static interactive Bokeh echogram or a live Panel app with plot controls.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import inspect
import logging
import os
import re
from pathlib import Path
from typing import Iterator, Sequence

import dask
import echopype as ep
import holoviews as hv
import hvplot.xarray  # noqa: F401  # Registers hvplot on xarray objects
import numpy as np
import xarray as xr
from bokeh.io import save
from bokeh.models import ColorBar, CustomJSHover, CustomJSTickFormatter, HoverTool
from bokeh.plotting import show
from bokeh.resources import CDN, INLINE

from channel_naming import channel_slug, infer_channel_frequency_khz


LOGGER = logging.getLogger("ek80_chunked_echogram")

DARK_BG = "#000000"
DARK_FG = "#e5e7eb"
DARK_MUTED = "#9ca3af"
DARK_GRID = "#4b5563"
PING_TIME_DISPLAY_COORD = "ping_time_display"
PING_TIME_ACTUAL_COORD = "ping_time_actual"


def _path_from_env(var_name: str) -> Path | None:
    """Return a Path from an environment variable when set."""
    value = os.getenv(var_name)
    if not value:
        return None
    return Path(value).expanduser()


def configure_logging(level: str = "INFO") -> None:
    """Configure application logging.

    Parameters
    ----------
    level : str, optional
        Logging level string, by default "INFO".
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def chunked(items: Sequence[Path], chunk_size: int) -> Iterator[list[Path]]:
    """Yield list slices of `items` with length up to `chunk_size`.

    Parameters
    ----------
    items : Sequence[Path]
        Input sequence of file paths.
    chunk_size : int
        Number of items per chunk.

    Yields
    ------
    Iterator[list[Path]]
        Chunked file lists.
    """
    for start in range(0, len(items), chunk_size):
        yield list(items[start : start + chunk_size])


FILENAME_TS_PATTERN = re.compile(r"D(?P<date>\d{8})-T(?P<time>\d{6})")


def extract_datetime_from_filename(file_path: Path) -> dt.datetime | None:
    """Extract `YYYYMMDD-HHMMSS` timestamp from raw filename text."""
    match = FILENAME_TS_PATTERN.search(file_path.name)
    if not match:
        return None
    return dt.datetime.strptime(
        f"{match.group('date')}{match.group('time')}",
        "%Y%m%d%H%M%S",
    )


def parse_datetime_input(value: str) -> dt.datetime:
    """Parse datetime text from common user formats.

    Supported examples include:
    - ``20250819``
    - ``2025-08-19``
    - ``20250819T230000``
    - ``2025-08-19T23:00:00``
    - ``2025-08-19 23:00:00``
    """
    text = value.strip()
    formats = [
        "%Y%m%dT%H%M%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y%m%d",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            return dt.datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(
        f"Unable to parse datetime: '{value}'. "
        "Use one of YYYYMMDD, YYYY-MM-DD, YYYYMMDDTHHMMSS, or YYYY-MM-DDTHH:MM:SS."
    )


def resolve_time_window(
    start_datetime_text: str | None,
    end_datetime_text: str | None,
    duration_days: int | None,
) -> tuple[dt.datetime | None, dt.datetime | None]:
    """Resolve CLI time-window arguments into concrete datetimes."""
    if duration_days is not None and duration_days < 1:
        raise ValueError(f"duration_days must be >= 1, got {duration_days}")
    if end_datetime_text and duration_days is not None:
        raise ValueError(
            "Use either --end-datetime or --duration-days, not both at the same time."
        )

    start_dt = parse_datetime_input(start_datetime_text) if start_datetime_text else None
    end_dt = parse_datetime_input(end_datetime_text) if end_datetime_text else None

    if duration_days is not None:
        if start_dt is None:
            raise ValueError("--duration-days requires --start-datetime.")
        end_dt = start_dt + dt.timedelta(days=duration_days)

    if start_dt and end_dt and end_dt <= start_dt:
        raise ValueError(
            f"Invalid time window: end ({end_dt.isoformat()}) must be after start ({start_dt.isoformat()})."
        )
    return start_dt, end_dt


def list_raw_files(
    raw_dir: Path,
    max_files: int,
    start_datetime: dt.datetime | None = None,
    end_datetime: dt.datetime | None = None,
) -> list[Path]:
    """Return filtered EK80 raw files from `raw_dir`.

    Parameters
    ----------
    raw_dir : Path
        Directory that contains raw and sidecar files.
    max_files : int
        Maximum number of `.raw` files to process when no datetime filter is used.
    start_datetime : dt.datetime | None, optional
        Optional inclusive lower bound for filename timestamp filtering.
    end_datetime : dt.datetime | None, optional
        Optional exclusive upper bound for filename timestamp filtering.

    Returns
    -------
    list[Path]
        Sorted list of `.raw` file paths.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")

    discovered_files = list(raw_dir.glob("*.raw"))
    raw_with_ts: list[tuple[dt.datetime, Path]] = []
    raw_without_ts: list[Path] = []
    for raw_file in discovered_files:
        ts = extract_datetime_from_filename(raw_file)
        if ts is None:
            raw_without_ts.append(raw_file)
        else:
            raw_with_ts.append((ts, raw_file))

    raw_with_ts.sort(key=lambda pair: (pair[0], pair[1].name))
    raw_without_ts.sort(key=lambda path: path.name)

    has_time_filter = start_datetime is not None or end_datetime is not None
    if has_time_filter:
        filtered_files = [
            file_path
            for file_dt, file_path in raw_with_ts
            if (start_datetime is None or file_dt >= start_datetime)
            and (end_datetime is None or file_dt < end_datetime)
        ]
        if raw_without_ts:
            LOGGER.warning(
                "Skipping %d .raw files without parseable timestamps while applying datetime filter.",
                len(raw_without_ts),
            )
        LOGGER.info(
            "Datetime filter active: processing all %d matched files (ignoring --max-files=%d).",
            len(filtered_files),
            max_files,
        )
        raw_files = filtered_files
    else:
        ordered_files = [file_path for _, file_path in raw_with_ts] + raw_without_ts
        raw_files = ordered_files[:max_files]

    if not raw_files:
        window_msg = ""
        if has_time_filter:
            window_msg = (
                f" within time window "
                f"[{start_datetime.isoformat() if start_datetime else '-inf'}, "
                f"{end_datetime.isoformat() if end_datetime else '+inf'})"
            )
        raise FileNotFoundError(
            f"No .raw files found in {raw_dir}{window_msg}. "
            "Check the folder path and extension filtering."
        )
    return raw_files


def _normalize_waveform_mode(value: str | None) -> str | None:
    """Normalize waveform mode user input for EK80 calibration."""
    if value is None:
        return None
    text = value.strip().upper()
    if not text or text == "AUTO":
        return None
    if text == "FM":
        return "BB"
    return text


def _normalize_encode_mode(value: str | None) -> str | None:
    """Normalize encode mode user input for EK80 calibration."""
    if value is None:
        return None
    text = value.strip().lower()
    if not text or text == "auto":
        return None
    return text


def _sv_candidate_kwargs(
    waveform_mode: str | None,
    encode_mode: str | None,
) -> list[dict[str, str]]:
    """Build ordered compute_Sv argument candidates for EK80 data."""
    if waveform_mode and encode_mode:
        return [{"waveform_mode": waveform_mode, "encode_mode": encode_mode}]

    candidates: list[dict[str, str]] = []

    if waveform_mode and not encode_mode:
        if waveform_mode == "CW":
            candidates = [
                {"waveform_mode": "CW", "encode_mode": "power"},
                {"waveform_mode": "CW", "encode_mode": "complex"},
            ]
        elif waveform_mode == "BB":
            candidates = [{"waveform_mode": "BB", "encode_mode": "complex"}]
        else:
            candidates = [{"waveform_mode": waveform_mode, "encode_mode": "complex"}]
    elif encode_mode and not waveform_mode:
        if encode_mode == "power":
            candidates = [{"waveform_mode": "CW", "encode_mode": "power"}]
        else:
            candidates = [
                {"waveform_mode": "CW", "encode_mode": "complex"},
                {"waveform_mode": "BB", "encode_mode": "complex"},
            ]
    else:
        candidates = [
            {"waveform_mode": "CW", "encode_mode": "power"},
            {"waveform_mode": "CW", "encode_mode": "complex"},
            {"waveform_mode": "BB", "encode_mode": "complex"},
        ]

    # De-duplicate while preserving order.
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        key = (candidate["waveform_mode"], candidate["encode_mode"])
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def compute_sv_with_fallback(
    echodata: ep.EchoData,
    waveform_mode: str | None,
    encode_mode: str | None,
) -> tuple[xr.Dataset, dict[str, str]]:
    """Compute Sv by trying valid EK80 mode combinations.

    Parameters
    ----------
    echodata : ep.EchoData
        Combined input EchoData for the current chunk.
    waveform_mode : str | None
        Requested waveform mode, or None for auto.
    encode_mode : str | None
        Requested encode mode, or None for auto.

    Returns
    -------
    tuple[xr.Dataset, dict[str, str]]
        Sv dataset and kwargs combination used successfully.
    """
    candidates = _sv_candidate_kwargs(waveform_mode=waveform_mode, encode_mode=encode_mode)
    errors: list[str] = []

    for candidate in candidates:
        LOGGER.info(
            "Trying compute_Sv with waveform_mode=%s encode_mode=%s",
            candidate["waveform_mode"],
            candidate["encode_mode"],
        )
        try:
            ds_sv = ep.calibrate.compute_Sv(echodata, **candidate)
            return ds_sv, candidate
        except Exception as exc:  # noqa: BLE001 - continue through known EK80 mode mismatches
            errors.append(
                f"{candidate['waveform_mode']}/{candidate['encode_mode']}: {type(exc).__name__}: {exc}"
            )

    joined_errors = "\n".join(errors)
    raise RuntimeError(
        "Unable to compute Sv for this chunk using EK80 mode combinations.\n"
        f"Attempted combinations: {candidates}\n"
        f"Errors:\n{joined_errors}"
    )


def _normalize_ping_time_bin(value: str) -> str:
    """Normalize time-bin strings for pandas/xarray compatibility.

    Pandas now prefers lowercase unit aliases (for example, ``30s`` over ``30S``).
    """
    text = value.strip()
    if not text:
        raise ValueError("ping_time_bin cannot be empty.")
    if text[-1].isalpha():
        return f"{text[:-1]}{text[-1].lower()}"
    return text


def compute_mvbs_with_compat(
    ds_sv: xr.Dataset,
    range_meter_bin: float,
    ping_time_bin: str,
) -> xr.Dataset:
    """Compute MVBS with compatibility across echopype API versions."""
    ping_time_bin_norm = _normalize_ping_time_bin(ping_time_bin)
    mvbs_signature = inspect.signature(ep.commongrid.compute_MVBS)

    if "range_meter_bin" in mvbs_signature.parameters:
        kwargs = {
            "range_meter_bin": range_meter_bin,
            "ping_time_bin": ping_time_bin_norm,
        }
    else:
        kwargs = {
            "range_bin": f"{range_meter_bin:g}m",
            "ping_time_bin": ping_time_bin_norm,
        }

    LOGGER.info("compute_MVBS kwargs resolved to: %s", kwargs)
    return ep.commongrid.compute_MVBS(ds_sv, **kwargs)


def normalize_color_limits(vmin: float, vmax: float) -> tuple[float, float]:
    """Return ascending (vmin, vmax) limits; swap if user provides reverse order."""
    if vmin <= vmax:
        return vmin, vmax
    LOGGER.warning(
        "Color limits were reversed (vmin=%s, vmax=%s); swapping to (%s, %s).",
        vmin,
        vmax,
        vmax,
        vmin,
    )
    return vmax, vmin


def build_channel_label(channel_name: str) -> str:
    """Build user-facing channel label including inferred frequency."""
    freq = infer_channel_frequency_khz(channel_name)
    if freq is None:
        return f"{channel_name} (frequency unknown)"
    return f"{channel_name} ({freq} kHz)"


def resolve_html_output_path(path_value: Path) -> Path:
    """Resolve HTML output path and ensure `.html` suffix."""
    path = path_value.resolve()
    if path.suffix.lower() != ".html":
        path = path.with_suffix(".html")
    return path


def apply_bokeh_plot_theme(plot_obj, plot_theme: str) -> None:
    """Apply dark/light styling to a Bokeh figure in place."""
    if plot_theme != "dark":
        return

    plot_obj.background_fill_color = DARK_BG
    plot_obj.border_fill_color = DARK_BG
    plot_obj.outline_line_color = DARK_MUTED
    plot_obj.toolbar.logo = None

    if plot_obj.title is not None:
        plot_obj.title.text_color = DARK_FG

    for axis in list(plot_obj.xaxis) + list(plot_obj.yaxis):
        axis.axis_label_text_color = DARK_FG
        axis.major_label_text_color = DARK_FG
        axis.axis_line_color = DARK_MUTED
        axis.major_tick_line_color = DARK_MUTED
        axis.minor_tick_line_color = DARK_MUTED

    for grid in list(plot_obj.xgrid) + list(plot_obj.ygrid):
        grid.grid_line_color = DARK_GRID
        grid.grid_line_alpha = 0.25

    for colorbar in plot_obj.select({"type": ColorBar}):
        colorbar.title_text_color = DARK_FG
        colorbar.major_label_text_color = DARK_FG
        colorbar.major_tick_line_color = DARK_MUTED
        colorbar.bar_line_color = DARK_MUTED
        colorbar.background_fill_color = DARK_BG


def save_bokeh_plot_html(
    plot_obj,
    output_path: Path,
    title: str,
    plot_theme: str,
    html_resources: str = "inline",
) -> Path:
    """Save a bokeh plot to HTML and return resolved path."""
    apply_bokeh_plot_theme(plot_obj, plot_theme=plot_theme)
    html_path = resolve_html_output_path(output_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    resources = INLINE if html_resources == "inline" else CDN
    save(plot_obj, filename=str(html_path), title=title, resources=resources)
    html_text = html_path.read_text(encoding="utf-8")
    page_bg = DARK_BG if plot_theme == "dark" else "#ffffff"
    page_fg = DARK_FG if plot_theme == "dark" else "#111827"
    extra_css = (
        "<style>"
        f"html, body {{ background: {page_bg}; color: {page_fg}; overflow: hidden; }}"
        ".bk-root { height: 100vh; }"
        "</style>"
    )
    if extra_css not in html_text:
        html_text = html_text.replace("</head>", f"{extra_css}\n</head>", 1)
        html_path.write_text(html_text, encoding="utf-8")
    return html_path


def _extract_echodata_channels(echodata: ep.EchoData) -> tuple[str, ...]:
    """Extract sorted channel labels present across all groups in an EchoData object."""
    channels: set[str] = set()
    for group_path in echodata.group_paths:
        group_ds = echodata[group_path]
        if "channel" in group_ds:
            channels.update(str(value) for value in group_ds["channel"].values.tolist())
    return tuple(sorted(channels))


def _compute_mvbs_from_echodata_group(
    echodata_group: Sequence[ep.EchoData],
    range_meter_bin: float,
    ping_time_bin: str,
    waveform_mode: str | None,
    encode_mode: str | None,
) -> xr.Dataset:
    """Compute MVBS from one channel-consistent EchoData subgroup."""
    if len(echodata_group) == 1:
        combined = echodata_group[0]
    else:
        combined = ep.combine_echodata(list(echodata_group))

    ds_sv, used_modes = compute_sv_with_fallback(
        combined,
        waveform_mode=waveform_mode,
        encode_mode=encode_mode,
    )
    LOGGER.info(
        "Using compute_Sv mode combination waveform_mode=%s encode_mode=%s",
        used_modes["waveform_mode"],
        used_modes["encode_mode"],
    )
    ds_mvbs = compute_mvbs_with_compat(
        ds_sv=ds_sv,
        range_meter_bin=range_meter_bin,
        ping_time_bin=ping_time_bin,
    ).compute()
    del ds_sv
    return ds_mvbs


def compute_mvbs_for_chunk(
    chunk_files: Sequence[Path],
    range_meter_bin: float,
    ping_time_bin: str,
    waveform_mode: str | None,
    encode_mode: str | None,
) -> xr.Dataset:
    """Compute MVBS for one chunk of EK80 `.raw` files.

    Parameters
    ----------
    chunk_files : Sequence[Path]
        Input raw files for the chunk.
    range_meter_bin : float
        Range bin size in meters passed to `compute_MVBS`.
    ping_time_bin : str
        Time bin size passed to `compute_MVBS` (for example, "30S").
    waveform_mode : str | None
        Optional EK80 waveform mode (for example, "CW" or "BB").
    encode_mode : str | None
        Optional EK80 encode mode (for example, "power" or "complex").

    Returns
    -------
    xr.Dataset
        Materialized MVBS dataset for the chunk.
    """
    LOGGER.info(
        "Opening %d files for chunk (%s ... %s)",
        len(chunk_files),
        chunk_files[0].name,
        chunk_files[-1].name,
    )
    echodata_list = [ep.open_raw(str(file_path), sonar_model="EK80") for file_path in chunk_files]

    requested_waveform_mode = _normalize_waveform_mode(waveform_mode)
    requested_encode_mode = _normalize_encode_mode(encode_mode)

    LOGGER.info(
        "Calibrating Sv for chunk | requested waveform_mode=%s encode_mode=%s",
        requested_waveform_mode or "auto",
        requested_encode_mode or "auto",
    )

    channel_signatures = [_extract_echodata_channels(ed) for ed in echodata_list]
    subgroup_boundaries: list[tuple[int, int]] = []
    start_idx = 0
    for idx in range(1, len(channel_signatures)):
        if channel_signatures[idx] != channel_signatures[idx - 1]:
            subgroup_boundaries.append((start_idx, idx))
            start_idx = idx
    subgroup_boundaries.append((start_idx, len(echodata_list)))

    LOGGER.info(
        "Chunk channel signatures detected: %d subgroup(s) for %d file(s).",
        len(subgroup_boundaries),
        len(echodata_list),
    )

    subgroup_results: list[xr.Dataset] = []
    for group_idx, (start, end) in enumerate(subgroup_boundaries, start=1):
        subgroup_files = chunk_files[start:end]
        subgroup_ed = echodata_list[start:end]
        subgroup_signature = channel_signatures[start]
        LOGGER.info(
            "Processing subgroup %d/%d | files=%d | channels=%s | first=%s | last=%s",
            group_idx,
            len(subgroup_boundaries),
            len(subgroup_files),
            list(subgroup_signature),
            subgroup_files[0].name,
            subgroup_files[-1].name,
        )
        try:
            subgroup_mvbs = _compute_mvbs_from_echodata_group(
                echodata_group=subgroup_ed,
                range_meter_bin=range_meter_bin,
                ping_time_bin=ping_time_bin,
                waveform_mode=requested_waveform_mode,
                encode_mode=requested_encode_mode,
            )
        except Exception as exc:  # noqa: BLE001 - fallback protects mixed/non-combinable batches
            if len(subgroup_ed) == 1:
                raise
            LOGGER.warning(
                "Subgroup combine/calibration failed (%s). Falling back to per-file processing for this subgroup.",
                exc,
            )
            per_file_results: list[xr.Dataset] = []
            for file_path, single_ed in zip(subgroup_files, subgroup_ed):
                LOGGER.info("Per-file fallback for %s", file_path.name)
                per_file_results.append(
                    _compute_mvbs_from_echodata_group(
                        echodata_group=[single_ed],
                        range_meter_bin=range_meter_bin,
                        ping_time_bin=ping_time_bin,
                        waveform_mode=requested_waveform_mode,
                        encode_mode=requested_encode_mode,
                    )
                )
            subgroup_mvbs = xr.concat(
                per_file_results,
                dim="ping_time",
                data_vars="minimal",
                coords="minimal",
                compat="override",
                combine_attrs="override",
                join="outer",
            ).sortby("ping_time")
        subgroup_results.append(subgroup_mvbs)

    if len(subgroup_results) == 1:
        ds_mvbs_chunk = subgroup_results[0]
    else:
        ds_mvbs_chunk = xr.concat(
            subgroup_results,
            dim="ping_time",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            combine_attrs="override",
            join="outer",
        ).sortby("ping_time")

    del echodata_list
    gc.collect()

    return ds_mvbs_chunk


def build_plot_dataarray(
    ds_mvbs: xr.Dataset, channel_name: str | None
) -> tuple[xr.DataArray, str, str]:
    """Select channel and return a plot-ready DataArray plus y-axis coordinate.

    Parameters
    ----------
    ds_mvbs : xr.Dataset
        Concatenated MVBS dataset.
    channel_name : str | None
        Requested channel label. If None, first available channel is used.

    Returns
    -------
    tuple[xr.DataArray, str, str]
        (selected Sv DataArray, y-axis coordinate name, selected channel name)
    """
    if "Sv" not in ds_mvbs.data_vars:
        raise KeyError(
            "Expected 'Sv' variable missing in MVBS output. "
            f"Available variables: {list(ds_mvbs.data_vars)}"
        )
    if "channel" not in ds_mvbs.dims:
        raise KeyError(
            "Expected 'channel' dimension missing in MVBS output. "
            f"Available dimensions: {dict(ds_mvbs.sizes)}"
        )

    channels = ds_mvbs["channel"].astype(str).values.tolist()
    if channel_name is None:
        channel_name = channels[0]
        LOGGER.info("No channel selected. Using first channel: %s", channel_name)
    elif channel_name not in channels:
        raise ValueError(
            "Requested channel not found.\n"
            f"Requested: {channel_name}\n"
            f"Available: {channels}"
        )

    sv_da = ds_mvbs["Sv"].sel(channel=channel_name)

    if "echo_range" in sv_da.coords:
        echo_range = sv_da["echo_range"]
        if "ping_time" in echo_range.dims:
            candidate = echo_range
            non_time_dims = [dim for dim in sv_da.dims if dim != "ping_time"]
            if non_time_dims:
                valid_ping_mask = sv_da.notnull().any(dim=non_time_dims)
                if int(valid_ping_mask.sum().item()) > 0:
                    candidate = candidate.sel(ping_time=valid_ping_mask)
            if "ping_time" in candidate.dims:
                range_non_time_dims = [dim for dim in candidate.dims if dim != "ping_time"]
                if range_non_time_dims:
                    valid_range_ping = candidate.notnull().any(dim=range_non_time_dims)
                    if int(valid_range_ping.sum().item()) > 0:
                        first_valid_idx = int(valid_range_ping.argmax(dim="ping_time").item())
                        candidate = candidate.isel(ping_time=first_valid_idx, drop=True)
                    else:
                        candidate = candidate.isel(ping_time=0, drop=True)
                else:
                    candidate = candidate.isel(ping_time=0, drop=True)
            echo_range = candidate

        if int(echo_range.notnull().sum().item()) > 1:
            sv_da = sv_da.assign_coords(echo_range=echo_range)
            y_coord = "echo_range"
        elif "range_sample" in sv_da.dims:
            LOGGER.warning(
                "echo_range coordinate is empty/invalid for channel '%s'; "
                "falling back to range_sample axis.",
                channel_name,
            )
            y_coord = "range_sample"
        else:
            y_coord = [dim for dim in sv_da.dims if dim != "ping_time"][0]
    elif "range_sample" in sv_da.dims:
        y_coord = "range_sample"
    else:
        y_coord = [dim for dim in sv_da.dims if dim != "ping_time"][0]

    return sv_da, y_coord, channel_name


def collapse_na_ping_gaps(sv_da: xr.DataArray) -> tuple[xr.DataArray, int]:
    """Drop all-NaN ping bins and swap plotting to a dense display axis.

    This is a display-only transform used to hide duty-cycle gaps visually.
    It never interpolates or modifies Sv values.
    """
    if "ping_time" not in sv_da.dims:
        return sv_da, 0

    non_time_dims = [dim for dim in sv_da.dims if dim != "ping_time"]
    if not non_time_dims:
        return sv_da, 0

    valid_ping_mask = sv_da.notnull().any(dim=non_time_dims)
    valid_count = int(valid_ping_mask.sum().item())
    total_count = int(sv_da.sizes.get("ping_time", 0))
    removed_count = max(total_count - valid_count, 0)
    if valid_count < 2:
        LOGGER.warning(
            "Requested NA-gap collapsing, but only %d valid ping bin(s) remain; "
            "falling back to original ping_time axis for stable rendering.",
            valid_count,
        )
        return sv_da, 0

    collapsed = sv_da.sel(ping_time=valid_ping_mask)
    collapsed = collapsed.assign_coords(
        {
            PING_TIME_ACTUAL_COORD: xr.DataArray(
                collapsed["ping_time"].values,
                dims=("ping_time",),
                coords={"ping_time": collapsed["ping_time"]},
            ),
            PING_TIME_DISPLAY_COORD: xr.DataArray(
                np.arange(collapsed.sizes["ping_time"], dtype=np.int64),
                dims=("ping_time",),
                coords={"ping_time": collapsed["ping_time"]},
            ),
        }
    )
    collapsed = collapsed.swap_dims({"ping_time": PING_TIME_DISPLAY_COORD})
    return collapsed, removed_count


def prepare_display_dataarray(
    sv_da: xr.DataArray, hide_na_gaps: bool
) -> tuple[xr.DataArray, str, str, int]:
    """Prepare plotting axis mode for display without mutating source data."""
    removed_count = 0
    if "ping_time" in sv_da.coords:
        valid_time_mask = sv_da["ping_time"].notnull()
        valid_time_count = int(valid_time_mask.sum().item())
        total_time_count = int(sv_da.sizes.get("ping_time", 0))
        if 0 < valid_time_count < total_time_count:
            LOGGER.warning(
                "Found %d NaT ping_time values; dropping them before plotting.",
                total_time_count - valid_time_count,
            )
            sv_da = sv_da.sel(ping_time=valid_time_mask)

    if hide_na_gaps:
        sv_da, removed_count = collapse_na_ping_gaps(sv_da)
    if hide_na_gaps and PING_TIME_DISPLAY_COORD in sv_da.dims:
        return sv_da, PING_TIME_DISPLAY_COORD, "Ping Time (NA gaps collapsed)", removed_count
    return sv_da, "ping_time", "Ping Time", removed_count


def describe_x_axis_mode(x_coord: str, hide_na_requested: bool, removed_count: int) -> str:
    """Describe x-axis display mode for logs and UI status text."""
    if x_coord == PING_TIME_DISPLAY_COORD:
        if removed_count > 0:
            return (
                f"Compressed ping index with real datetime labels "
                f"(removed {removed_count} all-NaN ping bins)."
            )
        return "Compressed ping index with real datetime labels (no all-NaN ping bins removed)."
    if hide_na_requested:
        return "Ping time axis (hide-na requested, but collapse was unavailable)."
    return "Ping time axis (includes duty-cycle gaps)."


def _resolve_axis_dimension(sv_da: xr.DataArray, axis_name: str) -> str | None:
    """Return the underlying dimension used by a plotting axis coordinate."""
    if axis_name in sv_da.dims:
        return axis_name
    coord = sv_da.coords.get(axis_name)
    if coord is not None and len(coord.dims) == 1 and coord.dims[0] in sv_da.dims:
        return coord.dims[0]
    return None


def _axis_valid_count(coord: xr.DataArray) -> int:
    """Count valid coordinate values (finite for numeric, non-NaT for datetime)."""
    values = np.asarray(coord.values)
    if values.size == 0:
        return 0
    if np.issubdtype(values.dtype, np.datetime64):
        return int((~np.isnat(values)).sum())
    if np.issubdtype(values.dtype, np.number):
        return int(np.isfinite(values).sum())
    return int(coord.notnull().sum().item())


def _assign_index_fallback_axis(
    sv_da: xr.DataArray,
    axis_name: str,
    fallback_coord_name: str,
) -> tuple[xr.DataArray, str]:
    """Assign an integer coordinate fallback for a target plotting axis."""
    axis_dim = _resolve_axis_dimension(sv_da, axis_name)
    if axis_dim is None:
        if not sv_da.dims:
            return sv_da, axis_name
        axis_dim = next(iter(sv_da.dims))
    axis_values = xr.DataArray(
        np.arange(sv_da.sizes[axis_dim], dtype=np.int64),
        dims=(axis_dim,),
        coords={axis_dim: sv_da[axis_dim]},
    )
    sv_da = sv_da.assign_coords({fallback_coord_name: axis_values})
    return sv_da, fallback_coord_name


def _build_collapsed_time_epoch_ms(sv_da: xr.DataArray) -> list[int]:
    """Build epoch-millisecond lookup for dynamic hide-gap x-axis tick formatting."""
    if PING_TIME_ACTUAL_COORD not in sv_da.coords:
        return []
    actual_values = np.asarray(sv_da[PING_TIME_ACTUAL_COORD].values)
    if actual_values.ndim != 1 or actual_values.size == 0:
        return []
    if not np.issubdtype(actual_values.dtype, np.datetime64):
        return []

    actual_ns = actual_values.astype("datetime64[ns]")
    epoch_ms: list[int] = []
    for dt_value in actual_ns:
        if np.isnat(dt_value):
            epoch_ms.append(-1)
            continue
        epoch_ms.append(int(dt_value.astype("datetime64[ms]").astype(np.int64)))
    return epoch_ms


def _build_collapsed_time_hook(
    epoch_ms_values: list[int],
):
    """Create a Holoviews hook with dynamic dense-axis datetime styling."""
    hover_formatter = CustomJSHover(
        args={"epoch_ms_values": epoch_ms_values},
        code="""
            const values = epoch_ms_values || [];
            if (!values.length || !Number.isFinite(value)) {
                return "NaT";
            }
            const idx = Math.round(value);
            if (idx < 0 || idx >= values.length) {
                return "NaT";
            }
            const millis = values[idx];
            if (!Number.isFinite(millis) || millis < 0) {
                return "NaT";
            }
            const current = new Date(millis);
            const pad2 = (num) => String(num).padStart(2, "0");
            const yyyy = current.getFullYear();
            const mm = pad2(current.getMonth() + 1);
            const dd = pad2(current.getDate());
            const hh = pad2(current.getHours());
            const min = pad2(current.getMinutes());
            const ss = pad2(current.getSeconds());
            return `${yyyy}-${mm}-${dd} ${hh}:${min}:${ss}`;
        """,
    )
    tick_formatter = CustomJSTickFormatter(
        args={"epoch_ms_values": epoch_ms_values},
        code="""
            const values = epoch_ms_values || [];
            if (!values.length || !Number.isFinite(tick)) {
                return "";
            }
            const idx = Math.max(0, Math.min(values.length - 1, Math.round(tick)));
            const millis = values[idx];
            if (!Number.isFinite(millis) || millis < 0) {
                return "NaT";
            }
            const current = new Date(millis);
            const timeText = current.toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
                hour12: false,
            });

            let showDate = idx === 0;
            if (!showDate && idx > 0) {
                const prevMillis = values[idx - 1];
                if (Number.isFinite(prevMillis) && prevMillis >= 0) {
                    const prev = new Date(prevMillis);
                    showDate =
                        current.getFullYear() !== prev.getFullYear() ||
                        current.getMonth() !== prev.getMonth() ||
                        current.getDate() !== prev.getDate();
                }
            }
            if (!showDate) {
                return timeText;
            }
            const dateText = current.toLocaleDateString([], {
                month: "short",
                day: "numeric",
                year: "numeric",
            });
            return `${timeText}\\n${dateText}`;
        """,
    )

    def _hook(plot, _element) -> None:
        hover_tools = list(plot.state.select({"type": HoverTool}))
        for hover_tool in hover_tools:
            existing_tooltips = list(hover_tool.tooltips or [])
            trailing_tooltips = existing_tooltips[1:] if existing_tooltips else [("value", "@image")]
            hover_tool.tooltips = [("ping_time", "$x{custom}"), *trailing_tooltips]
            updated_formatters = dict(hover_tool.formatters)
            updated_formatters["$x"] = hover_formatter
            hover_tool.formatters = updated_formatters

        x_axes = getattr(plot.state, "xaxis", [])
        if not x_axes:
            return
        axis = x_axes[0]
        axis.formatter = tick_formatter
        axis.major_label_orientation = 0.0
        axis.major_label_text_align = "center"
        axis.major_label_text_baseline = "top"
        axis.major_label_standoff = 10
        current_bottom = getattr(plot.state, "min_border_bottom", 0) or 0
        plot.state.min_border_bottom = max(current_bottom, 62)

    return _hook


def _sanitize_plot_axis(
    sv_da: xr.DataArray,
    axis_name: str,
    fallback_coord_name: str,
    fallback_label: str,
) -> tuple[xr.DataArray, str, str]:
    """Ensure a plotting axis has valid coordinates with at least 2 points."""
    axis_dim = _resolve_axis_dimension(sv_da, axis_name)
    if axis_dim is None or axis_dim not in sv_da.dims:
        sv_da, fallback_axis = _assign_index_fallback_axis(
            sv_da,
            axis_name=axis_name,
            fallback_coord_name=fallback_coord_name,
        )
        return sv_da, fallback_axis, fallback_label

    axis_size = int(sv_da.sizes.get(axis_dim, 0))
    if axis_size < 2:
        sv_da, fallback_axis = _assign_index_fallback_axis(
            sv_da,
            axis_name=axis_name,
            fallback_coord_name=fallback_coord_name,
        )
        return sv_da, fallback_axis, fallback_label

    coord = sv_da.coords.get(axis_name)
    if coord is None and axis_name in sv_da.dims and axis_name in sv_da.coords:
        coord = sv_da[axis_name]
    if coord is None:
        sv_da, fallback_axis = _assign_index_fallback_axis(
            sv_da,
            axis_name=axis_name,
            fallback_coord_name=fallback_coord_name,
        )
        return sv_da, fallback_axis, fallback_label

    if _axis_valid_count(coord) >= 2:
        return sv_da, axis_name, fallback_label.replace(" (fallback)", "")

    sv_da, fallback_axis = _assign_index_fallback_axis(
        sv_da,
        axis_name=axis_name,
        fallback_coord_name=fallback_coord_name,
    )
    return sv_da, fallback_axis, fallback_label


def create_echogram_plot(
    sv_da: xr.DataArray,
    y_coord: str,
    x_coord: str,
    x_label: str,
    vmin: float,
    vmax: float,
    cmap: str,
    width: int,
    height: int,
    title: str,
    plot_theme: str,
    plot_sizing: str,
) -> hv.core.Dimensioned:
    """Create an interactive hvPlot echogram.

    Parameters
    ----------
    sv_da : xr.DataArray
        Selected channel Sv data for plotting.
    y_coord : str
        Vertical coordinate to use on y-axis.
    vmin : float
        Lower colormap limit (dB).
    vmax : float
        Upper colormap limit (dB).
    cmap : str
        Matplotlib/holoviews colormap name.
    width : int
        Plot width in pixels.
    height : int
        Plot height in pixels.
    title : str
        Plot title text.
    plot_theme : str
        Visual theme for plots ("dark" or "light").
    plot_sizing : str
        Plot sizing mode ("responsive" or "fixed").

    Returns
    -------
    hv.core.Dimensioned
        Interactive Holoviews object.
    """
    sv_da, x_coord_safe, x_label_safe = _sanitize_plot_axis(
        sv_da,
        axis_name=x_coord,
        fallback_coord_name="ping_time_plot_index",
        fallback_label="Ping Index (fallback)",
    )
    if x_coord_safe != x_coord:
        LOGGER.warning(
            "Requested x-axis coordinate '%s' is invalid or unavailable; using '%s'.",
            x_coord,
            x_coord_safe,
        )
    else:
        x_label_safe = x_label

    sv_da, y_coord_safe, _ = _sanitize_plot_axis(
        sv_da,
        axis_name=y_coord,
        fallback_coord_name=f"{y_coord}_plot_index",
        fallback_label=f"{y_coord} Index (fallback)",
    )
    if y_coord_safe != y_coord:
        LOGGER.warning(
            "Y-axis coordinate '%s' has insufficient valid values; using numeric index fallback.",
            y_coord,
        )
    ylabel = "Depth (m)" if y_coord_safe == "echo_range" else y_coord_safe
    tick_hook = None
    if x_coord_safe == PING_TIME_DISPLAY_COORD:
        epoch_ms_values = _build_collapsed_time_epoch_ms(sv_da)
        if epoch_ms_values:
            tick_hook = _build_collapsed_time_hook(
                epoch_ms_values=epoch_ms_values,
            )
    hvplot_kwargs = {
        "x": x_coord_safe,
        "y": y_coord_safe,
        "rasterize": True,
        "cmap": cmap,
        "clim": (vmin, vmax),
        "xlabel": x_label_safe,
        "ylabel": ylabel,
        "title": title,
        "colorbar": True,
        "tools": ["pan", "wheel_zoom", "box_zoom", "reset", "save"],
    }
    if plot_sizing == "responsive":
        hvplot_kwargs["responsive"] = True
        hvplot_kwargs["min_width"] = width
        hvplot_kwargs["min_height"] = height
    else:
        hvplot_kwargs["width"] = width
        hvplot_kwargs["height"] = height

    try:
        plot = sv_da.hvplot(
            **hvplot_kwargs,
        )
    except ValueError as exc:
        if "cannot convert float NaN to integer" not in str(exc):
            raise
        LOGGER.warning(
            "hvplot bounds failed with NaN axis values; retrying with synthetic numeric axes."
        )
        fallback_da = sv_da
        fallback_da, fallback_x = _assign_index_fallback_axis(
            fallback_da,
            axis_name=x_coord_safe,
            fallback_coord_name="ping_time_plot_index",
        )
        fallback_da, fallback_y = _assign_index_fallback_axis(
            fallback_da,
            axis_name=y_coord_safe,
            fallback_coord_name=f"{y_coord_safe}_plot_index",
        )
        hvplot_kwargs["x"] = fallback_x
        hvplot_kwargs["y"] = fallback_y
        hvplot_kwargs["xlabel"] = "Ping Index (fallback)"
        hvplot_kwargs["ylabel"] = (
            "Depth Index (fallback)"
            if y_coord_safe == "echo_range"
            else f"{y_coord_safe} Index (fallback)"
        )
        plot = fallback_da.hvplot(
            **hvplot_kwargs,
        )
        tick_hook = None
    opt_kwargs = {
        "invert_yaxis": True,
        "active_tools": ["wheel_zoom"],
        "fontsize": {"title": "12pt", "labels": "10pt", "xticks": "9pt", "yticks": "9pt"},
    }
    if tick_hook is not None:
        opt_kwargs["hooks"] = [tick_hook]
    if plot_theme == "dark":
        opt_kwargs["bgcolor"] = DARK_BG
    return plot.opts(**opt_kwargs)


def create_panel_layout(
    ds_mvbs: xr.Dataset,
    channels: Sequence[str],
    initial_channel: str,
    initial_cmap: str,
    initial_vmin: float,
    initial_vmax: float,
    width: int,
    height: int,
    default_export_path: Path,
    plot_theme: str,
    plot_sizing: str,
    hide_na_gaps: bool,
    html_resources: str,
):
    """Create a Panel layout for interactive echogram parameter tuning."""
    import panel as pn

    channel_labels = [build_channel_label(channel) for channel in channels]
    label_to_channel = dict(zip(channel_labels, channels))
    initial_label = build_channel_label(initial_channel)

    cmap_options = list(
        dict.fromkeys(
            [
                initial_cmap,
                "viridis",
                "RdYlBu_r",
                "cividis",
                "magma",
                "inferno",
                "plasma",
                "turbo",
            ]
        )
    )

    channel_select = pn.widgets.Select(name="Channel", options=channel_labels, value=initial_label)
    cmap_select = pn.widgets.Select(name="Colormap", options=cmap_options, value=initial_cmap)
    vmin_slider = pn.widgets.FloatSlider(name="dB Min", start=-120.0, end=-10.0, step=1.0, value=initial_vmin)
    vmax_slider = pn.widgets.FloatSlider(name="dB Max", start=-120.0, end=-10.0, step=1.0, value=initial_vmax)
    export_path_input = pn.widgets.TextInput(
        name="Export HTML Path",
        value=str(default_export_path),
    )
    export_button = pn.widgets.Button(name="Export Current View to HTML", button_type="primary")
    export_status = pn.pane.Markdown("No export yet.", sizing_mode="stretch_width")

    def build_plot_for_settings(
        channel_label: str,
        cmap: str,
        vmin: float,
        vmax: float,
    ) -> tuple[hv.core.Dimensioned, str, str, float, float, str]:
        selected_channel = label_to_channel[channel_label]
        sv_da, y_coord, _ = build_plot_dataarray(ds_mvbs, selected_channel)
        sv_da, x_coord, x_label, removed_count = prepare_display_dataarray(
            sv_da,
            hide_na_gaps=hide_na_gaps,
        )
        x_axis_note = describe_x_axis_mode(
            x_coord=x_coord,
            hide_na_requested=hide_na_gaps,
            removed_count=removed_count,
        )
        if x_coord == PING_TIME_DISPLAY_COORD:
            LOGGER.info(
                "%s (%s)",
                x_axis_note,
                selected_channel,
            )
        plot_vmin, plot_vmax = normalize_color_limits(vmin, vmax)
        selected_freq = infer_channel_frequency_khz(selected_channel)
        title = f"EK80 MVBS Echogram | {selected_channel}"
        if selected_freq is not None:
            title = f"{title} ({selected_freq} kHz)"
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
        )
        return plot_obj, selected_channel, title, plot_vmin, plot_vmax, x_axis_note

    @pn.depends(
        channel_label=channel_select.param.value,
        cmap=cmap_select.param.value,
        vmin=vmin_slider.param.value,
        vmax=vmax_slider.param.value,
    )
    def plot_view(channel_label: str, cmap: str, vmin: float, vmax: float):
        plot_obj, _, _, _, _, _ = build_plot_for_settings(channel_label, cmap, vmin, vmax)
        return plot_obj

    @pn.depends(
        channel_label=channel_select.param.value,
        cmap=cmap_select.param.value,
        vmin=vmin_slider.param.value,
        vmax=vmax_slider.param.value,
    )
    def status_text(channel_label: str, cmap: str, vmin: float, vmax: float):
        _, selected_channel, _, plot_vmin, plot_vmax, x_axis_note = build_plot_for_settings(
            channel_label,
            cmap,
            vmin,
            vmax,
        )
        return (
            "### Display Settings\n"
            f"- Channel: `{selected_channel}`\n"
            f"- Colormap: `{cmap}`\n"
            f"- dB range: `{plot_vmin} to {plot_vmax}`\n"
            f"- X-axis: `{x_axis_note}`\n"
            "- Data source: `MVBS` (binned Sv), not raw ping-by-ping Sv."
        )

    def export_current_view(_: object) -> None:
        try:
            target_text = export_path_input.value.strip()
            if not target_text:
                raise ValueError("Export path cannot be empty.")
            export_path = Path(target_text).expanduser()
            if export_path.suffix.lower() != ".html":
                export_path = export_path.with_suffix(".html")
            export_path = export_path.resolve()
            export_path.parent.mkdir(parents=True, exist_ok=True)

            plot_obj, _, title, _, _, _ = build_plot_for_settings(
                channel_select.value,
                cmap_select.value,
                vmin_slider.value,
                vmax_slider.value,
            )
            bokeh_plot = hv.render(plot_obj, backend="bokeh")
            save_bokeh_plot_html(
                bokeh_plot,
                export_path,
                title=title,
                plot_theme=plot_theme,
                html_resources=html_resources,
            )
            export_status.object = f"Saved current view to `{export_path}`"
            LOGGER.info("Saved Panel-exported HTML snapshot to %s", export_path)
        except Exception as exc:  # noqa: BLE001 - show export errors inside UI
            export_status.object = f"Export failed: `{type(exc).__name__}: {exc}`"
            LOGGER.exception("Panel export failed.")

    export_button.on_click(export_current_view)

    controls = pn.WidgetBox(
        "## Plot Controls",
        channel_select,
        cmap_select,
        vmin_slider,
        vmax_slider,
        "### Export",
        export_path_input,
        export_button,
        export_status,
        width=360,
    )
    details = pn.pane.Markdown(status_text, sizing_mode="stretch_width")
    layout = pn.Row(
        controls,
        pn.Column(details, plot_view, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    )
    return layout


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Chunked in-memory EK80 MVBS echogram visualization."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=_path_from_env("EK80_RAW_DIR"),
        help=(
            "Directory containing EK80 .raw files. "
            "Required unless EK80_RAW_DIR is set."
        ),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=20,
        help="Maximum number of .raw files to process when no datetime window is used.",
    )
    parser.add_argument(
        "--start-datetime",
        type=str,
        default=None,
        help=(
            "Optional inclusive datetime lower bound for file selection. "
            "Examples: 20250819, 2025-08-19, 20250819T230000, 2025-08-19T23:00:00."
        ),
    )
    parser.add_argument(
        "--end-datetime",
        type=str,
        default=None,
        help=(
            "Optional exclusive datetime upper bound for file selection. "
            "Same formats as --start-datetime."
        ),
    )
    parser.add_argument(
        "--duration-days",
        type=int,
        default=None,
        help=(
            "Optional number of days from --start-datetime to process "
            "(for example, 1 for one day, 2 for two days)."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help="Number of .raw files per processing chunk.",
    )
    parser.add_argument(
        "--range-meter-bin",
        type=float,
        default=10.0,
        help="MVBS range bin size in meters.",
    )
    parser.add_argument(
        "--ping-time-bin",
        type=str,
        default="30s",
        help="MVBS time bin, e.g. 30s.",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default=None,
        help="Optional exact channel label to plot. Defaults to first available channel.",
    )
    parser.add_argument(
        "--waveform-mode",
        type=str,
        default="auto",
        help="EK80 waveform mode (CW/BB/FM) or 'auto' to detect.",
    )
    parser.add_argument(
        "--encode-mode",
        type=str,
        default="auto",
        help="EK80 encode mode (power/complex) or 'auto' to detect.",
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
        default="RdYlBu_r",
        help="Colormap for echogram rendering.",
    )
    parser.add_argument(
        "--plot-theme",
        type=str,
        choices=["dark", "light"],
        default="dark",
        help="Plot theme style for exported/displayed echograms.",
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
            "to hide duty-cycle time gaps while retaining datetime tick labels."
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
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--ui-mode",
        type=str,
        choices=["static", "panel"],
        default="static",
        help="Output mode: 'static' for one plot window, 'panel' for live controls.",
    )
    parser.add_argument(
        "--panel-port",
        type=int,
        default=0,
        help="Port for Panel app in panel mode (0 = auto-select).",
    )
    parser.add_argument(
        "--panel-no-browser",
        action="store_true",
        help="In panel mode, do not auto-open browser.",
    )
    parser.add_argument(
        "--save-html",
        type=Path,
        nargs="?",
        const=Path("ek80_chunked_echogram.html"),
        default=None,
        help=(
            "In static mode, save a standalone HTML snapshot immediately. "
            "In panel mode, sets the default path used by the 'Export Current View to HTML' button. "
            "Optionally provide a path; default is ./ek80_chunked_echogram.html."
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
        "--export-all-channels",
        action="store_true",
        help=(
            "In static mode, export one HTML per detected channel using --save-html "
            "as the filename base."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run chunked EK80 processing and open interactive echogram."""
    args = parse_args()
    configure_logging(args.log_level)
    if args.raw_dir is None:
        raise ValueError(
            "Missing raw input directory. Pass --raw-dir or set EK80_RAW_DIR."
        )

    hv.extension("bokeh")
    dask.config.set(scheduler="synchronous")
    LOGGER.info("Dask scheduler configured as 'synchronous' for Windows stability.")

    start_dt, end_dt = resolve_time_window(
        start_datetime_text=args.start_datetime,
        end_datetime_text=args.end_datetime,
        duration_days=args.duration_days,
    )
    if start_dt or end_dt:
        LOGGER.info(
            "Applying filename datetime window [start=%s, end=%s)",
            start_dt.isoformat() if start_dt else "-inf",
            end_dt.isoformat() if end_dt else "+inf",
        )

    raw_files = list_raw_files(
        raw_dir=args.raw_dir,
        max_files=args.max_files,
        start_datetime=start_dt,
        end_datetime=end_dt,
    )
    LOGGER.info("Selected %d .raw files from %s", len(raw_files), args.raw_dir)

    mvbs_chunks: list[xr.Dataset] = []
    for chunk_index, chunk_files in enumerate(chunked(raw_files, args.chunk_size), start=1):
        LOGGER.info("Processing chunk %d with %d files", chunk_index, len(chunk_files))
        ds_mvbs_chunk = compute_mvbs_for_chunk(
            chunk_files=chunk_files,
            range_meter_bin=args.range_meter_bin,
            ping_time_bin=args.ping_time_bin,
            waveform_mode=args.waveform_mode,
            encode_mode=args.encode_mode,
        )
        mvbs_chunks.append(ds_mvbs_chunk)

    ds_mvbs = xr.concat(
        mvbs_chunks,
        dim="ping_time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        combine_attrs="override",
        join="outer",
    ).sortby("ping_time")

    if "channel" not in ds_mvbs.dims:
        raise KeyError(
            "Missing expected 'channel' dimension after concatenation. "
            f"Dataset sizes: {dict(ds_mvbs.sizes)}"
        )

    channels = ds_mvbs["channel"].astype(str).values.tolist()
    print("\nAvailable channel names:")
    for channel in channels:
        print(f" - {build_channel_label(channel)}")

    sv_da, y_coord, selected_channel = build_plot_dataarray(ds_mvbs, args.channel)
    sv_da, x_coord, x_label, removed_bins = prepare_display_dataarray(
        sv_da,
        hide_na_gaps=args.hide_na_gaps,
    )
    x_axis_note = describe_x_axis_mode(
        x_coord=x_coord,
        hide_na_requested=args.hide_na_gaps,
        removed_count=removed_bins,
    )
    LOGGER.info("Display x-axis mode for %s: %s", selected_channel, x_axis_note)
    selected_freq = infer_channel_frequency_khz(selected_channel)
    selected_freq_label = f"{selected_freq} kHz" if selected_freq is not None else "frequency unknown"
    print(f"\nSelected channel for plot: {selected_channel} ({selected_freq_label})")
    print(f"X-axis mode: {x_axis_note}")

    plot_vmin, plot_vmax = normalize_color_limits(args.vmin, args.vmax)
    plot_title = f"EK80 MVBS Echogram | {selected_channel}"
    if selected_freq is not None:
        plot_title = f"{plot_title} ({selected_freq} kHz)"

    echogram = create_echogram_plot(
        sv_da=sv_da,
        y_coord=y_coord,
        x_coord=x_coord,
        x_label=x_label,
        vmin=plot_vmin,
        vmax=plot_vmax,
        cmap=args.cmap,
        width=args.width,
        height=args.height,
        title=plot_title,
        plot_theme=args.plot_theme,
        plot_sizing=args.plot_sizing,
    )
    bokeh_plot = hv.render(echogram, backend="bokeh")
    apply_bokeh_plot_theme(bokeh_plot, plot_theme=args.plot_theme)

    if args.export_all_channels and args.ui_mode == "panel":
        LOGGER.warning(
            "--export-all-channels applies to static mode only; ignoring in panel mode."
        )

    if args.ui_mode != "panel":
        if args.export_all_channels:
            base_output_path = resolve_html_output_path(
                args.save_html or Path("ek80_chunked_echogram.html")
            )
            saved_paths: list[Path] = []
            for channel_name in channels:
                channel_da, channel_y, _ = build_plot_dataarray(ds_mvbs, channel_name)
                channel_da, channel_x, channel_xlabel, channel_removed = prepare_display_dataarray(
                    channel_da,
                    hide_na_gaps=args.hide_na_gaps,
                )
                if args.hide_na_gaps:
                    LOGGER.info(
                        "Display x-axis mode for %s: %s",
                        channel_name,
                        describe_x_axis_mode(
                            x_coord=channel_x,
                            hide_na_requested=True,
                            removed_count=channel_removed,
                        ),
                    )
                channel_title = f"EK80 MVBS Echogram | {channel_name}"
                channel_freq = infer_channel_frequency_khz(channel_name)
                if channel_freq is not None:
                    channel_title = f"{channel_title} ({channel_freq} kHz)"
                channel_plot = create_echogram_plot(
                    sv_da=channel_da,
                    y_coord=channel_y,
                    x_coord=channel_x,
                    x_label=channel_xlabel,
                    vmin=plot_vmin,
                    vmax=plot_vmax,
                    cmap=args.cmap,
                    width=args.width,
                    height=args.height,
                    title=channel_title,
                    plot_theme=args.plot_theme,
                    plot_sizing=args.plot_sizing,
                )
                channel_bokeh = hv.render(channel_plot, backend="bokeh")
                channel_path = base_output_path.with_name(
                    f"{base_output_path.stem}__{channel_slug(channel_name)}{base_output_path.suffix}"
                )
                saved_path = save_bokeh_plot_html(
                    channel_bokeh,
                    channel_path,
                    channel_title,
                    plot_theme=args.plot_theme,
                    html_resources=args.html_resources,
                )
                saved_paths.append(saved_path)
            print("Saved HTML snapshots by channel:")
            for path in saved_paths:
                print(f" - {path}")
        elif args.save_html is not None:
            saved_path = save_bokeh_plot_html(
                bokeh_plot,
                args.save_html,
                plot_title,
                plot_theme=args.plot_theme,
                html_resources=args.html_resources,
            )
            print(f"Saved HTML snapshot: {saved_path}")

    print("\nProcessing summary:")
    print(f"Total files processed: {len(raw_files)}")
    print(f"Final MVBS dataset shape: {dict(ds_mvbs.sizes)}")
    print(f"Time range: {str(ds_mvbs['ping_time'].min().values)} -> {str(ds_mvbs['ping_time'].max().values)}")
    print(f"Channel names available: {channels}")
    if args.ui_mode == "panel":
        import panel as pn

        panel_vmin, panel_vmax = normalize_color_limits(args.vmin, args.vmax)
        default_export_path = args.save_html or Path("ek80_chunked_echogram.html")
        app = create_panel_layout(
            ds_mvbs=ds_mvbs,
            channels=channels,
            initial_channel=selected_channel,
            initial_cmap=args.cmap,
            initial_vmin=panel_vmin,
            initial_vmax=panel_vmax,
            width=args.width,
            height=args.height,
            default_export_path=default_export_path,
            plot_theme=args.plot_theme,
            plot_sizing=args.plot_sizing,
            hide_na_gaps=args.hide_na_gaps,
            html_resources=args.html_resources,
        )
        LOGGER.info(
            "Starting Panel app (port=%s, auto_open_browser=%s)",
            args.panel_port,
            not args.panel_no_browser,
        )
        pn.serve(
            app,
            title="EK80 MVBS Explorer",
            port=args.panel_port,
            show=not args.panel_no_browser,
        )
    else:
        show(bokeh_plot)


if __name__ == "__main__":
    main()
