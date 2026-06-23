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
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import dask
import echopype as ep
import holoviews as hv
import hvplot.xarray  # noqa: F401  # Registers hvplot on xarray objects
import numpy as np
import xarray as xr
from bokeh.io import save
from bokeh.models import ColorBar, CustomJSHover, CustomJSTickFormatter, FixedTicker, HoverTool
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
ANALYSIS_MODE_CHOICES = ("mean-vs-depth", "mean-vs-time", "scalar")
ANALYSIS_MODE_LABELS = {
    "mean-vs-depth": "Mean Sv vs Depth",
    "mean-vs-time": "Mean Sv vs Time",
    "scalar": "Scalar Mean Sv",
}
PLOT_DATA_EXPORT_CHOICES = ("none", "netcdf", "csv", "both")
PLOT_DATA_CSV_COMPRESSION_CHOICES = ("none", "gzip")


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


@dataclass(frozen=True)
class TransducerMetadataSnapshot:
    """Transducer metadata extracted from a single raw file."""

    raw_file: Path
    date_yyyymmdd: str
    transducer_names: tuple[str, ...]
    transducer_serial_numbers: tuple[str, ...]
    transceiver_serial_numbers: tuple[str, ...]
    frequency_nominal_hz: tuple[float, ...]
    beam_direction_z: tuple[float, ...]

    def signature(self) -> tuple[
        tuple[str, ...],
        tuple[str, ...],
        tuple[str, ...],
        tuple[float, ...],
        tuple[float, ...],
    ]:
        """Return signature used for consistency checks."""
        return (
            self.transducer_names,
            self.transducer_serial_numbers,
            self.transceiver_serial_numbers,
            self.frequency_nominal_hz,
            self.beam_direction_z,
        )


@dataclass(frozen=True)
class DailyTransducerReport:
    """Per-day transducer metadata summary for plotting checks."""

    date_yyyymmdd: str
    sampled_files: tuple[Path, ...]
    metadata_available: bool
    consistent: bool
    inferred_facing: str | None
    facing_reason: str
    primary_snapshot: TransducerMetadataSnapshot | None


@dataclass(frozen=True)
class MVBSAnalysisResult:
    """Container for MVBS-based mean Sv analysis outputs."""

    selected_channel: str
    y_coord: str
    y_label: str
    requested_time_start: str | None
    requested_time_end: str | None
    requested_depth_min: float | None
    requested_depth_max: float | None
    actual_time_start: str | None
    actual_time_end: str | None
    actual_depth_min: float | None
    actual_depth_max: float | None
    ping_time_count: int
    depth_bin_count: int
    total_cell_count: int
    valid_cell_count: int
    scalar_mean_sv_db: float | None
    ping_time_bin: str | None
    range_meter_bin: float | None
    mean_sv_by_depth_db: xr.DataArray
    mean_sv_by_time_db: xr.DataArray

    def to_record(self, analysis_mode: str) -> dict[str, object]:
        """Serialize scalar analysis metadata to a machine-readable record."""
        return {
            "analysis_mode": analysis_mode,
            "analysis_mode_label": ANALYSIS_MODE_LABELS.get(analysis_mode, analysis_mode),
            "selected_channel": self.selected_channel,
            "y_coord": self.y_coord,
            "y_label": self.y_label,
            "requested_time_start": self.requested_time_start,
            "requested_time_end": self.requested_time_end,
            "requested_depth_min": self.requested_depth_min,
            "requested_depth_max": self.requested_depth_max,
            "actual_time_start": self.actual_time_start,
            "actual_time_end": self.actual_time_end,
            "actual_depth_min": self.actual_depth_min,
            "actual_depth_max": self.actual_depth_max,
            "ping_time_count": self.ping_time_count,
            "depth_bin_count": self.depth_bin_count,
            "total_cell_count": self.total_cell_count,
            "valid_cell_count": self.valid_cell_count,
            "scalar_mean_sv_db": self.scalar_mean_sv_db,
            "ping_time_bin": self.ping_time_bin,
            "range_meter_bin": self.range_meter_bin,
        }


def _extract_date_token(file_path: Path) -> str:
    """Extract YYYYMMDD date token from filename, or 'unknown_date'."""
    file_dt = extract_datetime_from_filename(file_path)
    if file_dt is None:
        return "unknown_date"
    return file_dt.strftime("%Y%m%d")


def _normalize_text_values(values: np.ndarray) -> tuple[str, ...]:
    """Normalize string-like metadata arrays to sorted unique non-empty values."""
    normalized: set[str] = set()
    for item in np.asarray(values).ravel().tolist():
        text = str(item).strip()
        if not text:
            continue
        if text.lower() in {"nan", "none"}:
            continue
        normalized.add(text)
    return tuple(sorted(normalized))


def _normalize_float_values(values: np.ndarray, ndigits: int = 6) -> tuple[float, ...]:
    """Normalize numeric metadata arrays to sorted unique finite floats."""
    arr = np.asarray(values, dtype=float).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return tuple()
    rounded = np.round(finite, ndigits)
    unique_values = np.unique(rounded)
    return tuple(float(value) for value in unique_values.tolist())


def _dataset_text_values(group_ds: xr.Dataset, var_name: str) -> tuple[str, ...]:
    """Extract normalized text metadata values from an xarray dataset variable."""
    if var_name not in group_ds:
        return tuple()
    return _normalize_text_values(group_ds[var_name].values)


def _dataset_float_values(
    group_ds: xr.Dataset, var_name: str, ndigits: int = 6
) -> tuple[float, ...]:
    """Extract normalized float metadata values from an xarray dataset variable."""
    if var_name not in group_ds:
        return tuple()
    try:
        return _normalize_float_values(group_ds[var_name].values, ndigits=ndigits)
    except (TypeError, ValueError):
        return tuple()


def _probe_transducer_metadata(raw_file: Path) -> TransducerMetadataSnapshot | None:
    """Probe transducer metadata from one EK80 raw file."""
    date_token = _extract_date_token(raw_file)
    try:
        echodata = ep.open_raw(str(raw_file), sonar_model="EK80")
    except Exception as exc:  # noqa: BLE001 - non-fatal metadata probe
        LOGGER.warning(
            "Transducer metadata probe failed for %s (%s: %s).",
            raw_file.name,
            type(exc).__name__,
            exc,
        )
        return None

    try:
        sonar_group = echodata["Sonar"]
        beam_group = echodata["Sonar/Beam_group1"]
        snapshot = TransducerMetadataSnapshot(
            raw_file=raw_file,
            date_yyyymmdd=date_token,
            transducer_names=_dataset_text_values(sonar_group, "transducer_name"),
            transducer_serial_numbers=_dataset_text_values(
                sonar_group, "transducer_serial_number"
            ),
            transceiver_serial_numbers=_dataset_text_values(
                sonar_group, "transceiver_serial_number"
            ),
            frequency_nominal_hz=_dataset_float_values(
                sonar_group, "frequency_nominal", ndigits=3
            ),
            beam_direction_z=_dataset_float_values(beam_group, "beam_direction_z"),
        )
        return snapshot
    except Exception as exc:  # noqa: BLE001 - non-fatal metadata probe
        LOGGER.warning(
            "Transducer metadata probe failed for %s (%s: %s).",
            raw_file.name,
            type(exc).__name__,
            exc,
        )
        return None
    finally:
        del echodata
        gc.collect()


def _infer_facing_from_beam_direction_z(
    beam_direction_z_values: Sequence[float],
) -> tuple[str | None, str]:
    """Infer transducer facing from beam_direction_z values when available."""
    if not beam_direction_z_values:
        return None, "beam_direction_z metadata is unavailable"

    positives = [value for value in beam_direction_z_values if value > 0]
    negatives = [value for value in beam_direction_z_values if value < 0]
    if positives and not negatives:
        return "down", "beam_direction_z > 0 (assumed down-looking)"
    if negatives and not positives:
        return "up", "beam_direction_z < 0 (assumed up-looking)"
    return None, "beam_direction_z contains mixed/zero values"


def _sample_daily_files_for_metadata(day_files: Sequence[Path]) -> list[Path]:
    """Pick representative files within a day for metadata consistency checks."""
    if not day_files:
        return []
    if len(day_files) == 1:
        return [day_files[0]]
    return [day_files[0], day_files[-1]]


def collect_daily_transducer_reports(raw_files: Sequence[Path]) -> list[DailyTransducerReport]:
    """Collect per-day transducer metadata reports from representative files."""
    grouped_files: dict[str, list[Path]] = {}
    for file_path in raw_files:
        grouped_files.setdefault(_extract_date_token(file_path), []).append(file_path)

    reports: list[DailyTransducerReport] = []
    for date_token in sorted(grouped_files):
        day_files = grouped_files[date_token]
        sampled_files = tuple(_sample_daily_files_for_metadata(day_files))
        snapshots: list[TransducerMetadataSnapshot] = []
        for sample_file in sampled_files:
            snapshot = _probe_transducer_metadata(sample_file)
            if snapshot is not None:
                snapshots.append(snapshot)

        if not snapshots:
            reports.append(
                DailyTransducerReport(
                    date_yyyymmdd=date_token,
                    sampled_files=sampled_files,
                    metadata_available=False,
                    consistent=False,
                    inferred_facing=None,
                    facing_reason="metadata probe unavailable for sampled files",
                    primary_snapshot=None,
                )
            )
            continue

        base_signature = snapshots[0].signature()
        consistent = all(item.signature() == base_signature for item in snapshots[1:])

        inferred_values: set[str] = set()
        facing_reasons: set[str] = set()
        for snapshot in snapshots:
            inferred_facing, reason = _infer_facing_from_beam_direction_z(
                snapshot.beam_direction_z
            )
            facing_reasons.add(reason)
            if inferred_facing is not None:
                inferred_values.add(inferred_facing)

        if len(inferred_values) == 1:
            inferred_facing = next(iter(inferred_values))
        else:
            inferred_facing = None

        if len(inferred_values) > 1:
            facing_reason = "conflicting beam_direction_z sign across sampled files"
        else:
            facing_reason = "; ".join(sorted(facing_reasons))
        if not consistent:
            facing_reason = (
                f"{facing_reason}; transducer metadata differs across sampled files"
            )

        reports.append(
            DailyTransducerReport(
                date_yyyymmdd=date_token,
                sampled_files=sampled_files,
                metadata_available=True,
                consistent=consistent,
                inferred_facing=inferred_facing,
                facing_reason=facing_reason,
                primary_snapshot=snapshots[0],
            )
        )

    return reports


def log_daily_transducer_reports(reports: Sequence[DailyTransducerReport]) -> None:
    """Log per-day transducer metadata report details."""
    if not reports:
        LOGGER.warning("Transducer metadata report skipped: no files were selected.")
        print("Transducer metadata check: no files selected.")
        return

    LOGGER.info("=== Daily transducer metadata check ===")
    print("\nTransducer metadata check by day:")
    for report in reports:
        sampled_names = [path.name for path in report.sampled_files]
        LOGGER.info(
            "Date %s | sampled_files=%s | metadata_available=%s | consistent=%s",
            report.date_yyyymmdd,
            sampled_names,
            report.metadata_available,
            report.consistent,
        )
        print(
            f"- Date {report.date_yyyymmdd} | sampled_files={sampled_names} | "
            f"metadata_available={report.metadata_available} | consistent={report.consistent}"
        )

        snapshot = report.primary_snapshot
        if snapshot is None:
            LOGGER.warning("  No transducer metadata could be read for sampled files.")
            print("  - Metadata: unavailable")
        else:
            LOGGER.info(
                "  transducer_names=%s",
                list(snapshot.transducer_names) or ["(missing)"],
            )
            LOGGER.info(
                "  transducer_serial_numbers=%s",
                list(snapshot.transducer_serial_numbers) or ["(missing)"],
            )
            LOGGER.info(
                "  transceiver_serial_numbers=%s",
                list(snapshot.transceiver_serial_numbers) or ["(missing)"],
            )
            LOGGER.info(
                "  frequency_nominal_hz=%s",
                list(snapshot.frequency_nominal_hz) or ["(missing)"],
            )
            LOGGER.info(
                "  beam_direction_z=%s",
                list(snapshot.beam_direction_z) or ["(missing/NaN)"],
            )
            print(
                "  - transducer_names="
                f"{list(snapshot.transducer_names) or ['(missing)']}"
            )
            print(
                "  - transducer_serial_numbers="
                f"{list(snapshot.transducer_serial_numbers) or ['(missing)']}"
            )
            print(
                "  - transceiver_serial_numbers="
                f"{list(snapshot.transceiver_serial_numbers) or ['(missing)']}"
            )
            print(
                "  - frequency_nominal_hz="
                f"{list(snapshot.frequency_nominal_hz) or ['(missing)']}"
            )
            print(
                "  - beam_direction_z="
                f"{list(snapshot.beam_direction_z) or ['(missing/NaN)']}"
            )

        if report.inferred_facing is not None:
            LOGGER.info(
                "  inferred_transducer_facing=%s | reason=%s",
                report.inferred_facing,
                report.facing_reason,
            )
            print(
                "  - inferred_transducer_facing="
                f"{report.inferred_facing} | reason={report.facing_reason}"
            )
        else:
            LOGGER.warning(
                "  inferred_transducer_facing=unknown | reason=%s",
                report.facing_reason,
            )
            print(
                "  - inferred_transducer_facing=unknown "
                f"| reason={report.facing_reason}"
            )

        if not report.consistent:
            LOGGER.warning(
                "  Metadata consistency warning for %s; verify instrument setup changes.",
                report.date_yyyymmdd,
            )
            print(
                "  - WARNING: metadata differs across sampled files; "
                "verify instrument setup changes."
            )


def resolve_transducer_facing(
    requested_facing: str,
    reports: Sequence[DailyTransducerReport],
) -> tuple[str, str]:
    """Resolve effective transducer facing used for plotting."""
    if requested_facing in {"down", "up"}:
        return requested_facing, "cli_override"

    inferred = sorted(
        {
            report.inferred_facing
            for report in reports
            if report.inferred_facing in {"down", "up"}
        }
    )
    if len(inferred) == 1:
        return inferred[0], "metadata_auto"
    if len(inferred) > 1:
        LOGGER.warning(
            "Conflicting inferred transducer facing values across days: %s. "
            "Falling back to down-looking plotting.",
            inferred,
        )
        return "down", "metadata_conflict_default_down"

    LOGGER.warning(
        "Unable to infer transducer facing from metadata. "
        "Falling back to down-looking plotting."
    )
    return "down", "metadata_unavailable_default_down"


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


def _datetime_to_iso_text(value: dt.datetime | np.datetime64 | None) -> str | None:
    """Convert datetime-like input to a compact ISO8601 string."""
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.replace(microsecond=0).isoformat(sep=" ")

    datetime_value = np.datetime64(value, "ns")
    if np.isnat(datetime_value):
        return None
    return np.datetime_as_string(datetime_value, unit="s").replace("T", " ")


def _coerce_datetime_value(
    value: str | dt.datetime | np.datetime64 | None,
) -> dt.datetime | None:
    """Normalize optional datetime-like inputs to Python datetime objects."""
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            return None
        text = np.datetime_as_string(np.datetime64(value, "ns"), unit="s")
        return dt.datetime.fromisoformat(text.replace("T", " "))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        return parse_datetime_input(text)
    raise TypeError(f"Unsupported datetime input type: {type(value).__name__}")


def _normalize_numeric_bounds(
    lower: float | None, upper: float | None, label: str
) -> tuple[float | None, float | None]:
    """Return ascending numeric bounds and warn when user order is reversed."""
    if lower is None or upper is None:
        return lower, upper
    if lower <= upper:
        return lower, upper
    LOGGER.warning(
        "%s bounds were reversed (%s > %s); swapping for analysis selection.",
        label,
        lower,
        upper,
    )
    return upper, lower


def _coordinate_float_bounds(coord: xr.DataArray) -> tuple[float | None, float | None]:
    """Return finite min/max values from a numeric coordinate."""
    try:
        values = np.asarray(coord.values, dtype=float).ravel()
    except (TypeError, ValueError):
        return None, None

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None
    return float(finite.min()), float(finite.max())


def _coordinate_time_bounds(coord: xr.DataArray) -> tuple[str | None, str | None]:
    """Return min/max ISO timestamps from a datetime coordinate."""
    values = np.asarray(coord.values)
    if values.size == 0:
        return None, None
    try:
        datetime_values = values.astype("datetime64[ns]").ravel()
    except (TypeError, ValueError):
        return None, None
    valid = datetime_values[~np.isnat(datetime_values)]
    if valid.size == 0:
        return None, None
    return _datetime_to_iso_text(valid.min()), _datetime_to_iso_text(valid.max())


def _sv_db_to_linear(sv_da: xr.DataArray) -> xr.DataArray:
    """Convert Sv in dB to linear units for physically consistent averaging."""
    return np.power(10.0, sv_da / 10.0)


def _linear_to_sv_db(linear_da: xr.DataArray) -> xr.DataArray:
    """Convert linear backscatter values back to Sv in dB."""
    safe_linear = linear_da.where(linear_da > 0)
    return 10.0 * np.log10(safe_linear)


def compute_linear_mean_sv_db(
    sv_da: xr.DataArray,
    dims: Sequence[str],
) -> xr.DataArray:
    """Compute mean Sv over dims by averaging in linear space then converting to dB."""
    if not dims:
        return sv_da
    linear = _sv_db_to_linear(sv_da)
    mean_linear = linear.mean(dim=list(dims), skipna=True)
    return _linear_to_sv_db(mean_linear)


def subset_mvbs_for_analysis(
    sv_da: xr.DataArray,
    y_coord: str,
    time_start: dt.datetime | None,
    time_end: dt.datetime | None,
    depth_min: float | None,
    depth_max: float | None,
) -> xr.DataArray:
    """Apply optional time/depth filters to an MVBS Sv DataArray."""
    subset = sv_da

    if (time_start is not None or time_end is not None) and "ping_time" not in subset.dims:
        raise ValueError("Selected dataset does not expose a 'ping_time' axis for time filtering.")

    if time_start is not None and time_end is not None and time_end <= time_start:
        raise ValueError(
            f"Invalid analysis time window: end ({time_end.isoformat()}) "
            f"must be after start ({time_start.isoformat()})."
        )
    if time_start is not None or time_end is not None:
        start_value = np.datetime64(time_start) if time_start is not None else None
        end_value = np.datetime64(time_end) if time_end is not None else None
        subset = subset.sel(ping_time=slice(start_value, end_value))

    depth_min_norm, depth_max_norm = _normalize_numeric_bounds(depth_min, depth_max, "Depth")
    if depth_min_norm is not None or depth_max_norm is not None:
        if y_coord not in subset.coords and y_coord not in subset.dims:
            raise ValueError(
                f"Cannot apply depth bounds because coordinate '{y_coord}' is unavailable."
            )
        coord = subset[y_coord]
        mask = coord.notnull()
        if depth_min_norm is not None:
            mask = mask & (coord >= depth_min_norm)
        if depth_max_norm is not None:
            mask = mask & (coord <= depth_max_norm)
        subset = subset.where(mask, drop=True)

    if subset.size == 0 or any(size == 0 for size in subset.sizes.values()):
        raise ValueError("Selected analysis window contains no MVBS bins.")
    return subset


def compute_mvbs_mean_sv_analysis(
    ds_mvbs: xr.Dataset,
    channel_name: str | None,
    time_start: str | dt.datetime | np.datetime64 | None = None,
    time_end: str | dt.datetime | np.datetime64 | None = None,
    depth_min: float | None = None,
    depth_max: float | None = None,
    ping_time_bin: str | None = None,
    range_meter_bin: float | None = None,
) -> MVBSAnalysisResult:
    """Compute mean-Sv analysis products from an MVBS dataset."""
    sv_da, y_coord, selected_channel = build_plot_dataarray(ds_mvbs, channel_name)
    time_start_dt = _coerce_datetime_value(time_start)
    time_end_dt = _coerce_datetime_value(time_end)
    subset = subset_mvbs_for_analysis(
        sv_da=sv_da,
        y_coord=y_coord,
        time_start=time_start_dt,
        time_end=time_end_dt,
        depth_min=depth_min,
        depth_max=depth_max,
    )

    y_dim = _resolve_axis_dimension(subset, y_coord)
    if y_dim is None or y_dim not in subset.dims:
        raise ValueError(f"Unable to resolve analysis depth axis from '{y_coord}'.")

    mean_sv_by_depth_db = compute_linear_mean_sv_db(
        subset,
        dims=("ping_time",) if "ping_time" in subset.dims else tuple(),
    )
    mean_sv_by_time_db = compute_linear_mean_sv_db(subset, dims=(y_dim,))
    scalar_da = compute_linear_mean_sv_db(subset, dims=tuple(subset.dims))

    scalar_mean_sv_db: float | None = None
    scalar_values = np.asarray(scalar_da.values).ravel()
    if scalar_values.size > 0 and np.isfinite(scalar_values[0]):
        scalar_mean_sv_db = float(scalar_values[0])

    valid_cell_count = int(subset.notnull().sum().values.item())
    if valid_cell_count == 0:
        raise ValueError("Selected analysis window has no valid Sv bins (all NaN).")

    total_cell_count = int(np.prod(list(subset.sizes.values()), dtype=np.int64))
    ping_time_count = int(subset.sizes.get("ping_time", 0))
    depth_bin_count = int(subset.sizes.get(y_dim, 0))

    actual_time_start, actual_time_end = (
        _coordinate_time_bounds(subset["ping_time"])
        if "ping_time" in subset.coords
        else (None, None)
    )
    actual_depth_min, actual_depth_max = _coordinate_float_bounds(subset[y_coord])
    y_label = "Depth (m)" if y_coord == "echo_range" else f"{y_coord} index"

    return MVBSAnalysisResult(
        selected_channel=selected_channel,
        y_coord=y_coord,
        y_label=y_label,
        requested_time_start=_datetime_to_iso_text(time_start_dt),
        requested_time_end=_datetime_to_iso_text(time_end_dt),
        requested_depth_min=depth_min,
        requested_depth_max=depth_max,
        actual_time_start=actual_time_start,
        actual_time_end=actual_time_end,
        actual_depth_min=actual_depth_min,
        actual_depth_max=actual_depth_max,
        ping_time_count=ping_time_count,
        depth_bin_count=depth_bin_count,
        total_cell_count=total_cell_count,
        valid_cell_count=valid_cell_count,
        scalar_mean_sv_db=scalar_mean_sv_db,
        ping_time_bin=ping_time_bin,
        range_meter_bin=range_meter_bin,
        mean_sv_by_depth_db=mean_sv_by_depth_db,
        mean_sv_by_time_db=mean_sv_by_time_db,
    )


def format_analysis_summary_markdown(
    result: MVBSAnalysisResult,
    analysis_mode: str,
) -> str:
    """Build markdown summary text for analysis results."""
    scalar_text = (
        f"{result.scalar_mean_sv_db:.2f} dB"
        if result.scalar_mean_sv_db is not None and np.isfinite(result.scalar_mean_sv_db)
        else "N/A"
    )
    mode_label = ANALYSIS_MODE_LABELS.get(analysis_mode, analysis_mode)
    return (
        "### Analysis Summary\n"
        f"- Mode: `{mode_label}`\n"
        f"- Channel: `{result.selected_channel}`\n"
        f"- Time window requested: `{result.requested_time_start or '-inf'} -> {result.requested_time_end or '+inf'}`\n"
        f"- Time window used: `{result.actual_time_start or 'n/a'} -> {result.actual_time_end or 'n/a'}`\n"
        f"- Depth window requested: `{result.requested_depth_min if result.requested_depth_min is not None else '-inf'} -> {result.requested_depth_max if result.requested_depth_max is not None else '+inf'}`\n"
        f"- Depth window used: `{result.actual_depth_min if result.actual_depth_min is not None else 'n/a'} -> {result.actual_depth_max if result.actual_depth_max is not None else 'n/a'}`\n"
        f"- Valid bins: `{result.valid_cell_count}/{result.total_cell_count}`\n"
        f"- Scalar mean Sv: `{scalar_text}`\n"
        f"- MVBS bins: `range={result.range_meter_bin if result.range_meter_bin is not None else 'n/a'} m, ping_time={result.ping_time_bin or 'n/a'}`"
    )


def append_analysis_record_jsonl(output_path: Path, record: dict[str, object]) -> Path:
    """Append one analysis record to a JSONL output file and return resolved path."""
    resolved = output_path.resolve()
    if resolved.suffix.lower() != ".jsonl":
        resolved = resolved.with_suffix(".jsonl")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\n")
    return resolved


def analysis_summary_lines(
    result: MVBSAnalysisResult,
    analysis_mode: str,
) -> list[str]:
    """Build plain-text lines for stdout/log reporting."""
    scalar_text = (
        f"{result.scalar_mean_sv_db:.2f} dB"
        if result.scalar_mean_sv_db is not None and np.isfinite(result.scalar_mean_sv_db)
        else "N/A"
    )
    return [
        f"Mode: {ANALYSIS_MODE_LABELS.get(analysis_mode, analysis_mode)}",
        f"Channel: {result.selected_channel}",
        (
            "Time window (requested -> used): "
            f"{result.requested_time_start or '-inf'} -> {result.requested_time_end or '+inf'} | "
            f"{result.actual_time_start or 'n/a'} -> {result.actual_time_end or 'n/a'}"
        ),
        (
            "Depth window (requested -> used): "
            f"{result.requested_depth_min if result.requested_depth_min is not None else '-inf'} "
            f"-> {result.requested_depth_max if result.requested_depth_max is not None else '+inf'} | "
            f"{result.actual_depth_min if result.actual_depth_min is not None else 'n/a'} "
            f"-> {result.actual_depth_max if result.actual_depth_max is not None else 'n/a'}"
        ),
        f"Valid bins: {result.valid_cell_count}/{result.total_cell_count}",
        f"Scalar mean Sv: {scalar_text}",
        (
            "MVBS bins: "
            f"range={result.range_meter_bin if result.range_meter_bin is not None else 'n/a'} m, "
            f"ping_time={result.ping_time_bin or 'n/a'}"
        ),
    ]


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
        Directory that contains raw and related output files.
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


def _resolve_data_output_dir(
    html_path: Path,
    data_output_dir: Path | None,
) -> Path:
    """Resolve MVBS data output directory for export files."""
    if data_output_dir is None:
        output_dir = html_path.parent
    else:
        output_dir = data_output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _data_output_base_path(
    html_path: Path,
    data_output_dir: Path | None,
) -> Path:
    """Return base path stem used for MVBS data output files."""
    output_dir = _resolve_data_output_dir(html_path, data_output_dir)
    return output_dir / html_path.stem


def _build_data_output_dataset(
    sv_da: xr.DataArray,
    selected_channel: str,
    x_coord: str,
    y_coord: str,
    x_axis_note: str,
    hide_na_gaps: bool,
    flip_vertical: bool,
    transducer_facing: str,
    ping_time_bin: str,
    range_meter_bin: float,
) -> xr.Dataset:
    """Build a metadata-rich MVBS output dataset from plotted Sv data."""
    plot_ds = sv_da.to_dataset(name="Sv")
    plot_ds.attrs.update(
        {
            "selected_channel": selected_channel,
            "x_coord": x_coord,
            "y_coord": y_coord,
            "x_axis_note": x_axis_note,
            "hide_na_gaps": "true" if hide_na_gaps else "false",
            "flip_vertical": "true" if flip_vertical else "false",
            "transducer_facing": transducer_facing,
            "mvbs_ping_time_bin": ping_time_bin,
            "mvbs_range_meter_bin": float(range_meter_bin),
            "exported_at": dt.datetime.now().replace(microsecond=0).isoformat(sep=" "),
        }
    )
    plot_ds["Sv"].attrs.setdefault("long_name", "Mean Volume Backscattering Strength")
    plot_ds["Sv"].attrs.setdefault("units", "dB")
    return plot_ds


def _export_data_output_netcdf(
    plot_ds: xr.Dataset,
    output_path: Path,
) -> Path:
    """Write NetCDF MVBS output, preferring compression when available."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plot_ds.to_netcdf(
            output_path,
            encoding={"Sv": {"zlib": True, "complevel": 4}},
        )
    except Exception as exc:  # noqa: BLE001 - fallback when backend lacks compression support
        LOGGER.warning(
            "Compressed NetCDF output export failed for %s (%s). "
            "Retrying without compression.",
            output_path.name,
            exc,
        )
        plot_ds.to_netcdf(output_path)
    return output_path


def _export_data_output_csv(
    plot_ds: xr.Dataset,
    output_path: Path,
    csv_compression: str,
) -> Path:
    """Write CSV MVBS output from plotted Sv data."""
    csv_output_path = output_path
    if csv_compression == "gzip" and not csv_output_path.name.endswith(".gz"):
        csv_output_path = Path(f"{csv_output_path}.gz")

    dataframe = plot_ds["Sv"].to_dataframe().reset_index()
    compression_arg = "gzip" if csv_compression == "gzip" else None
    dataframe.to_csv(csv_output_path, index=False, compression=compression_arg)
    return csv_output_path


def export_data_outputs(
    sv_da: xr.DataArray,
    selected_channel: str,
    x_coord: str,
    y_coord: str,
    x_axis_note: str,
    hide_na_gaps: bool,
    flip_vertical: bool,
    transducer_facing: str,
    ping_time_bin: str,
    range_meter_bin: float,
    html_path: Path,
    export_mode: str,
    data_output_dir: Path | None,
    csv_compression: str,
) -> list[Path]:
    """Export plotted MVBS data outputs (NetCDF/CSV) aligned with saved HTML/base path."""
    if export_mode == "none":
        return []
    if export_mode not in PLOT_DATA_EXPORT_CHOICES:
        raise ValueError(
            f"Unsupported export mode '{export_mode}'. "
            f"Expected one of: {PLOT_DATA_EXPORT_CHOICES}"
        )
    if csv_compression not in PLOT_DATA_CSV_COMPRESSION_CHOICES:
        raise ValueError(
            f"Unsupported CSV compression '{csv_compression}'. "
            f"Expected one of: {PLOT_DATA_CSV_COMPRESSION_CHOICES}"
        )

    plot_ds = _build_data_output_dataset(
        sv_da=sv_da,
        selected_channel=selected_channel,
        x_coord=x_coord,
        y_coord=y_coord,
        x_axis_note=x_axis_note,
        hide_na_gaps=hide_na_gaps,
        flip_vertical=flip_vertical,
        transducer_facing=transducer_facing,
        ping_time_bin=ping_time_bin,
        range_meter_bin=range_meter_bin,
    )
    base_path = _data_output_base_path(html_path=html_path, data_output_dir=data_output_dir)
    saved_paths: list[Path] = []

    if export_mode in {"netcdf", "both"}:
        netcdf_path = Path(f"{base_path}.mvbs.nc")
        saved_paths.append(
            _export_data_output_netcdf(
                plot_ds=plot_ds,
                output_path=netcdf_path,
            )
        )
    if export_mode in {"csv", "both"}:
        csv_path = Path(f"{base_path}.mvbs.csv")
        saved_paths.append(
            _export_data_output_csv(
                plot_ds=plot_ds,
                output_path=csv_path,
                csv_compression=csv_compression,
            )
        )
    return saved_paths




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
    if "channel" in ds_mvbs.dims:
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
    else:
        sv_da = ds_mvbs["Sv"]
        inferred_channel: str | None = None
        if "channel" in ds_mvbs.coords:
            try:
                inferred_channel = str(np.asarray(ds_mvbs["channel"].values).item())
            except (TypeError, ValueError):
                inferred_channel = None
        if inferred_channel is None:
            selected_attr = ds_mvbs.attrs.get("selected_channel")
            if selected_attr:
                inferred_channel = str(selected_attr)

        if channel_name is None:
            channel_name = inferred_channel or "unknown_channel"
        elif inferred_channel is not None and channel_name != inferred_channel:
            raise ValueError(
                "Requested channel does not match this MVBS NetCDF output.\n"
                f"Requested: {channel_name}\n"
                f"Output channel: {inferred_channel}"
            )

    if "echo_range" in sv_da.coords:
        echo_range = sv_da["echo_range"]
        if "ping_time" in echo_range.dims:
            candidate = echo_range
            non_time_dims = [dim for dim in sv_da.dims if dim != "ping_time"]
            if non_time_dims:
                valid_ping_mask = sv_da.notnull().any(dim=non_time_dims)
                if int(valid_ping_mask.sum().values.item()) > 0:
                    candidate = candidate.sel(ping_time=valid_ping_mask)
            if "ping_time" in candidate.dims:
                range_non_time_dims = [dim for dim in candidate.dims if dim != "ping_time"]
                if range_non_time_dims:
                    valid_range_ping = candidate.notnull().any(dim=range_non_time_dims)
                    if int(valid_range_ping.sum().values.item()) > 0:
                        first_valid_idx = int(
                            valid_range_ping.argmax(dim="ping_time").values.item()
                        )
                        candidate = candidate.isel(ping_time=first_valid_idx, drop=True)
                    else:
                        candidate = candidate.isel(ping_time=0, drop=True)
                else:
                    candidate = candidate.isel(ping_time=0, drop=True)
            echo_range = candidate

        if int(echo_range.notnull().sum().values.item()) > 1:
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
    valid_count = int(valid_ping_mask.sum().values.item())
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
        valid_time_count = int(valid_time_mask.sum().values.item())
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
    return int(coord.notnull().sum().values.item())


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


def _build_collapsed_time_tick_indices(
    epoch_ms_values: Sequence[int], target_tick_count: int = 12
) -> list[int]:
    """Build readable tick positions while forcing day-transition labels."""
    total_count = len(epoch_ms_values)
    if total_count == 0:
        return []

    step = max(1, total_count // max(target_tick_count, 1))
    min_spacing = max(1, step // 2)
    regular_ticks: list[int] = list(range(0, total_count, step))
    if regular_ticks[-1] != total_count - 1:
        regular_ticks.append(total_count - 1)

    transition_ticks: set[int] = set()
    previous_day: dt.date | None = None
    for idx, millis in enumerate(epoch_ms_values):
        if not np.isfinite(millis) or millis < 0:
            continue
        # Use UTC day boundaries so hide-gap labels match dataset timestamps.
        current_day = dt.datetime.utcfromtimestamp(millis / 1000.0).date()
        if previous_day is None or current_day != previous_day:
            transition_ticks.add(idx)
            previous_day = current_day

    selected: list[int] = sorted({0, total_count - 1, *transition_ticks})
    selected_set = set(selected)
    for idx in regular_ticks:
        if idx in selected_set:
            continue
        if any(abs(idx - anchor) < min_spacing for anchor in selected):
            continue
        selected.append(idx)
        selected_set.add(idx)

    return sorted(selected)


def _build_collapsed_time_hook(
    epoch_ms_values: list[int],
):
    """Create a Holoviews hook with dynamic dense-axis datetime styling."""
    tick_indices = _build_collapsed_time_tick_indices(epoch_ms_values)

    hover_formatter = CustomJSHover(
        args={"epoch_ms_values": epoch_ms_values},
        code="""
            const values = epoch_ms_values || [];
            const hoverX =
                (typeof special_vars !== "undefined" && Number.isFinite(special_vars?.x))
                    ? special_vars.x
                    : value;
            if (!values.length || !Number.isFinite(hoverX)) {
                return "NaT";
            }
            const idx = Math.round(hoverX);
            if (idx < 0 || idx >= values.length) {
                return "NaT";
            }
            const millis = values[idx];
            if (!Number.isFinite(millis) || millis < 0) {
                return "NaT";
            }
            const current = new Date(millis);
            if (!Number.isFinite(current.getTime())) {
                return "NaT";
            }
            const pad2 = (num) => String(num).padStart(2, "0");
            const yyyy = current.getUTCFullYear();
            const mm = pad2(current.getUTCMonth() + 1);
            const dd = pad2(current.getUTCDate());
            const hh = pad2(current.getUTCHours());
            const min = pad2(current.getUTCMinutes());
            const ss = pad2(current.getUTCSeconds());
            return `${yyyy}-${mm}-${dd} ${hh}:${min}:${ss}`;
        """,
    )
    y_hover_formatter = CustomJSHover(
        code="""
            const hoverY =
                (typeof special_vars !== "undefined" && Number.isFinite(special_vars?.y))
                    ? special_vars.y
                    : value;
            if (!Number.isFinite(hoverY)) {
                return "n/a";
            }
            return Number(hoverY).toFixed(2);
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
            if (!Number.isFinite(current.getTime())) {
                return "NaT";
            }
            const pad2 = (num) => String(num).padStart(2, "0");
            const timeText = `${pad2(current.getUTCHours())}:${pad2(current.getUTCMinutes())}`;

            let showDate = false;
            if (idx > 0) {
                const prevMillis = values[idx - 1];
                if (Number.isFinite(prevMillis) && prevMillis >= 0) {
                    const prev = new Date(prevMillis);
                    showDate =
                        current.getUTCFullYear() !== prev.getUTCFullYear() ||
                        current.getUTCMonth() !== prev.getUTCMonth() ||
                        current.getUTCDate() !== prev.getUTCDate();
                }
            }
            if (!showDate) {
                return timeText;
            }
            const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
            const monthIdx = current.getUTCMonth();
            const monthText = monthNames[monthIdx] ?? "???";
            const dateText = `${monthText} ${pad2(current.getUTCDate())}`;
            return `${timeText}\\n${dateText}`;
        """,
    )

    def _hook(plot, _element) -> None:
        hover_tools = list(plot.state.select({"type": HoverTool}))
        for hover_tool in hover_tools:
            existing_tooltips = list(hover_tool.tooltips or [])
            trailing_tooltips = existing_tooltips[1:] if existing_tooltips else [("value", "@image")]
            rewritten_tooltips: list[tuple[str, str]] = [("ping_time", "$x{custom}")]
            for label, field in trailing_tooltips:
                if isinstance(field, str) and field.strip() == "$y":
                    rewritten_tooltips.append((label, "$y{custom}"))
                else:
                    rewritten_tooltips.append((label, field))
            hover_tool.tooltips = rewritten_tooltips
            updated_formatters = dict(hover_tool.formatters)
            updated_formatters["$x"] = hover_formatter
            updated_formatters["$y"] = y_hover_formatter
            hover_tool.formatters = updated_formatters

        x_axes = getattr(plot.state, "xaxis", [])
        if not x_axes:
            return
        axis = x_axes[0]
        if tick_indices:
            axis.ticker = FixedTicker(ticks=tick_indices)
        axis.formatter = tick_formatter
        axis.major_label_orientation = 0.0
        axis.major_label_text_align = "center"
        axis.major_label_text_baseline = "top"
        axis.major_label_standoff = 10

        x_range = getattr(plot.state, "x_range", None)
        if x_range is not None and hasattr(x_range, "start") and hasattr(x_range, "end"):
            try:
                start = float(x_range.start)
                end = float(x_range.end)
                if np.isfinite(start) and np.isfinite(end) and end > start:
                    pad = max((end - start) * 0.01, 1.0)
                    x_range.start = start - pad
                    x_range.end = end + pad
            except (TypeError, ValueError):
                pass

        current_left = getattr(plot.state, "min_border_left", 0) or 0
        plot.state.min_border_left = max(current_left, 74)
        current_bottom = getattr(plot.state, "min_border_bottom", 0) or 0
        plot.state.min_border_bottom = max(current_bottom, 68)

    return _hook


def _build_dark_theme_hook():
    """Create a Holoviews hook that applies dark Bokeh styling live."""

    def _hook(plot, _element) -> None:
        state = getattr(plot, "state", None)
        if state is None:
            return
        apply_bokeh_plot_theme(state, plot_theme="dark")

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
    transducer_facing: str,
    flip_vertical: bool,
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
    transducer_facing : str
        Effective transducer facing mode ("down" or "up").
    flip_vertical : bool
        If True, flip the rendered y-axis orientation from its default.

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
    down_looking = transducer_facing != "up"
    invert_yaxis = down_looking
    if flip_vertical:
        invert_yaxis = not invert_yaxis
    if y_coord_safe == "echo_range":
        ylabel = "Depth (m)" if down_looking else "Range Above Transducer (m)"
    else:
        ylabel = y_coord_safe
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
        hvplot_kwargs["min_height"] = max(280, min(height, 600))
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
            (
                "Depth Index (fallback)"
                if down_looking
                else "Range Index (fallback)"
            )
            if y_coord_safe == "echo_range"
            else f"{y_coord_safe} Index (fallback)"
        )
        plot = fallback_da.hvplot(
            **hvplot_kwargs,
        )
        tick_hook = None
    opt_kwargs = {
        "invert_yaxis": invert_yaxis,
        "active_tools": ["wheel_zoom"],
        "fontsize": {"title": "12pt", "labels": "10pt", "xticks": "9pt", "yticks": "9pt"},
    }
    hooks = []
    if tick_hook is not None:
        hooks.append(tick_hook)
    if plot_theme == "dark":
        hooks.append(_build_dark_theme_hook())
    if hooks:
        opt_kwargs["hooks"] = hooks
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
    transducer_facing: str,
    flip_vertical: bool,
    ping_time_bin: str,
    range_meter_bin: float,
    data_output_format: str,
    data_output_dir: Path | None,
    data_csv_compression: str,
):
    """Create a Panel layout for interactive echogram parameter tuning."""
    import panel as pn

    is_dark_theme = plot_theme == "dark"
    dark_text = "#f3f4f6"
    dark_surface = "#111827"
    dark_style_vars = (
        {
            "--design-background-color": DARK_BG,
            "--design-background-text-color": dark_text,
            "--design-surface-color": dark_surface,
            "--design-surface-text-color": dark_text,
            "--panel-background-color": DARK_BG,
            "--panel-on-background-color": dark_text,
            "--panel-surface-color": dark_surface,
            "--panel-on-surface-color": dark_text,
            "--text-color": dark_text,
            "background": DARK_BG,
            "color": dark_text,
        }
        if is_dark_theme
        else {}
    )
    dark_surface_vars = (
        {
            **dark_style_vars,
            "background": dark_surface,
            "color": dark_text,
        }
        if is_dark_theme
        else {}
    )
    accordion_dark_stylesheet = (
        """
:host {
  --design-background-color: #000000;
  --design-background-text-color: #f3f4f6;
  --design-surface-color: #111827;
  --design-surface-text-color: #f3f4f6;
  --text-color: #f3f4f6;
  color: #f3f4f6;
}
.bk-header,
.bk-accordion-header,
.accordion-header {
  background-color: #111827 !important;
  color: #f3f4f6 !important;
}
"""
        if is_dark_theme
        else None
    )
    widget_dark_stylesheet = (
        """
:host {
  --design-background-color: #111827;
  --design-background-text-color: #f3f4f6;
  --design-surface-color: #111827;
  --design-surface-text-color: #f3f4f6;
  --text-color: #f3f4f6;
  color: #f3f4f6;
}
label, span, p {
  color: #f3f4f6 !important;
}
"""
        if is_dark_theme
        else None
    )

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

    if is_dark_theme:
        widget_surface_styles = {
            "--text-color": dark_text,
            "--design-background-text-color": dark_text,
            "--design-surface-text-color": dark_text,
            "background": dark_surface,
            "color": dark_text,
        }
        for widget in (
            channel_select,
            cmap_select,
            vmin_slider,
            vmax_slider,
            export_path_input,
            export_button,
        ):
            widget.styles = widget_surface_styles
            widget.stylesheets = [widget_dark_stylesheet]

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
            transducer_facing=transducer_facing,
            flip_vertical=flip_vertical,
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
            f"- Transducer facing: `{transducer_facing}`\n"
            f"- Vertical flip: `{'enabled' if flip_vertical else 'disabled'}`\n"
            f"- MVBS data output format: `{data_output_format}`\n"
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

            selected_channel = label_to_channel[channel_select.value]
            plot_obj, _, title, _, _, _ = build_plot_for_settings(
                channel_select.value,
                cmap_select.value,
                vmin_slider.value,
                vmax_slider.value,
            )
            bokeh_plot = hv.render(plot_obj, backend="bokeh")
            saved_html_path = save_bokeh_plot_html(
                bokeh_plot,
                export_path,
                title=title,
                plot_theme=plot_theme,
                html_resources=html_resources,
            )

            export_sv_da, export_y_coord, _ = build_plot_dataarray(ds_mvbs, selected_channel)
            export_sv_da, export_x_coord, _, export_removed = prepare_display_dataarray(
                export_sv_da,
                hide_na_gaps=hide_na_gaps,
            )
            export_x_axis_note = describe_x_axis_mode(
                x_coord=export_x_coord,
                hide_na_requested=hide_na_gaps,
                removed_count=export_removed,
            )
            data_output_paths = export_data_outputs(
                sv_da=export_sv_da,
                selected_channel=selected_channel,
                x_coord=export_x_coord,
                y_coord=export_y_coord,
                x_axis_note=export_x_axis_note,
                hide_na_gaps=hide_na_gaps,
                flip_vertical=flip_vertical,
                transducer_facing=transducer_facing,
                ping_time_bin=ping_time_bin,
                range_meter_bin=range_meter_bin,
                html_path=saved_html_path,
                export_mode=data_output_format,
                data_output_dir=data_output_dir,
                csv_compression=data_csv_compression,
            )

            status_lines = [f"Saved current view to `{saved_html_path}`"]
            if data_output_paths:
                status_lines.append("Saved MVBS data outputs:")
                status_lines.extend(f"- `{path}`" for path in data_output_paths)
            export_status.object = "\n".join(status_lines)
            LOGGER.info("Saved Panel-exported HTML snapshot to %s", saved_html_path)
            for output_path in data_output_paths:
                LOGGER.info("Saved Panel MVBS data output to %s", output_path)
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
        sizing_mode="stretch_width",
        styles={**dark_surface_vars, "padding": "8px"} if is_dark_theme else None,
        stylesheets=[widget_dark_stylesheet] if is_dark_theme else None,
    )
    details = pn.pane.Markdown(
        status_text,
        sizing_mode="stretch_width",
        styles={**dark_surface_vars, "padding": "8px"} if is_dark_theme else None,
        stylesheets=[widget_dark_stylesheet] if is_dark_theme else None,
    )
    controls_menu = pn.Accordion(
        ("Controls", controls),
        active=[],
        sizing_mode="stretch_width",
        styles=dark_style_vars if is_dark_theme else None,
        stylesheets=[accordion_dark_stylesheet] if is_dark_theme else None,
    )
    details_menu = pn.Accordion(
        ("Display Settings", details),
        active=[],
        sizing_mode="stretch_width",
        styles=dark_style_vars if is_dark_theme else None,
        stylesheets=[accordion_dark_stylesheet] if is_dark_theme else None,
    )
    top_menus = pn.Row(
        pn.Column(controls_menu, sizing_mode="stretch_width"),
        pn.Column(details_menu, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
        styles=dark_style_vars if is_dark_theme else None,
    )
    layout = pn.Column(
        top_menus,
        plot_view,
        sizing_mode="stretch_both",
        styles=dark_style_vars if is_dark_theme else None,
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Display-only option: drop all-NaN ping bins and plot a dense x-axis "
            "to hide duty-cycle time gaps while retaining datetime tick labels. "
            "Enabled by default; use --no-hide-na-gaps to disable."
        ),
    )
    parser.add_argument(
        "--transducer-facing",
        type=str,
        choices=["auto", "down", "up"],
        default="auto",
        help=(
            "Vertical orientation mode for echogram rendering. "
            "'auto' tries per-day metadata inference and falls back to down-looking."
        ),
    )
    parser.add_argument(
        "--flip-vertical",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Flip rendered echogram vertically after orientation is resolved "
            "(applies to static exports and Panel views). "
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
        "--skip-html",
        action="store_true",
        help=(
            "In static mode, skip HTML writing and export only MVBS data outputs "
            "configured by --data-output-format. "
            "The --save-html path is still used as the output filename base."
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
            "In static mode, export one output set per detected channel using "
            "--save-html as the filename base."
        ),
    )
    parser.add_argument(
        "--data-output-format",
        dest="data_output_format",
        type=str,
        choices=PLOT_DATA_EXPORT_CHOICES,
        default="netcdf",
        help=(
            "Export plot-consistent MVBS data outputs in static/panel export flows. "
            "'netcdf' (default) writes .mvbs.nc, 'csv' writes .mvbs.csv(.gz), "
            "'both' writes both, and 'none' disables data export."
        ),
    )
    parser.add_argument(
        "--data-output-dir",
        dest="data_output_dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory for MVBS data outputs. "
            "Defaults to the same directory as each saved HTML (or base output path in --skip-html mode)."
        ),
    )
    parser.add_argument(
        "--data-csv-compression",
        dest="data_csv_compression",
        type=str,
        choices=PLOT_DATA_CSV_COMPRESSION_CHOICES,
        default="gzip",
        help="Compression mode for CSV outputs when --data-output-format includes csv.",
    )
    return parser.parse_args()


def main() -> None:
    """Run chunked EK80 processing and open interactive echogram."""
    args = parse_args()
    configure_logging(args.log_level)
    if args.ui_mode == "panel" and args.skip_html:
        LOGGER.warning("--skip-html applies to static mode only; ignoring in panel mode.")
    if args.ui_mode != "panel" and args.skip_html and args.data_output_format == "none":
        raise ValueError(
            "--skip-html requires --data-output-format netcdf/csv/both in static mode "
            "(otherwise no output files would be produced)."
        )
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
    daily_transducer_reports = collect_daily_transducer_reports(raw_files)
    log_daily_transducer_reports(daily_transducer_reports)
    transducer_facing, transducer_facing_source = resolve_transducer_facing(
        requested_facing=args.transducer_facing,
        reports=daily_transducer_reports,
    )
    LOGGER.info(
        "Plot transducer facing resolved to '%s' (source=%s).",
        transducer_facing,
        transducer_facing_source,
    )

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
    print(
        "Transducer facing for plotting: "
        f"{transducer_facing} ({transducer_facing_source})"
    )

    plot_vmin, plot_vmax = normalize_color_limits(args.vmin, args.vmax)
    plot_title = f"EK80 MVBS Echogram | {selected_channel}"
    if selected_freq is not None:
        plot_title = f"{plot_title} ({selected_freq} kHz)"

    bokeh_plot = None
    if args.ui_mode != "panel" and not args.export_all_channels and not args.skip_html:
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
            transducer_facing=transducer_facing,
            flip_vertical=args.flip_vertical,
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
            saved_data_output_paths: list[Path] = []
            for channel_name in channels:
                channel_da, channel_y, _ = build_plot_dataarray(ds_mvbs, channel_name)
                channel_da, channel_x, channel_xlabel, channel_removed = prepare_display_dataarray(
                    channel_da,
                    hide_na_gaps=args.hide_na_gaps,
                )
                channel_x_axis_note = describe_x_axis_mode(
                    x_coord=channel_x,
                    hide_na_requested=args.hide_na_gaps,
                    removed_count=channel_removed,
                )
                if args.hide_na_gaps:
                    LOGGER.info(
                        "Display x-axis mode for %s: %s",
                        channel_name,
                        channel_x_axis_note,
                    )
                channel_title = f"EK80 MVBS Echogram | {channel_name}"
                channel_freq = infer_channel_frequency_khz(channel_name)
                if channel_freq is not None:
                    channel_title = f"{channel_title} ({channel_freq} kHz)"
                channel_path = base_output_path.with_name(
                    f"{base_output_path.stem}__{channel_slug(channel_name)}{base_output_path.suffix}"
                )
                data_output_base_path = channel_path
                if not args.skip_html:
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
                        transducer_facing=transducer_facing,
                        flip_vertical=args.flip_vertical,
                    )
                    channel_bokeh = hv.render(channel_plot, backend="bokeh")
                    saved_path = save_bokeh_plot_html(
                        channel_bokeh,
                        channel_path,
                        channel_title,
                        plot_theme=args.plot_theme,
                        html_resources=args.html_resources,
                    )
                    saved_paths.append(saved_path)
                    data_output_base_path = saved_path
                data_output_paths = export_data_outputs(
                    sv_da=channel_da,
                    selected_channel=channel_name,
                    x_coord=channel_x,
                    y_coord=channel_y,
                    x_axis_note=channel_x_axis_note,
                    hide_na_gaps=args.hide_na_gaps,
                    flip_vertical=args.flip_vertical,
                    transducer_facing=transducer_facing,
                    ping_time_bin=args.ping_time_bin,
                    range_meter_bin=args.range_meter_bin,
                    html_path=data_output_base_path,
                    export_mode=args.data_output_format,
                    data_output_dir=args.data_output_dir,
                    csv_compression=args.data_csv_compression,
                )
                saved_data_output_paths.extend(data_output_paths)
            if saved_paths:
                print("Saved HTML snapshots by channel:")
                for path in saved_paths:
                    print(f" - {path}")
            elif args.skip_html:
                print("Skipped HTML snapshot export by channel (--skip-html).")
            if saved_data_output_paths:
                print("Saved MVBS data outputs:")
                for output_path in saved_data_output_paths:
                    print(f" - {output_path}")
        else:
            base_output_path = resolve_html_output_path(
                args.save_html or Path("ek80_chunked_echogram.html")
            )
            saved_path: Path | None = None
            if args.save_html is not None and not args.skip_html:
                if bokeh_plot is None:
                    raise RuntimeError("Expected Bokeh plot for HTML export, but none was generated.")
                saved_path = save_bokeh_plot_html(
                    bokeh_plot,
                    args.save_html,
                    plot_title,
                    plot_theme=args.plot_theme,
                    html_resources=args.html_resources,
                )
                print(f"Saved HTML snapshot: {saved_path}")
            elif args.skip_html and args.save_html is not None:
                print("Skipped HTML snapshot export (--skip-html).")

            data_output_paths: list[Path] = []
            if args.data_output_format != "none" and (args.skip_html or args.save_html is not None):
                data_output_paths = export_data_outputs(
                    sv_da=sv_da,
                    selected_channel=selected_channel,
                    x_coord=x_coord,
                    y_coord=y_coord,
                    x_axis_note=x_axis_note,
                    hide_na_gaps=args.hide_na_gaps,
                    flip_vertical=args.flip_vertical,
                    transducer_facing=transducer_facing,
                    ping_time_bin=args.ping_time_bin,
                    range_meter_bin=args.range_meter_bin,
                    html_path=saved_path or base_output_path,
                    export_mode=args.data_output_format,
                    data_output_dir=args.data_output_dir,
                    csv_compression=args.data_csv_compression,
                )
            if data_output_paths:
                print("Saved MVBS data outputs:")
                for output_path in data_output_paths:
                    print(f" - {output_path}")

    print("\nProcessing summary:")
    print(f"Total files processed: {len(raw_files)}")
    print(f"Final MVBS dataset shape: {dict(ds_mvbs.sizes)}")
    print(f"Time range: {str(ds_mvbs['ping_time'].min().values)} -> {str(ds_mvbs['ping_time'].max().values)}")
    print(f"Channel names available: {channels}")
    print(
        f"Transducer facing used: {transducer_facing} "
        f"(source: {transducer_facing_source})"
    )
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
            transducer_facing=transducer_facing,
            flip_vertical=args.flip_vertical,
            ping_time_bin=args.ping_time_bin,
            range_meter_bin=args.range_meter_bin,
            data_output_format=args.data_output_format,
            data_output_dir=args.data_output_dir,
            data_csv_compression=args.data_csv_compression,
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
    elif not args.skip_html and not args.export_all_channels and args.save_html is None and bokeh_plot is not None:
        show(bokeh_plot)


if __name__ == "__main__":
    main()
