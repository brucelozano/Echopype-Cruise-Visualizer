"""Post-process MVBS NetCDF outputs for mean-Sv analysis outputs.

This script analyzes `.mvbs.nc` files exported by `ek80_chunked_echogram.py`
without reprocessing raw `.raw` files.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr

from ek80_chunked_echogram import (
    ANALYSIS_MODE_CHOICES,
    ANALYSIS_MODE_LABELS,
    analysis_summary_lines,
    append_analysis_record_jsonl,
    compute_mvbs_mean_sv_analysis,
    parse_datetime_input,
)


def _path_from_env(var_name: str) -> Path | None:
    """Return Path from environment variable when set."""
    import os

    value = os.getenv(var_name)
    if not value:
        return None
    return Path(value).expanduser()


def _coerce_datetime_text(value: str | None) -> dt.datetime | None:
    """Parse optional datetime text using shared project formats."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return parse_datetime_input(text)


def _infer_selected_channel(ds: xr.Dataset) -> str:
    """Infer selected channel label from output metadata."""
    if "selected_channel" in ds.attrs and ds.attrs["selected_channel"]:
        return str(ds.attrs["selected_channel"])
    if "channel" in ds.coords:
        values = np.asarray(ds["channel"].values)
        if values.size == 1:
            return str(values.item())
    return "unknown_channel"


def _iter_output_files(input_dir: Path, glob_pattern: str) -> Iterable[Path]:
    """Yield MVBS output files matching pattern recursively."""
    yield from sorted(input_dir.rglob(glob_pattern))


def _coerce_optional_float(value: object) -> float | None:
    """Convert metadata values to float when possible."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze exported MVBS NetCDF outputs without rerunning raw processing."
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
        "--analysis-mode",
        type=str,
        choices=ANALYSIS_MODE_CHOICES,
        required=True,
        help="Analysis mode to run on each output file.",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default=None,
        help="Optional exact channel label to analyze.",
    )
    parser.add_argument(
        "--channel-filter",
        type=str,
        default=None,
        help="Optional substring filter for output channel labels (case-insensitive).",
    )
    parser.add_argument(
        "--analysis-time-start",
        type=str,
        default=None,
        help=(
            "Optional analysis window start datetime. "
            "Formats: YYYYMMDD, YYYY-MM-DD, YYYYMMDDTHHMMSS, YYYY-MM-DDTHH:MM:SS."
        ),
    )
    parser.add_argument(
        "--analysis-time-end",
        type=str,
        default=None,
        help="Optional analysis window end datetime (same formats as --analysis-time-start).",
    )
    parser.add_argument(
        "--analysis-depth-min",
        type=float,
        default=None,
        help="Optional minimum analysis depth/y-value.",
    )
    parser.add_argument(
        "--analysis-depth-max",
        type=float,
        default=None,
        help="Optional maximum analysis depth/y-value.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL path for machine-readable analysis records.",
    )
    parser.add_argument(
        "--profile-csv-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory for profile CSV exports when "
            "--analysis-mode is mean-vs-depth or mean-vs-time."
        ),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on number of matching outputs to analyze.",
    )
    return parser.parse_args()


def main() -> None:
    """Analyze MVBS NetCDF outputs with selected mode and bounds."""
    args = parse_args()
    if args.input_dir is None:
        raise ValueError(
            "Missing MVBS output directory. Pass --input-dir or set EK80_OUTPUT_DIR."
        )
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    start_dt = _coerce_datetime_text(args.analysis_time_start)
    end_dt = _coerce_datetime_text(args.analysis_time_end)
    if start_dt and end_dt and end_dt <= start_dt:
        raise ValueError(
            f"Invalid analysis time window: end ({end_dt}) must be after start ({start_dt})."
        )

    files = list(_iter_output_files(args.input_dir, args.glob_pattern))
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise RuntimeError(
            f"No outputs found in {args.input_dir} matching pattern '{args.glob_pattern}'."
        )

    if args.profile_csv_dir is not None and args.analysis_mode in {"mean-vs-depth", "mean-vs-time"}:
        args.profile_csv_dir.mkdir(parents=True, exist_ok=True)

    matched = 0
    succeeded = 0
    skipped = 0
    for output_path in files:
        with xr.open_dataset(output_path) as ds:
            selected_channel = _infer_selected_channel(ds)
            if args.channel is not None and selected_channel != args.channel:
                skipped += 1
                continue
            if args.channel_filter and args.channel_filter.lower() not in selected_channel.lower():
                skipped += 1
                continue
            matched += 1

            result = compute_mvbs_mean_sv_analysis(
                ds_mvbs=ds,
                channel_name=args.channel or selected_channel,
                time_start=start_dt,
                time_end=end_dt,
                depth_min=args.analysis_depth_min,
                depth_max=args.analysis_depth_max,
                ping_time_bin=str(ds.attrs.get("mvbs_ping_time_bin", "unknown")),
                range_meter_bin=_coerce_optional_float(ds.attrs.get("mvbs_range_meter_bin")),
            )

            print(f"\n{output_path.name}")
            for line in analysis_summary_lines(result=result, analysis_mode=args.analysis_mode):
                print(f" - {line}")

            record = result.to_record(args.analysis_mode)
            record.update(
                {
                    "analysis_mode_label": ANALYSIS_MODE_LABELS[args.analysis_mode],
                    "source_file": str(output_path.resolve()),
                    "source_name": output_path.name,
                    "generated_at": dt.datetime.now().replace(microsecond=0).isoformat(sep=" "),
                }
            )

            if args.profile_csv_dir is not None and args.analysis_mode in {"mean-vs-depth", "mean-vs-time"}:
                profile_da = (
                    result.mean_sv_by_depth_db
                    if args.analysis_mode == "mean-vs-depth"
                    else result.mean_sv_by_time_db
                )
                csv_name = f"{output_path.stem}.{args.analysis_mode}.csv"
                csv_path = args.profile_csv_dir / csv_name
                profile_da.to_dataframe(name="mean_sv_db").reset_index().to_csv(
                    csv_path,
                    index=False,
                )
                record["profile_csv"] = str(csv_path.resolve())
                print(f" - Profile CSV: {csv_path}")

            if args.output_jsonl is not None:
                output_path = append_analysis_record_jsonl(
                    output_path=args.output_jsonl,
                    record=record,
                )
                print(f" - JSONL record appended: {output_path}")

            succeeded += 1

    print("\nMVBS output analysis summary:")
    print(f" - Files scanned: {len(files)}")
    print(f" - Files matched filters: {matched}")
    print(f" - Files succeeded: {succeeded}")
    print(f" - Files skipped by filters: {skipped}")

    if matched == 0:
        raise RuntimeError("No outputs matched the provided channel filters.")


if __name__ == "__main__":
    main()
