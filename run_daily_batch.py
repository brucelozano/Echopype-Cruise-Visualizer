"""Batch runner for daily EK80 output exports.

This utility discovers unique date stamps in `.raw` filenames and runs
`ek80_chunked_echogram.py` once per day. Failures are isolated per day so
successful days remain completed even if one day fails.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from channel_naming import channel_slug


DATE_PATTERN = re.compile(r"D(?P<date>\d{8})-T\d{6}", flags=re.IGNORECASE)
DEFAULT_OUTPUT_PREFIX = "cruise"
OUTPUT_TYPE_CHOICES = ("netcdf", "both", "html")


@dataclass
class DayRunResult:
    """Container for one daily run result."""

    date_yyyymmdd: str
    command: list[str]
    output_path: str
    log_path: str
    status: str
    return_code: int
    elapsed_seconds: float


@dataclass
class ViewerExportResult:
    """Container for optional post-batch headless viewer export."""

    status: str
    command: list[str]
    export_dir: str | None
    log_path: str | None
    return_code: int
    elapsed_seconds: float
    message: str | None = None


def _path_from_env(var_name: str) -> Path | None:
    """Return a Path from an environment variable when set."""
    value = os.getenv(var_name)
    if not value:
        return None
    return Path(value).expanduser()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run ek80_chunked_echogram.py once per discovered date."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=_path_from_env("EK80_RAW_DIR"),
        help=(
            "Directory containing all EK80 .raw files. "
            "Required unless EK80_RAW_DIR is set."
        ),
    )
    parser.add_argument(
        "--script-path",
        type=Path,
        default=Path(__file__).with_name("ek80_chunked_echogram.py"),
        help="Path to ek80_chunked_echogram.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "daily",
        help="Directory for per-day outputs and logs.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help=(
            "Prefix used for exported daily output files. "
            "Defaults to prefix inferred from raw filenames before DYYYYMMDD-THHMMSS "
            "(falls back to 'cruise'). Case from raw filenames is preserved."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2,
        help="Chunk size passed through to ek80_chunked_echogram.py.",
    )
    parser.add_argument(
        "--range-meter-bin",
        type=float,
        default=10.0,
        help="MVBS range bin passed through to the main script.",
    )
    parser.add_argument(
        "--ping-time-bin",
        type=str,
        default="30s",
        help="MVBS ping time bin passed through to the main script.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Colormap passed through to the main script.",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=-90.0,
        help="Lower dB limit passed through to the main script.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=-55.0,
        help="Upper dB limit passed through to the main script.",
    )
    parser.add_argument(
        "--plot-theme",
        type=str,
        choices=["dark", "light"],
        default="dark",
        help="Plot theme passed through to the main script.",
    )
    parser.add_argument(
        "--plot-sizing",
        type=str,
        choices=["responsive", "fixed"],
        default="responsive",
        help="Plot sizing mode passed through to the main script.",
    )
    parser.add_argument(
        "--html-resources",
        type=str,
        choices=["inline", "cdn"],
        default="inline",
        help=(
            "Bokeh resource mode passed through to the main script. "
            "'inline' embeds JS/CSS for offline portability; 'cdn' keeps files smaller."
        ),
    )
    parser.add_argument(
        "--data-output-format",
        dest="data_output_format",
        type=str,
        choices=["none", "netcdf", "csv", "both"],
        default="netcdf",
        help=(
            "MVBS data output format passed through to the main script. "
            "'netcdf' (default) writes .mvbs.nc outputs."
        ),
    )
    parser.add_argument(
        "--output-type",
        dest="output_type",
        type=str,
        choices=OUTPUT_TYPE_CHOICES,
        default="netcdf",
        help=(
            "Primary output type to keep per day. "
            "'netcdf' (default) skips HTML and exports data outputs only, "
            "'both' keeps HTML + data outputs, and 'html' keeps HTML only."
        ),
    )
    parser.add_argument(
        "--data-output-dir",
        dest="data_output_dir",
        type=Path,
        default=None,
        help=(
            "Optional MVBS data output directory passed through to the main script. "
            "Defaults to each output directory."
        ),
    )
    parser.add_argument(
        "--data-csv-compression",
        dest="data_csv_compression",
        type=str,
        choices=["none", "gzip"],
        default="gzip",
        help=(
            "CSV compression mode passed through when "
            "--data-output-format includes csv."
        ),
    )
    parser.add_argument(
        "--hide-na-gaps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pass through display-only NA-gap collapsing to the main script "
            "(hides duty-cycle time gaps by plotting a dense index with datetime labels). "
            "Enabled by default; use --no-hide-na-gaps to disable. "
            "Use a dedicated --output-dir for this mode to avoid mixing with time-axis exports."
        ),
    )
    parser.add_argument(
        "--transducer-facing",
        type=str,
        choices=["auto", "down", "up"],
        default="auto",
        help=(
            "Vertical orientation mode passed through to the main script. "
            "'auto' uses per-day metadata checks with safe fallback to down-looking."
        ),
    )
    parser.add_argument(
        "--flip-vertical",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pass through vertical flip to the main script after orientation is resolved. "
            "Enabled by default; use --no-flip-vertical to disable."
        ),
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional lower date bound in YYYYMMDD.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional upper date bound in YYYYMMDD.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip dates whose target outputs already exist. "
            "When combined with --hide-na-gaps, use a separate --output-dir so mode variants do not collide."
        ),
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop batch immediately on first failed day.",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default=None,
        help=(
            "Optional exact channel name to export (for example, "
            "'EKA 288458-70 ES200-7CDK-Split'). "
            "If omitted, all detected channels are exported."
        ),
    )
    parser.add_argument(
        "--viewer-export-html",
        action="store_true",
        help=(
            "After daily processing completes, run `view_mvbs_outputs.py` in headless mode "
            "to generate ready-to-open HTML snapshots from `.mvbs.nc` outputs."
        ),
    )
    parser.add_argument(
        "--viewer-script-path",
        type=Path,
        default=Path(__file__).with_name("view_mvbs_outputs.py"),
        help="Path to view_mvbs_outputs.py used by --viewer-export-html.",
    )
    parser.add_argument(
        "--viewer-export-dir",
        type=Path,
        default=None,
        help=(
            "Directory for headless viewer HTML snapshots. "
            "Defaults to <output-dir>/viewer_html."
        ),
    )
    parser.add_argument(
        "--viewer-export-glob-pattern",
        type=str,
        default=None,
        help=(
            "Optional recursive glob pattern for selecting `.mvbs.nc` files in "
            "headless viewer export mode. Defaults to '<output-prefix>_*.mvbs.nc'."
        ),
    )
    parser.add_argument(
        "--viewer-export-index-name",
        type=str,
        default="index.html",
        help=(
            "Index filename passed through to headless viewer export "
            "(default: index.html)."
        ),
    )
    parser.add_argument(
        "--viewer-export-no-index",
        action="store_true",
        help="Disable index page generation in headless viewer export mode.",
    )
    return parser.parse_args()


def parse_yyyymmdd(value: str) -> datetime:
    """Parse YYYYMMDD date string to datetime."""
    return datetime.strptime(value, "%Y%m%d")


def discover_dates(raw_dir: Path) -> list[str]:
    """Discover unique YYYYMMDD date tokens from `.raw` filenames."""
    raw_files = sorted(raw_dir.glob("*.raw"))
    discovered: set[str] = set()
    for file_path in raw_files:
        match = DATE_PATTERN.search(file_path.name)
        if match:
            discovered.add(match.group("date"))
    return sorted(discovered)


def sanitize_output_prefix(value: str) -> str:
    """Normalize output prefix text into a safe filename token."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    if not cleaned:
        raise ValueError(
            f"Invalid output prefix '{value}'. Include at least one alphanumeric character."
        )
    return cleaned


def infer_output_prefix(raw_dir: Path, fallback: str = DEFAULT_OUTPUT_PREFIX) -> str:
    """Infer output filename prefix from raw filenames."""
    raw_files = sorted(raw_dir.glob("*.raw"))
    candidates: list[str] = []
    for file_path in raw_files:
        match = DATE_PATTERN.search(file_path.stem)
        if not match:
            continue
        raw_prefix = file_path.stem[: match.start()].rstrip("-_ .")
        if not raw_prefix:
            continue
        cleaned = re.sub(r"[^A-Za-z0-9]+", "_", raw_prefix).strip("_")
        if cleaned:
            candidates.append(cleaned)

    if not candidates:
        return sanitize_output_prefix(fallback)

    counts = Counter(candidates)
    most_common_count = counts.most_common(1)[0][1]
    for candidate in candidates:
        if counts[candidate] == most_common_count:
            return candidate
    return candidates[0]


def filter_dates(dates: list[str], start_date: str | None, end_date: str | None) -> list[str]:
    """Apply optional inclusive date bounds to discovered dates."""
    start_dt = parse_yyyymmdd(start_date) if start_date else None
    end_dt = parse_yyyymmdd(end_date) if end_date else None
    filtered: list[str] = []
    for date_text in dates:
        current = parse_yyyymmdd(date_text)
        if start_dt and current < start_dt:
            continue
        if end_dt and current > end_dt:
            continue
        filtered.append(date_text)
    return filtered


def _data_output_paths_for_channel(base_output_path: Path, export_mode: str) -> list[Path]:
    """Return expected MVBS data output paths for one channel base path."""
    base_text = str(base_output_path.with_suffix(""))
    expected: list[Path] = []
    if export_mode in {"netcdf", "both"}:
        expected.append(Path(f"{base_text}.mvbs.nc"))
    if export_mode in {"csv", "both"}:
        expected.append(Path(f"{base_text}.mvbs.csv"))
        expected.append(Path(f"{base_text}.mvbs.csv.gz"))
    return expected


def _data_output_patterns_for_date(output_prefix: str, date_text: str, export_mode: str) -> list[str]:
    """Return glob patterns for date-level data output existence checks."""
    patterns: list[str] = []
    if export_mode in {"netcdf", "both"}:
        patterns.append(f"{output_prefix}_{date_text}__*.mvbs.nc")
    if export_mode in {"csv", "both"}:
        patterns.append(f"{output_prefix}_{date_text}__*.mvbs.csv")
        patterns.append(f"{output_prefix}_{date_text}__*.mvbs.csv.gz")
    return patterns


def run_headless_viewer_export(
    args: argparse.Namespace,
    python_exe: str,
    logs_dir: Path,
) -> ViewerExportResult:
    """Run optional headless NetCDF-to-HTML export via view_mvbs_outputs.py."""
    input_dir = (args.data_output_dir or args.output_dir).resolve()
    export_dir = (args.viewer_export_dir or (args.output_dir / "viewer_html")).resolve()
    glob_pattern = args.viewer_export_glob_pattern or f"{args.output_prefix}_*.mvbs.nc"
    matching_outputs = sorted(input_dir.rglob(glob_pattern))
    log_path = logs_dir / "viewer_headless_export.log"
    command = [
        python_exe,
        str(args.viewer_script_path),
        "--input-dir",
        str(input_dir),
        "--glob-pattern",
        glob_pattern,
        "--export-html-dir",
        str(export_dir),
        "--vmin",
        str(args.vmin),
        "--vmax",
        str(args.vmax),
        "--cmap",
        args.cmap,
        "--plot-theme",
        args.plot_theme,
        "--plot-sizing",
        args.plot_sizing,
        "--html-resources",
        args.html_resources,
        "--transducer-facing",
        args.transducer_facing,
    ]
    if args.hide_na_gaps:
        command.append("--hide-na-gaps")
    if args.flip_vertical:
        command.append("--flip-vertical")
    if args.channel:
        command.extend(["--channel-filter", args.channel])
    if args.viewer_export_no_index:
        command.append("--export-html-no-index")
    else:
        command.extend(["--export-html-index-name", args.viewer_export_index_name])

    if not matching_outputs:
        return ViewerExportResult(
            status="skipped_no_netcdf",
            command=command,
            export_dir=str(export_dir),
            log_path=None,
            return_code=0,
            elapsed_seconds=0.0,
            message=(
                f"No `.mvbs.nc` files matched '{glob_pattern}' in {input_dir}; "
                "skipping headless viewer export."
            ),
        )

    start = datetime.now()
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = (datetime.now() - start).total_seconds()
    status = "success" if proc.returncode == 0 else "failed"
    message = (
        f"Headless viewer export completed for {len(matching_outputs)} matched NetCDF file(s)."
        if status == "success"
        else "Headless viewer export failed. See log for details."
    )
    return ViewerExportResult(
        status=status,
        command=command,
        export_dir=str(export_dir),
        log_path=str(log_path),
        return_code=proc.returncode,
        elapsed_seconds=elapsed,
        message=message,
    )


def run_one_day(
    date_text: str,
    args: argparse.Namespace,
    python_exe: str,
    logs_dir: Path,
) -> DayRunResult:
    """Run one day export and return structured result."""
    day_iso = parse_yyyymmdd(date_text).strftime("%Y-%m-%d")
    effective_export_mode = "none" if args.output_type == "html" else args.data_output_format
    html_base_path = args.output_dir / f"{args.output_prefix}_{date_text}.html"
    html_pattern = args.output_dir / f"{args.output_prefix}_{date_text}__*.html"
    selected_channel_path: Path | None = None
    log_path = logs_dir / f"{args.output_prefix}_{date_text}.log"

    command = [
        python_exe,
        str(args.script_path),
        "--raw-dir",
        str(args.raw_dir),
        "--start-datetime",
        day_iso,
        "--duration-days",
        "1",
        "--chunk-size",
        str(args.chunk_size),
        "--range-meter-bin",
        str(args.range_meter_bin),
        "--ping-time-bin",
        args.ping_time_bin,
        "--cmap",
        args.cmap,
        "--vmin",
        str(args.vmin),
        "--vmax",
        str(args.vmax),
        "--plot-theme",
        args.plot_theme,
        "--plot-sizing",
        args.plot_sizing,
        "--html-resources",
        args.html_resources,
        "--data-output-format",
        effective_export_mode,
        "--data-csv-compression",
        args.data_csv_compression,
        "--transducer-facing",
        args.transducer_facing,
        "--ui-mode",
        "static",
    ]
    if args.data_output_dir is not None:
        command.extend(["--data-output-dir", str(args.data_output_dir)])
    if args.output_type == "netcdf":
        command.append("--skip-html")

    if args.channel:
        selected_channel_path = args.output_dir / (
            f"{args.output_prefix}_{date_text}__{channel_slug(args.channel)}.html"
        )
        command.extend(
            [
                "--channel",
                args.channel,
                "--save-html",
                str(selected_channel_path),
            ]
        )
    else:
        command.extend(
            [
                "--save-html",
                str(html_base_path),
                "--export-all-channels",
            ]
        )

    if args.hide_na_gaps:
        command.append("--hide-na-gaps")
    if args.flip_vertical:
        command.append("--flip-vertical")

    output_reference = str(selected_channel_path or html_pattern)
    has_existing_outputs = False
    if args.output_type in {"html", "both"}:
        existing_exports = sorted(args.output_dir.glob(f"{args.output_prefix}_{date_text}__*.html"))
        has_existing_outputs = (
            (selected_channel_path is not None and selected_channel_path.exists())
            or (selected_channel_path is None and len(existing_exports) > 0)
        )
    elif args.output_type == "netcdf":
        data_output_dir = args.data_output_dir or args.output_dir
        if selected_channel_path is not None:
            data_output_candidates = [
                path if path.is_absolute() else data_output_dir / path.name
                for path in _data_output_paths_for_channel(selected_channel_path, effective_export_mode)
            ]
            has_existing_outputs = any(path.exists() for path in data_output_candidates)
            output_reference = ", ".join(str(path) for path in data_output_candidates)
        else:
            data_output_patterns = _data_output_patterns_for_date(
                output_prefix=args.output_prefix,
                date_text=date_text,
                export_mode=effective_export_mode,
            )
            matched_paths: list[Path] = []
            for pattern in data_output_patterns:
                matched_paths.extend(sorted(data_output_dir.glob(pattern)))
            has_existing_outputs = len(matched_paths) > 0
            output_reference = ", ".join(str(data_output_dir / pattern) for pattern in data_output_patterns)

    if args.skip_existing and has_existing_outputs:
        return DayRunResult(
            date_yyyymmdd=date_text,
            command=command,
            output_path=output_reference,
            log_path=str(log_path),
            status="skipped_existing",
            return_code=0,
            elapsed_seconds=0.0,
        )

    start = datetime.now()
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = (datetime.now() - start).total_seconds()
    status = "success" if proc.returncode == 0 else "failed"
    return DayRunResult(
        date_yyyymmdd=date_text,
        command=command,
        output_path=output_reference,
        log_path=str(log_path),
        status=status,
        return_code=proc.returncode,
        elapsed_seconds=elapsed,
    )


def main() -> None:
    """Execute daily batch processing."""
    args = parse_args()
    if args.raw_dir is None:
        raise ValueError(
            "Missing raw input directory. Pass --raw-dir or set EK80_RAW_DIR."
        )
    if not args.raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {args.raw_dir}")
    if not args.script_path.exists():
        raise FileNotFoundError(f"Script path does not exist: {args.script_path}")
    if args.viewer_export_html and not args.viewer_script_path.exists():
        raise FileNotFoundError(f"Viewer script path does not exist: {args.viewer_script_path}")

    if args.output_prefix:
        args.output_prefix = sanitize_output_prefix(args.output_prefix)
    else:
        args.output_prefix = infer_output_prefix(args.raw_dir)

    effective_export_mode = "none" if args.output_type == "html" else args.data_output_format
    if args.output_type in {"netcdf", "both"} and args.data_output_format == "none":
        raise ValueError(
            f"--output-type {args.output_type} requires --data-output-format netcdf/csv/both."
        )
    if args.output_type == "html" and args.data_output_format != "none":
        print("INFO: --output-type html selected; forcing MVBS data output format to none.")
    if args.viewer_export_html and effective_export_mode not in {"netcdf", "both"}:
        raise ValueError(
            "--viewer-export-html requires NetCDF outputs. "
            "Use --data-output-format netcdf/both and --output-type netcdf/both."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if args.hide_na_gaps:
        print(
            "INFO: hide-na-gaps enabled (default display mode; compressed timeline with datetime labels). "
            "Use --no-hide-na-gaps to disable. Use a dedicated output directory for this variant."
        )
        if args.skip_existing:
            if args.output_type == "netcdf":
                data_output_dir = args.data_output_dir or args.output_dir
                existing_outputs = sorted(data_output_dir.glob(f"{args.output_prefix}_*.mvbs.*"))
            else:
                existing_outputs = sorted(args.output_dir.glob(f"{args.output_prefix}_*.html"))
            if existing_outputs:
                print(
                    "WARNING: --skip-existing found existing outputs in this target directory. "
                    "Those files may come from non-hide-na runs and will be treated as already complete. "
                    "Use a separate output directory for hide-na exports."
                )

    all_dates = discover_dates(args.raw_dir)
    run_dates = filter_dates(all_dates, args.start_date, args.end_date)
    if not run_dates:
        raise RuntimeError("No matching dates found to process.")

    print(f"Discovered {len(all_dates)} date(s), running {len(run_dates)} date(s).")
    print(f"Output directory: {args.output_dir}")
    print(f"Output filename prefix: {args.output_prefix}")
    print(f"Output type: {args.output_type}")
    print(f"MVBS data output format: {effective_export_mode}")
    if args.viewer_export_html:
        print("Post-batch viewer export: enabled")

    python_exe = sys.executable
    results: list[DayRunResult] = []
    for date_text in run_dates:
        print(f"\n=== Processing {date_text} ===")
        result = run_one_day(date_text, args=args, python_exe=python_exe, logs_dir=logs_dir)
        results.append(result)
        print(
            f"{result.status.upper()} | date={result.date_yyyymmdd} | "
            f"elapsed={result.elapsed_seconds:.1f}s | output={result.output_path}"
        )
        if result.status == "skipped_existing" and args.hide_na_gaps:
            print(
                "  NOTE: skipped because matching output pattern already exists in this output directory "
                "(mode-specific outputs are not distinguished by filename)."
            )
        if result.status == "failed":
            print(f"  See log: {result.log_path}")
            if args.stop_on_error:
                break

    viewer_export_result = ViewerExportResult(
        status="not_requested",
        command=[],
        export_dir=None,
        log_path=None,
        return_code=0,
        elapsed_seconds=0.0,
        message=None,
    )
    if args.viewer_export_html:
        print("\n=== Headless Viewer Export ===")
        viewer_export_result = run_headless_viewer_export(
            args=args,
            python_exe=python_exe,
            logs_dir=logs_dir,
        )
        print(
            f"{viewer_export_result.status.upper()} | "
            f"elapsed={viewer_export_result.elapsed_seconds:.1f}s"
        )
        if viewer_export_result.message:
            print(f"  {viewer_export_result.message}")
        if viewer_export_result.export_dir:
            print(f"  HTML export dir: {viewer_export_result.export_dir}")
        if viewer_export_result.log_path:
            print(f"  Export log: {viewer_export_result.log_path}")

    summary_path = args.output_dir / "batch_summary.json"
    summary = {
        "raw_dir": str(args.raw_dir),
        "script_path": str(args.script_path),
        "output_dir": str(args.output_dir),
        "run_config": {
            "output_prefix": args.output_prefix,
            "chunk_size": args.chunk_size,
            "range_meter_bin": args.range_meter_bin,
            "ping_time_bin": args.ping_time_bin,
            "transducer_facing": args.transducer_facing,
            "flip_vertical": args.flip_vertical,
            "html_resources": args.html_resources,
            "data_output_format": args.data_output_format,
            "effective_data_output_format": effective_export_mode,
            "data_output_dir": str(args.data_output_dir) if args.data_output_dir else None,
            "data_csv_compression": args.data_csv_compression,
            "output_type": args.output_type,
            "channel": args.channel,
            "hide_na_gaps": args.hide_na_gaps,
            "skip_existing": args.skip_existing,
            "viewer_export_html": args.viewer_export_html,
            "viewer_script_path": str(args.viewer_script_path),
            "viewer_export_dir": str(args.viewer_export_dir) if args.viewer_export_dir else None,
            "viewer_export_glob_pattern": args.viewer_export_glob_pattern,
            "viewer_export_index_name": args.viewer_export_index_name,
            "viewer_export_no_index": args.viewer_export_no_index,
        },
        "results": [asdict(item) for item in results],
        "viewer_export": asdict(viewer_export_result),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    success_count = sum(1 for item in results if item.status == "success")
    fail_count = sum(1 for item in results if item.status == "failed")
    skip_count = sum(1 for item in results if item.status == "skipped_existing")
    print("\n=== Batch Summary ===")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Skipped existing: {skip_count}")
    print(f"Viewer export: {viewer_export_result.status}")
    print(f"Summary file: {summary_path}")

    if fail_count > 0 or viewer_export_result.status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
