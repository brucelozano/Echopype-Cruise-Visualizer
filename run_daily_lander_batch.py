"""Batch runner for daily EK80 HTML exports.

This utility discovers unique date stamps in `.raw` filenames and runs
`ek80_chunked_echogram.py` once per day. Failures are isolated per day so
successful days remain completed even if one day fails. By default it exports
all detected channels for each day.
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
DEFAULT_OUTPUT_PREFIX = "lander"


@dataclass
class DayRunResult:
    """Container for one daily run result."""

    date_yyyymmdd: str
    command: list[str]
    html_path: str
    log_path: str
    status: str
    return_code: int
    elapsed_seconds: float


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
        help="Directory for per-day HTML outputs and logs.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help=(
            "Prefix used for exported daily HTML files. "
            "Defaults to prefix inferred from raw filenames before DYYYYMMDD-THHMMSS "
            "(falls back to 'lander'). Case from raw filenames is preserved."
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
        default=-80.0,
        help="Lower dB limit passed through to the main script.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=-60.0,
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
        "--hide-na-gaps",
        action="store_true",
        help=(
            "Pass through display-only NA-gap collapsing to the main script "
            "(hides duty-cycle time gaps by plotting a dense index with datetime labels). "
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
            "Skip dates whose output HTML already exists. "
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


def run_one_day(
    date_text: str,
    args: argparse.Namespace,
    python_exe: str,
    logs_dir: Path,
) -> DayRunResult:
    """Run one day export and return structured result."""
    day_iso = parse_yyyymmdd(date_text).strftime("%Y-%m-%d")
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
        "--transducer-facing",
        args.transducer_facing,
        "--ui-mode",
        "static",
    ]

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

    existing_exports = sorted(args.output_dir.glob(f"{args.output_prefix}_{date_text}__*.html"))
    if args.skip_existing and (
        (selected_channel_path is not None and selected_channel_path.exists())
        or (selected_channel_path is None and len(existing_exports) > 0)
    ):
        return DayRunResult(
            date_yyyymmdd=date_text,
            command=command,
            html_path=str(selected_channel_path or html_pattern),
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
        html_path=str(selected_channel_path or html_pattern),
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

    if args.output_prefix:
        args.output_prefix = sanitize_output_prefix(args.output_prefix)
    else:
        args.output_prefix = infer_output_prefix(args.raw_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if args.hide_na_gaps:
        print(
            "INFO: --hide-na-gaps enabled (display-only compressed timeline with datetime labels). "
            "Use a dedicated output directory for this variant."
        )
        if args.skip_existing:
            existing_html = sorted(args.output_dir.glob(f"{args.output_prefix}_*.html"))
            if existing_html:
                print(
                    "WARNING: --skip-existing found existing daily exports in this output directory. "
                    "Those files may come from non-hide-na runs and will be treated as already complete. "
                    "Use a separate --output-dir for hide-na exports."
                )

    all_dates = discover_dates(args.raw_dir)
    run_dates = filter_dates(all_dates, args.start_date, args.end_date)
    if not run_dates:
        raise RuntimeError("No matching dates found to process.")

    print(f"Discovered {len(all_dates)} date(s), running {len(run_dates)} date(s).")
    print(f"Output directory: {args.output_dir}")
    print(f"Output filename prefix: {args.output_prefix}")

    python_exe = sys.executable
    results: list[DayRunResult] = []
    for date_text in run_dates:
        print(f"\n=== Processing {date_text} ===")
        result = run_one_day(date_text, args=args, python_exe=python_exe, logs_dir=logs_dir)
        results.append(result)
        print(
            f"{result.status.upper()} | date={result.date_yyyymmdd} | "
            f"elapsed={result.elapsed_seconds:.1f}s | html={result.html_path}"
        )
        if result.status == "skipped_existing" and args.hide_na_gaps:
            print(
                "  NOTE: skipped because matching HTML pattern already exists in this output directory "
                "(mode-specific outputs are not distinguished by filename)."
            )
        if result.status == "failed":
            print(f"  See log: {result.log_path}")
            if args.stop_on_error:
                break

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
            "html_resources": args.html_resources,
            "channel": args.channel,
            "hide_na_gaps": args.hide_na_gaps,
            "skip_existing": args.skip_existing,
        },
        "results": [asdict(item) for item in results],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    success_count = sum(1 for item in results if item.status == "success")
    fail_count = sum(1 for item in results if item.status == "failed")
    skip_count = sum(1 for item in results if item.status == "skipped_existing")
    print("\n=== Batch Summary ===")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Skipped existing: {skip_count}")
    print(f"Summary file: {summary_path}")

    if fail_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
