"""Build a lightweight local timeline viewer for daily echogram HTML exports.

This utility scans a directory of daily HTML outputs (for example, files produced
by ``run_daily_lander_batch.py``) and creates a single HTML page with:
- Channel selector
- Date stepping (prev/next)
- Autoplay through daily pages
- Embedded iframe to inspect each day's interactive echogram
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote


DAILY_PATTERN = re.compile(
    r"^(?P<prefix>.+)_(?P<date>\d{8})(?:__(?P<channel>.+))?\.html$",
    re.IGNORECASE,
)


def _path_from_env(*var_names: str) -> Path | None:
    """Return a Path from the first set environment variable in var_names."""
    for var_name in var_names:
        value = os.getenv(var_name)
        if value:
            return Path(value).expanduser()
    return None


@dataclass
class DailyHtml:
    """Represents one daily exported HTML file."""

    prefix: str
    date_yyyymmdd: str
    channel_label: str
    file_path: Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build a local timeline viewer for daily echogram HTML files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_path_from_env("EK80_TIMELINE_INPUT_DIR", "EK80_OUTPUT_DIR"),
        help=(
            "Directory containing daily echogram HTML files. "
            "Required unless EK80_TIMELINE_INPUT_DIR or EK80_OUTPUT_DIR is set."
        ),
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=None,
        help=(
            "Output viewer HTML path. Defaults to <input-dir>/<prefix>_timeline_viewer.html "
            "for single-prefix directories, otherwise <input-dir>/timeline_viewer.html."
        ),
    )
    parser.add_argument(
        "--channel-filter",
        type=str,
        default=None,
        help="Optional substring filter for channel label (case-insensitive).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Daily Echogram Timeline",
        help="Viewer page title.",
    )
    parser.add_argument(
        "--file-prefix",
        type=str,
        default=None,
        help=(
            "Optional filename prefix filter (case-insensitive). "
            "For example, 'lander' or 'DSB2'."
        ),
    )
    parser.add_argument(
        "--autoplay-ms",
        type=int,
        default=1500,
        help="Default autoplay interval in milliseconds.",
    )
    return parser.parse_args()


def prettify_channel_label(raw: str | None) -> str:
    """Convert channel suffix slug to readable text."""
    if not raw:
        return "selected_channel"
    text = raw.replace("_", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def discover_daily_html(
    input_dir: Path,
    channel_filter: str | None,
    file_prefix: str | None,
) -> list[DailyHtml]:
    """Discover daily echogram HTML files and parse date/channel labels."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    prefix_filter_norm = file_prefix.lower() if file_prefix else None
    discovered: list[DailyHtml] = []
    for file_path in sorted(input_dir.glob("*.html")):
        match = DAILY_PATTERN.match(file_path.name)
        if not match:
            continue
        prefix = match.group("prefix").strip()
        if not prefix:
            continue
        if prefix_filter_norm and prefix.lower() != prefix_filter_norm:
            continue
        date_text = match.group("date")
        channel_label = prettify_channel_label(match.group("channel"))
        if channel_filter and channel_filter.lower() not in channel_label.lower():
            continue
        discovered.append(
            DailyHtml(
                prefix=prefix,
                date_yyyymmdd=date_text,
                channel_label=channel_label,
                file_path=file_path,
            )
        )

    if not discovered:
        filters: list[str] = []
        if file_prefix:
            filters.append(f"prefix '{file_prefix}'")
        if channel_filter:
            filters.append(f"channel '{channel_filter}'")
        filter_msg = f" with {' and '.join(filters)}" if filters else ""
        raise RuntimeError(f"No matching daily HTML files found in {input_dir}{filter_msg}.")
    return discovered


def group_for_frontend(entries: list[DailyHtml], base_dir: Path) -> dict[str, list[dict[str, str]]]:
    """Group entries by channel and serialize relative paths for browser use."""
    grouped: dict[str, list[dict[str, str]]] = {}
    distinct_prefixes = {entry.prefix for entry in entries}
    include_prefix_in_label = len(distinct_prefixes) > 1
    for entry in entries:
        rel_path = entry.file_path.relative_to(base_dir).as_posix()
        rel_url = quote(rel_path, safe="/")
        group_label = (
            f"{entry.prefix} | {entry.channel_label}"
            if include_prefix_in_label
            else entry.channel_label
        )
        grouped.setdefault(group_label, []).append(
            {
                "prefix": entry.prefix,
                "date": entry.date_yyyymmdd,
                "path": rel_url,
                "file": entry.file_path.name,
            }
        )

    for _, items in grouped.items():
        items.sort(key=lambda x: x["date"])
    return dict(sorted(grouped.items(), key=lambda pair: pair[0]))


def build_viewer_html(title: str, grouped_json: str, autoplay_ms: int) -> str:
    """Build the standalone viewer HTML text."""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 0;
      background: #111827;
      color: #e5e7eb;
    }}
    .top {{
      padding: 12px 14px;
      border-bottom: 1px solid #374151;
      background: #1f2937;
      display: grid;
      gap: 8px;
    }}
    .row {{
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }}
    select, button, input {{
      background: #111827;
      color: #e5e7eb;
      border: 1px solid #4b5563;
      border-radius: 6px;
      padding: 6px 8px;
    }}
    .meta {{
      color: #9ca3af;
      font-size: 13px;
    }}
    iframe {{
      width: 100%;
      height: calc(100vh - 142px);
      border: none;
      background: #fff;
    }}
  </style>
</head>
<body>
  <div class="top">
    <div class="row">
      <strong>{title}</strong>
    </div>
    <div class="row">
      <label for="channelSelect">Channel:</label>
      <select id="channelSelect"></select>
      <button id="prevBtn" type="button">Prev</button>
      <button id="nextBtn" type="button">Next</button>
      <button id="playBtn" type="button">Play</button>
      <label for="speedInput">ms/frame:</label>
      <input id="speedInput" type="number" min="200" step="100" value="{autoplay_ms}" style="width: 90px;" />
    </div>
    <div class="meta" id="meta"></div>
  </div>
  <iframe id="viewerFrame" title="Daily echogram"></iframe>

  <script>
    const grouped = {grouped_json};
    const channels = Object.keys(grouped);
    const channelSelect = document.getElementById("channelSelect");
    const prevBtn = document.getElementById("prevBtn");
    const nextBtn = document.getElementById("nextBtn");
    const playBtn = document.getElementById("playBtn");
    const speedInput = document.getElementById("speedInput");
    const meta = document.getElementById("meta");
    const frame = document.getElementById("viewerFrame");

    let currentChannel = channels[0];
    let currentIndex = 0;
    let timerId = null;

    function fillChannelOptions() {{
      channels.forEach((channel) => {{
        const option = document.createElement("option");
        option.value = channel;
        option.textContent = channel;
        channelSelect.appendChild(option);
      }});
      channelSelect.value = currentChannel;
    }}

    function getItems() {{
      return grouped[currentChannel] || [];
    }}

    function setFrame() {{
      const items = getItems();
      if (!items.length) {{
        frame.removeAttribute("src");
        meta.textContent = "No entries for selected channel.";
        return;
      }}
      if (currentIndex < 0) currentIndex = items.length - 1;
      if (currentIndex >= items.length) currentIndex = 0;

      const item = items[currentIndex];
      frame.src = item.path;
      meta.textContent = `Date: ${{item.date}} | Prefix: ${{item.prefix}} | Channel: ${{currentChannel}} | File: ${{item.file}} | ${{currentIndex + 1}} / ${{items.length}}`;
    }}

    function step(delta) {{
      currentIndex += delta;
      setFrame();
    }}

    function setChannelPreserveDate(newChannel) {{
      const oldItems = getItems();
      const targetDate = oldItems[currentIndex]?.date || null;
      currentChannel = newChannel;
      const newItems = getItems();
      if (!newItems.length) {{
        currentIndex = 0;
        setFrame();
        return;
      }}
      if (targetDate !== null) {{
        const exactIdx = newItems.findIndex((item) => item.date === targetDate);
        if (exactIdx !== -1) {{
          currentIndex = exactIdx;
        }} else {{
          currentIndex = Math.min(currentIndex, newItems.length - 1);
        }}
      }} else {{
        currentIndex = Math.min(currentIndex, newItems.length - 1);
      }}
      setFrame();
    }}

    function stopPlayback() {{
      if (timerId !== null) {{
        clearInterval(timerId);
        timerId = null;
      }}
      playBtn.textContent = "Play";
    }}

    function startPlayback() {{
      const interval = Math.max(200, Number(speedInput.value) || {autoplay_ms});
      stopPlayback();
      timerId = setInterval(() => step(1), interval);
      playBtn.textContent = "Pause";
    }}

    channelSelect.addEventListener("change", () => {{
      setChannelPreserveDate(channelSelect.value);
    }});

    prevBtn.addEventListener("click", () => step(-1));
    nextBtn.addEventListener("click", () => step(1));
    playBtn.addEventListener("click", () => {{
      if (timerId === null) startPlayback();
      else stopPlayback();
    }});

    fillChannelOptions();
    setFrame();
  </script>
</body>
</html>
"""


def main() -> None:
    """Build the timeline viewer HTML."""
    args = parse_args()
    if args.input_dir is None:
        raise ValueError(
            "Missing input directory. Pass --input-dir or set EK80_TIMELINE_INPUT_DIR."
        )
    input_dir = args.input_dir.resolve()
    entries = discover_daily_html(
        input_dir=input_dir,
        channel_filter=args.channel_filter,
        file_prefix=args.file_prefix,
    )
    if args.output_html is not None:
        output_html = args.output_html.resolve()
    else:
        prefixes = sorted({entry.prefix for entry in entries})
        default_name = "timeline_viewer.html"
        if len(prefixes) == 1:
            default_name = f"{prefixes[0]}_timeline_viewer.html"
        output_html = (input_dir / default_name).resolve()
    grouped = group_for_frontend(entries=entries, base_dir=output_html.parent)
    viewer_html = build_viewer_html(
        title=args.title,
        grouped_json=json.dumps(grouped, indent=2),
        autoplay_ms=args.autoplay_ms,
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(viewer_html, encoding="utf-8")
    print(f"Built timeline viewer: {output_html}")
    print(f"Detected prefixes: {sorted({entry.prefix for entry in entries})}")
    print(f"Detected channels: {list(grouped.keys())}")


if __name__ == "__main__":
    main()
