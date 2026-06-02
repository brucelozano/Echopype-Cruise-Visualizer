"""Shared channel naming helpers used across scripts."""

from __future__ import annotations

import re


def infer_channel_frequency_khz(channel_name: str) -> int | None:
    """Infer channel nominal frequency in kHz from channel label text."""
    patterns = [
        r"(\d+)\s*kHz",
        r"ES(\d+)(?:[-_]|$)",
        r"GPT\s+(\d+)\s*kHz",
    ]
    for pattern in patterns:
        match = re.search(pattern, channel_name, flags=re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
    return None


def channel_slug(channel_name: str, max_length: int = 80) -> str:
    """Create a filesystem-safe slug for channel-specific filenames."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", channel_name).strip("_").lower()
    if not slug:
        slug = "channel"
    freq = infer_channel_frequency_khz(channel_name)
    if freq is not None and f"{freq}khz" not in slug:
        slug = f"{slug}_{freq}khz"
    return slug[:max_length]
