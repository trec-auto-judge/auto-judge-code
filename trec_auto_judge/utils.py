"""Shared utility functions."""

import subprocess
from typing import Sequence


def get_git_info() -> dict[str, str]:
    """
    Get git repository information for reproducibility.

    Returns dict with:
        - commit: SHA or "unknown"
        - dirty: "true", "false", or "unknown"
        - remote: remote URL or "none" or "unknown"
    """
    result = {}

    # Get commit SHA
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        result["commit"] = commit.stdout.strip() if commit.returncode == 0 else "unknown"
    except Exception:
        result["commit"] = "unknown"

    # Check if dirty
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if status.returncode == 0:
            result["dirty"] = "true" if status.stdout.strip() else "false"
        else:
            result["dirty"] = "unknown"
    except Exception:
        result["dirty"] = "unknown"

    # Get remote URL (origin)
    try:
        remote = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if remote.returncode == 0 and remote.stdout.strip():
            result["remote"] = remote.stdout.strip()
        else:
            result["remote"] = "none"
    except Exception:
        result["remote"] = "unknown"

    return result


def format_preview(
    items: Sequence[str],
    limit: int = 10,
    separator: str = ", ",
) -> str:
    """
    Format a list of items with preview and 'more' suffix.

    Args:
        items: Items to format (must already be strings)
        limit: Maximum items to show before truncating
        separator: Separator between items (e.g., ", " or "\\n  ")

    Returns:
        Formatted string like "a, b, c ... (7 more)" or "a, b, c" if under limit

    Examples:
        >>> format_preview(["a", "b", "c", "d", "e"], limit=3)
        'a, b, c ... (2 more)'
        >>> format_preview(["a", "b"], limit=3)
        'a, b'
        >>> format_preview(["line1", "line2", "line3"], limit=2, separator="\\n  ")
        'line1\\n  line2\\n  ... (1 more)'
    """
    preview = separator.join(items[:limit])
    if len(items) > limit:
        # For newline separators, use the separator for "more" suffix
        # For inline separators like ", ", use " "
        if "\n" in separator:
            more_prefix = separator
        else:
            more_prefix = " "
        preview += f"{more_prefix}... ({len(items) - limit} more)"
    return preview