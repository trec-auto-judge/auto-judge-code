"""Shared utility functions."""

from typing import Sequence


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