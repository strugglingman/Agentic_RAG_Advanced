"""
UTF-8 Encoding Fix for Windows Console.

Import this at the top of any standalone script that prints to console.
The backend already handles this via logging_config.py.

Usage:
    from src.utils.encoding_fix import fix_windows_encoding
    fix_windows_encoding()
"""

import sys
import io


def fix_windows_encoding():
    """
    Fix Windows console encoding to UTF-8.

    Prevents UnicodeEncodeError when printing Unicode characters
    (Chinese, Japanese, emoji, special symbols) on Windows.

    Safe to call on Linux/Mac (does nothing).
    """
    if sys.platform == "win32":
        try:
            # Wrap stdout and stderr with UTF-8 encoding
            if hasattr(sys.stdout, "buffer"):
                sys.stdout = io.TextIOWrapper(
                    sys.stdout.buffer,
                    encoding="utf-8",
                    errors="replace",  # Replace problematic characters
                    line_buffering=True,
                )
            if hasattr(sys.stderr, "buffer"):
                sys.stderr = io.TextIOWrapper(
                    sys.stderr.buffer,
                    encoding="utf-8",
                    errors="replace",
                    line_buffering=True,
                )
        except (AttributeError, ValueError):
            # Already wrapped or running in environment that doesn't support it
            pass
