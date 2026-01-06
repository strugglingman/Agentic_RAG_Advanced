"""
URL formatting utilities for converting bare URLs to markdown links.

This module provides functions to:
1. Convert bare URLs to clickable markdown links
2. Extract meaningful titles from surrounding text
3. Map known domains to readable display names
"""

import re
from urllib.parse import urlparse


# Map of known domains to readable names
DOMAIN_NAMES = {
    "wikipedia.org": "Wikipedia",
    "en.wikipedia.org": "Wikipedia",
    "zh.wikipedia.org": "维基百科",
    "chinadiscovery.com": "China Discovery",
    "travelchinaguide.com": "Travel China Guide",
    "tripadvisor.com": "TripAdvisor",
    "lonelyplanet.com": "Lonely Planet",
    "ctrip.com": "Ctrip",
    "booking.com": "Booking.com",
    "agoda.com": "Agoda",
    "klook.com": "Klook",
    "viator.com": "Viator",
    "gov.cn": "Official Gov",
    "github.com": "GitHub",
    "stackoverflow.com": "Stack Overflow",
    "medium.com": "Medium",
    "youtube.com": "YouTube",
    "reddit.com": "Reddit",
}


def _get_domain_display(url: str) -> str:
    """Extract readable display name from domain.

    Args:
        url: Full URL string

    Returns:
        Human-readable domain name
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")

        # Check known domains first
        for known_domain, name in DOMAIN_NAMES.items():
            if domain.endswith(known_domain):
                return name

        # Fallback: convert domain to readable name
        parts = domain.split(".")
        if len(parts) >= 2:
            main = (
                parts[-2]
                if parts[-1] in ("com", "org", "net", "cn", "io", "co")
                else parts[0]
            )
            display = re.sub(r"([a-z])([A-Z])", r"\1 \2", main)
            return display.replace("-", " ").replace("_", " ").title()
        return domain.title()
    except Exception:
        return "Link"


def _merge_titles(text: str) -> str:
    """Merge 'Title: [Domain](url)' patterns into '[Title](url)'.

    Args:
        text: Text with markdown links that may have titles before them

    Returns:
        Text with merged titles
    """
    lines = text.split("\n")
    result_lines = []

    for line in lines:
        # Pattern: captures (optional list marker)(title text): [domain](url)
        # Group 1: list marker like "1. " or "- "
        # Group 2: title text before ": ["
        # Group 3: domain text (discarded)
        # Group 4: url
        # Group 5: rest of line after the link
        pattern = r"^(\d+\.\s*|[-*•]\s*)?(.+?):\s*\[([^\]]+)\]\(([^)]+)\)(.*)$"
        match = re.match(pattern, line)

        if match:
            list_marker = match.group(1) or ""
            title = match.group(2).strip()
            # domain = match.group(3)  # Not used
            url = match.group(4)
            rest = match.group(5) or ""

            # Check if title is meaningful (has letters, reasonable length)
            if title and len(title) >= 2 and re.search(r"[a-zA-Z\u4e00-\u9fff]", title):
                result_lines.append(f"{list_marker}[{title}]({url}){rest}")
            else:
                result_lines.append(line)
        else:
            result_lines.append(line)

    return "\n".join(result_lines)


def format_urls_as_markdown(text: str) -> str:
    """Convert bare URLs in text to markdown links with meaningful titles.

    Matches URLs that are NOT already inside markdown link syntax [text](url).
    Extracts title from preceding text pattern "Title: URL" when available,
    falls back to domain name otherwise.

    IMPORTANT: Only handles http/https URLs. Does NOT touch:
    - Already-formatted markdown links [text](url)
    - Internal file paths like /api/files/xxx

    Also fixes LLM incorrectly adding https:// to relative /api/files/ URLs
    (common with smaller models like gpt-4o-mini).

    Examples:
        "Wikipedia: https://en.wikipedia.org/wiki/Test" -> "[Wikipedia](https://...)"
        "https://www.example.com/page" -> "[Example](https://...)" (fallback to domain)

    Args:
        text: Text that may contain bare URLs

    Returns:
        Text with bare URLs converted to [Title](url) format
    """
    if not text:
        return text

    # Fix LLM incorrectly adding https:// to relative /api/ URLs
    text = text.replace("https://api/", "/api/")
    text = text.replace("http://api/", "/api/")

    # Pattern: Match bare URLs, optionally preceded by "Title: "
    # Only matches http/https URLs, skips already-formatted markdown links
    url_pattern = r'(?<!\]\()(https?://[^\s\)\]<>"]+)'

    def replace_url(match):
        url = match.group(1)

        # Strip trailing punctuation from URL
        trailing = ""
        while url and url[-1] in ".,;:!?)":
            trailing = url[-1] + trailing
            url = url[:-1]

        # Use domain name as display
        display = _get_domain_display(url)

        return f"[{display}]({url}){trailing}"

    # First pass: convert bare URLs to markdown links with domain names
    result = re.sub(url_pattern, replace_url, text)

    # Second pass: merge "Title: [Domain](url)" into "[Title](url)"
    result = _merge_titles(result)

    return result
