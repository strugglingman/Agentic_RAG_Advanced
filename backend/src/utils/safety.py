from __future__ import annotations
import re
from typing import Iterable, Tuple, List, Dict, Any

# --- 1) Enhanced prompt-injection heuristics with categorization ---
DANGEROUS_PATTERNS: Dict[str, List[str]] = {
    "instruction_override": [
        r"\b(ignore|disregard|bypass|neglect|remove|delete|forget|skip).{1,3}(all|any|previous|above|your).{1,3}(instructions|rules|commands|orders)\b",
        r"\bdo.{1,3}not.{1,3}(follow|obey).{1,3}(instructions|rules|orders)\b",
        r"\b(ignore|disregard|forget).{1,10}(previous|above|all|any).{1,10}(instructions|rules|commands)\b",
    ],
    "safety_bypass": [
        r"\b(override|bypass|disable|deactivate).{1,3}(system|safety|security|filter|restriction)\b",
        r"\bturn.{1,3}off.{1,3}(safety|security|filter)\b",
    ],
    "prompt_leakage": [
        r"\b(reveal|show|display|tell).{1,10}(system|developer|initial).{1,3}(prompt|instructions)\b",
        r"\bshow.{1,3}(me|your).{1,3}(prompt|instructions)\b",
        r"\bwhat.{1,10}(system|initial|original).{1,3}(prompt|instructions)\b",
    ],
    "data_exfiltration": [
        r"\b(leak|steal|extract|exfiltrat).{1,10}(confidential|secret|sensitive|private).{1,3}(data|information)\b",
    ],
    "code_execution": [
        r"\b(run|execute).{1,3}(shell|code|command|script|bash|python)\b",
        r"\bos\.(system|exec)\b",
    ],
    "external_requests": [
        r"\b(make|send).{1,10}(http|api|external|network).{1,3}(request|call)\b",
        r"\bconnect.{1,3}to.{1,10}(external|remote).{1,3}(server|api|url)\b",
    ],
    "role_manipulation": [
        r"\b(act.{1,3}as|pretend.{1,5}to.{1,3}be|you.{1,3}are.{1,3}now).{1,10}(developer|admin|god|different|another)\b",
        r"\benable.{1,3}(developer|admin|debug|god).{1,3}mode\b",
        r"\byou.{1,3}are.{1,3}(no.{1,3}longer|not).{1,3}(assistant|ai)\b",
    ],
    "jailbreak_attempts": [
        r"\b(DAN|AIM|DUDE|STAN|SWITCH|AlphaBreak|BasedGPT)\b",
        r"\b(unfiltered|uncensored|unrestricted).{1,3}(mode|version|access)\b",
    ],
    "instruction_injection": [
        r"={3,}|#{3,}|\*{3,}|-{5,}",  # Suspicious delimiter patterns
        r"\[(SYSTEM|INST)\]|\[/INST\]",  # Model-specific tokens
        r"\bend.{1,3}of.{1,10}(instructions|prompt|system|rules|context)\b",
        r"\bignore.{1,3}above\b",
        r"\bnew.{1,3}prompt\b",
    ],
    "information_disclosure": [
        r"\b(list|show|display).{1,10}(all|your).{1,3}(files|documents|secrets|credentials|passwords)\b",
    ],
}

# Compile all patterns with case-insensitive flag
COMPILED_PATTERNS: Dict[str, re.Pattern] = {}
for category, patterns in DANGEROUS_PATTERNS.items():
    combined = "|".join(patterns)
    COMPILED_PATTERNS[category] = re.compile(combined, re.IGNORECASE)


def looks_like_injection(text: str, max_len: int = 4000) -> Tuple[bool, str]:
    """
    Check if text contains prompt injection patterns.
    Returns (is_flagged, error_message_or_category).
    """
    if not text:
        return False, ""

    # Check length
    if len(text) > max_len:
        return True, "Input too long (possible overflow attack)"

    # Check for excessive repetition
    if len(text) > 100:
        # Check for repeated characters
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        max_char_pct = max(char_counts.values()) / len(text)
        if max_char_pct > 0.4:  # More than 40% same character
            return (
                True,
                "Suspicious repetition detected (possible denial-of-service attack)",
            )

    # Check each category
    for cat, pattern in COMPILED_PATTERNS.items():
        match = pattern.search(text)
        if match:
            matched_text = match.group(0)[:100]  # Limit match display
            category_names = {
                "instruction_override": "Instruction Override",
                "safety_bypass": "Safety Bypass",
                "prompt_leakage": "Prompt Leakage",
                "data_exfiltration": "Data Exfiltration",
                "code_execution": "Code Execution",
                "external_requests": "External Request",
                "role_manipulation": "Role Manipulation",
                "jailbreak_attempts": "Jailbreak Attempt",
                "instruction_injection": "Instruction Injection",
                "information_disclosure": "Information Disclosure",
                "repetition_attack": "Repetition Attack",
            }
            return True, f"{category_names.get(cat, cat)} detected: '{matched_text}...'"

    return False, ""


# --- 2) Scrub risky "instructions" inside retrieved documents ---
STRIP_PATTERNS = [
    r"\bignore (?:previous|above|all) instructions\b",
    r"\bdo not obey\b",
    r"\byou are chatgpt\b",
    r"\byou are now\b",
    r"\bact as\b",
    r"\bpretend to be\b",
    r"\[SYSTEM\]|\[INST\]|\[/INST\]",
    r"\<\|im_start\|\>|\<\|im_end\|\>",
]
STRIP_RE = re.compile("|".join(STRIP_PATTERNS), re.IGNORECASE)


def scrub_context(text: str) -> str:
    if not text:
        return ""
    # Neutralize obvious instruction-like lines
    text = STRIP_RE.sub("[removed: unsafe instruction text]", text)
    return text


# --- 3) Confidence gating helpers (coverage, not just max) ---
def coverage_ok(
    scores: Iterable[float],
    topk: int = 5,
    score_avg: float = 0.28,
    score_min: float = 0.38,
) -> bool:
    s = sorted(scores or [], reverse=True)[:topk]
    if len(s) == 0:
        return False
    if s[0] < score_min:
        return False
    avg = sum(s) / len(s)
    return avg >= score_avg


# --- 4) Simple post-check: require citations per sentence ---
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
CIT_RE = re.compile(r"\[(\d+)\]")
# Pattern to detect "Sources:" line at the end
SOURCES_LINE = re.compile(r"\n*\s*Sources?:.*$", re.IGNORECASE | re.DOTALL)


def enforce_citations(answer: str, valid_ids: List[int]) -> Tuple[str, bool]:
    """
    Return (clean_answer, all_sentences_supported).

    Groups sentences with their following citations. A sentence is supported if:
    - It contains a citation inline, OR
    - The next segment starts with citations that apply to it

    This handles patterns like:
    - "His wife's name is Sonja. [1]" → kept (citation follows)
    - "His wife's name is Sonja [1]." → kept (citation inline)
    - "His wife's name is Sonja." → dropped (no citation)
    """
    if not answer:
        return "", False

    valid = set(valid_ids)

    # Extract and preserve "Sources:" line at the end
    sources_match = SOURCES_LINE.search(answer)
    sources_line = sources_match.group(0).strip() if sources_match else ""
    main_answer = SOURCES_LINE.sub("", answer).strip() if sources_match else answer.strip()

    # Split into segments (sentences + citation-only fragments)
    segments = SENT_SPLIT.split(main_answer)
    segments = [s.strip() for s in segments if s and s.strip()]

    if not segments:
        return sources_line, False  # Return just sources if no valid content

    # Process segments: group content with following citations
    keep: List[str] = []
    all_supported = True
    i = 0

    while i < len(segments):
        seg = segments[i]
        seg_cites = {int(m.group(1)) for m in CIT_RE.finditer(seg)}

        # Check if this segment is citation-only (like "[1]" or "[1][2]")
        text_without_cites = CIT_RE.sub("", seg).strip()
        is_cite_only = len(text_without_cites) == 0

        if is_cite_only:
            # Skip standalone citations - they should have been attached to previous
            i += 1
            continue

        # This is a content sentence - check if it has citations
        if seg_cites and (seg_cites & valid):
            # Has valid inline citation
            keep.append(seg)
            i += 1
            continue

        # No inline citation - check if next segment STARTS with citations
        if i + 1 < len(segments):
            next_seg = segments[i + 1]
            # Check if next segment starts with citation(s)
            next_cites = {int(m.group(1)) for m in CIT_RE.finditer(next_seg)}

            if next_cites and (next_cites & valid):
                # Next segment has citations - check if it starts with them
                next_text_before_cite = CIT_RE.split(next_seg)[0].strip()
                if len(next_text_before_cite) == 0:
                    # Next segment starts with citation - attach the citation part to current sentence
                    # Find leading citations
                    cite_match = re.match(r"^(\s*\[\d+\]\s*)+", next_seg)
                    if cite_match:
                        leading_cites = cite_match.group(0).strip()
                        remaining = next_seg[cite_match.end():].strip()
                        keep.append(f"{seg} {leading_cites}")
                        # Put remaining text back for next iteration if non-empty
                        if remaining:
                            segments[i + 1] = remaining
                        else:
                            i += 1  # Skip the now-consumed segment
                        i += 1
                        continue

        # No citation found - drop this sentence
        all_supported = False
        i += 1

    result = " ".join(keep)
    if sources_line:
        result = f"{result}\n\n{sources_line}" if result else sources_line

    return result, all_supported


def add_sources_from_citations(
    answer: str,
    contexts: List[Dict[str, Any]]
) -> Tuple[str, List[str]]:
    """
    Extract citation numbers from answer and append accurate Sources line.

    This replaces LLM-generated "Sources:" with programmatic extraction,
    ensuring the Sources line matches the actual citations used.

    Args:
        answer: The LLM response text (may contain [1], [2] citations)
        contexts: List of context dicts with 'filename' or 'source' keys

    Returns:
        (answer_with_sources, list_of_cited_filenames)
    """
    if not answer or not contexts:
        return answer, []

    # Extract all citation numbers from the answer
    cited_nums = sorted({int(m.group(1)) for m in CIT_RE.finditer(answer)})

    if not cited_nums:
        return answer, []

    # Map citation numbers to filenames (1-indexed)
    cited_files = []
    for num in cited_nums:
        idx = num - 1  # Convert to 0-indexed
        if 0 <= idx < len(contexts):
            ctx = contexts[idx]
            filename = ctx.get("filename") or ctx.get("source") or f"Context {num}"
            if filename not in cited_files:
                cited_files.append(filename)

    if not cited_files:
        return answer, []

    # Remove any existing Sources: line (in case LLM still added one)
    answer_clean = SOURCES_LINE.sub("", answer).strip()

    # Add accurate Sources line
    sources_line = f"Sources: {', '.join(cited_files)}"
    final_answer = f"{answer_clean}\n\nSources: {', '.join(cited_files)}"

    return final_answer, cited_files


def renumber_citations(answer: str, offset: int) -> str:
    """
    Renumber all [n] citations in an answer by adding an offset.

    Used for multi-step queries to ensure global citation numbering
    matches the combined all_contexts order sent to frontend.

    Args:
        answer: Text with [n] citations
        offset: Number to add to each citation (e.g., 3 means [1] -> [4])

    Returns:
        Answer with renumbered citations
    """
    if offset == 0 or not answer:
        return answer

    def replace_citation(match):
        old_num = int(match.group(1))
        new_num = old_num + offset
        return f"[{new_num}]"

    return CIT_RE.sub(replace_citation, answer)


# --- 5) Sanitize historical messages before sending to LLM ---
_SANITIZE_PATTERNS = [
    (r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions?", "[FILTERED]"),
    (r"disregard\s+(all\s+)?(previous|above|prior)\s+instructions?", "[FILTERED]"),
    (r"forget\s+(all\s+)?(previous|above|prior)\s+instructions?", "[FILTERED]"),
    (r"you\s+are\s+now\s+(a|an)", "[FILTERED]"),
    (r"new\s+instructions?:", "[FILTERED]"),
    (r"system\s*:\s*you", "[FILTERED]"),
    (r"<\|im_start\|>", ""),
    (r"<\|im_end\|>", ""),
    (r"\[INST\]|\[/INST\]", ""),
]


def sanitize_text(text: str, max_length: int = 10000) -> str:
    """Sanitize text input to prevent prompt injection in historical messages."""
    if not text:
        return ""

    text = text[:max_length]

    for pattern, replacement in _SANITIZE_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Remove excessive newlines
    text = re.sub(r'\n{4,}', '\n\n\n', text)

    return text.strip()
