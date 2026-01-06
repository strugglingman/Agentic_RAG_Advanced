"""
Multilingual text processing utilities.

Provides language detection, tokenization, and sentence splitting
for multiple languages including:
- English, Swedish, Finnish, Spanish, German, French (space-separated)
- Chinese, Japanese (requires word segmentation)

This module keeps language-specific logic isolated from the main retrieval code.
"""

import re
import logging
from typing import List, Optional, Tuple
from functools import lru_cache

logger = logging.getLogger(__name__)

# Language detection patterns
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")  # Chinese characters
JAPANESE_PATTERN = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")  # Hiragana + Katakana
KOREAN_PATTERN = re.compile(r"[\uac00-\ud7af]")  # Korean Hangul

# Sentence ending patterns for different languages
SENTENCE_ENDINGS = {
    "default": r"(?<=[.!?])\s+",
    "chinese": r"(?<=[。！？；])",
    "japanese": r"(?<=[。！？])",
}


class MultilingualTokenizer:
    """
    Tokenizer that handles multiple languages intelligently.

    For space-separated languages (English, Swedish, Finnish, Spanish):
        - Uses whitespace tokenization with optional stemming

    For CJK languages (Chinese, Japanese):
        - Uses jieba for Chinese word segmentation
        - Falls back to character-level tokenization if jieba unavailable
    """

    def __init__(self):
        self._jieba_loaded = False
        self._jieba = None
        self._load_jieba()

    def _load_jieba(self):
        """Lazy load jieba for Chinese tokenization."""
        try:
            import jieba

            # Suppress jieba's initialization messages
            jieba.setLogLevel(logging.WARNING)
            self._jieba = jieba
            self._jieba_loaded = True
            logger.info(
                "[MULTILINGUAL] jieba loaded successfully for Chinese tokenization"
            )
        except ImportError:
            logger.warning(
                "[MULTILINGUAL] jieba not installed. Chinese tokenization will use "
                "character-level fallback. Install with: pip install jieba"
            )
            self._jieba_loaded = False

    def detect_language_type(self, text: str) -> str:
        """
        Detect the primary language type of the text.

        Returns:
            'cjk' for Chinese/Japanese/Korean
            'space_separated' for English, Swedish, Finnish, Spanish, etc.
        """
        if not text:
            return "space_separated"

        # Sample first 500 characters for detection
        sample = text[:500]

        # Count CJK characters
        cjk_count = len(CJK_PATTERN.findall(sample))
        japanese_count = len(JAPANESE_PATTERN.findall(sample))

        # If more than 10% of characters are CJK, treat as CJK text
        total_chars = len(sample.replace(" ", "").replace("\n", ""))
        if total_chars > 0:
            cjk_ratio = (cjk_count + japanese_count) / total_chars
            if cjk_ratio > 0.1:
                return "cjk"

        return "space_separated"

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text based on detected language.

        Args:
            text: Input text to tokenize

        Returns:
            List of tokens
        """
        if not text:
            return []

        lang_type = self.detect_language_type(text)

        if lang_type == "cjk":
            return self._tokenize_cjk(text)
        else:
            return self._tokenize_space_separated(text)

    def _tokenize_space_separated(self, text: str) -> List[str]:
        """
        Tokenize space-separated languages (English, Swedish, Finnish, Spanish, etc.).

        Uses simple whitespace splitting with basic normalization.
        Works well for: English, Swedish, Finnish, Spanish, German, French, etc.
        """
        # Lowercase and split on whitespace
        text = text.lower()
        # Split on whitespace and punctuation, keeping alphanumeric tokens
        tokens = re.findall(r"\b\w+\b", text, re.UNICODE)
        return tokens

    def _tokenize_cjk(self, text: str) -> List[str]:
        """
        Tokenize CJK (Chinese/Japanese/Korean) text.

        Uses jieba for Chinese if available, otherwise falls back to
        character-level tokenization combined with any space-separated words.
        """
        tokens = []

        # Split text into CJK and non-CJK segments
        segments = self._split_cjk_segments(text)

        for segment_type, segment_text in segments:
            if segment_type == "cjk":
                if self._jieba_loaded and self._jieba:
                    # Use jieba for Chinese word segmentation
                    cjk_tokens = list(self._jieba.cut(segment_text))
                    tokens.extend([t.strip() for t in cjk_tokens if t.strip()])
                else:
                    # Fallback: character-level tokenization for CJK
                    # This is less accurate but works without dependencies
                    tokens.extend(list(segment_text.replace(" ", "")))
            else:
                # Non-CJK segment: use space tokenization
                space_tokens = self._tokenize_space_separated(segment_text)
                tokens.extend(space_tokens)

        return tokens

    def _split_cjk_segments(self, text: str) -> List[Tuple[str, str]]:
        """
        Split text into CJK and non-CJK segments.

        Returns list of tuples: (segment_type, segment_text)
        where segment_type is 'cjk' or 'other'
        """
        segments = []
        current_type = None
        current_text = []

        for char in text:
            is_cjk = bool(CJK_PATTERN.match(char) or JAPANESE_PATTERN.match(char))
            char_type = "cjk" if is_cjk else "other"

            if char_type != current_type and current_text:
                segments.append((current_type, "".join(current_text)))
                current_text = []

            current_type = char_type
            current_text.append(char)

        if current_text:
            segments.append((current_type, "".join(current_text)))

        return segments


class MultilingualSentenceSplitter:
    """
    Sentence splitter that handles multiple languages.

    Recognizes sentence boundaries for:
    - Latin languages: . ! ? followed by space and capital
    - Chinese: 。 ！ ？ ；
    - Japanese: 。 ！ ？
    """

    # Combined pattern for multiple language sentence endings
    UNIVERSAL_SENTENCE_PATTERN = re.compile(
        r"(?<=[.!?。！？；])\s*(?=[A-Z0-9\u4e00-\u9fff\u3040-\u30ff])|"  # After punctuation, before new sentence
        r"(?<=[。！？])"  # Chinese/Japanese sentence endings (no space required)
    )

    def split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences, handling multiple languages.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        if not text:
            return []

        text = text.strip()

        # Detect if text contains CJK characters
        has_cjk = bool(CJK_PATTERN.search(text))

        if has_cjk:
            return self._split_cjk_sentences(text)
        else:
            return self._split_latin_sentences(text)

    def _split_latin_sentences(self, text: str) -> List[str]:
        """Split sentences for Latin-script languages."""
        # Original pattern + support for more cases
        # Matches: . ! ? followed by whitespace and uppercase/number
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
        return [p.strip() for p in parts if p and p.strip()]

    def _split_cjk_sentences(self, text: str) -> List[str]:
        """
        Split sentences for CJK text.

        Handles:
        - Chinese: 。 ！ ？ ；
        - Japanese: 。 ！ ？
        - Mixed CJK + Latin text
        """
        sentences = []

        # Split on CJK sentence endings
        # Also handle Latin punctuation for mixed text
        pattern = r"([。！？；.!?]+)"

        parts = re.split(pattern, text)

        current_sentence = ""
        for i, part in enumerate(parts):
            if re.match(r"^[。！？；.!?]+$", part):
                # This is punctuation - append to current sentence
                current_sentence += part
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
            else:
                current_sentence += part

        # Don't forget the last sentence if it doesn't end with punctuation
        if current_sentence.strip():
            sentences.append(current_sentence.strip())

        return sentences


# Global singleton instances for efficiency
_tokenizer: Optional[MultilingualTokenizer] = None
_sentence_splitter: Optional[MultilingualSentenceSplitter] = None


def get_tokenizer() -> MultilingualTokenizer:
    """Get or create the global tokenizer instance."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = MultilingualTokenizer()
    return _tokenizer


def get_sentence_splitter() -> MultilingualSentenceSplitter:
    """Get or create the global sentence splitter instance."""
    global _sentence_splitter
    if _sentence_splitter is None:
        _sentence_splitter = MultilingualSentenceSplitter()
    return _sentence_splitter


# Convenience functions for direct use
def tokenize(text: str) -> List[str]:
    """
    Tokenize text with automatic language detection.

    Args:
        text: Input text (any language)

    Returns:
        List of tokens appropriate for the detected language

    Example:
        >>> tokenize("Hello world")
        ['hello', 'world']
        >>> tokenize("什么是人工智能")
        ['什么', '是', '人工', '智能']  # with jieba
        >>> tokenize("Hej världen")  # Swedish
        ['hej', 'världen']
    """
    return get_tokenizer().tokenize(text)


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences with automatic language detection.

    Args:
        text: Input text (any language)

    Returns:
        List of sentences

    Example:
        >>> split_sentences("Hello. World!")
        ['Hello.', 'World!']
        >>> split_sentences("这是第一句。这是第二句。")
        ['这是第一句。', '这是第二句。']
    """
    return get_sentence_splitter().split_sentences(text)


def detect_language_type(text: str) -> str:
    """
    Detect whether text is CJK or space-separated language.

    Args:
        text: Input text

    Returns:
        'cjk' or 'space_separated'
    """
    return get_tokenizer().detect_language_type(text)
