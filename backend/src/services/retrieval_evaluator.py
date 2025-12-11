"""
Retrieval quality evaluator for the self-reflection system.

This module implements the RetrievalEvaluator class which assesses whether
retrieved contexts are sufficient to answer a user's query.

Supports three evaluation modes:
- FAST: Heuristic-based evaluation (~50ms, no LLM)
- BALANCED: Heuristics + light LLM check (~500ms, 1 LLM call)
- THOROUGH: Full LLM evaluation (~2s, detailed analysis)
"""

import re
import json
from typing import Dict, Any, List, Optional, Tuple
from openai import OpenAI
from langsmith import traceable

from src.models.evaluation import (
    EvaluationCriteria,
    EvaluationResult,
    ReflectionConfig,
    QualityLevel,
    RecommendationAction,
    ReflectionMode,
)


class RetrievalEvaluator:
    """
    Evaluates retrieval quality and recommends next actions.

    This class provides three evaluation modes with different speed/accuracy tradeoffs:
    - FAST: Pure heuristics, no LLM calls
    - BALANCED: Heuristics + quick LLM validation
    - THOROUGH: Comprehensive LLM-based evaluation

    Attributes:
        config: ReflectionConfig instance with thresholds and mode settings
        openai_client: OpenAI client for LLM-based evaluation (required for BALANCED/THOROUGH)
    """

    def __init__(
        self, config: ReflectionConfig, openai_client: Optional[OpenAI] = None
    ):
        """
        Initialize the evaluator.

        Args:
            config: Runtime configuration for evaluation
            openai_client: OpenAI client (required for BALANCED/THOROUGH modes)

        Raises:
            ValueError: If mode requires LLM but openai_client is None

        TODO: Implement initialization
        Steps:
        1. Store config as self.config
        2. Store openai_client as self.client
        3. Validate: if config.mode is BALANCED or THOROUGH:
           - Check if openai_client is not None
           - Raise ValueError if None with message:
             "OpenAI client required for BALANCED/THOROUGH modes"
        """
        self.config = config
        self.client = openai_client
        if self.config.mode in [ReflectionMode.BALANCED, ReflectionMode.THOROUGH]:
            if self.client is None:
                raise ValueError("OpenAI client required for BALANCED/THOROUGH modes")

    @traceable
    def evaluate(self, criteria: EvaluationCriteria) -> EvaluationResult:
        """
        Evaluate retrieval quality.

        Routes to appropriate evaluation method based on mode.

        Args:
            criteria: Input containing query, contexts, metadata, and mode

        Returns:
            EvaluationResult with quality assessment and recommendation

        TODO: Implement evaluation routing
        Steps:
        1. Get mode from criteria.mode (fallback to self.config.mode if None)
        2. Route to appropriate method:
           - ReflectionMode.FAST -> return self._evaluate_fast(criteria)
           - ReflectionMode.BALANCED -> return self._evaluate_balanced(criteria)
           - ReflectionMode.THOROUGH -> return self._evaluate_thorough(criteria)
        3. If mode is invalid, raise ValueError or fallback to FAST mode
        """
        mode = criteria.mode or self.config.mode
        if mode == ReflectionMode.FAST:
            return self._evaluate_fast(criteria)
        elif mode == ReflectionMode.BALANCED:
            return self._evaluate_balanced(criteria)
        elif mode == ReflectionMode.THOROUGH:
            return self._evaluate_thorough(criteria)
        else:
            raise ValueError(f"Invalid reflection mode: {mode}")

    # =========================================================================
    # FAST MODE: Heuristic-Based Evaluation
    # =========================================================================

    def _evaluate_fast(self, criteria: EvaluationCriteria) -> EvaluationResult:
        """
        Fast heuristic-based evaluation (no LLM calls).

        Calculates confidence based on:
        - Keyword overlap between query and contexts
        - Average retrieval scores
        - Number of contexts retrieved

        Args:
            criteria: Evaluation criteria with query and contexts

        Returns:
            EvaluationResult with heuristic-based assessment

        TODO: Implement FAST mode evaluation
        This is the most important method to implement first!

        Steps:
        1. Extract basic metrics:
           - context_count = len(criteria.contexts)
           - Extract query keywords (call _extract_keywords helper)
           - Extract all context text (call _extract_context_text helper)

        2. Calculate keyword overlap:
           - Call _calculate_keyword_overlap(query_keywords, context_text)
           - Returns float 0.0-1.0

        3. Extract relevance scores:
           - Call _extract_relevance_scores(criteria.contexts)
           - Uses priority: rerank > hybrid > sem_sim (consistent across all contexts)
           - Returns List[float], defaults to 0.0 if missing
           - Calculate avg_score and min_score from relevance_scores

        4. Calculate confidence:
           - Formula: keyword_overlap * 0.4 + avg_score * 0.3 + min_score * 0.2 + context_presence * 0.1
           - context_presence = 1.0 if context_count > 0 else 0.0
           - Clamp result to [0.0, 1.0]

        5. Calculate coverage:
           - coverage = keyword_overlap if context_count > 0 else 0.0

        6. Detect issues:
           - Call _detect_issues(context_count, avg_score, keyword_overlap)
           - Returns List[str] of issue descriptions

        7. Identify missing aspects:
           - Call _identify_missing_aspects(query_keywords, context_text)
           - Returns List[str] of keywords not found

        8. Determine recommendation:
           - Call _determine_recommendation(confidence, context_count)
           - Returns (RecommendationAction, reasoning: str)

        9. Build EvaluationResult:
            - quality = self.config.get_quality_level(confidence)
            - Fill all fields
            - metrics = {"keyword_overlap": ..., "avg_score": ..., etc.}
            - mode_used = ReflectionMode.FAST

        10. Return result
        """
        context_count = len(criteria.contexts)
        query_keywords = self._extract_keywords(criteria.query)
        context_text = self._extract_context_text(criteria.contexts)
        keyword_overlap = self._calculate_keyword_overlap(query_keywords, context_text)
        relevance_scores = self._extract_relevance_scores(criteria.contexts)
        avg_score = (
            sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        )
        min_score = min(relevance_scores) if relevance_scores else 0.0
        context_presence = 1.0 if context_count > 0 else 0.0
        # Calculate evaluation confidence
        evaluation_confidence = min(
            1.0,
            (
                keyword_overlap * 0.4
                + avg_score * 0.3
                + min_score * 0.2
                + context_presence * 0.1
            ),
        )
        coverage = keyword_overlap if context_count > 0 else 0.0
        issues = self._detect_issues(context_count, avg_score, keyword_overlap)
        missing_aspects = self._identify_missing_aspects(query_keywords, context_text)
        recommendation, reasoning = self._determine_recommendation(
            evaluation_confidence, context_count, criteria.query
        )
        quality_level = self.config.get_quality_level(evaluation_confidence)

        result = EvaluationResult(
            quality=quality_level,
            confidence=evaluation_confidence,
            coverage=coverage,
            recommendation=recommendation,
            reasoning=reasoning,
            relevance_scores=relevance_scores,
            issues=issues,
            missing_aspects=missing_aspects,
            metrics={
                "keyword_overlap": keyword_overlap,
                "avg_score": avg_score,
                "min_score": min_score,
                "context_count": context_count,
                "context_presence": context_presence,
            },
            mode_used=ReflectionMode.FAST,
        )

        return result

    # =========================================================================
    # BALANCED MODE: Heuristics + Light LLM Check
    # =========================================================================

    def _evaluate_balanced(self, criteria: EvaluationCriteria) -> EvaluationResult:
        """
        Balanced evaluation: heuristics + light LLM validation.

        Runs FAST mode first, then uses LLM to validate borderline cases.

        Args:
            criteria: Evaluation criteria

        Returns:
            EvaluationResult with LLM-adjusted assessment

        TODO: Implement BALANCED mode evaluation
        Steps:
        1. Call self._evaluate_fast(criteria) to get baseline result
        2. Check if confidence is borderline (partial <= confidence < good threshold):
           - If yes, call _quick_llm_check() to validate
           - Get adjusted_confidence and llm_reasoning
           - Update result.confidence = adjusted_confidence
           - Update result.reasoning = llm_reasoning
           - Update result.quality = self.config.get_quality_level(adjusted_confidence)
        3. Update result.mode_used = ReflectionMode.BALANCED
        4. Return result
        """
        result = self._evaluate_fast(criteria=criteria)
        partial_threshold = self.config.thresholds["partial"]
        good_threshold = self.config.thresholds["good"]
        if partial_threshold <= result.confidence < good_threshold:
            adjusted_confidence, llm_reasoning = self._quick_llm_check(
                criteria.query, criteria.contexts, result.confidence
            )
            result.confidence = adjusted_confidence
            result.reasoning = llm_reasoning
            result.quality = self.config.get_quality_level(adjusted_confidence)

        result.mode_used = ReflectionMode.BALANCED
        return result

    def _quick_llm_check(
        self, query: str, contexts: List[Dict[str, Any]], baseline_confidence: float
    ) -> Tuple[float, str]:
        """
        Quick LLM check for borderline cases.

        Asks LLM: "Can these contexts answer the query? (yes/partial/no)"

        Args:
            query: User's question
            contexts: Retrieved contexts
            baseline_confidence: Confidence from heuristics

        Returns:
            Tuple of (adjusted_confidence, llm_reasoning)

        TODO: Implement quick LLM check
        Steps:
        1. Format contexts into readable text:
           - For each context: f"[{i+1}] {context.get('chunk', '')[:300]}"
           - Join with newlines

        2. Build prompt (see template below):
           prompt = f'''You are evaluating whether retrieved contexts can answer a user's query.

           USER QUERY: {query}

           RETRIEVED CONTEXTS:
           {formatted_contexts}

           QUESTION: Can these contexts adequately answer the user's query?

           Respond in this exact format:
           ANSWER: [yes/partial/no]
           REASONING: [one sentence explanation]

           Be strict: only answer 'yes' if contexts directly and completely answer the query.
           '''

        3. Call OpenAI API:
           - model="gpt-4o-mini"
           - temperature=0.0
           - max_tokens=150
           - messages=[{"role": "user", "content": prompt}]

        4. Parse response:
           - Extract answer (yes/partial/no) using regex or string search
           - Extract reasoning (text after "REASONING:")

        5. Adjust confidence:
           - If "yes": adjusted = min(1.0, baseline_confidence + 0.1)
           - If "no": adjusted = max(0.0, baseline_confidence - 0.1)
           - If "partial": adjusted = baseline_confidence (no change)

        6. Return (adjusted_confidence, reasoning)

        Error handling:
        - If LLM call fails, return (baseline_confidence, "LLM check failed")
        """
        try:
            formatted_contexts = "\n".join(
                [f"{i+1}] {c.get('chunk', '')[:300]}" for i, c in enumerate(contexts)]
            )
            prompt = f"""
                You are evaluating whether retrieved contexts can answer a user's query.

                USER QUERY: {query}

                RETRIEVED CONTEXTS:
                {formatted_contexts}

                QUESTION: Can these contexts adequately answer the user's query?

                Respond in this exact format:
                ANSWER: [yes/partial/no]
                REASONING: [one sentence explanation]

                Be strict: only answer 'yes' if contexts directly and completely answer the query.
            """
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.0,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content
            answer_match = re.search(
                r"ANSWER:\s*(yes|partial|no)", content, re.IGNORECASE
            )
            reasoning_match = re.search(r"REASONING:\s*(.*)", content, re.IGNORECASE)

            if not answer_match or answer_match.group(1).lower() not in [
                "yes",
                "partial",
                "no",
            ]:
                return baseline_confidence, "LLM check failed"
            answer = answer_match.group(1).lower()
            adjusted = baseline_confidence
            if answer == "yes":
                adjusted = min(1.0, baseline_confidence + 0.1)
            elif answer == "no":
                adjusted = max(0.0, baseline_confidence - 0.1)

            reasoning = (
                reasoning_match.group(1).strip()
                if reasoning_match
                else "No reasoning provided"
            )

            return adjusted, reasoning
        except Exception as e:
            return baseline_confidence, f"LLM check failed: {str(e)}"

    # =========================================================================
    # THOROUGH MODE: Full LLM Evaluation
    # =========================================================================

    def _evaluate_thorough(self, criteria: EvaluationCriteria) -> EvaluationResult:
        """
        Thorough LLM-based evaluation.

        Uses detailed LLM prompt to comprehensively evaluate retrieval quality.

        Args:
            criteria: Evaluation criteria

        Returns:
            EvaluationResult with comprehensive LLM analysis

        TODO: Implement THOROUGH mode evaluation
        Steps:
        1. Format contexts into numbered list with full text

        2. Build detailed evaluation prompt (see template below):
           prompt = f'''You are an expert retrieval quality evaluator. Analyze whether the retrieved contexts can answer the user's query.

           USER QUERY:
           {query}

           RETRIEVED CONTEXTS:
           {formatted_contexts}

           Evaluate the following:

           1. RELEVANCE: Rate each context's relevance (0.0-1.0 per context)
           2. COVERAGE: Do contexts fully cover all aspects of the query? (0.0-1.0)
           3. CONFIDENCE: Overall confidence these contexts can answer the query (0.0-1.0)
           4. ISSUES: List any problems (e.g., "context too generic", "missing key information")
           5. MISSING: What query aspects are not covered?
           6. RECOMMENDATION: What should we do?
              - ANSWER: Contexts are sufficient
              - REFINE: Reformulate query to get better results
              - EXTERNAL: Search external sources
              - CLARIFY: Query is ambiguous

           Respond in JSON format:
           {{
               "relevance_scores": [0.0-1.0, ...],
               "coverage": 0.0-1.0,
               "confidence": 0.0-1.0,
               "issues": ["issue1", "issue2"],
               "missing_aspects": ["aspect1", "aspect2"],
               "recommendation": "ANSWER|REFINE|EXTERNAL|CLARIFY",
               "reasoning": "one sentence explanation"
           }}
           '''

        3. Call OpenAI API:
           - model="gpt-4o-mini" or "gpt-4o"
           - temperature=0.0
           - response_format={"type": "json_object"} (if available)
           - messages=[{"role": "user", "content": prompt}]

        4. Parse LLM JSON response:
           - json.loads(response.choices[0].message.content)
           - Extract all fields
           - Convert recommendation string to RecommendationAction enum
           - Validate scores are in [0.0, 1.0]

        5. Build EvaluationResult:
           - quality = self.config.get_quality_level(confidence)
           - Use LLM-provided values for all fields
           - mode_used = ReflectionMode.THOROUGH

        6. Error handling:
           - If LLM call fails: fallback to self._evaluate_balanced(criteria)
           - If JSON parsing fails: fallback to self._evaluate_balanced(criteria)
           - Log errors for debugging

        7. Return result
        """
        # Step 1: Format contexts into numbered list with full text
        formatted_contexts = "\n\n".join(
            [
                f"Context {i+1}:\n{c.get('chunk', '')}"
                for i, c in enumerate(criteria.contexts)
            ]
        )

        # Step 2: Build detailed evaluation prompt
        prompt = (
            "You are an expert retrieval quality evaluator. "
            "Analyze whether the retrieved contexts can answer the user's query.\n\n"
            f"USER QUERY:\n{criteria.query}\n\n"
            f"RETRIEVED CONTEXTS:\n{formatted_contexts}\n\n"
            "Evaluate the following:\n\n"
            "1. RELEVANCE: Rate each context's relevance (0.0-1.0 per context)\n"
            "2. COVERAGE: Do contexts fully cover all aspects of the query? (0.0-1.0)\n"
            "3. CONFIDENCE: Overall confidence these contexts can answer the query (0.0-1.0)\n"
            '4. ISSUES: List any problems (e.g., "context too generic", "missing key information")\n'
            "5. MISSING: What query aspects are not covered?\n"
            "6. RECOMMENDATION: What should we do?\n"
            "   - ANSWER: Contexts are sufficient\n"
            "   - REFINE: Reformulate query to get better results\n"
            "   - EXTERNAL: Search external sources\n"
            "   - CLARIFY: Query is ambiguous\n\n"
            "Respond in JSON format:\n"
            "{\n"
            '    "relevance_scores": [0.0-1.0, ...],\n'
            '    "coverage": 0.0-1.0,\n'
            '    "confidence": 0.0-1.0,\n'
            '    "issues": ["issue1", "issue2"],\n'
            '    "missing_aspects": ["aspect1", "aspect2"],\n'
            '    "recommendation": "ANSWER|REFINE|EXTERNAL|CLARIFY",\n'
            '    "reasoning": "one sentence explanation"\n'
            "}"
        )

        # Step 3: Call OpenAI API
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )

            # Step 4: Parse LLM JSON response
            content = response.choices[0].message.content
            llm_result = json.loads(content)

            # Extract and validate fields
            relevance_scores = llm_result.get("relevance_scores", [])
            coverage = float(llm_result.get("coverage", 0.0))
            confidence = float(llm_result.get("confidence", 0.0))
            issues = llm_result.get("issues", [])
            missing_aspects = llm_result.get("missing_aspects", [])
            recommendation_str = llm_result.get("recommendation", "CLARIFY").upper()
            reasoning = llm_result.get("reasoning", "LLM evaluation completed")

            # Clamp scores to [0.0, 1.0]
            relevance_scores = [max(0.0, min(1.0, float(s))) for s in relevance_scores]
            coverage = max(0.0, min(1.0, coverage))
            confidence = max(0.0, min(1.0, confidence))

            # Convert recommendation string to enum
            recommendation_map = {
                "ANSWER": RecommendationAction.ANSWER,
                "REFINE": RecommendationAction.REFINE,
                "EXTERNAL": RecommendationAction.EXTERNAL,
                "CLARIFY": RecommendationAction.CLARIFY,
            }
            recommendation = recommendation_map.get(
                recommendation_str, RecommendationAction.CLARIFY
            )

            # Step 5: Build EvaluationResult
            quality = self.config.get_quality_level(confidence)

            result = EvaluationResult(
                quality=quality,
                confidence=confidence,
                coverage=coverage,
                recommendation=recommendation,
                reasoning=reasoning,
                relevance_scores=relevance_scores,
                issues=issues,
                missing_aspects=missing_aspects,
                metrics={
                    "llm_evaluation": True,
                    "context_count": len(criteria.contexts),
                },
                mode_used=ReflectionMode.THOROUGH,
            )

            return result

        except json.JSONDecodeError as e:
            # Step 6: Error handling - JSON parsing failed
            print(f"THOROUGH mode JSON parsing failed: {e}")
            return self._evaluate_balanced(criteria)
        except Exception as e:
            # Step 6: Error handling - LLM call failed
            print(f"THOROUGH mode failed: {e}")
            return self._evaluate_balanced(criteria)

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _contains_cjk(self, text: str) -> bool:
        """
        Check if text contains CJK (Chinese, Japanese, Korean) characters.

        Args:
            text: Text to check

        Returns:
            True if text contains CJK characters
        """
        if not text:
            return False
        # Unicode ranges for CJK characters
        for char in text:
            if "\u4e00" <= char <= "\u9fff":  # CJK Unified Ideographs
                return True
            if "\u3400" <= char <= "\u4dbf":  # CJK Extension A
                return True
            if "\uf900" <= char <= "\ufaff":  # CJK Compatibility Ideographs
                return True
        return False

    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract keywords from query (remove stopwords).

        Args:
            query: User's question

        Returns:
            List of keywords (lowercase, no stopwords)

        TODO: Implement keyword extraction
        Steps:
        1. Define stopwords list:
           stopwords = {"the", "a", "an", "is", "are", "was", "were",
                        "what", "when", "where", "how", "why", "who",
                        "in", "on", "at", "to", "for", "of", "with"}

        2. Lowercase query and split by non-alphanumeric chars:
           - Use re.split(r'\W+', query.lower())

        3. Filter out stopwords and empty strings

        4. Return list of keywords
        """
        # For CJK languages, use character-based extraction
        if self._contains_cjk(query):
            # Extract all CJK characters as individual tokens
            # This is better than word-splitting which breaks CJK text incorrectly
            cjk_chars = [
                char
                for char in query
                if "\u4e00" <= char <= "\u9fff"
                or "\u3400" <= char <= "\u4dbf"
                or "\uf900" <= char <= "\ufaff"
            ]
            # Also extract English words if present
            english_words = re.findall(r"[a-zA-Z]+", query.lower())
            return cjk_chars + english_words

        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "what",
            "when",
            "where",
            "how",
            "why",
            "who",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
        }
        tokens = re.split(r"\W+", query.lower())
        return [t for t in tokens if t and t not in stopwords]

    def _extract_context_text(self, contexts: List[Dict[str, Any]]) -> str:
        """
        Extract all text from contexts into single string.

        Args:
            contexts: List of context dictionaries

        Returns:
            Combined text from all contexts (lowercase)

        TODO: Implement context text extraction
        Steps:
        1. Initialize empty list to collect text
        2. For each context in contexts:
           - Get text from context.get("chunk", "") or context.get("text", "")
           - Add to list
        3. Join all text with spaces
        4. Return lowercase version
        """
        texts = []
        for c in contexts:
            chunk = c.get("chunk", "")
            texts.append(chunk)

        return " ".join(texts).lower()

    def _calculate_keyword_overlap(
        self, keywords: List[str], context_text: str
    ) -> float:
        """
        Calculate keyword overlap score.

        Args:
            keywords: Query keywords
            context_text: Combined context text

        Returns:
            Overlap score (0.0-1.0)

        TODO: Implement keyword overlap calculation
        Steps:
        1. If no keywords, return 0.0
        2. Count how many keywords appear in context_text:
           - found_count = sum(1 for kw in keywords if kw in context_text)
        3. Calculate overlap = found_count / len(keywords)
        4. Return overlap (already in 0.0-1.0 range)
        """
        if not keywords:
            return 0.0

        # For CJK character-based keywords, check character presence
        # For English keywords, use case-insensitive word matching
        found_count = 0
        for kw in keywords:
            if len(kw) == 1 and self._contains_cjk(kw):
                # Single CJK character - check presence
                if kw in context_text:
                    found_count += 1
            else:
                # English word or multi-char - case insensitive
                if kw.lower() in context_text.lower():
                    found_count += 1

        return found_count / len(keywords)

    def _detect_issues(
        self, context_count: int, avg_score: float, keyword_overlap: float
    ) -> List[str]:
        """
        Detect issues with retrieval quality.

        Args:
            context_count: Number of contexts retrieved
            avg_score: Average retrieval score
            keyword_overlap: Keyword overlap score

        Returns:
            List of issue descriptions

        TODO: Implement issue detection
        Steps:
        1. Initialize empty issues list
        2. Check conditions and append issues:
           - If context_count == 0: "No contexts retrieved"
           - If context_count < self.config.min_contexts: f"Only {context_count} contexts (min: {min_contexts})"
           - If avg_score < self.config.avg_score: f"Low average relevance: {avg_score:.2f}"
           - If keyword_overlap < self.config.keyword_overlap: f"Poor keyword match: {keyword_overlap:.2f}"
        3. Return issues list
        """
        issues = []
        if context_count == 0:
            issues.append("No contexts retrieved")
        if context_count < self.config.min_contexts:
            issues.append(
                f"Only {context_count} contexts retrieved (minimum recommended: {self.config.min_contexts})"
            )
        if avg_score < self.config.avg_score:
            issues.append(f"Low average relevance score: {avg_score:.2f}")
        if keyword_overlap < self.config.keyword_overlap:
            issues.append(f"Poor keyword match: {keyword_overlap:.2f}")

        return issues

    def _identify_missing_aspects(
        self, keywords: List[str], context_text: str
    ) -> List[str]:
        """
        Identify query aspects not covered by contexts.

        Args:
            keywords: Query keywords
            context_text: Combined context text

        Returns:
            List of keywords not found in contexts

        TODO: Implement missing aspects identification
        Steps:
        1. Filter keywords to find those NOT in context_text:
           missing = [kw for kw in keywords if kw not in context_text]
        2. Return missing list
        """
        missing = []
        for kw in keywords:
            if len(kw) == 1 and self._contains_cjk(kw):
                # Single CJK character check
                if kw not in context_text:
                    missing.append(kw)
            else:
                # English word or multi-char - case insensitive
                if kw.lower() not in context_text.lower():
                    missing.append(kw)
        return missing

    def _is_external_query(self, query: str) -> bool:
        """
        Check if query appears to need external/real-time information.

        Detects queries that likely cannot be answered from internal documents:
        - Real-time data (prices, weather, news)
        - Current/latest information
        - General knowledge not in company docs

        Args:
            query: User query string

        Returns:
            True if query appears to need external information

        Examples:
            "What is the current inflation rate?" → True
            "What is the weather today?" → True
            "What is our vacation policy?" → False
            "How do I apply for leave?" → False
        """
        if not query:
            return False

        query_lower = query.lower()

        # Real-time / current data indicators
        time_indicators = [
            "current",
            "latest",
            "today",
            "now",
            "live",
            "real-time",
            "recent",
            "this week",
            "this month",
            "this year",
        ]

        # External data types
        external_data_types = [
            "weather",
            "stock price",
            "stock market",
            "news",
            "trending",
            "inflation rate",
            "exchange rate",
            "cryptocurrency",
            "bitcoin",
        ]

        # General knowledge patterns (unlikely in company docs)
        general_knowledge = [
            "who is",
            "what is a ",
            "what is an ",
            "what are ",
            "how does",
            "define ",
            "meaning of",
            "history of",
        ]

        # Check time indicators
        for indicator in time_indicators:
            if indicator in query_lower:
                return True

        # Check external data types
        for data_type in external_data_types:
            if data_type in query_lower:
                return True

        # Check general knowledge patterns
        for pattern in general_knowledge:
            if query_lower.startswith(pattern):
                return True

        return False

    def _determine_recommendation(
        self, confidence: float, context_count: int, query: str = ""
    ) -> Tuple[RecommendationAction, str]:
        """
        Determine recommendation based on confidence, context count, and query.

        Args:
            confidence: Confidence score (0.0-1.0)
            context_count: Number of contexts
            query: Original user query (for EXTERNAL detection)

        Returns:
            Tuple of (RecommendationAction, reasoning)

        Decision tree:
        1. If context_count == 0:
           - If query looks like external request → EXTERNAL
           - Else → REFINE

        2. If confidence >= threshold["excellent"]:
           - return ANSWER

        3. If confidence >= threshold["good"]:
           - return ANSWER

        4. If confidence >= threshold["partial"]:
           - return REFINE

        5. Else (confidence < partial):
           - If query looks like external request → EXTERNAL
           - Else → CLARIFY
        """
        # No contexts found
        if context_count == 0:
            if self._is_external_query(query):
                return (
                    RecommendationAction.EXTERNAL,
                    "No contexts found - query appears to need external/real-time information",
                )
            return (
                RecommendationAction.REFINE,
                "No contexts found - query refinement recommended",
            )

        # High confidence - answer directly
        if confidence >= self.config.thresholds["excellent"]:
            return (
                RecommendationAction.ANSWER,
                "High confidence - contexts directly answer query",
            )
        if confidence >= self.config.thresholds["good"]:
            return (
                RecommendationAction.ANSWER,
                "Good confidence - contexts provide sufficient information",
            )

        # Partial confidence - try refinement
        if confidence >= self.config.thresholds["partial"]:
            return (
                RecommendationAction.REFINE,
                "Partial confidence - query refinement may help",
            )

        # Low confidence - check if external or clarify
        if self._is_external_query(query):
            return (
                RecommendationAction.EXTERNAL,
                "Low confidence - query appears to need external/real-time information",
            )
        return (
            RecommendationAction.CLARIFY,
            "Low confidence - query may be ambiguous or out of scope",
        )

    def _extract_relevance_scores(self, contexts: List[Dict[str, Any]]) -> List[float]:
        """
        Extract relevance scores from contexts.

        Uses consistent scoring across all contexts to avoid mixing score types:
        - If reranker was used (any context has rerank != 0), use rerank for ALL
        - Else if hybrid was used (any context has hybrid != 0), use hybrid for ALL
        - Else use sem_sim for ALL

        This ensures we don't mix different score scales in the same evaluation.

        Args:
            contexts: List of context dictionaries

        Returns:
            List of relevance scores (0.0-1.0)
        """
        if not contexts:
            return []

        # Detect which scoring method was used by checking if ANY context has the score
        # This ensures consistent scoring across all contexts
        has_rerank = any(c.get("rerank", 0.0) != 0.0 for c in contexts)
        has_hybrid = any(c.get("hybrid", 0.0) != 0.0 for c in contexts)

        scores = []
        for c in contexts:
            if has_rerank:
                # Use rerank for all contexts (even if some are 0.0)
                score = c.get("rerank", 0.0)
            elif has_hybrid:
                # Use hybrid for all contexts (even if some are 0.0)
                score = c.get("hybrid", 0.0)
            else:
                # Fall back to semantic similarity
                # NOTE: sem_sim should always exist. If missing, default to 0.0 (not 0.5)
                # to avoid artificially boosting missing data
                score = c.get("sem_sim", 0.0)

            # Clamp to [0.0, 1.0] range
            scores.append(max(0.0, min(1.0, score)))

        return scores


# =============================================================================
# TESTING BLOCK - Run: python -m src.services.retrieval_evaluator
# =============================================================================

if __name__ == "__main__":
    """Test the retrieval evaluator."""
    from src.config.settings import Config

    print("=" * 70)
    print("TESTING RETRIEVAL EVALUATOR")
    print("=" * 70)

    # Load config
    config = ReflectionConfig.from_settings(Config)
    print(f"\nConfig loaded: mode={config.mode.value}")

    # Test 1: Create evaluator (FAST mode, no LLM needed)
    print("\n[Test 1] Creating evaluator (FAST mode)...")
    try:
        config.mode = ReflectionMode.FAST
        evaluator = RetrievalEvaluator(config=config, openai_client=None)
        print("  [OK] Evaluator created")
    except Exception as e:
        print(f"  [FAIL] {e}")

    # Test 2: Evaluate with good contexts
    print("\n[Test 2] Evaluating good retrieval...")
    try:
        criteria = EvaluationCriteria(
            query="What is the company vacation policy?",
            contexts=[
                {
                    "chunk": "The company provides 15 days of vacation per year. Employees can accrue vacation time...",
                    "score": 0.92,
                },
                {
                    "chunk": "Vacation policy details: All employees are eligible for paid vacation after 90 days...",
                    "score": 0.88,
                },
            ],
            search_metadata={"hybrid": True, "top_k": 5},
        )

        result = evaluator.evaluate(criteria)
        print(f"  Quality: {result.quality.value}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Recommendation: {result.recommendation.value}")
        print(f"  Reasoning: {result.reasoning}")
        print("  [OK] Evaluation complete")
    except Exception as e:
        print(f"  [FAIL] {e}")

    # Test 3: Evaluate with poor contexts
    print("\n[Test 3] Evaluating poor retrieval...")
    try:
        criteria = EvaluationCriteria(
            query="What is quantum computing?",
            contexts=[
                {
                    "chunk": "The company has various employee benefits including health insurance...",
                    "score": 0.15,
                }
            ],
            search_metadata={"hybrid": True, "top_k": 5},
        )

        result = evaluator.evaluate(criteria)
        print(f"  Quality: {result.quality.value}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Recommendation: {result.recommendation.value}")
        print("  [OK] Evaluation complete")
    except Exception as e:
        print(f"  [FAIL] {e}")

    print("\n" + "=" * 70)
    print("TESTS COMPLETE!")
    print("=" * 70)
    print("\nNext: Implement the TODO methods marked in the code")
