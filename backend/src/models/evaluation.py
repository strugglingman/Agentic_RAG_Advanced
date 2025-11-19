"""
Data models for retrieval quality evaluation (Self-Reflection System).

TODO: Implement these dataclasses to define the structure for evaluation.

This module should define:
1. EvaluationCriteria - Input: what we're evaluating (query + contexts)
2. EvaluationResult - Output: quality assessment + recommendation
3. ReflectionConfig - Runtime settings for evaluation

Week 1 Focus: Pure evaluation, no action-taking yet.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


# ============================================================================
# STEP 1: Define Enums
# ============================================================================
# These to define the retrieval quality levels
class QualityLevel(str, Enum):
    """Quality levels for retrieval evaluation."""
    EXCELLENT = "excellent"
    GOOD = "good"
    PARTIAL = "partial"
    POOR = "poor"

# These to define the action recommended based on the result of evaluation
class RecommendationAction(str, Enum):
    ANSWER = "answer"
    REFINE = "refine"
    EXTERNAL = "external"
    CLARIFY = "clarify"

# These control evaluation speed vs accuracy tradeoff
class ReflectionMode(str , Enum):
    FAST = "fast"
    BALANCED = "balanced"
    THOROUGH = "thorough"

# ============================================================================
# STEP 2: Define Input Model (EvaluationCriteria)
# ============================================================================

# TODO: Create EvaluationCriteria dataclass
#
# Purpose: Package everything needed to evaluate retrieval quality
#
# Required fields:
#   - query: str (the user's question)
#   - contexts: List[Dict] (retrieved chunks from search_documents)
#   - search_metadata: Optional[Dict] (scores, hybrid info, etc.)
#   - mode: Optional[ReflectionMode] (which evaluation mode to use)
#
# Guidelines:
#   - Use @dataclass decorator
#   - Add __post_init__ to validate query is not empty
#   - Contexts can be empty list (valid case - no results found)
#
# Example usage:
#   criteria = EvaluationCriteria(
#       query="What was Q1 revenue?",
#       contexts=[{"chunk": "...", "score": 0.89}],
#       search_metadata={"hybrid": True}
#   )
@dataclass
class EvaluationCriteria:
    """Data class for evaluation criteria."""
    query: str
    contexts: List[Dict[str, Any]] = field(default_factory=list)
    search_metadata: Optional[Dict[str, Any]] = None
    mode: Optional[ReflectionMode] = ReflectionMode.BALANCED

    def __post_init__(self):
        if not self.query or not self.query.strip():
            raise ValueError("Query must be a non-empty string.")


# ============================================================================
# STEP 3: Define Output Model (EvaluationResult)
# ============================================================================

# TODO: Create EvaluationResult dataclass
#
# Purpose: Return evaluation results + recommendation
#
# Required fields:
#   Core assessment:
#   - quality: QualityLevel (EXCELLENT/GOOD/PARTIAL/POOR)
#   - confidence: float (0.0-1.0 overall confidence score)
#   - coverage: float (0.0-1.0 how well query is covered)
#
#   Details:
#   - relevance_scores: List[float] (score per context)
#   - issues: List[str] (problems detected, e.g., "context too generic")
#   - missing_aspects: List[str] (what's not covered)
#
#   Recommendation:
#   - recommendation: RecommendationAction (what to do next)
#   - reasoning: str (explain why this recommendation)
#
#   Metadata:
#   - metrics: Dict (raw scores for debugging)
#   - timestamp: datetime (when evaluation happened)
#   - mode_used: ReflectionMode (which mode was used)
#
# Methods to add:
#   - to_dict() -> Dict: Convert to JSON-serializable dict
#   - Properties: should_answer, should_refine, should_search_external
#
# Validation in __post_init__:
#   - Confidence must be 0.0-1.0
#   - Coverage must be 0.0-1.0
#   - All relevance_scores must be 0.0-1.0
@dataclass
class EvaluationResult:
    """Data class for evaluation result."""
    quality: QualityLevel
    confidence: float
    coverage: float
    relevance_scores: list[float] = field(default_factory=list)

# ============================================================================
# STEP 4: Define Configuration Model (ReflectionConfig)
# ============================================================================

# TODO: Create ReflectionConfig dataclass
#
# Purpose: Runtime configuration for evaluation (overrides Config settings)
#
# Required fields:
#   - mode: ReflectionMode (FAST/BALANCED/THOROUGH)
#   - thresholds: Dict[str, float] (quality thresholds)
#       Example: {"excellent": 0.85, "good": 0.70, "partial": 0.50}
#   - auto_refine: bool (auto-trigger refinement? Week 2+)
#   - auto_external: bool (auto-trigger external search? Week 3+)
#   - min_contexts: int (minimum contexts needed)
#   - max_refinement_attempts: int (prevent infinite loops)
#
# Methods to add:
#   - from_settings(config_class) -> ReflectionConfig
#     Load values from src.config.settings.Config
#
#   - get_quality_level(confidence: float) -> QualityLevel
#     Map confidence score to quality level using thresholds
#     Example: 0.88 -> EXCELLENT, 0.75 -> GOOD, 0.45 -> POOR
#
# Default thresholds should match Config defaults:
#   excellent: 0.85, good: 0.70, partial: 0.50


# ============================================================================
# IMPLEMENTATION GUIDELINES
# ============================================================================

"""
IMPLEMENTATION ORDER:
1. First, define the 3 enums (QualityLevel, RecommendationAction, ReflectionMode)
2. Then, define EvaluationCriteria (input model)
3. Then, define EvaluationResult (output model)
4. Finally, define ReflectionConfig (settings model)

TESTING:
- After implementing, run: python -m src.models.evaluation
- Add a __main__ block with example usage
- Create sample instances to verify structure

VALIDATION:
- Use __post_init__ to validate ranges (0-1 for scores)
- Raise ValueError with clear messages if invalid

DOCUMENTATION:
- Add docstrings to each class explaining purpose
- Add type hints to all fields
- Include usage examples in docstrings

QUESTIONS TO ASK:
- Is my dataclass structure correct?
- Are the validation rules appropriate?
- Should I add more helper methods?
"""


# ============================================================================
# YOUR IMPLEMENTATION STARTS HERE
# ============================================================================

# Step 1: Define enums here


# Step 2: Define EvaluationCriteria here


# Step 3: Define EvaluationResult here


# Step 4: Define ReflectionConfig here


# ============================================================================
# TESTING BLOCK (optional, for your testing)
# ============================================================================

if __name__ == "__main__":
    """
    Test your implementations here.

    Example tests:
    1. Create a sample EvaluationCriteria
    2. Create a sample EvaluationResult
    3. Test to_dict() method
    4. Test quality level mapping with different scores
    """
    print("Testing evaluation models...")

    # TODO: Add your tests here

    print("Tests complete!")
