"""
Centralized prompt management for the RAG system.

This module provides a single source of truth for all LLM prompts used across:
- Planning (query decomposition)
- Generation (answer creation)
- Evaluation (quality assessment)
- Refinement (query improvement)
- Tool calling (web_search, code_execution, etc.)
"""

from src.prompts.planning import PlanningPrompts
from src.prompts.generation import GenerationPrompts
from src.prompts.evaluation import EvaluationPrompts
from src.prompts.refinement import RefinementPrompts
from src.prompts.tools import ToolPrompts

__all__ = [
    "PlanningPrompts",
    "GenerationPrompts",
    "EvaluationPrompts",
    "RefinementPrompts",
    "ToolPrompts",
]
