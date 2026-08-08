"""Voluntas: a BDI (Belief-Desire-Intention) agent framework.

A modular BDI agent implementation built on Pydantic AI, with support for:
- Belief management and extraction
- Desire-driven planning
- Single-stage intention generation
- Step-by-step execution with outcome analysis
- Plan reconsideration and monitoring

Main exports:
- BDI: The main agent class
- Belief, BeliefSet, Desire, DesireStatus, Intention, Plan: Core schemas
"""

__version__ = "0.1.1"

# Main agent class
from voluntas.agent import BDI
from voluntas.usage import BDIUsageTracker

# Core schemas (most commonly used)
from voluntas.schemas import (
    Belief,
    BeliefSet,
    Desire,
    DesireStatus,
    Intention,
    Plan,
    PlanStatus,
    PlanStep,
    PlanStepHistory,
    # Also export other useful schemas
    PlanningDecision,
    ReconsiderResult,
    BeliefExtractionResult,
)

__all__ = [
    # Agent
    "BDI",
    "BDIUsageTracker",
    "__version__",
    # Core schemas
    "Belief",
    "BeliefSet",
    "Desire",
    "DesireStatus",
    "Intention",
    "Plan",
    "PlanStatus",
    "PlanStep",
    "PlanStepHistory",
    # Additional schemas
    "PlanningDecision",
    "ReconsiderResult",
    "BeliefExtractionResult",
]
