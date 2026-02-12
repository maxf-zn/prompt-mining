"""
Shared data types used across prompt_mining modules.
"""
from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass
class FeatureInfo:
    """
    Information about a feature (SAE or raw activation).

    Used by both InferencePipeline (for influence analysis) and
    SAEFeatureInterpreter (for interpretations).
    """
    feature_idx: int
    """Index of the feature."""

    # From classifier analysis (populated by InferencePipeline)
    coefficient: Optional[float] = None
    """Classifier coefficient for this feature."""

    activation: Optional[float] = None
    """Feature activation value for the input."""

    contribution: Optional[float] = None
    """Influence score (coefficient * activation). Positive = toward malicious."""

    # Interpretation (populated by SAEFeatureInterpreter)
    interpretation: Optional[str] = None
    """Human-readable description of what this feature detects."""

    interpretation_source: Optional[str] = None
    """Source of interpretation: 'neuronpedia' or 'llm'."""

    # Neuronpedia-specific (populated by SAEFeatureInterpreter)
    neuronpedia_url: Optional[str] = None
    """URL to view this feature on Neuronpedia."""

    auto_interp: Optional[str] = None
    """Original auto-interpretation from Neuronpedia."""

    top_activations: Optional[List[Dict]] = None
    """Top activating examples from Neuronpedia."""