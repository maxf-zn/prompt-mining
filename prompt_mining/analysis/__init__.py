"""
Analysis module for prompt mining platform.

This module provides exploratory analysis tools:
- AnalysisData: Unified data accessor for loading feature/activation matrices
- Analyzer: Abstract base class for all analyzers
- ComparisonAnalyzer: Frequency and co-occurrence analysis
- DirectionsAnalyzer: Steering vector computation
- UnsupervisedAnalyzer: PCA and clustering

For classification, use prompt_mining.classifiers instead:
    >>> from prompt_mining.classifiers import (
    ...     ClassificationDataset,
    ...     LinearClassifier,
    ...     DANNClassifier,
    ...     lodo_evaluate,
    ... )

Example Usage:
    >>> from prompt_mining.analysis import (
    ...     AnalysisData,
    ...     ComparisonAnalyzer,
    ...     DirectionsAnalyzer,
    ...     UnsupervisedAnalyzer
    ... )
    >>> from prompt_mining.registry import SQLiteRegistry
    >>> from prompt_mining.storage import LocalStorage
    >>>
    >>> # Setup
    >>> registry = SQLiteRegistry("registry.sqlite")
    >>> storage = LocalStorage("file:///path/to/storage")
    >>>
    >>> # Load data
    >>> success_data = AnalysisData(
    ...     registry, storage,
    ...     filters={'dataset_id': 'injecagent', 'labels': {'success': True}}
    ... )
    >>>
    >>> # Run comparison analysis
    >>> comparison = ComparisonAnalyzer(top_k=200)
    >>> results = comparison.run(success_data)
    >>> comparison.save_results(results, Path("output/comparison"))
"""

from prompt_mining.analysis.base import (
    ActivationMatrix,
    AnalysisData,
    Analyzer,
    FeatureMatrix,
)
from prompt_mining.analysis.comparison import ComparisonAnalyzer
from prompt_mining.analysis.directions import DirectionsAnalyzer
from prompt_mining.analysis.unsupervised import UnsupervisedAnalyzer
from prompt_mining.analysis.attention_inspector import AttentionInspector, AttentionRow
from prompt_mining.analysis.sae_feature_interpreter import (
    LLMInterpreter,
    LLMStrategy,
    NeuronpediaStrategy,
    SAEFeatureInterpreter,
)
from prompt_mining.types import FeatureInfo


__all__ = [
    # Base classes
    "AnalysisData",
    "Analyzer",
    "FeatureMatrix",
    "ActivationMatrix",
    # Analyzers
    "ComparisonAnalyzer",
    "DirectionsAnalyzer",
    "UnsupervisedAnalyzer",
    # Feature interpretation
    "SAEFeatureInterpreter",
    "FeatureInfo",
    "LLMStrategy",
    "NeuronpediaStrategy",
    "LLMInterpreter",
    # Attention utilities
    "AttentionInspector",
    "AttentionRow",
]
