"""
Inference pipeline for prompt classification.

Provides end-to-end prompt → prediction flow with pluggable
feature extraction and classification components.

Example:
    >>> from prompt_mining.pipeline import (
    ...     InferencePipeline,
    ...     SAEFeatureExtractor,
    ...     RawActivationExtractor,
    ... )
    >>> from prompt_mining.classifiers import LinearClassifier
    >>>
    >>> # SAE-based classification
    >>> extractor = SAEFeatureExtractor(layer=27)
    >>> pipeline = InferencePipeline(
    ...     model_wrapper=model,
    ...     feature_extractor=extractor,
    ...     classifier=trained_clf,
    ...     threshold=0.85,
    ... )
    >>> result = pipeline.classify("How do I hack a computer?")
    >>> print(f"Malicious: {result.is_malicious}, Score: {result.score:.3f}")
"""
from prompt_mining.pipeline.inference_pipeline import (
    InferencePipeline,
    ClassificationResult,
)
from prompt_mining.pipeline.feature_extractors import (
    FeatureExtractorProtocol,
    PositionType,
    RawActivationExtractor,
    SAEFeatureExtractor,
)
from prompt_mining.types import FeatureInfo

__all__ = [
    "InferencePipeline",
    "ClassificationResult",
    "FeatureInfo",
    "FeatureExtractorProtocol",
    "PositionType",
    "RawActivationExtractor",
    "SAEFeatureExtractor",
]
