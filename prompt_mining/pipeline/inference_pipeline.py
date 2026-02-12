"""
Inference pipeline for prompt classification.

Orchestrates feature extraction and classification without coupling
to specific feature types or classifier implementations.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Literal

from prompt_mining.model import ModelWrapper
from prompt_mining.classifiers.base_classifier import ClassifierProtocol
from prompt_mining.pipeline.feature_extractors import FeatureExtractorProtocol
from prompt_mining.types import FeatureInfo


@dataclass
class ClassificationResult:
    """Result from pipeline classification."""
    score: float
    """Raw score from classifier (typically probability in [0, 1])."""

    prediction: int
    """Binary prediction (0 or 1)."""

    is_malicious: bool
    """Human-readable prediction (True if malicious)."""

    features: np.ndarray
    """Extracted features (for downstream analysis/interpretation)."""


class InferencePipeline:
    """
    End-to-end inference pipeline for prompt classification.

    Orchestrates feature extraction and classification without coupling
    to specific feature types or classifier implementations.

    Example:
        >>> from prompt_mining.pipeline import (
        ...     InferencePipeline, SAEFeatureExtractor, RawActivationExtractor
        ... )
        >>> from prompt_mining.classifiers import LinearClassifier
        >>>
        >>> # SAE-based classification
        >>> extractor = SAEFeatureExtractor(layer=27, position='last')
        >>> pipeline = InferencePipeline(
        ...     model_wrapper=model,
        ...     feature_extractor=extractor,
        ...     classifier=trained_clf,
        ...     threshold=0.85,
        ... )
        >>> result = pipeline.classify("How do I hack a computer?")
        >>> print(f"Malicious: {result.is_malicious}, Score: {result.score:.3f}")
        Malicious: True, Score: 0.923
        >>>
        >>> # Raw activation classification with DANN
        >>> extractor = RawActivationExtractor(layer=19)
        >>> pipeline = InferencePipeline(
        ...     model_wrapper=model,
        ...     feature_extractor=extractor,
        ...     classifier=dann_clf,
        ...     threshold=0.5,
        ... )
        >>> result = pipeline.classify("What's the weather today?")
        >>> print(f"Malicious: {result.is_malicious}")
        Malicious: False
    """

    def __init__(
        self,
        model_wrapper: ModelWrapper,
        feature_extractor: FeatureExtractorProtocol,
        classifier: ClassifierProtocol,
        threshold: float = 0.5,
        apply_chat_template: bool = True,
    ):
        """
        Initialize the inference pipeline.

        Args:
            model_wrapper: Loaded ModelWrapper for running forward passes
            feature_extractor: Extractor to convert prompts to feature vectors
            classifier: Trained classifier implementing ClassifierProtocol
            threshold: Decision threshold for binary prediction (default 0.5)
            apply_chat_template: Whether to wrap prompt in chat template (default True)
        """
        self.model_wrapper = model_wrapper
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.threshold = threshold
        self.apply_chat_template = apply_chat_template

    def classify(self, prompt: str) -> ClassificationResult:
        """
        Classify a single prompt.

        Args:
            prompt: Text prompt to classify (raw user message)

        Returns:
            ClassificationResult with score, prediction, and extracted features
        """
        # Apply chat template if enabled
        if self.apply_chat_template:
            prompt = self.model_wrapper.apply_chat_template(
                [{"role": "user", "content": prompt}]
            )

        # Extract features
        features = self.feature_extractor.extract(prompt, self.model_wrapper)

        # Classify (reshape to 2D for sklearn-style classifiers)
        score = self.classifier.predict_scores(features.reshape(1, -1))[0]
        prediction = int(score >= self.threshold)

        return ClassificationResult(
            score=float(score),
            prediction=prediction,
            is_malicious=bool(prediction),
            features=features,
        )

    def get_feature_influence(self, result: ClassificationResult) -> np.ndarray:
        """
        Compute feature influence scores for a classification result.

        Influence = coefficient * activation. Positive values push toward
        malicious, negative toward benign.

        Args:
            result: ClassificationResult from classify()

        Returns:
            1D array of influence scores, same shape as result.features

        Raises:
            AttributeError: If classifier doesn't have coef_ property
        """
        if not hasattr(self.classifier, 'coef_'):
            raise AttributeError(
                f"Classifier {type(self.classifier).__name__} doesn't have coef_ property. "
                "Feature influence requires a linear classifier."
            )
        coef = self.classifier.coef_.flatten()
        return coef * result.features

    def get_top_influential_features(
        self,
        result: ClassificationResult,
        top_k: int = 10,
        direction: Literal['positive', 'negative', 'abs'] = 'positive',
    ) -> List[FeatureInfo]:
        """
        Get the most influential features for a classification result.

        Args:
            result: ClassificationResult from classify()
            top_k: Number of top features to return
            direction: Which features to return:
                - 'positive': highest positive influence (toward malicious)
                - 'negative': most negative influence (toward benign)
                - 'abs': highest absolute influence (either direction)

        Returns:
            List of FeatureInfo with coefficient, activation, and contribution populated

        Example:
            >>> result = pipeline.classify("How do I hack?")
            >>> top_features = pipeline.get_top_influential_features(result, top_k=5)
            >>> for f in top_features:
            ...     print(f"Feature {f.feature_idx}: {f.contribution:+.2f} = {f.coefficient:+.2f} × {f.activation:.2f}")
        """
        coef = self.classifier.coef_.flatten()
        influence = coef * result.features

        if direction == 'positive':
            indices = np.argsort(influence)[-top_k:][::-1]
        elif direction == 'negative':
            indices = np.argsort(influence)[:top_k]
        else:  # abs
            indices = np.argsort(np.abs(influence))[-top_k:][::-1]

        return [
            FeatureInfo(
                feature_idx=int(idx),
                coefficient=float(coef[idx]),
                activation=float(result.features[idx]),
                contribution=float(influence[idx]),
            )
            for idx in indices
        ]

    def __repr__(self) -> str:
        return (
            f"InferencePipeline(\n"
            f"  extractor={self.feature_extractor},\n"
            f"  classifier={self.classifier},\n"
            f"  threshold={self.threshold},\n"
            f"  apply_chat_template={self.apply_chat_template}\n"
            f")"
        )
