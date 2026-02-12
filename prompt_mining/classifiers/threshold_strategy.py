"""
Threshold selection strategies for classifier evaluation.

Provides a clean abstraction for different threshold selection methods:
- ConstantThreshold: Fixed threshold value
- TargetPrecisionThreshold: Achieve target precision on benign class
- MaxF1Threshold: Maximize F1 score

Also includes CVConfig for controlling cross-validation score generation.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union
import numpy as np


class ThresholdStrategy(ABC):
    """
    Base class for threshold selection strategies.

    Subclasses implement different methods for selecting a decision
    threshold from training scores and labels.

    Example:
        >>> strategy = TargetPrecisionThreshold(precision=0.95)
        >>> threshold = strategy.select_threshold(train_scores, y_train)
    """

    @abstractmethod
    def select_threshold(self, scores: np.ndarray, y: np.ndarray) -> float:
        """
        Select threshold from training scores and labels.

        Args:
            scores: Prediction scores on training set
            y: True labels for training set

        Returns:
            Selected threshold value
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for the strategy."""
        ...


@dataclass
class ConstantThreshold(ThresholdStrategy):
    """
    Use a fixed threshold value.

    Useful when you have a known good threshold from prior analysis
    or want to use a standard value like 0.5.

    Example:
        >>> strategy = ConstantThreshold(0.4)
        >>> threshold = strategy.select_threshold(scores, y)  # Always returns 0.4
    """
    value: float = 0.5

    def select_threshold(self, scores: np.ndarray, y: np.ndarray) -> float:
        return self.value

    @property
    def name(self) -> str:
        return f"constant({self.value})"


@dataclass
class TargetPrecisionThreshold(ThresholdStrategy):
    """
    Select threshold to achieve target precision (controls FPR).

    Uses the percentile of benign (negative class) scores to find
    a threshold that achieves the target precision. For example,
    precision=0.95 means 95% of benign samples will be correctly
    classified as benign.

    This is useful when you want to control false positive rate.

    Example:
        >>> strategy = TargetPrecisionThreshold(precision=0.95)
        >>> threshold = strategy.select_threshold(scores, y)
    """
    precision: float = 0.95

    def select_threshold(self, scores: np.ndarray, y: np.ndarray) -> float:
        benign_scores = scores[y == 0]
        if len(benign_scores) > 0:
            return float(np.percentile(benign_scores, 100 * self.precision))
        # Fallback to median if no benign samples
        return float(np.median(scores))

    @property
    def name(self) -> str:
        return f"precision({self.precision})"


@dataclass
class MaxF1Threshold(ThresholdStrategy):
    """
    Select threshold that maximizes F1 score on training data.

    Searches all possible thresholds from the precision-recall curve
    and selects the one with highest F1 score.

    Example:
        >>> strategy = MaxF1Threshold()
        >>> threshold = strategy.select_threshold(scores, y)
    """

    def select_threshold(self, scores: np.ndarray, y: np.ndarray) -> float:
        from sklearn.metrics import precision_recall_curve

        prec, rec, thresh = precision_recall_curve(y, scores)
        # F1 = 2 * precision * recall / (precision + recall)
        f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-10)

        if len(f1s) == 0:
            return float(np.median(scores))

        return float(thresh[f1s.argmax()])

    @property
    def name(self) -> str:
        return "max_f1"


@dataclass
class CVConfig:
    """
    Configuration for cross-validation score generation.

    When enabled, uses cross-validation to generate out-of-fold
    predictions for threshold selection, which avoids overfitting
    the threshold to the training data.

    Attributes:
        enabled: Whether to use CV for score generation
        folds: Number of CV folds (default 3)
        n_jobs: Number of parallel jobs for CV (default 8)

    Example:
        >>> cv = CVConfig(enabled=True, folds=5, n_jobs=4)
        >>> results = lodo_evaluate(clf, X, y, datasets, cv=cv)
    """
    enabled: bool = False
    folds: int = 3
    n_jobs: int = 8


# Type alias for threshold strategy parameter
ThresholdStrategyType = Union[ThresholdStrategy, None]
