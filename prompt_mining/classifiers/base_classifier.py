"""
Base classifier protocol and shared evaluation utilities.

All classifiers (DANN, LogisticRegression, XGBoost, etc.) implement
ClassifierProtocol to enable unified evaluation via LODO.
"""
from typing import Protocol, Dict, Any, Optional, Union, runtime_checkable
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import (
    precision_recall_curve, f1_score, accuracy_score,
    roc_auc_score, auc
)

from prompt_mining.classifiers.threshold_strategy import (
    ThresholdStrategy,
    TargetPrecisionThreshold,
)


@runtime_checkable
class ClassifierProtocol(Protocol):
    """
    Interface that all classifiers must implement.

    This enables generic LODO evaluation across different classifier types
    (DANN, LogisticRegression, XGBoost, etc.).

    Example:
        >>> class MyClassifier:
        ...     def fit(self, X, y, **kwargs): ...
        ...     def predict_scores(self, X): ...
        ...     def clone(self): ...
        >>>
        >>> clf = MyClassifier()
        >>> assert isinstance(clf, ClassifierProtocol)
    """

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs
    ) -> "ClassifierProtocol":
        """
        Train the classifier.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            sample_weight: Optional sample weights
            **kwargs: Classifier-specific arguments (e.g., datasets for DANN)

        Returns:
            self (for method chaining)
        """
        ...

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Return continuous prediction scores.

        These scores are used for threshold selection and ROC/PR curves.
        Higher scores indicate higher confidence in positive class.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Scores array (n_samples,) - values typically in [0, 1] or unbounded
        """
        ...

    def clone(self) -> "ClassifierProtocol":
        """
        Create a fresh copy of this classifier (unfitted).

        Used by LODO to create independent classifiers per fold.
        """
        ...


@dataclass
class EvaluationResult:
    """Results from classifier evaluation."""
    accuracy: float
    malicious_f1: float  # F1 for positive class (%)
    benign_f1: float     # F1 for negative class (%)
    macro_f1: float      # Macro-averaged F1 (%)
    roc_auc: float       # ROC AUC (%)
    pr_auc: float        # Precision-Recall AUC (%)
    threshold: float     # Selected decision threshold
    recall: float        # Recall at threshold
    precision: float     # Precision at threshold

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization."""
        return {
            'acc': self.accuracy,
            'malicious_f1': self.malicious_f1,
            'benign_f1': self.benign_f1,
            'macro_f1': self.macro_f1,
            'roc_auc': self.roc_auc,
            'pr_auc': self.pr_auc,
            'threshold': self.threshold,
            'recall': self.recall,
            'precision': self.precision,
        }


def evaluate_predictions(
    train_scores: np.ndarray,
    y_train: np.ndarray,
    test_scores: np.ndarray,
    y_test: np.ndarray,
    threshold_strategy: Optional[ThresholdStrategy] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate classifier predictions with threshold selection on train set.

    Works with any model that produces scores. Threshold is selected on
    train_scores using the provided strategy, avoiding test set leakage.

    Args:
        train_scores: Prediction scores on training set (for threshold selection)
        y_train: True labels for training set
        test_scores: Prediction scores on test set
        y_test: True labels for test set
        threshold_strategy: Strategy for threshold selection. Defaults to
                           TargetPrecisionThreshold(0.95) if not provided.
        verbose: Print threshold and F1 info

    Returns:
        Dictionary with evaluation metrics:
        - acc: Accuracy
        - malicious_f1: F1 for malicious class (%)
        - benign_f1: F1 for benign class (%)
        - macro_f1: Macro-averaged F1 (%)
        - roc_auc: ROC AUC score (%)
        - pr_auc: Precision-Recall AUC (%)
        - threshold: Selected threshold
        - recall: Recall at threshold
        - precision: Precision at threshold

    Example:
        >>> from prompt_mining.classifiers.threshold_strategy import MaxF1Threshold
        >>> train_scores = clf.predict_scores(X_train)
        >>> test_scores = clf.predict_scores(X_test)
        >>> results = evaluate_predictions(
        ...     train_scores, y_train, test_scores, y_test,
        ...     threshold_strategy=MaxF1Threshold()
        ... )
        >>> print(f"Test F1: {results['malicious_f1']:.1f}%")
    """
    # Default to target precision strategy for backward compatibility
    if threshold_strategy is None:
        threshold_strategy = TargetPrecisionThreshold(precision=0.95)

    # Select threshold using the strategy
    best_t = threshold_strategy.select_threshold(train_scores, y_train)

    train_preds = (train_scores >= best_t).astype(int)
    if verbose:
        print(f"Threshold ({threshold_strategy.name}): {best_t:.4f}, "
              f"Train F1: {f1_score(y_train, train_preds)*100:.1f}%")

    # Evaluate on test set
    preds = (test_scores >= best_t).astype(int)
    acc = accuracy_score(y_test, preds)

    prec_test, rec_test, thresh_test = precision_recall_curve(y_test, test_scores)
    best_idx = np.argmin(np.abs(thresh_test - best_t)) if len(thresh_test) > 0 else 0

    return {
        'acc': acc,
        'malicious_f1': f1_score(y_test, preds, pos_label=1) * 100,
        'benign_f1': f1_score(y_test, preds, pos_label=0) * 100,
        'macro_f1': f1_score(y_test, preds, average='macro') * 100,
        'roc_auc': roc_auc_score(y_test, test_scores) * 100,
        'pr_auc': auc(rec_test, prec_test) * 100,
        'threshold': best_t,
        'recall': rec_test[best_idx] if len(rec_test) > best_idx else 0.0,
        'precision': prec_test[best_idx] if len(prec_test) > best_idx else 0.0,
    }
