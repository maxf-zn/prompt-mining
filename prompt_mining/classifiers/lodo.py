"""
Leave-One-Dataset-Out (LODO) evaluation protocol.

LODO tests classifier generalization by training on N-1 datasets
and evaluating on the held-out dataset, rotating through all datasets.
This is the standard evaluation protocol for cross-dataset generalization.
"""
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
from sklearn.model_selection import cross_val_predict, KFold

from prompt_mining.classifiers.base_classifier import ClassifierProtocol, evaluate_predictions
from prompt_mining.classifiers.threshold_strategy import (
    ThresholdStrategy,
    TargetPrecisionThreshold,
    CVConfig,
)
from joblib.externals.loky import get_reusable_executor

@dataclass
class LODOResult:
    """
    Results from LODO evaluation.

    Attributes:
        per_dataset: Dict mapping dataset name to evaluation metrics
        weighted_average: Sample-weighted average metrics across all datasets
    """
    per_dataset: Dict[str, Dict[str, float]]
    weighted_average: Dict[str, float]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = ["LODO Results:", "=" * 50]

        for ds, metrics in sorted(self.per_dataset.items()):
            lines.append(
                f"  {ds:25s}: acc={metrics['acc']:.1%}, "
                f"F1={metrics['malicious_f1']:.1f}%, "
                f"AUC={metrics['roc_auc']:.1f}%"
            )

        lines.append("-" * 50)
        lines.append(
            f"  {'Weighted Average':25s}: acc={self.weighted_average['acc']:.1%}, "
            f"F1={self.weighted_average['malicious_f1']:.1f}%, "
            f"AUC={self.weighted_average['roc_auc']:.1f}%"
        )

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'per_dataset': self.per_dataset,
            'weighted_average': self.weighted_average,
        }


def lodo_evaluate(
    classifier: ClassifierProtocol,
    X: np.ndarray,
    y: np.ndarray,
    datasets: np.ndarray,
    threshold_strategy: Optional[ThresholdStrategy] = None,
    cv: Optional[CVConfig] = None,
    merge_datasets: Optional[Dict[str, str]] = None,
    verbose: bool = True,
) -> LODOResult:
    """
    Run Leave-One-Dataset-Out evaluation with any classifier.

    For each dataset D:
    1. Train classifier on all datasets except D
    2. Evaluate on D using threshold selected from training data
    3. Record metrics

    Args:
        classifier: Any classifier implementing ClassifierProtocol
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        datasets: Dataset IDs (n_samples,)
        threshold_strategy: Strategy for threshold selection. Defaults to
                           TargetPrecisionThreshold(0.95).
        cv: Cross-validation config for score generation. If cv.enabled=True,
            uses out-of-fold predictions for threshold selection (recommended
            for sklearn classifiers). Defaults to CVConfig(enabled=False).
        merge_datasets: Optional dict mapping dataset names to merge targets
                       e.g., {'gandalf_summarization': 'mosscap'}
        verbose: Print progress

    Returns:
        LODOResult with per-dataset and weighted average metrics

    Example:
        >>> from prompt_mining.classifiers import LinearClassifier, lodo_evaluate
        >>> from prompt_mining.classifiers.threshold_strategy import (
        ...     ConstantThreshold, MaxF1Threshold, CVConfig
        ... )
        >>>
        >>> clf = LinearClassifier()
        >>>
        >>> # Use CV with default precision-based threshold
        >>> results = lodo_evaluate(clf, X, y, datasets, cv=CVConfig(enabled=True))
        >>>
        >>> # Use constant threshold
        >>> results = lodo_evaluate(clf, X, y, datasets,
        ...     threshold_strategy=ConstantThreshold(0.4))
        >>>
        >>> # Use max F1 threshold with CV
        >>> results = lodo_evaluate(clf, X, y, datasets,
        ...     threshold_strategy=MaxF1Threshold(),
        ...     cv=CVConfig(enabled=True, folds=5))
    """
    # Apply defaults
    if threshold_strategy is None:
        threshold_strategy = TargetPrecisionThreshold(precision=0.95)
    if cv is None:
        cv = CVConfig(enabled=False)

    datasets = datasets.copy()

    # Merge small datasets if specified
    if merge_datasets:
        for src, dst in merge_datasets.items():
            datasets[datasets == src] = dst

    unique_datasets = np.unique(datasets)
    per_dataset_results = {}

    if verbose:
        print(f"Threshold strategy: {threshold_strategy.name}, CV: {cv.enabled}")

    for test_ds in unique_datasets:
        test_mask = datasets == test_ds
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        datasets_train = datasets[train_mask]

        if verbose:
            n_test = test_mask.sum()
            print(f"Evaluating on {test_ds} ({n_test} samples)...", end=" ")

        # Clone and fit classifier for this fold
        clf = classifier.clone()

        if cv.enabled:
            # Get CV scores for threshold selection before final fit
            train_scores = _get_cv_scores(clf, X_train, y_train, cv.folds, cv.n_jobs)
            # Now fit on full training data
            clf.fit(X_train, y_train, datasets=datasets_train, verbose=verbose)
        else:
            # Fit and use train predictions (for DANN or when CV not needed)
            clf.fit(X_train, y_train, datasets=datasets_train, verbose=verbose)
            train_scores = clf.predict_scores(X_train)

        # Get test scores
        test_scores = clf.predict_scores(X_test)

        # Evaluate using shared evaluation function
        fold_results = evaluate_predictions(
            train_scores, y_train,
            test_scores, y_test,
            threshold_strategy=threshold_strategy,
            verbose=False,
        )

        per_dataset_results[test_ds] = fold_results

        if verbose:
            print(
                f"acc={fold_results['acc']:.1%}, "
                f"F1={fold_results['malicious_f1']:.1f}%, "
                f"thr={fold_results['threshold']:.3f}"
            )

    # Compute weighted averages
    weighted_avg = calculate_weighted_average(per_dataset_results, datasets)

    if verbose:
        print("-" * 50)
        print(
            f"Weighted average: acc={weighted_avg['acc']:.1%}, "
            f"F1={weighted_avg['malicious_f1']:.1f}%"
        )

    return LODOResult(
        per_dataset=per_dataset_results,
        weighted_average=weighted_avg,
    )


def _get_cv_scores(
    classifier: ClassifierProtocol,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 3,
    n_jobs: int = 8,
) -> np.ndarray:
    """
    Get cross-validation prediction scores for threshold selection.

    Uses sklearn's cross_val_predict to get out-of-fold predictions,
    avoiding overfitting in threshold selection.

    Args:
        classifier: Classifier implementing ClassifierProtocol
        X: Feature matrix
        y: Labels
        cv_folds: Number of CV folds
        n_jobs: Number of parallel jobs for cross-validation

    Returns:
        Out-of-fold prediction scores (n_samples,)
    """
    from prompt_mining.classifiers.sklearn_classifiers import BaseSklearnClassifier

    # For sklearn-based classifiers, we need to access the underlying sklearn model
    if isinstance(classifier, BaseSklearnClassifier):
        # Create a fresh sklearn model with same config
        clf_clone = classifier.clone()

        # Handle normalization by pre-normalizing data
        X_norm = clf_clone._normalize(X, fit=True)
        sklearn_model = clf_clone._create_model()

        # Get method for scoring
        method = 'predict_proba' if hasattr(sklearn_model, 'predict_proba') else 'decision_function'

        # Get CV predictions with shuffled folds to avoid data ordering issues
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_predict(
            sklearn_model, X_norm, y,
            cv=cv, n_jobs=n_jobs, method=method
        )
        get_reusable_executor().shutdown(wait=True)

        if method == 'predict_proba':
            cv_scores = cv_scores[:, 1]

        return cv_scores
    else:
        # For other classifiers (DANN, etc.), fall back to train predictions
        # since CV doesn't make sense for complex models
        clf_clone = classifier.clone()
        clf_clone.fit(X, y)
        return clf_clone.predict_scores(X)


def calculate_weighted_average(
    results: Dict[str, Dict[str, float]],
    datasets: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate weighted average metrics across datasets.

    Weights each dataset by its sample count, giving larger datasets
    more influence on the final average.

    Args:
        results: Dict mapping dataset name to metrics dict
        datasets: Dataset IDs array (for computing weights)

    Returns:
        Dict with same keys as individual results, weighted averaged
    """
    if not results:
        return {}

    metric_keys = list(next(iter(results.values())).keys())
    weighted = {k: 0.0 for k in metric_keys}
    total_samples = 0

    for ds, metrics in results.items():
        n_samples = (datasets == ds).sum()
        total_samples += n_samples
        for k in metric_keys:
            weighted[k] += metrics[k] * n_samples

    return {k: v / total_samples for k, v in weighted.items()}


def lodo_evaluate_multiple(
    classifiers: Dict[str, ClassifierProtocol],
    X: np.ndarray,
    y: np.ndarray,
    datasets: np.ndarray,
    threshold_strategy: Optional[ThresholdStrategy] = None,
    cv: Optional[CVConfig] = None,
    merge_datasets: Optional[Dict[str, str]] = None,
    verbose: bool = True,
) -> Dict[str, LODOResult]:
    """
    Run LODO evaluation for multiple classifiers.

    Useful for comparing different classifier configurations.

    Args:
        classifiers: Dict mapping classifier name to classifier instance
        X, y, datasets: Data (see lodo_evaluate)
        threshold_strategy: Strategy for threshold selection
        cv: Cross-validation config for score generation
        merge_datasets: Optional dataset merging
        verbose: Print progress

    Returns:
        Dict mapping classifier name to LODOResult

    Example:
        >>> from prompt_mining.classifiers import LinearClassifier, DANNClassifier
        >>>
        >>> classifiers = {
        ...     'logistic': LinearClassifier(),
        ...     'dann': DANNClassifier(),
        ... }
        >>> results = lodo_evaluate_multiple(classifiers, X, y, datasets)
        >>> for name, result in results.items():
        ...     print(f"{name}: {result.weighted_average['malicious_f1']:.1f}%")
    """
    results = {}

    for name, clf in classifiers.items():
        if verbose:
            print(f"\n{'='*50}")
            print(f"Evaluating {name}")
            print('='*50)

        results[name] = lodo_evaluate(
            clf, X, y, datasets,
            threshold_strategy=threshold_strategy,
            cv=cv,
            merge_datasets=merge_datasets,
            verbose=verbose,
        )

    return results
