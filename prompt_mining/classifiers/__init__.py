"""
Classifier module for prompt mining platform.

Provides unified interface for training and evaluating classifiers:

- ClassifierProtocol: Interface all classifiers implement
- LinearClassifier: Sklearn-based linear models (LogisticRegression, SGD)
- DANNClassifier: Domain-adversarial neural network
- ClassificationDataset: Data loading with LODO splits
- lodo_evaluate: Leave-One-Dataset-Out evaluation
- ThresholdStrategy: Pluggable threshold selection (Constant, TargetPrecision, MaxF1)

Example - Basic classification:
    >>> from prompt_mining.classifiers import (
    ...     ClassificationDataset,
    ...     LinearClassifier,
    ...     lodo_evaluate,
    ... )
    >>>
    >>> # Load data
    >>> dataset = ClassificationDataset.from_path("/path/to/ingestion/output")
    >>> data = dataset.load(layer=23, space='raw')
    >>>
    >>> # Train and evaluate
    >>> clf = LinearClassifier()
    >>> results = lodo_evaluate(clf, data.X, data.y, data.datasets)
    >>> print(results.summary())

Example - Custom threshold strategies:
    >>> from prompt_mining.classifiers import (
    ...     LinearClassifier, lodo_evaluate,
    ...     ConstantThreshold, MaxF1Threshold, CVConfig,
    ... )
    >>>
    >>> # Use constant threshold
    >>> results = lodo_evaluate(clf, X, y, datasets,
    ...     threshold_strategy=ConstantThreshold(0.4))
    >>>
    >>> # Use max F1 threshold with cross-validation
    >>> results = lodo_evaluate(clf, X, y, datasets,
    ...     threshold_strategy=MaxF1Threshold(),
    ...     cv=CVConfig(enabled=True, folds=5, n_jobs=8))

Example - DANN with domain adversarial training:
    >>> from prompt_mining.classifiers import DANNClassifier, DANNConfig
    >>>
    >>> clf = DANNClassifier(DANNConfig(
    ...     normalize='l2',
    ...     domain_weight=0.5,
    ...     max_epochs=40,
    ... ))
    >>> results = lodo_evaluate(clf, data.X, data.y, data.datasets)

Example - Feature selection:
    >>> from prompt_mining.classifiers import npmi_feature_selection
    >>>
    >>> mask = npmi_feature_selection(X_train, y_train, datasets_train)
    >>> X_selected = X[:, mask]
"""

# Base protocol and evaluation
from prompt_mining.classifiers.base_classifier import (
    ClassifierProtocol,
    EvaluationResult,
    evaluate_predictions,
)

# Data loading
from prompt_mining.classifiers.dataset import (
    ClassificationDataset,
    ClassificationData,
)

# Classifiers
from prompt_mining.classifiers.sklearn_classifiers import (
    LinearClassifier,
    LinearConfig,
    XGBoostClassifier,
    XGBoostConfig,
)
from prompt_mining.classifiers.dann import (
    DANNClassifier,
    DANNConfig,
    DANNModule,
)

# LPM (Latent Prototype Moderation) - training-free baseline
from prompt_mining.classifiers.lpm_classifier import (
    LPMClassifier,
    LPMConfig,
)

# Feature selection
from prompt_mining.classifiers.feature_selection import (
    compute_npmi,
    npmi_feature_selection,
    get_feature_stats,
)

# LODO evaluation
from prompt_mining.classifiers.lodo import (
    lodo_evaluate,
    lodo_evaluate_multiple,
    calculate_weighted_average,
    LODOResult,
)

# Threshold strategies
from prompt_mining.classifiers.threshold_strategy import (
    ThresholdStrategy,
    ConstantThreshold,
    TargetPrecisionThreshold,
    MaxF1Threshold,
    CVConfig,
)


__all__ = [
    # Protocol
    'ClassifierProtocol',
    'EvaluationResult',
    'evaluate_predictions',
    # Data
    'ClassificationDataset',
    'ClassificationData',
    # Classifiers
    'LinearClassifier',
    'LinearConfig',
    'XGBoostClassifier',
    'XGBoostConfig',
    'DANNClassifier',
    'DANNConfig',
    'DANNModule',
    # Feature selection
    'compute_npmi',
    'npmi_feature_selection',
    'get_feature_stats',
    # LODO
    'lodo_evaluate',
    'lodo_evaluate_multiple',
    'calculate_weighted_average',
    'LODOResult',
    # Threshold strategies
    'ThresholdStrategy',
    'ConstantThreshold',
    'TargetPrecisionThreshold',
    'MaxF1Threshold',
    'CVConfig',
    # LPM (training-free baseline)
    'LPMClassifier',
    'LPMConfig',
]
