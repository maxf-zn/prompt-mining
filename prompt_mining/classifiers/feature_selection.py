"""
Feature selection methods for SAE/PLT features.

Provides NPMI-based selection that identifies class-predictive but
dataset-agnostic features for improved cross-dataset generalization.
"""
import numpy as np
from typing import Optional, Dict, Any


def compute_npmi(
    X: np.ndarray,
    labels: np.ndarray,
    min_freq: float = 0.01,
) -> np.ndarray:
    """
    Compute Normalized Pointwise Mutual Information between features and labels.

    NPMI measures the association between each feature and the target label,
    normalized to range [-1, 1]:
    - NPMI = 1: Perfect positive association
    - NPMI = 0: Independence
    - NPMI = -1: Perfect negative association

    Args:
        X: Binary feature matrix (n_samples, n_features)
        labels: Binary labels (n_samples,)
        min_freq: Minimum feature frequency to compute NPMI (default 0.01)

    Returns:
        NPMI values (n_features,) - positive values indicate association with label=1

    Example:
        >>> npmi = compute_npmi(X_binary, y)
        >>> top_features = np.argsort(npmi)[-10:]  # Top 10 positive associations
    """
    n_samples = X.shape[0]

    # Feature frequencies P(feature=1)
    p_feature = np.asarray(X.mean(axis=0)).flatten()
    active_mask = p_feature > min_freq

    # Label probability P(label=1)
    p_label = labels.mean()

    # Joint probability P(feature=1, label=1)
    # = count(feature & label) / n_samples
    p_joint = np.asarray(X[labels == 1].sum(axis=0)).flatten() / n_samples

    # NPMI = PMI / -log(P(joint))
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.log2((p_joint + 1e-10) / ((p_feature + 1e-10) * p_label))
        npmi = pmi / (-np.log2(p_joint + 1e-10))
        npmi = np.nan_to_num(npmi, nan=0, posinf=0, neginf=0)

    npmi[~active_mask] = 0
    return npmi


def npmi_feature_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    datasets_train: np.ndarray,
    class_threshold: float = 0.1,
    dataset_threshold: float = 0.5,
    min_freq: float = 0.01,
) -> np.ndarray:
    """
    Select features using NPMI-based criteria for cross-dataset generalization.

    Selects features where:
    1. |NPMI(feature, class)| > class_threshold (class-predictive)
    2. max|NPMI(feature, dataset)| < dataset_threshold (not dataset-specific)

    This identifies features that predict the target class but are not
    spuriously correlated with specific datasets.

    Args:
        X_train: Training feature matrix (binary for SAE features)
        y_train: Training labels (binary)
        datasets_train: Training dataset IDs
        class_threshold: Minimum |NPMI| with class label (default 0.1)
        dataset_threshold: Maximum |NPMI| with any dataset (default 0.5)
        min_freq: Minimum feature frequency (default 0.01)

    Returns:
        Boolean mask of selected features (n_features,)

    Example:
        >>> mask = npmi_feature_selection(X_train, y_train, datasets_train)
        >>> X_train_selected = X_train[:, mask]
        >>> print(f"Selected {mask.sum()} of {len(mask)} features")
    """
    n_features = X_train.shape[1]

    # Compute class NPMI
    npmi_class = compute_npmi(X_train, y_train, min_freq)

    # Compute max dataset NPMI across all datasets
    max_npmi_dataset = np.zeros(n_features)
    for ds in np.unique(datasets_train):
        ds_mask = (datasets_train == ds).astype(int)
        npmi_ds = compute_npmi(X_train, ds_mask, min_freq)
        max_npmi_dataset = np.maximum(max_npmi_dataset, np.abs(npmi_ds))

    # Feature frequency for filtering
    freq = np.asarray(X_train.mean(axis=0)).flatten()

    # Select features meeting all criteria
    select_mask = (
        (np.abs(npmi_class) > class_threshold) &
        (max_npmi_dataset < dataset_threshold) &
        (freq > min_freq)
    )

    return select_mask


def get_feature_stats(
    X: np.ndarray,
    y: np.ndarray,
    datasets: np.ndarray,
    min_freq: float = 0.01,
) -> Dict[str, Any]:
    """
    Compute comprehensive feature statistics for analysis.

    Args:
        X: Binary feature matrix
        y: Labels
        datasets: Dataset IDs
        min_freq: Minimum feature frequency

    Returns:
        Dictionary with:
        - npmi_class: NPMI with class label
        - npmi_datasets: Dict of NPMI per dataset
        - max_npmi_dataset: Max |NPMI| across datasets
        - frequencies: Feature frequencies
        - n_features: Total features
        - n_active: Features above min_freq
    """
    npmi_class = compute_npmi(X, y, min_freq)

    npmi_datasets = {}
    max_npmi_dataset = np.zeros(X.shape[1])
    for ds in np.unique(datasets):
        ds_mask = (datasets == ds).astype(int)
        npmi_ds = compute_npmi(X, ds_mask, min_freq)
        npmi_datasets[str(ds)] = npmi_ds
        max_npmi_dataset = np.maximum(max_npmi_dataset, np.abs(npmi_ds))

    freq = np.asarray(X.mean(axis=0)).flatten()

    return {
        'npmi_class': npmi_class,
        'npmi_datasets': npmi_datasets,
        'max_npmi_dataset': max_npmi_dataset,
        'frequencies': freq,
        'n_features': X.shape[1],
        'n_active': (freq > min_freq).sum(),
    }
