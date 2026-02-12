"""
Latent Prototype Moderation (LPM) classifier implementing ClassifierProtocol.

A training-free baseline that classifies via Mahalanobis distance to class
prototypes (centroids) in activation space.

Reference:
    "Do LLMs Understand the Safety of Their Inputs? Training-Free Moderation
    via Latent Prototypes" (arXiv:2502.16174)
"""
import numpy as np
from typing import Optional, Literal, Dict, Any, List
from dataclasses import dataclass
import copy

from sklearn.covariance import LedoitWolf, EmpiricalCovariance


@dataclass
class LPMConfig:
    """Configuration for LPM classifier."""

    distance_metric: Literal['mahalanobis', 'euclidean', 'cosine'] = 'mahalanobis'
    """Distance metric for prototype comparison."""

    covariance_estimator: Literal['bayes_ridge', 'ledoit_wolf', 'empirical', 'shared', 'per_class'] = 'bayes_ridge'
    """
    Covariance estimation strategy:
    - 'bayes_ridge': Paper's Bayes ridge-type estimator (Eq. 3) - fast, recommended
    - 'ledoit_wolf': Shrinkage estimator (slower for high-dim)
    - 'empirical': Standard MLE covariance
    - 'shared': Single covariance for both classes using Ledoit-Wolf
    - 'per_class': Separate covariance per class
    """

    score_transform: Literal['ratio', 'softmax', 'raw'] = 'softmax'
    """
    How to convert distances to scores:
    - 'softmax': GDA posterior P(unsafe|x) ∝ exp(-dist²/2) (paper Eq. 4, default)
    - 'ratio': dist_safe / (dist_safe + dist_unsafe) -> [0, 1]
    - 'raw': -dist_unsafe (negative distance to unsafe, for ranking)
    """

    regularization: float = 1e-6
    """Regularization added to covariance diagonal for numerical stability."""

    max_samples_covariance: Optional[int] = None
    """Max samples for covariance estimation. None = use all. Subsampling speeds up Ledoit-Wolf dramatically."""


class LPMClassifier:
    """
    Latent Prototype Moderation classifier.

    Training-free approach that computes class centroids (prototypes) from
    training data and classifies test samples based on distance to prototypes.

    Example:
        >>> clf = LPMClassifier(LPMConfig(distance_metric='mahalanobis'))
        >>> clf.fit(X_train, y_train)
        >>> scores = clf.predict_scores(X_test)  # Higher = more likely malicious
    """

    def __init__(self, config: Optional[LPMConfig] = None):
        self.config = config or LPMConfig()
        self._fitted = False

        # Computed during fit
        self._safe_prototype: Optional[np.ndarray] = None      # (d,)
        self._unsafe_prototype: Optional[np.ndarray] = None    # (d,)
        self._precision_matrix: Optional[np.ndarray] = None    # (d, d) for Mahalanobis
        self._precision_chol: Optional[np.ndarray] = None      # Cholesky factor for fast distance
        self._safe_precision: Optional[np.ndarray] = None      # Per-class precision
        self._unsafe_precision: Optional[np.ndarray] = None
        self._safe_precision_chol: Optional[np.ndarray] = None
        self._unsafe_precision_chol: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs
    ) -> "LPMClassifier":
        """
        Compute class prototypes (centroids) from training data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (n_samples,) where 1=malicious, 0=benign
            sample_weight: Optional sample weights (not used, for interface compat)
            **kwargs: Ignored

        Returns:
            self
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Split by class
        safe_mask = (y == 0)
        unsafe_mask = (y == 1)

        X_safe = X[safe_mask]
        X_unsafe = X[unsafe_mask]

        if len(X_safe) == 0 or len(X_unsafe) == 0:
            raise ValueError("Training data must contain both safe and unsafe samples")

        # Compute prototypes (class means)
        self._safe_prototype = X_safe.mean(axis=0)
        self._unsafe_prototype = X_unsafe.mean(axis=0)

        # Compute covariance/precision for Mahalanobis distance
        if self.config.distance_metric == 'mahalanobis':
            self._fit_covariance(X_safe, X_unsafe)

        self._fitted = True
        return self

    def _fit_covariance(self, X_safe: np.ndarray, X_unsafe: np.ndarray) -> None:
        """Fit covariance estimator(s) for Mahalanobis distance."""

        def get_estimator():
            if self.config.covariance_estimator in ('ledoit_wolf', 'shared'):
                return LedoitWolf()
            else:
                return EmpiricalCovariance()

        def subsample(X: np.ndarray, max_n: Optional[int]) -> np.ndarray:
            """Subsample array if larger than max_n. Uses uniform random sampling."""
            if max_n is None or len(X) <= max_n:
                return X
            # Uniform random subsample - fine for covariance since we want
            # global activation structure, not per-dataset patterns.
            # Prototypes (means) still use ALL data.
            rng = np.random.default_rng(42)
            indices = rng.choice(len(X), max_n, replace=False)
            return X[indices]

        def compute_cholesky(precision: np.ndarray) -> np.ndarray:
            """Compute Cholesky factor of precision matrix for fast distance."""
            # For numerical stability, add small regularization if needed
            try:
                return np.linalg.cholesky(precision)
            except np.linalg.LinAlgError:
                # Add regularization and retry
                reg = self.config.regularization * np.eye(precision.shape[0])
                return np.linalg.cholesky(precision + reg)

        max_n = self.config.max_samples_covariance

        if self.config.covariance_estimator == 'bayes_ridge':
            # Paper's Bayes ridge-type estimator (Eq. 3):
            # Σ_d^(-1) = d * ((N-1)Σ̂ + tr(Σ̂)·I_d)^(-1)
            # This is much faster than Ledoit-Wolf for high-dimensional data
            X_all = np.vstack([X_safe, X_unsafe])
            X_sub = subsample(X_all, max_n)
            n, d = X_sub.shape

            # Compute sample covariance
            X_centered = X_sub - X_sub.mean(axis=0)
            cov = (X_centered.T @ X_centered) / (n - 1)
            trace_cov = np.trace(cov)

            # Regularized covariance: (n-1)*cov + trace(cov)*I
            reg_cov = (n - 1) * cov + trace_cov * np.eye(d)

            # Precision = d * inv(reg_cov)
            self._precision_matrix = d * np.linalg.inv(reg_cov + self.config.regularization * np.eye(d))
            self._precision_chol = compute_cholesky(self._precision_matrix)

        elif self.config.covariance_estimator == 'per_class':
            # Separate covariance per class
            safe_cov = get_estimator().fit(subsample(X_safe, max_n))
            unsafe_cov = get_estimator().fit(subsample(X_unsafe, max_n))
            self._safe_precision = safe_cov.precision_ + self.config.regularization * np.eye(X_safe.shape[1])
            self._unsafe_precision = unsafe_cov.precision_ + self.config.regularization * np.eye(X_unsafe.shape[1])
            self._safe_precision_chol = compute_cholesky(self._safe_precision)
            self._unsafe_precision_chol = compute_cholesky(self._unsafe_precision)
        else:
            # Shared covariance (pooled) - subsample from combined
            X_all = np.vstack([X_safe, X_unsafe])
            X_sub = subsample(X_all, max_n)
            cov_estimator = get_estimator().fit(X_sub)
            self._precision_matrix = cov_estimator.precision_ + self.config.regularization * np.eye(X_all.shape[1])
            self._precision_chol = compute_cholesky(self._precision_matrix)

    def _mahalanobis_distance(
        self,
        X: np.ndarray,
        prototype: np.ndarray,
        precision: Optional[np.ndarray] = None,
        precision_chol: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute Mahalanobis distance from X to prototype.

        Uses Cholesky decomposition for efficiency when available:
        d_M(x, μ) = ||L^T (x - μ)||_2 where P = L @ L^T

        Args:
            X: (n_samples, d) or (d,)
            prototype: (d,)
            precision: (d, d) precision matrix, uses shared if None
            precision_chol: (d, d) Cholesky factor of precision (faster if provided)

        Returns:
            distances: (n_samples,) or scalar
        """
        diff = X - prototype  # (n, d) or (d,)

        # Prefer Cholesky factor for speed: ||diff @ L||_2
        if precision_chol is not None:
            chol = precision_chol
        elif self._precision_chol is not None:
            chol = self._precision_chol
        else:
            chol = None

        if chol is not None:
            # Fast path using Cholesky: d = ||diff @ L||
            transformed = diff @ chol  # (n, d) or (d,)
            if transformed.ndim == 1:
                return np.linalg.norm(transformed)
            else:
                return np.linalg.norm(transformed, axis=1)
        else:
            # Fallback to direct computation
            if precision is None:
                precision = self._precision_matrix

            if diff.ndim == 1:
                return np.sqrt(diff @ precision @ diff)
            else:
                # Vectorized: sqrt(sum_j sum_k diff_ij * P_jk * diff_ik)
                return np.sqrt(np.einsum('ij,jk,ik->i', diff, precision, diff))

    def _euclidean_distance(self, X: np.ndarray, prototype: np.ndarray) -> np.ndarray:
        """Compute Euclidean distance from X to prototype."""
        diff = X - prototype
        if diff.ndim == 1:
            return np.linalg.norm(diff)
        return np.linalg.norm(diff, axis=1)

    def _cosine_distance(self, X: np.ndarray, prototype: np.ndarray) -> np.ndarray:
        """Compute cosine distance (1 - cosine_similarity) from X to prototype."""
        # Normalize
        X_norm = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-10)
        p_norm = prototype / (np.linalg.norm(prototype) + 1e-10)

        similarity = X_norm @ p_norm
        return 1.0 - similarity

    def _compute_distances(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute distances to both prototypes."""
        metric = self.config.distance_metric

        if metric == 'mahalanobis':
            if self.config.covariance_estimator == 'per_class':
                dist_safe = self._mahalanobis_distance(
                    X, self._safe_prototype, self._safe_precision, self._safe_precision_chol
                )
                dist_unsafe = self._mahalanobis_distance(
                    X, self._unsafe_prototype, self._unsafe_precision, self._unsafe_precision_chol
                )
            else:
                # Uses shared precision/cholesky via defaults
                dist_safe = self._mahalanobis_distance(X, self._safe_prototype)
                dist_unsafe = self._mahalanobis_distance(X, self._unsafe_prototype)
        elif metric == 'euclidean':
            dist_safe = self._euclidean_distance(X, self._safe_prototype)
            dist_unsafe = self._euclidean_distance(X, self._unsafe_prototype)
        elif metric == 'cosine':
            dist_safe = self._cosine_distance(X, self._safe_prototype)
            dist_unsafe = self._cosine_distance(X, self._unsafe_prototype)
        else:
            raise ValueError(f"Unknown distance metric: {metric}")

        return dist_safe, dist_unsafe

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Return prediction scores (higher = more likely malicious).

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Scores array (n_samples,) in [0, 1] for ratio/softmax transforms
        """
        if not self._fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        X = np.asarray(X)
        dist_safe, dist_unsafe = self._compute_distances(X)

        # Convert distances to scores (higher = more malicious)
        transform = self.config.score_transform

        if transform == 'ratio':
            # Closer to unsafe (smaller dist_unsafe) -> higher score
            # score = dist_safe / (dist_safe + dist_unsafe)
            total = dist_safe + dist_unsafe + 1e-10
            scores = dist_safe / total

        elif transform == 'softmax':
            # Paper Eq. 4: GDA posterior P(c|x) ∝ exp(-dist²/2)
            # P(unsafe) = exp(-dist_unsafe²/2) / (exp(-dist_safe²/2) + exp(-dist_unsafe²/2))
            # Use log-sum-exp for numerical stability
            neg_sq_dists = np.stack([-dist_safe**2 / 2, -dist_unsafe**2 / 2], axis=-1)  # (n, 2) or (2,)
            max_neg = neg_sq_dists.max(axis=-1, keepdims=True)
            exp_neg = np.exp(neg_sq_dists - max_neg)
            probs = exp_neg / exp_neg.sum(axis=-1, keepdims=True)
            # Handle both batched and single-sample cases
            scores = probs[..., 1] if probs.ndim > 1 else probs[1]

        elif transform == 'raw':
            # Raw negative distance to unsafe (for ranking only)
            scores = -dist_unsafe

        else:
            raise ValueError(f"Unknown score transform: {transform}")

        return scores

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Return binary predictions.

        Args:
            X: Feature matrix (n_samples, n_features)
            threshold: Decision threshold (default 0.5)

        Returns:
            Binary predictions (n_samples,)
        """
        scores = self.predict_scores(X)
        return (scores >= threshold).astype(int)

    def clone(self) -> "LPMClassifier":
        """Create unfitted copy with same configuration."""
        return LPMClassifier(config=copy.deepcopy(self.config))

    def get_params(self) -> Dict[str, Any]:
        """Return configuration as dict for reproducibility."""
        return {
            'distance_metric': self.config.distance_metric,
            'covariance_estimator': self.config.covariance_estimator,
            'score_transform': self.config.score_transform,
            'regularization': self.config.regularization,
            'max_samples_covariance': self.config.max_samples_covariance,
        }

    def get_top_features(
        self,
        top_k: int = 10,
        direction: Literal['positive', 'negative', 'both', 'abs'] = 'abs',
    ) -> List[int]:
        """
        Get indices of features with largest prototype difference.

        For LPM, "important" features are those where the prototypes differ most.

        Args:
            top_k: Number of top features to return
            direction:
                - 'positive': features where unsafe > safe
                - 'negative': features where safe > unsafe
                - 'both': top_k from each direction
                - 'abs': by absolute difference (default)

        Returns:
            List of feature indices
        """
        if not self._fitted:
            raise RuntimeError("Classifier not fitted.")

        diff = self._unsafe_prototype - self._safe_prototype  # (d,)

        if direction == 'positive':
            return np.argsort(diff)[-top_k:][::-1].tolist()
        elif direction == 'negative':
            return np.argsort(diff)[:top_k].tolist()
        elif direction == 'both':
            pos = np.argsort(diff)[-top_k:][::-1].tolist()
            neg = np.argsort(diff)[:top_k].tolist()
            return pos + neg
        elif direction == 'abs':
            return np.argsort(np.abs(diff))[-top_k:][::-1].tolist()
        else:
            raise ValueError(f"Invalid direction: {direction}")

    @property
    def prototypes(self) -> Dict[str, np.ndarray]:
        """Return the computed prototypes."""
        if not self._fitted:
            raise RuntimeError("Classifier not fitted.")
        return {
            'safe': self._safe_prototype.copy(),
            'unsafe': self._unsafe_prototype.copy(),
        }

    @property
    def prototype_distance(self) -> float:
        """Return distance between the two prototypes."""
        if not self._fitted:
            raise RuntimeError("Classifier not fitted.")

        if self.config.distance_metric == 'mahalanobis':
            return float(self._mahalanobis_distance(
                self._safe_prototype,
                self._unsafe_prototype
            ))
        elif self.config.distance_metric == 'cosine':
            return float(self._cosine_distance(
                self._safe_prototype.reshape(1, -1),
                self._unsafe_prototype
            )[0])
        else:
            return float(np.linalg.norm(self._safe_prototype - self._unsafe_prototype))

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return (f"LPMClassifier(distance={self.config.distance_metric}, "
                f"transform={self.config.score_transform}, {status})")