"""
Sklearn-compatible classifiers implementing ClassifierProtocol.

Contains wrappers for sklearn and XGBoost classifiers with standardized
normalization and scoring interface.
"""
import numpy as np
from typing import Optional, Literal, Dict, Any, TypeVar, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
import copy

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler, normalize
import xgboost as xgb


T = TypeVar('T', bound='BaseSklearnClassifier')


# =============================================================================
# Base Class
# =============================================================================

class BaseSklearnClassifier(ABC):
    """
    Base class for sklearn-compatible classifiers.

    Handles normalization, fit/predict pattern, and cloning.
    Subclasses only need to implement _create_model() and config handling.
    """

    def __init__(self, normalize: Literal['l2', 'standard', 'none'] = 'none'):
        self._model = None
        self._scaler = None
        self._fitted = False
        self._normalize_mode = normalize

    @abstractmethod
    def _create_model(self):
        """Create fresh model instance. Implemented by subclasses."""
        ...

    @abstractmethod
    def clone(self: T) -> T:
        """Create unfitted copy with same configuration."""
        ...

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Return configuration as dict for reproducibility."""
        ...

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Apply normalization to input features."""
        if self._normalize_mode == 'l2':
            return normalize(X, norm='l2')
        elif self._normalize_mode == 'standard':
            if fit:
                self._scaler = StandardScaler()
                return self._scaler.fit_transform(X)
            else:
                if self._scaler is None:
                    raise RuntimeError("Scaler not fitted. Call fit() first.")
                return self._scaler.transform(X)
        else:
            return X

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs
    ) -> "BaseSklearnClassifier":
        """
        Train the classifier.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            sample_weight: Optional sample weights
            **kwargs: Ignored (for interface compatibility)

        Returns:
            self
        """
        X_norm = self._normalize(X, fit=True)
        self._model = self._create_model()
        self._model.fit(X_norm, y, sample_weight=sample_weight)
        self._fitted = True
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Return prediction scores (probability of positive class).

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Scores array (n_samples,)
        """
        if not self._fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        X_norm = self._normalize(X, fit=False)

        if hasattr(self._model, 'predict_proba'):
            return self._model.predict_proba(X_norm)[:, 1]
        else:
            return self._model.decision_function(X_norm)

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

    @abstractmethod
    def get_top_features(
        self,
        top_k: int = 10,
        direction: Literal['positive', 'negative', 'both', 'abs'] = 'positive',
    ) -> List[int]:
        """
        Get indices of top features by importance/coefficient.

        Args:
            top_k: Number of top features to return
            direction: How to select features:
                - 'positive': highest values (for linear: positive class predictors)
                - 'negative': lowest values (for linear: negative class predictors)
                - 'both': top_k from each direction (returns 2*top_k)
                - 'abs': by absolute value (default for tree models)

        Returns:
            List of feature indices
        """
        ...


# =============================================================================
# Linear Classifier
# =============================================================================

@dataclass
class LinearConfig:
    """Configuration for linear classifiers."""
    model: Literal['logistic', 'sgd'] = 'logistic'
    normalize: Literal['l2', 'standard', 'none'] = 'l2'
    C: float = 1.0
    max_iter: int = 1000
    class_weight: Optional[str] = 'balanced'
    random_state: int = 42


class LinearClassifier(BaseSklearnClassifier):
    """
    Sklearn linear classifier implementing ClassifierProtocol.

    Example:
        >>> clf = LinearClassifier(LinearConfig(model='logistic', normalize='l2'))
        >>> clf.fit(X_train, y_train)
        >>> scores = clf.predict_scores(X_test)
    """

    def __init__(self, config: Optional[LinearConfig] = None):
        self.config = config or LinearConfig()
        super().__init__(normalize=self.config.normalize)

    def _create_model(self):
        if self.config.model == 'logistic':
            return LogisticRegression(
                C=self.config.C,
                max_iter=self.config.max_iter,
                class_weight=self.config.class_weight,
                random_state=self.config.random_state,
            )
        elif self.config.model == 'sgd':
            return SGDClassifier(
                loss='log_loss',
                alpha=1.0 / self.config.C,
                max_iter=self.config.max_iter,
                class_weight=self.config.class_weight,
                random_state=self.config.random_state,
                tol=1e-3,
            )
        else:
            raise ValueError(f"Unknown model: {self.config.model}")

    def clone(self) -> "LinearClassifier":
        return LinearClassifier(config=copy.deepcopy(self.config))

    @property
    def coef_(self) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Classifier not fitted.")
        return self._model.coef_

    @property
    def intercept_(self) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Classifier not fitted.")
        return self._model.intercept_

    def get_params(self) -> Dict[str, Any]:
        return {
            'model': self.config.model,
            'normalize': self.config.normalize,
            'C': self.config.C,
            'max_iter': self.config.max_iter,
            'class_weight': self.config.class_weight,
            'random_state': self.config.random_state,
        }

    def get_top_features(
        self,
        top_k: int = 10,
        direction: Literal['positive', 'negative', 'both', 'abs'] = 'positive',
    ) -> List[int]:
        """
        Get indices of top features by coefficient value.

        Args:
            top_k: Number of top features to return
            direction:
                - 'positive': highest coefficients (positive class predictors)
                - 'negative': lowest coefficients (negative class predictors)
                - 'both': top_k from each direction (returns 2*top_k)
                - 'abs': by absolute coefficient value

        Returns:
            List of feature indices
        """
        if not self._fitted:
            raise RuntimeError("Classifier not fitted.")

        coef = self._model.coef_.flatten()

        if direction == 'positive':
            return np.argsort(coef)[-top_k:][::-1].tolist()
        elif direction == 'negative':
            return np.argsort(coef)[:top_k].tolist()
        elif direction == 'both':
            pos = np.argsort(coef)[-top_k:][::-1].tolist()
            neg = np.argsort(coef)[:top_k].tolist()
            return pos + neg
        elif direction == 'abs':
            return np.argsort(np.abs(coef))[-top_k:][::-1].tolist()
        else:
            raise ValueError(f"Invalid direction: {direction}")

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return f"LinearClassifier(model={self.config.model}, normalize={self.config.normalize}, {status})"


# =============================================================================
# XGBoost Classifier
# =============================================================================

@dataclass
class XGBoostConfig:
    """Configuration for XGBoost classifier."""
    normalize: Literal['l2', 'standard', 'none'] = 'none'
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.3
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    scale_pos_weight: Optional[float] = None
    random_state: int = 42
    n_jobs: int = -1
    verbosity: int = 0


class XGBoostClassifier(BaseSklearnClassifier):
    """
    XGBoost classifier implementing ClassifierProtocol.

    Example:
        >>> clf = XGBoostClassifier(XGBoostConfig(max_depth=4, n_estimators=200))
        >>> clf.fit(X_train, y_train)
        >>> scores = clf.predict_scores(X_test)
    """

    def __init__(self, config: Optional[XGBoostConfig] = None):
        self.config = config or XGBoostConfig()
        super().__init__(normalize=self.config.normalize)

    def _create_model(self):
        return xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            scale_pos_weight=self.config.scale_pos_weight,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbosity=self.config.verbosity,
            eval_metric='logloss',
        )

    def clone(self) -> "XGBoostClassifier":
        return XGBoostClassifier(config=copy.deepcopy(self.config))

    @property
    def feature_importances_(self) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Classifier not fitted.")
        return self._model.feature_importances_

    def get_params(self) -> Dict[str, Any]:
        return {
            'normalize': self.config.normalize,
            'n_estimators': self.config.n_estimators,
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'subsample': self.config.subsample,
            'colsample_bytree': self.config.colsample_bytree,
            'reg_alpha': self.config.reg_alpha,
            'reg_lambda': self.config.reg_lambda,
            'scale_pos_weight': self.config.scale_pos_weight,
            'random_state': self.config.random_state,
        }

    def get_top_features(
        self,
        top_k: int = 10,
        direction: Literal['positive', 'negative', 'both', 'abs'] = 'abs',
    ) -> List[int]:
        """
        Get indices of top features by importance.

        Note: XGBoost feature_importances_ are always positive (gain-based),
        so 'positive'/'negative'/'abs' all return the same result.
        Use 'abs' for clarity.

        Args:
            top_k: Number of top features to return
            direction: Ignored for XGBoost (always uses absolute importance)

        Returns:
            List of feature indices
        """
        if not self._fitted:
            raise RuntimeError("Classifier not fitted.")

        importances = self._model.feature_importances_
        return np.argsort(importances)[-top_k:][::-1].tolist()

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return f"XGBoostClassifier(n_estimators={self.config.n_estimators}, max_depth={self.config.max_depth}, {status})"
