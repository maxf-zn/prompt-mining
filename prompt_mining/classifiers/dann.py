"""
Domain-Adversarial Neural Network (DANN) implementing ClassifierProtocol.

DANN learns dataset-invariant representations by adversarially training
against a domain classifier while optimizing for the main classification task.

Reference: notebooks/2025-12-02-12-56_DANN_Adversarial.ipynb
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize, MaxAbsScaler
from sklearn.metrics import f1_score
from typing import Optional, Literal, Dict, Any, Tuple, List
from dataclasses import dataclass, field, asdict
from tqdm import tqdm
import copy


# =============================================================================
# DANN Architecture
# =============================================================================

class GradientReversalFunction(torch.autograd.Function):
    """Gradient reversal layer for domain-adversarial training."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """Wrapper module for gradient reversal."""

    def __init__(self):
        super().__init__()
        self.lambda_ = 1.0

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float):
        self.lambda_ = lambda_


class DANNModule(nn.Module):
    """
    DANN PyTorch module.

    Architecture:
        Input → Feature Extractor → Shared Features
                                        ↓
                          ┌─────────────┴─────────────┐
                          ↓                           ↓
                    Class Head (1)           Domain Head (N)
                    (malicious?)             (which dataset?)
                                            [gradient reversal]

    Args:
        input_dim: Input feature dimension
        hidden_layers: List of hidden layer dimensions
        n_datasets: Number of datasets for domain classifier
        dropout: Dropout rate
        activation: Activation function ('gelu' or 'relu')
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        n_datasets: int,
        dropout: float = 0.0,
        activation: str = 'gelu',
    ):
        super().__init__()

        act_fn = nn.GELU if activation == 'gelu' else nn.ReLU

        # Build feature extractor
        extractor_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            extractor_layers.append(nn.Linear(prev_dim, hidden_dim))
            extractor_layers.append(act_fn())
            if dropout > 0:
                extractor_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*extractor_layers)

        head_dim = hidden_layers[-1]

        # Class classifier (malicious/benign) - 2 layer head
        class_layers = [nn.Linear(hidden_layers[-1], head_dim), act_fn()]
        if dropout > 0:
            class_layers.append(nn.Dropout(dropout))
        class_layers.append(nn.Linear(head_dim, 1))
        self.class_classifier = nn.Sequential(*class_layers)

        # Domain classifier with gradient reversal - 2 layer head
        self.grl = GradientReversalLayer()
        domain_layers = [nn.Linear(hidden_layers[-1], head_dim), act_fn()]
        if dropout > 0:
            domain_layers.append(nn.Dropout(dropout))
        domain_layers.append(nn.Linear(head_dim, n_datasets))
        self.domain_classifier = nn.Sequential(*domain_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            class_logits: Shape (batch,) - logits for malicious classification
            domain_logits: Shape (batch, n_datasets) - logits for dataset prediction
        """
        features = self.feature_extractor(x)
        class_logits = self.class_classifier(features).squeeze(-1)

        reversed_features = self.grl(features)
        domain_logits = self.domain_classifier(reversed_features)

        return class_logits, domain_logits

    def set_lambda(self, lambda_: float):
        """Set gradient reversal strength."""
        self.grl.set_lambda(lambda_)


# =============================================================================
# DANN Configuration
# =============================================================================

@dataclass
class DANNConfig:
    """
    Configuration for DANN training.

    Attributes:
        hidden_layers: Network architecture
        activation: Activation function ('gelu' or 'relu')
        dropout: Dropout rate
        lr: Learning rate
        weight_decay: L2 regularization
        batch_size: Training batch size
        max_epochs: Maximum training epochs
        early_stopping_patience: Stop if no improvement for N epochs
        val_split: Fraction of training data for validation
        domain_weight: Weight for domain adversarial loss
        normalize: Input normalization method
        random_state: Random seed
        max_grad_norm: Gradient clipping threshold (None = no clipping)
    """
    hidden_layers: List[int] = field(default_factory=lambda: [512, 256, 64])
    activation: str = 'gelu'
    dropout: float = 0.1
    lr: float = 1e-4
    weight_decay: float = 0.00027
    batch_size: int = 1024
    max_epochs: int = 100
    early_stopping_patience: int = 10
    val_split: float = 0.15
    domain_weight: float = 0.0
    normalize: Literal['l2', 'standard', 'maxabs', 'none'] = 'maxabs'
    max_grad_norm: Optional[float] = 0.1
    random_state: int = 42

# =============================================================================
# DANN Classifier (implements ClassifierProtocol)
# =============================================================================

class DANNClassifier:
    """
    DANN classifier implementing ClassifierProtocol.

    Domain-adversarial training learns features that are predictive of the
    target class but invariant to the source dataset, improving generalization
    across different data distributions.

    Example:
        >>> from prompt_mining.classifiers import DANNClassifier, DANNConfig
        >>>
        >>> clf = DANNClassifier(DANNConfig(normalize='l2', domain_weight=0.5))
        >>> clf.fit(X_train, y_train, datasets=dataset_ids)
        >>> scores = clf.predict_scores(X_test)
    """

    def __init__(
        self,
        config: Optional[DANNConfig] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize DANNClassifier.

        Args:
            config: Training configuration (uses defaults if None)
            device: PyTorch device (auto-detects CUDA if None)
        """
        self.config = config or DANNConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model: Optional[DANNModule] = None
        self._scaler = None
        self._label_encoder: Optional[LabelEncoder] = None
        self._input_dim: Optional[int] = None
        self._fitted = False

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Apply normalization to input features."""
        if self.config.normalize == 'l2':
            return normalize(X, norm='l2').astype(np.float32)
        elif self.config.normalize == 'standard':
            if fit:
                self._scaler = StandardScaler()
                return self._scaler.fit_transform(X).astype(np.float32)
            return self._scaler.transform(X).astype(np.float32)
        elif self.config.normalize == 'maxabs':
            if fit:
                self._scaler = MaxAbsScaler()
                return self._scaler.fit_transform(X).astype(np.float32)
            return self._scaler.transform(X).astype(np.float32)
        return X.astype(np.float32)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        datasets: Optional[np.ndarray] = None,
        verbose: bool = True,
        **kwargs
    ) -> "DANNClassifier":
        """
        Train DANN classifier.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            sample_weight: Not used (for interface compatibility)
            datasets: Dataset IDs for domain adversarial training.
                     If None, creates single-domain (no adversarial training).
            verbose: Show progress bar
            **kwargs: Ignored

        Returns:
            self
        """
        np.random.seed(self.config.random_state)
        torch.manual_seed(self.config.random_state)
        torch.cuda.manual_seed(self.config.random_state)

        self._input_dim = X.shape[1]

        # Handle datasets - if None, disable domain adversarial training
        if datasets is None:
            datasets = np.zeros(len(y), dtype=int)

        # Encode datasets to contiguous integers
        self._label_encoder = LabelEncoder()
        dataset_ids = self._label_encoder.fit_transform(datasets)
        n_datasets = len(self._label_encoder.classes_)

        # Normalize input
        X_norm = self._normalize(X, fit=True)

        # Train/val split for early stopping
        n_val = int(len(X) * self.config.val_split)
        indices = np.random.permutation(len(X))
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        X_train, X_val = X_norm[train_idx], X_norm[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        d_train = dataset_ids[train_idx]

        # Create data loader
        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long),
                torch.tensor(d_train, dtype=torch.long),
            ),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            prefetch_factor=1,
            persistent_workers=True,
        )

        # Initialize model
        self._model = DANNModule(
            input_dim=X.shape[1],
            hidden_layers=self.config.hidden_layers,
            n_datasets=n_datasets,
            dropout=self.config.dropout,
            activation=self.config.activation,
        ).to(self.device)

        optimizer = optim.AdamW(
            self._model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        class_criterion = nn.BCEWithLogitsLoss()
        domain_criterion = nn.CrossEntropyLoss()

        # Training with early stopping
        best_f1 = 0.0
        best_state = None
        patience_counter = 0

        iterator = tqdm(range(self.config.max_epochs), desc="Training DANN") if verbose else range(self.config.max_epochs)
        loss_history = {
            'class_loss': [],
            'domain_loss': [],
            'total_loss': [],
        }

        for epoch in iterator:
            # Lambda schedule: 0 → 1 over first half of training
            p = epoch / (self.config.max_epochs / 2)
            lambda_ = min(1.0, 2.0 / (1.0 + np.exp(-10 * p)) - 1)
            self._model.set_lambda(lambda_)

            # Training
            self._model.train()
            for X_batch, y_batch, d_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                d_batch = d_batch.to(self.device)

                optimizer.zero_grad()
                class_logits, domain_logits = self._model(X_batch)

                class_loss = class_criterion(class_logits, y_batch.float())
                domain_loss = domain_criterion(domain_logits, d_batch)
                loss = class_loss + self.config.domain_weight * domain_loss

                loss.backward()

                if self.config.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(), self.config.max_grad_norm
                    )

                optimizer.step()

                loss_history['class_loss'].append(class_loss.item())
                loss_history['domain_loss'].append(domain_loss.item())
                loss_history['total_loss'].append(loss.item())

            # Validation for early stopping
            self._model.eval()
            with torch.no_grad():
                X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
                val_logits, _ = self._model(X_val_t)
                val_preds = (torch.sigmoid(val_logits) > 0.5).cpu().numpy().astype(int)
                val_f1 = f1_score(y_val, val_preds, average='macro')

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        # Restore best model
        if best_state is not None:
            self._model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        self._fitted = True
        return loss_history

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Return prediction scores (sigmoid probabilities).

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Scores array (n_samples,) in range [0, 1]
        """
        if not self._fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        X_norm = self._normalize(X, fit=False)

        self._model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_norm, dtype=torch.float32).to(self.device)
            logits, _ = self._model(X_t)
            scores = torch.sigmoid(logits).cpu().numpy()

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

    def clone(self) -> "DANNClassifier":
        """Create unfitted copy with same configuration."""
        return DANNClassifier(
            config=copy.deepcopy(self.config),
            device=self.device,
        )

    def get_params(self) -> Dict[str, Any]:
        """Return configuration as dict for reproducibility."""
        return asdict(self.config)

    def save(self, path: str):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted model.")

        checkpoint = {
            'model_state_dict': self._model.state_dict(),
            'config': asdict(self.config),
            'input_dim': self._input_dim,
            'label_encoder_classes': self._label_encoder.classes_.tolist(),
        }

        if self._scaler is not None:
            checkpoint['scaler_mean'] = self._scaler.mean_ if hasattr(self._scaler, 'mean_') else None
            checkpoint['scaler_scale'] = self._scaler.scale_ if hasattr(self._scaler, 'scale_') else None

        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> "DANNClassifier":
        """
        Load model from checkpoint.

        Args:
            path: Path to checkpoint
            device: PyTorch device

        Returns:
            Loaded DANNClassifier
        """
        checkpoint = torch.load(path, map_location='cpu')

        config = DANNConfig(**checkpoint['config'])
        classifier = cls(config=config, device=device)
        classifier._input_dim = checkpoint['input_dim']

        # Restore label encoder
        classifier._label_encoder = LabelEncoder()
        classifier._label_encoder.classes_ = np.array(checkpoint['label_encoder_classes'])

        # Restore model
        n_datasets = len(classifier._label_encoder.classes_)
        classifier._model = DANNModule(
            input_dim=classifier._input_dim,
            hidden_layers=config.hidden_layers,
            n_datasets=n_datasets,
            dropout=config.dropout,
            activation=config.activation,
        ).to(classifier.device)
        classifier._model.load_state_dict(checkpoint['model_state_dict'])

        classifier._fitted = True
        return classifier

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return f"DANNClassifier(normalize={self.config.normalize}, domain_weight={self.config.domain_weight}, {status})"
