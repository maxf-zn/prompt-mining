"""
CompactGraph: Minimal run artifact with sparse features and metadata.
FeatureInfo: Single sparse tensor element (one position in activation space).
"""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import torch
import datetime


@dataclass(frozen=True)
class FeatureInfo:
    """
    One feature at one position - represents a single element in the sparse activation tensor.

    The activation matrix is (n_layers, seq_len, d_transcoder) and highly sparse.
    Each non-zero element becomes one FeatureInfo.

    Attributes:
        layer: Layer index (0-27 for 28-layer model)
        position: Token position in sequence
        feature_idx: Index into d_transcoder (0-16383)
        influence: Total influence on output (from graph traversal)
        activation_value: Magnitude at this (layer, position, feature_idx)
        logit_attribution: Direct effect on target logits

    Examples:
        # activation_matrix[10, 31, 5421] = 2.47  (sparse tensor)
        # This becomes:
        FeatureInfo(
            layer=10,
            position=31,
            feature_idx=5421,
            influence=0.034,
            activation_value=2.47,
            logit_attribution=0.021
        )

        # If the same feature (layer=10, idx=5421) fires at position 44:
        FeatureInfo(
            layer=10,
            position=44,  # Different position = separate FeatureInfo
            feature_idx=5421,
            influence=0.019,
            activation_value=1.9,
            logit_attribution=0.015
        )
    """
    layer: int
    position: int
    feature_idx: int
    influence: float
    activation_value: float
    logit_attribution: float

    def __post_init__(self):
        """Validate field values."""
        if self.layer < 0:
            raise ValueError(f"layer must be >= 0, got {self.layer}")
        if self.position < 0:
            raise ValueError(f"position must be >= 0, got {self.position}")
        if self.feature_idx < 0:
            raise ValueError(f"feature_idx must be >= 0, got {self.feature_idx}")

    def to_tuple(self) -> tuple:
        """Convert to tuple for tensor serialization."""
        return (
            self.layer,
            self.position,
            self.feature_idx,
            self.influence,
            self.activation_value,
            self.logit_attribution
        )

    @classmethod
    def from_tuple(cls, t: tuple) -> "FeatureInfo":
        """Load from tuple."""
        return cls(
            layer=int(t[0]),
            position=int(t[1]),
            feature_idx=int(t[2]),
            influence=float(t[3]),
            activation_value=float(t[4]),
            logit_attribution=float(t[5])
        )


@dataclass(frozen=True)
class CompactGraph:
    """
    Minimal run artifact - sparse features only, no adjacency matrix.

    This is the primary artifact stored per run. It contains:
    - Input tokens and generated logits
    - ~200 top features (sparse tensor elements)
    - Metadata for reproducibility

    Storage format: PyTorch .pt (native, fast, compact)
    Storage size: ~5KB per run (vs. ~100MB for full graph with adjacency)

    Attributes:
        prompt_id: Unique identifier for the prompt
        input_string: Original text input
        input_tokens: Tokenized input (shape: seq_len)
        logit_tokens: Top-K next token predictions
        logit_probabilities: Probabilities for top-K tokens
        features: List of ~200 FeatureInfo objects (configurable via top_k)
        n_layers: Number of layers in the model
        model_hash: Hash of model weights for versioning
        clt_hash: Hash of transcoder weights (if applicable)
        seed: Random seed for reproducibility
        created_at: ISO timestamp
        scan: Optional transcoder identifier (e.g., "gemma-scope-2b")
    """
    prompt_id: str
    input_string: str
    input_tokens: np.ndarray
    logit_tokens: np.ndarray
    logit_probabilities: np.ndarray
    features: List[FeatureInfo]
    n_layers: int
    model_hash: str
    clt_hash: str
    seed: int
    created_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat())
    scan: Optional[str] = None

    def __post_init__(self):
        """Validate CompactGraph fields."""
        if not self.prompt_id:
            raise ValueError("prompt_id cannot be empty")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be > 0, got {self.n_layers}")
        if len(self.input_tokens.shape) != 1:
            raise ValueError(f"input_tokens must be 1D array, got shape {self.input_tokens.shape}")
        # Features can be empty for raw-only runs (attribution disabled)

    def _features_to_tensor(self) -> torch.Tensor:
        """Convert features list to tensor for serialization (n_features, 6)."""
        feature_tuples = [f.to_tuple() for f in self.features]
        return torch.tensor(feature_tuples, dtype=torch.float32)

    @staticmethod
    def _tensor_to_features(tensor: torch.Tensor) -> List[FeatureInfo]:
        """Convert tensor back to features list."""
        features = []
        for row in tensor:
            features.append(FeatureInfo.from_tuple(tuple(row.tolist())))
        return features

    def to_pt(self, path: str):
        """
        Serialize to PyTorch .pt file.

        Args:
            path: Output file path (should end with .pt)
        """
        torch.save({
            'prompt_id': self.prompt_id,
            'input_string': self.input_string,
            'input_tokens': self.input_tokens,
            'logit_tokens': self.logit_tokens,
            'logit_probabilities': self.logit_probabilities,
            'feature_tensor': self._features_to_tensor(),
            'n_layers': self.n_layers,
            'model_hash': self.model_hash,
            'clt_hash': self.clt_hash,
            'seed': self.seed,
            'created_at': self.created_at,
            'scan': self.scan
        }, path)

    @staticmethod
    def from_pt(path: str) -> 'CompactGraph':
        """
        Deserialize from PyTorch .pt file.

        Args:
            path: Input file path

        Returns:
            CompactGraph instance
        """
        data = torch.load(path, weights_only=False)
        features = CompactGraph._tensor_to_features(data['feature_tensor'])

        return CompactGraph(
            prompt_id=data['prompt_id'],
            input_string=data['input_string'],
            input_tokens=data['input_tokens'],
            logit_tokens=data['logit_tokens'],
            logit_probabilities=data['logit_probabilities'],
            features=features,
            n_layers=data['n_layers'],
            model_hash=data['model_hash'],
            clt_hash=data['clt_hash'],
            seed=data['seed'],
            created_at=data['created_at'],
            scan=data.get('scan')
        )

    def to_dict(self) -> dict:
        """Convert to dictionary (for JSON serialization if needed)."""
        return {
            'prompt_id': self.prompt_id,
            'input_string': self.input_string,
            'input_tokens': self.input_tokens.tolist(),
            'logit_tokens': self.logit_tokens.tolist(),
            'logit_probabilities': self.logit_probabilities.tolist(),
            'features': [f.to_tuple() for f in self.features],
            'n_layers': self.n_layers,
            'model_hash': self.model_hash,
            'clt_hash': self.clt_hash,
            'seed': self.seed,
            'created_at': self.created_at,
            'scan': self.scan
        }

    def get_features_at_layer(self, layer: int) -> List[FeatureInfo]:
        """Get all features at a specific layer."""
        return [f for f in self.features if f.layer == layer]

    def get_features_at_position(self, position: int) -> List[FeatureInfo]:
        """Get all features at a specific position."""
        return [f for f in self.features if f.position == position]

    def get_top_features(self, k: int = 10) -> List[FeatureInfo]:
        """Get top-K features by influence."""
        sorted_features = sorted(self.features, key=lambda f: f.influence, reverse=True)
        return sorted_features[:k]

    @classmethod
    def from_graph(
        cls,
        graph,
        prompt_id: str,
        input_string: str,
        topk_tokens: np.ndarray,
        topk_probs: np.ndarray,
        model_hash: str,
        clt_hash: str,
        seed: int,
        n_layers: int,
        node_mask: torch.Tensor,
        node_influence: torch.Tensor
    ) -> 'CompactGraph':
        """
        Create CompactGraph from pruned circuit_tracer Graph object.

        Uses pre-computed node influences from prune_graph() to avoid recalculation.

        Args:
            graph: circuit_tracer Graph object with attribution data
            prompt_id: Unique identifier for the prompt
            input_string: Original text input
            topk_tokens: Top-K token IDs (numpy array)
            topk_probs: Top-K token probabilities (numpy array)
            model_hash: Model weight hash
            clt_hash: Transcoder weight hash
            seed: Random seed
            n_layers: Number of layers in model
            node_mask: Boolean mask from prune_graph() indicating which nodes to keep
            node_influence: Pre-computed node influences from compute_node_influence()

        Returns:
            CompactGraph instance with extracted features
        """
        device = graph.adjacency_matrix.device

        n_tokens = len(graph.input_tokens)
        n_logits = len(graph.logit_tokens)
        n_features = len(graph.selected_features)

        # Extract feature-specific data using pre-computed influences
        feature_influences = node_influence[:n_features]
        feature_activations = graph.activation_values[:n_features]

        # Get direct logit attributions (from adjacency_matrix rows for logits)
        logit_rows = graph.adjacency_matrix[-n_logits:, :n_features]
        logit_probs_on_device = graph.logit_probabilities.to(device)
        logit_attributions = (logit_rows.T * logit_probs_on_device).sum(dim=1)

        # Select features based on node_mask (only keep pruned features)
        feature_mask = node_mask[:n_features]

        # Build FeatureInfo list for all kept features
        features = []
        for idx in range(n_features):
            if not feature_mask[idx]:
                continue  # Skip features not in pruned graph

            # Get the actual feature coordinates from selected_features
            selected_idx = graph.selected_features[idx]
            layer, pos, feat_idx = graph.active_features[selected_idx].tolist()

            features.append(FeatureInfo(
                layer=int(layer),
                position=int(pos),
                feature_idx=int(feat_idx),
                influence=float(feature_influences[idx].item()),
                activation_value=float(feature_activations[idx].item()),
                logit_attribution=float(logit_attributions[idx].item())
            ))

        # Sort by influence (descending)
        features.sort(key=lambda f: f.influence, reverse=True)

        return cls(
            prompt_id=prompt_id,
            input_string=input_string,
            input_tokens=graph.input_tokens.cpu().numpy(),
            logit_tokens=topk_tokens,
            logit_probabilities=topk_probs,
            features=features,
            n_layers=n_layers,
            model_hash=model_hash,
            clt_hash=clt_hash,
            seed=seed,
            scan=str(graph.scan) if graph.scan else None
        )
