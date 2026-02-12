"""
Data structures for analysis pipeline.

Contains container classes for feature and activation matrices.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class FeatureMatrix:
    """
    Prompt×Feature matrix built from feature_incidence tables.

    Attributes:
        matrix: Binary/influence/activation matrix (n_prompts, n_features) in bfloat16
        prompt_ids: List of prompt IDs corresponding to rows
        feature_coords: List of (layer, feature_idx) tuples corresponding to columns
        metadata: Additional information (top_k, format, filters, etc.)
        run_ids: List of run IDs corresponding to rows (for traceability)
    """
    matrix: np.ndarray  # Shape: (n_prompts, n_features)
    prompt_ids: List[str]
    feature_coords: List[Tuple[int, int]]  # (layer, feature_idx)
    metadata: Dict[str, Any]
    run_ids: List[str]

    def __post_init__(self):
        """Validate dimensions."""
        assert self.matrix.shape[0] == len(self.prompt_ids) == len(self.run_ids), \
            "Matrix rows must match prompt_ids and run_ids length"
        assert self.matrix.shape[1] == len(self.feature_coords), \
            "Matrix columns must match feature_coords length"


@dataclass
class ActivationMatrix:
    """
    Prompt×d_sae matrix built from Zarr activation arrays or sparse parquet.

    Works with both circuit_tracer transcoders and SAELens SAEs.

    Attributes:
        matrix: Activation matrix. Shape depends on position:
            - Single position (str): (n_prompts, d_model) or (n_prompts, d_sae)
            - Multiple positions (List[int], raw only): (n_prompts, n_positions, d_model)
              (Special case: a singleton list like [-5] is treated as a single position and
               returns (n_prompts, d_model) for convenience.)
            Can be dense numpy array or scipy.sparse.csr_matrix (sparse only for single position)
        prompt_ids: List of prompt IDs corresponding to rows
        metadata: Additional information (layer, position, space, filters, is_sparse, etc.)
        run_ids: List of run IDs corresponding to rows (for traceability)
    """
    matrix: Any  # np.ndarray or scipy.sparse.csr_matrix
    prompt_ids: List[str]
    metadata: Dict[str, Any]
    run_ids: List[str]

    def __post_init__(self):
        """Validate dimensions."""
        assert self.matrix.shape[0] == len(self.prompt_ids) == len(self.run_ids), \
            "Matrix rows must match prompt_ids and run_ids length"
