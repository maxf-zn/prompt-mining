"""
DirectionsAnalyzer: Compute steering vectors from activation matrices.

This analyzer computes direction vectors between two clusters (e.g., success vs. failure)
and evaluates their separability.
"""
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json
import matplotlib.pyplot as plt

from prompt_mining.analysis.base import Analyzer, AnalysisData
from prompt_mining.analysis.utilities import (
    compute_direction,
    compute_separability,
    compute_projection
)


class DirectionsAnalyzer(Analyzer):
    """
    Compute steering vectors between two clusters of activations.

    This analyzer:
    1. Loads activation matrices for two groups (e.g., success vs. failure)
    2. Computes direction vector: d = normalize(mean_B - mean_A)
    3. Evaluates separability (L2 distance, cosine similarity, etc.)
    4. Projects all activations onto direction for visualization

    Example:
        >>> # Create two filtered data accessors
        >>> success_data = AnalysisData(registry, storage, filters={'labels': {'success': True}})
        >>> failure_data = AnalysisData(registry, storage, filters={'labels': {'success': False}})
        >>>
        >>> # Compute direction
        >>> analyzer = DirectionsAnalyzer(layer=10, position='last')
        >>> results = analyzer.run_pairwise(success_data, failure_data)
        >>> print(results['direction'].shape)  # (d_transcoder,)
        >>> print(results['separability_metrics'])
    """

    def __init__(
        self,
        layer: int,
        position: str = 'last',
        space: str = 'plt',
        normalize: bool = True,
        load_sparse: bool = True
    ):
        """
        Initialize DirectionsAnalyzer.

        Args:
            layer: Which layer to analyze (0-27 for 28-layer model)
            position: Token position ('last', 'first', or index)
            space: 'raw' (d_model) or 'plt' (d_transcoder)
            normalize: Whether to normalize direction vectors
            load_sparse: If True, load PLT activations in sparse format for 2-5x speedup
                        (automatically converted to dense for analysis)
        """
        self.layer = layer
        self.position = position
        self.space = space
        self.normalize = normalize
        self.load_sparse = load_sparse

    def run(self, data: AnalysisData, **kwargs) -> Dict[str, Any]:
        """
        Run direction analysis on a single data accessor.

        This is primarily for loading activations; use run_pairwise() for
        computing directions between two clusters.

        Args:
            data: AnalysisData accessor
            **kwargs: Additional parameters (unused)

        Returns:
            Dictionary with activation matrix and metadata
        """
        # Load activation matrix (sparse for PLT space if enabled)
        return_sparse = self.load_sparse and self.space == 'plt'

        activation_matrix = data.get_activation_matrix(
            layer=self.layer,
            position=self.position,
            space=self.space,
            return_sparse=return_sparse
        )

        return {
            'activation_matrix': activation_matrix.matrix,
            'prompt_ids': activation_matrix.prompt_ids,
            'run_ids': activation_matrix.run_ids,
            'metadata': activation_matrix.metadata,
        }

    def run_pairwise(
        self,
        data_a: AnalysisData,
        data_b: AnalysisData,
        label_a: str = 'cluster_a',
        label_b: str = 'cluster_b'
    ) -> Dict[str, Any]:
        """
        Compute direction between two clusters.

        Args:
            data_a: AnalysisData for first cluster (e.g., successful attacks)
            data_b: AnalysisData for second cluster (e.g., unsuccessful attacks)
            label_a: Human-readable label for cluster A
            label_b: Human-readable label for cluster B

        Returns:
            Dictionary with:
                - direction: Direction vector (d_transcoder,)
                - separability_metrics: L2 distance, cosine similarity, etc.
                - cluster_a_stats: Statistics for cluster A
                - cluster_b_stats: Statistics for cluster B
                - projections_a: Projections of cluster A onto direction
                - projections_b: Projections of cluster B onto direction
                - metadata: Analysis metadata
        """
        # Load activation matrices (sparse for PLT space if enabled)
        return_sparse = self.load_sparse and self.space == 'plt'

        act_matrix_a = data_a.get_activation_matrix(
            layer=self.layer,
            position=self.position,
            space=self.space,
            return_sparse=return_sparse
        )

        act_matrix_b = data_b.get_activation_matrix(
            layer=self.layer,
            position=self.position,
            space=self.space,
            return_sparse=return_sparse
        )

        # Convert to dense if needed (for torch operations)
        matrix_a = act_matrix_a.matrix
        matrix_b = act_matrix_b.matrix

        # Convert sparse to dense for torch operations
        from scipy.sparse import issparse
        if issparse(matrix_a):
            matrix_a = matrix_a.toarray()
        if issparse(matrix_b):
            matrix_b = matrix_b.toarray()

        # Convert to torch tensors
        cluster_a = torch.from_numpy(matrix_a)
        cluster_b = torch.from_numpy(matrix_b)

        # Compute direction
        direction = compute_direction(cluster_a, cluster_b, normalize=self.normalize)

        # Compute separability metrics
        separability_metrics = compute_separability(cluster_a, cluster_b)

        # Compute cluster statistics
        cluster_a_stats = self._compute_cluster_stats(cluster_a, label_a)
        cluster_b_stats = self._compute_cluster_stats(cluster_b, label_b)

        # Project activations onto direction (use dense matrices)
        direction_np = direction.cpu().numpy()
        projections_a = compute_projection(matrix_a, direction_np)
        projections_b = compute_projection(matrix_b, direction_np)

        return {
            'direction': direction_np,
            'separability_metrics': separability_metrics,
            'cluster_a_stats': cluster_a_stats,
            'cluster_b_stats': cluster_b_stats,
            'projections_a': projections_a,
            'projections_b': projections_b,
            'metadata': {
                'layer': self.layer,
                'position': self.position,
                'space': self.space,
                'normalize': self.normalize,
                'load_sparse': self.load_sparse,
                'label_a': label_a,
                'label_b': label_b,
                'n_samples_a': len(act_matrix_a.prompt_ids),
                'n_samples_b': len(act_matrix_b.prompt_ids),
                'd_model': matrix_a.shape[1],
                'filters_a': data_a.filters,
                'filters_b': data_b.filters,
            }
        }

    def _compute_cluster_stats(
        self,
        cluster: torch.Tensor,
        label: str
    ) -> Dict[str, Any]:
        """
        Compute statistics for a cluster of activations.

        Args:
            cluster: Activation tensor (n_samples, d_model)
            label: Cluster label

        Returns:
            Dictionary with statistics
        """
        cluster_np = cluster.cpu().numpy()

        return {
            'label': label,
            'n_samples': cluster.shape[0],
            'd_model': cluster.shape[1],
            'mean_norm': float(np.linalg.norm(cluster_np.mean(axis=0))),
            'mean_activation': float(cluster_np.mean()),
            'std_activation': float(cluster_np.std()),
            'min_activation': float(cluster_np.min()),
            'max_activation': float(cluster_np.max()),
        }

    def save_results(self, results: Dict[str, Any], output_dir: Path):
        """
        Save direction analysis results.

        Saves:
        - direction_vector.npy: Direction vector
        - separability_metrics.json: Separability metrics
        - cluster_stats.json: Statistics for both clusters
        - projection_histogram.png: Histogram of projections
        - metadata.json: Analysis metadata

        Args:
            results: Results dictionary from run_pairwise()
            output_dir: Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save direction vector
        np.save(output_dir / 'direction_vector.npy', results['direction'])

        # Save JSON results
        json_results = {
            'separability_metrics': results['separability_metrics'],
            'cluster_a_stats': results['cluster_a_stats'],
            'cluster_b_stats': results['cluster_b_stats'],
            'metadata': results['metadata'],
        }

        with open(output_dir / 'directions_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)

        # Save projections
        np.save(output_dir / 'projections_a.npy', results['projections_a'])
        np.save(output_dir / 'projections_b.npy', results['projections_b'])

        # Plot projection histogram
        self._plot_projection_histogram(
            results['projections_a'],
            results['projections_b'],
            results['metadata']['label_a'],
            results['metadata']['label_b'],
            output_dir / 'projection_histogram.png'
        )

        print(f"✅ Direction analysis results saved to {output_dir}/")
        print(f"   - direction_vector.npy")
        print(f"   - directions_results.json")
        print(f"   - projections_a.npy, projections_b.npy")
        print(f"   - projection_histogram.png")

    def _plot_projection_histogram(
        self,
        projections_a: np.ndarray,
        projections_b: np.ndarray,
        label_a: str,
        label_b: str,
        output_path: Path
    ):
        """
        Plot histogram of projections onto direction.

        Args:
            projections_a: Projections for cluster A
            projections_b: Projections for cluster B
            label_a: Label for cluster A
            label_b: Label for cluster B
            output_path: Path to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot histograms
        bins = 50
        ax.hist(projections_a, bins=bins, alpha=0.6, label=label_a, color='blue', density=True)
        ax.hist(projections_b, bins=bins, alpha=0.6, label=label_b, color='red', density=True)

        # Add vertical lines for means
        ax.axvline(projections_a.mean(), color='blue', linestyle='--', linewidth=2,
                   label=f'{label_a} mean')
        ax.axvline(projections_b.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'{label_b} mean')

        ax.set_xlabel('Projection onto Direction')
        ax.set_ylabel('Density')
        ax.set_title(f'Activation Projections: {label_a} vs. {label_b}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
