"""
UnsupervisedAnalyzer: Dimensionality reduction and clustering.

This analyzer performs unsupervised analysis on feature matrices:
- PCA/UMAP dimensionality reduction
- K-means/DBSCAN clustering
- 2D/3D scatter plot visualization
"""
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from prompt_mining.analysis.base import Analyzer, AnalysisData


class UnsupervisedAnalyzer(Analyzer):
    """
    Perform unsupervised analysis on feature incidence or activation matrices.

    This analyzer:
    1. Reduces dimensionality via PCA
    2. Clusters prompts via K-means or DBSCAN
    3. Visualizes in 2D/3D scatter plots
    4. Identifies cluster characteristics

    Supports two modes:
    - Feature incidence (via run_on_features()): Analyze sparse SAE feature activations
    - Dense activations (via run_on_activations()): Analyze raw/PLT activation vectors

    Example (feature incidence):
        >>> analyzer = UnsupervisedAnalyzer(n_components=50, n_clusters=5)
        >>> results = analyzer.run_on_features(
        ...     analysis_data, top_k=200, max_ubiquity=0.8, matrix_format='binary'
        ... )
        >>> print(results['cluster_sizes'])

    Example (PLT activations):
        >>> analyzer = UnsupervisedAnalyzer(n_components=50, n_clusters=5)
        >>> results = analyzer.run_on_activations(
        ...     analysis_data, layer=10, position='last', space='plt'
        ... )
        >>> print(f"Silhouette score: {results['silhouette_score']}")
    """

    def __init__(
        self,
        n_components: int = 50,
        clustering_method: str = 'kmeans',
        n_clusters: int = 5,
        random_state: int = 42
    ):
        """
        Initialize UnsupervisedAnalyzer.

        Args:
            n_components: Number of PCA components
            clustering_method: 'kmeans' or 'dbscan'
            n_clusters: Number of clusters (for k-means)
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters
        self.random_state = random_state

    def run(self, data: AnalysisData, **kwargs) -> Dict[str, Any]:
        """
        Use run_on_features() or run_on_activations() instead.
        """
        raise NotImplementedError(
            "Use run_on_features() or run_on_activations() for unsupervised analysis"
        )

    def run_on_features(
        self,
        data: AnalysisData,
        top_k: int = 200,
        max_ubiquity: float = 1.0,
        matrix_format: str = 'binary',
        standardize: bool = True
    ) -> Dict[str, Any]:
        """
        Run unsupervised analysis on feature incidence.

        Args:
            data: AnalysisData accessor
            top_k: Number of top features to use
            max_ubiquity: Exclude features appearing in more than this fraction of prompts
                         (e.g., 0.4 filters features in >40% of prompts)
            matrix_format: Format for feature matrix ('binary', 'influence', 'activation')
            standardize: Whether to standardize features before PCA (default: True)

        Returns:
            Dictionary with:
                - pca: Fitted PCA object
                - reduced_data: Reduced data (n_prompts, n_components)
                - cluster_labels: Cluster assignments (n_prompts,)
                - cluster_centers: Cluster centers (n_clusters, n_components)
                - cluster_sizes: Number of prompts per cluster
                - explained_variance: Explained variance ratio
                - silhouette_score: Clustering quality metric
                - metadata: Analysis metadata
        """
        # Load feature matrix
        feature_matrix = data.get_feature_matrix(
            top_k=top_k,
            max_ubiquity=max_ubiquity,
            format=matrix_format
        )

        X = feature_matrix.matrix  # (n_prompts, top_k)

        # Build metadata
        metadata = {
            'mode': 'features',
            'top_k': top_k,
            'max_ubiquity': max_ubiquity,
            'matrix_format': matrix_format,
            'filters': data.filters,
        }

        # Run shared analysis pipeline
        return self._run_analysis_pipeline(
            X=X,
            prompt_ids=feature_matrix.prompt_ids,
            run_ids=feature_matrix.run_ids,
            standardize=standardize,
            metadata=metadata
        )

    def run_on_activations(
        self,
        data: AnalysisData,
        layer: int,
        position: str = 'last',
        space: str = 'plt',
        standardize: bool = True
    ) -> Dict[str, Any]:
        """
        Run unsupervised analysis on dense activation vectors.

        This mode uses get_activation_matrix instead of get_feature_matrix,
        allowing PCA/clustering on raw model activations or PLT space.

        Args:
            data: AnalysisData accessor
            layer: Which layer to analyze (0-27 for 28-layer model)
            position: Token position ('last', 'first', or index)
            space: Activation space ('plt' or 'raw')
            standardize: Whether to standardize features before PCA (default: True)

        Returns:
            Dictionary with PCA/clustering results (same structure as run_on_features())
        """

        # Load activation matrix
        activation_matrix = data.get_activation_matrix(
            layer=layer,
            position=position,
            space=space
        )

        X = activation_matrix.matrix  # (n_prompts, d_model or d_transcoder)

        # Build metadata
        metadata = {
            'mode': 'activations',
            'layer': layer,
            'position': position,
            'space': space,
            'filters': data.filters,
        }

        # Run shared analysis pipeline
        return self._run_analysis_pipeline(
            X=X,
            prompt_ids=activation_matrix.prompt_ids,
            run_ids=activation_matrix.run_ids,
            standardize=standardize,
            metadata=metadata
        )

    def _run_analysis_pipeline(
        self,
        X: np.ndarray,
        prompt_ids: List[str],
        run_ids: List[str],
        standardize: bool,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Unified analysis pipeline for both feature and activation modes.

        Args:
            X: Data matrix (n_prompts, n_features)
            prompt_ids: List of prompt IDs
            run_ids: List of run IDs
            standardize: Whether to standardize before PCA
            metadata: Mode-specific metadata dict

        Returns:
            Dictionary with PCA/clustering results
        """
        # Standardize features (recommended for PCA)
        if standardize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X
            scaler = None

        # PCA dimensionality reduction
        n_components = min(self.n_components, X_scaled.shape[0], X_scaled.shape[1])
        pca = PCA(n_components=n_components, random_state=self.random_state)
        X_reduced = pca.fit_transform(X_scaled)

        # Clustering
        if self.clustering_method == 'kmeans':
            cluster_labels, cluster_centers = self._kmeans_clustering(X_reduced)
        elif self.clustering_method == 'dbscan':
            cluster_labels, cluster_centers = self._dbscan_clustering(X_reduced)
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")

        # Compute cluster sizes
        cluster_sizes = self._compute_cluster_sizes(cluster_labels)

        # Compute silhouette score (if enough samples)
        silhouette_score = None
        if len(set(cluster_labels)) > 1 and len(cluster_labels) > 10:
            from sklearn.metrics import silhouette_score as compute_silhouette
            silhouette_score = float(compute_silhouette(X_reduced, cluster_labels))

        # Merge metadata
        full_metadata = {
            'n_components': n_components,
            'clustering_method': self.clustering_method,
            'n_clusters': self.n_clusters,
            'standardize': standardize,
            'n_prompts': X.shape[0],
            'n_features': X.shape[1],
            **metadata
        }

        return {
            'pca': pca,
            'scaler': scaler,
            'reduced_data': X_reduced,
            'cluster_labels': cluster_labels,
            'cluster_centers': cluster_centers,
            'cluster_sizes': cluster_sizes,
            'explained_variance': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'silhouette_score': silhouette_score,
            'prompt_ids': prompt_ids,
            'run_ids': run_ids,
            'metadata': full_metadata
        }

    def _kmeans_clustering(self, X: np.ndarray) -> tuple:
        """
        Perform K-means clustering.

        Args:
            X: Reduced data (n_prompts, n_components)

        Returns:
            (cluster_labels, cluster_centers)
        """
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        cluster_labels = kmeans.fit_predict(X)
        cluster_centers = kmeans.cluster_centers_

        return cluster_labels, cluster_centers

    def _dbscan_clustering(self, X: np.ndarray) -> tuple:
        """
        Perform DBSCAN clustering.

        Args:
            X: Reduced data (n_prompts, n_components)

        Returns:
            (cluster_labels, cluster_centers)
        """
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        cluster_labels = dbscan.fit_predict(X)

        # Compute cluster centers (excluding noise points labeled -1)
        unique_labels = set(cluster_labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        cluster_centers = []
        for label in sorted(unique_labels):
            mask = cluster_labels == label
            center = X[mask].mean(axis=0)
            cluster_centers.append(center)

        cluster_centers = np.array(cluster_centers) if cluster_centers else np.array([])

        return cluster_labels, cluster_centers

    def _compute_cluster_sizes(self, cluster_labels: np.ndarray) -> Dict[int, int]:
        """
        Compute number of samples per cluster.

        Args:
            cluster_labels: Cluster assignments (n_prompts,)

        Returns:
            Dictionary mapping cluster_id -> count
        """
        unique, counts = np.unique(cluster_labels, return_counts=True)
        return {int(label): int(count) for label, count in zip(unique, counts)}

    def save_results(self, results: Dict[str, Any], output_dir: Path):
        """
        Save unsupervised analysis results.

        Saves:
        - pca_components.npy: PCA components
        - reduced_data.npy: Reduced data
        - cluster_labels.npy: Cluster assignments
        - cluster_analysis.json: Cluster statistics
        - explained_variance_plot.png: Scree plot
        - scatter_2d.png: 2D scatter plot
        - scatter_3d.png: 3D scatter plot (if n_components >= 3)
        - metadata.json: Analysis metadata

        Args:
            results: Results dictionary from run()
            output_dir: Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save numpy arrays
        np.save(output_dir / 'pca_components.npy', results['pca'].components_)
        np.save(output_dir / 'reduced_data.npy', results['reduced_data'])
        np.save(output_dir / 'cluster_labels.npy', results['cluster_labels'])

        # Save JSON results
        json_results = {
            'cluster_sizes': results['cluster_sizes'],
            'explained_variance': results['explained_variance'][:20],  # First 20 components
            'cumulative_variance': results['cumulative_variance'][:20],
            'silhouette_score': results['silhouette_score'],
            'metadata': results['metadata'],
        }

        with open(output_dir / 'unsupervised_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)

        # Save cluster assignments with prompt IDs
        import pandas as pd
        cluster_df = pd.DataFrame({
            'prompt_id': results['prompt_ids'],
            'run_id': results['run_ids'],
            'cluster': results['cluster_labels'],
        })
        cluster_df.to_csv(output_dir / 'cluster_assignments.csv', index=False)

        # Plot explained variance
        self._plot_explained_variance(
            results['explained_variance'],
            results['cumulative_variance'],
            output_dir / 'explained_variance_plot.png'
        )

        # Plot 2D scatter
        self._plot_scatter_2d(
            results['reduced_data'],
            results['cluster_labels'],
            output_dir / 'scatter_2d.png'
        )

        # Plot 3D scatter (if enough components)
        if results['reduced_data'].shape[1] >= 3:
            self._plot_scatter_3d(
                results['reduced_data'],
                results['cluster_labels'],
                output_dir / 'scatter_3d.png'
            )

        print(f"✅ Unsupervised analysis results saved to {output_dir}/")
        print(f"   - unsupervised_results.json")
        print(f"   - cluster_assignments.csv")
        print(f"   - pca_components.npy, reduced_data.npy, cluster_labels.npy")
        print(f"   - explained_variance_plot.png")
        print(f"   - scatter_2d.png, scatter_3d.png")
        if results['silhouette_score'] is not None:
            print(f"\n   Silhouette Score: {results['silhouette_score']:.3f}")

    def _plot_explained_variance(
        self,
        explained_variance: List[float],
        cumulative_variance: List[float],
        output_path: Path
    ):
        """
        Plot explained variance (scree plot).

        Args:
            explained_variance: Explained variance ratio per component
            cumulative_variance: Cumulative explained variance
            output_path: Path to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        n_components = len(explained_variance)
        x = np.arange(1, n_components + 1)

        # Plot explained variance per component
        ax1.bar(x, explained_variance, alpha=0.7, color='blue')
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Explained Variance per Component')
        ax1.grid(axis='y', alpha=0.3)

        # Plot cumulative explained variance
        ax2.plot(x, cumulative_variance, marker='o', linewidth=2, color='red')
        ax2.axhline(y=0.9, color='gray', linestyle='--', label='90% threshold')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Explained Variance')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_scatter_2d(
        self,
        reduced_data: np.ndarray,
        cluster_labels: np.ndarray,
        output_path: Path
    ):
        """
        Plot 2D scatter of first two principal components.

        Args:
            reduced_data: Reduced data (n_prompts, n_components)
            cluster_labels: Cluster assignments
            output_path: Path to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Get unique cluster labels
        unique_labels = np.unique(cluster_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        # Plot each cluster
        for i, label in enumerate(unique_labels):
            mask = cluster_labels == label
            label_str = f"Cluster {label}" if label >= 0 else "Noise"
            ax.scatter(
                reduced_data[mask, 0],
                reduced_data[mask, 1],
                c=[colors[i]],
                label=label_str,
                alpha=0.6,
                s=50
            )

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('2D Projection (First Two Principal Components)')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_scatter_3d(
        self,
        reduced_data: np.ndarray,
        cluster_labels: np.ndarray,
        output_path: Path
    ):
        """
        Plot 3D scatter of first three principal components.

        Args:
            reduced_data: Reduced data (n_prompts, n_components)
            cluster_labels: Cluster assignments
            output_path: Path to save plot
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Get unique cluster labels
        unique_labels = np.unique(cluster_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        # Plot each cluster
        for i, label in enumerate(unique_labels):
            mask = cluster_labels == label
            label_str = f"Cluster {label}" if label >= 0 else "Noise"
            ax.scatter(
                reduced_data[mask, 0],
                reduced_data[mask, 1],
                reduced_data[mask, 2],
                c=[colors[i]],
                label=label_str,
                alpha=0.6,
                s=50
            )

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('3D Projection (First Three Principal Components)')
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
