"""
ComparisonAnalyzer: Frequency and co-occurrence analysis across prompts.

This analyzer computes:
- Feature frequency (how many prompts activate each feature)
- PMI (Pointwise Mutual Information) between feature pairs
- Co-occurrence matrix
- Top motifs (frequently co-occurring feature pairs)
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json

from prompt_mining.analysis.base import Analyzer, AnalysisData, FeatureMatrix


class ComparisonAnalyzer(Analyzer):
    """
    Analyze feature frequency and co-occurrence patterns across prompts.

    This is the simplest and most useful analyzer - it identifies which features
    are common across many prompts and which features tend to co-occur.

    Example:
        >>> analyzer = ComparisonAnalyzer(top_k=200, max_ubiquity=0.9)
        >>> results = analyzer.run(analysis_data)
        >>> print(results['frequency_stats']['most_frequent'][:5])
        >>> print(results['pmi_matrix'].shape)
    """

    def __init__(
        self,
        top_k: int = 200,
        max_ubiquity: float = 1.0,
        pmi_top_pairs: int = 100
    ):
        """
        Initialize ComparisonAnalyzer.

        Args:
            top_k: Number of top features to analyze
            max_ubiquity: Exclude features appearing in more than this fraction of prompts
                         (e.g., 0.9 = exclude features in >90% of prompts)
            pmi_top_pairs: Number of top PMI pairs to report
        """
        self.top_k = top_k
        self.max_ubiquity = max_ubiquity
        self.pmi_top_pairs = pmi_top_pairs

    def run(self, data: AnalysisData, **kwargs) -> Dict[str, Any]:
        """
        Run comparison analysis.

        Args:
            data: AnalysisData accessor
            **kwargs: Additional parameters (unused)

        Returns:
            Dictionary with:
                - frequency_stats: Per-feature frequency statistics
                - pmi_matrix: Pointwise mutual information matrix (top_k × top_k)
                - top_pmi_pairs: Top PMI feature pairs
                - co_occurrence_matrix: Co-occurrence counts (top_k × top_k)
                - metadata: Analysis metadata
        """
        # Load feature matrix (binary format for frequency/co-occurrence)
        feature_matrix = data.get_feature_matrix(
            top_k=self.top_k,
            format='binary',
            max_ubiquity=self.max_ubiquity
        )

        n_prompts, n_features = feature_matrix.matrix.shape

        # 1. Feature Frequency
        frequency_stats = self._compute_frequency(feature_matrix)

        # 2. Co-occurrence Matrix
        co_occurrence_matrix = self._compute_cooccurrence(feature_matrix.matrix)

        # 3. PMI Matrix
        pmi_matrix = self._compute_pmi(
            co_occurrence_matrix,
            frequency_stats['frequencies'],
            n_prompts
        )

        # 4. Top PMI Pairs
        top_pmi_pairs = self._extract_top_pairs(
            pmi_matrix,
            feature_matrix.feature_coords,
            self.pmi_top_pairs
        )

        return {
            'frequency_stats': frequency_stats,
            'pmi_matrix': pmi_matrix,
            'top_pmi_pairs': top_pmi_pairs,
            'co_occurrence_matrix': co_occurrence_matrix,
            'metadata': {
                'top_k': self.top_k,
                'max_ubiquity': self.max_ubiquity,
                'n_prompts': n_prompts,
                'n_features': n_features,
                'filters': feature_matrix.metadata.get('filters', {}),
            }
        }

    def _compute_frequency(self, feature_matrix: FeatureMatrix) -> Dict[str, Any]:
        """
        Compute feature frequency statistics.

        Args:
            feature_matrix: Binary feature matrix

        Returns:
            Dictionary with frequency stats
        """
        # Count prompts per feature (column sums)
        frequencies = feature_matrix.matrix.sum(axis=0)  # (n_features,)

        # Create feature -> count mapping
        feature_to_count = {
            feature_matrix.feature_coords[i]: int(frequencies[i])
            for i in range(len(frequencies))
        }

        # Sort by frequency (descending)
        sorted_features = sorted(
            feature_to_count.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            'frequencies': frequencies,
            'feature_to_count': feature_to_count,
            'most_frequent': sorted_features[:20],  # Top 20
            'least_frequent': sorted_features[-20:],  # Bottom 20
            'mean_frequency': float(frequencies.mean()),
            'median_frequency': float(np.median(frequencies)),
            'std_frequency': float(frequencies.std()),
        }

    def _compute_cooccurrence(self, binary_matrix: np.ndarray) -> np.ndarray:
        """
        Compute co-occurrence matrix.

        Co-occurrence[i, j] = number of prompts where both feature i and j are active.

        Args:
            binary_matrix: Binary feature matrix (n_prompts, n_features)

        Returns:
            Co-occurrence matrix (n_features, n_features)
        """
        # Matrix multiplication: F^T @ F
        # Where F is the binary feature matrix
        co_occurrence = binary_matrix.T @ binary_matrix  # (n_features, n_features)

        return co_occurrence.astype(np.int32)

    def _compute_pmi(
        self,
        co_occurrence: np.ndarray,
        frequencies: np.ndarray,
        n_prompts: int
    ) -> np.ndarray:
        """
        Compute Pointwise Mutual Information (PMI) matrix.

        PMI(i, j) = log2( P(i, j) / (P(i) * P(j)) )
                  = log2( co_occurrence[i,j] * n_prompts / (freq[i] * freq[j]) )

        Where:
        - P(i, j) = co_occurrence[i, j] / n_prompts
        - P(i) = freq[i] / n_prompts

        Args:
            co_occurrence: Co-occurrence matrix (n_features, n_features)
            frequencies: Feature frequencies (n_features,)
            n_prompts: Total number of prompts

        Returns:
            PMI matrix (n_features, n_features)
        """
        n_features = len(frequencies)

        # Compute P(i, j) - joint probability
        p_joint = co_occurrence / n_prompts

        # Compute P(i) * P(j) - independent probabilities
        p_i = frequencies / n_prompts
        p_independent = np.outer(p_i, p_i)  # (n_features, n_features)

        # PMI = log2(P(i,j) / (P(i) * P(j)))
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        pmi_matrix = np.log2((p_joint + epsilon) / (p_independent + epsilon))

        # Set diagonal to 0 (PMI of feature with itself is not meaningful)
        np.fill_diagonal(pmi_matrix, 0)

        return pmi_matrix

    def _extract_top_pairs(
        self,
        pmi_matrix: np.ndarray,
        feature_coords: List[Tuple[int, int]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Extract top-K feature pairs by PMI score.

        Args:
            pmi_matrix: PMI matrix (n_features, n_features)
            feature_coords: List of (layer, feature_idx) tuples
            top_k: Number of top pairs to extract

        Returns:
            List of dictionaries with feature pair information
        """
        n_features = len(feature_coords)

        # Get upper triangle indices (avoid duplicates and diagonal)
        triu_indices = np.triu_indices(n_features, k=1)

        # Get PMI values for upper triangle
        pmi_values = pmi_matrix[triu_indices]

        # Sort by PMI (descending)
        sorted_indices = np.argsort(pmi_values)[::-1][:top_k]

        # Extract top pairs
        top_pairs = []
        for idx in sorted_indices:
            i = triu_indices[0][idx]
            j = triu_indices[1][idx]
            pmi = float(pmi_values[idx])

            top_pairs.append({
                'feature_1': feature_coords[i],
                'feature_2': feature_coords[j],
                'pmi': pmi,
                'feature_1_str': f"layer{feature_coords[i][0]}_feat{feature_coords[i][1]}",
                'feature_2_str': f"layer{feature_coords[j][0]}_feat{feature_coords[j][1]}",
            })

        return top_pairs

    def save_results(self, results: Dict[str, Any], output_dir: Path):
        """
        Save comparison analysis results.

        Saves:
        - frequency_stats.json: Feature frequency statistics
        - top_pmi_pairs.json: Top PMI feature pairs
        - pmi_matrix.npy: Full PMI matrix
        - co_occurrence_matrix.npy: Full co-occurrence matrix
        - metadata.json: Analysis metadata

        Args:
            results: Results dictionary from run()
            output_dir: Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON results
        json_results = {
            'frequency_stats': {
                'most_frequent': results['frequency_stats']['most_frequent'],
                'least_frequent': results['frequency_stats']['least_frequent'],
                'mean_frequency': results['frequency_stats']['mean_frequency'],
                'median_frequency': results['frequency_stats']['median_frequency'],
                'std_frequency': results['frequency_stats']['std_frequency'],
            },
            'top_pmi_pairs': results['top_pmi_pairs'],
            'metadata': results['metadata'],
        }

        with open(output_dir / 'comparison_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)

        # Save numpy arrays
        np.save(output_dir / 'pmi_matrix.npy', results['pmi_matrix'])
        np.save(output_dir / 'co_occurrence_matrix.npy', results['co_occurrence_matrix'])

        # Save feature frequencies as CSV for easy inspection
        import pandas as pd
        freq_data = []
        feature_to_count = results['frequency_stats']['feature_to_count']
        for (layer, feat_idx), count in feature_to_count.items():
            freq_data.append({
                'layer': layer,
                'feature_idx': feat_idx,
                'count': count,
                'fraction': count / results['metadata']['n_prompts']
            })

        df = pd.DataFrame(freq_data).sort_values('count', ascending=False)
        df.to_csv(output_dir / 'feature_frequencies.csv', index=False)

        print(f"✅ Comparison analysis results saved to {output_dir}/")
        print(f"   - comparison_results.json")
        print(f"   - feature_frequencies.csv")
        print(f"   - pmi_matrix.npy")
        print(f"   - co_occurrence_matrix.npy")

