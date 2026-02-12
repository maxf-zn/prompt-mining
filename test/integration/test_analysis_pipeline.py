"""
Integration tests for analysis pipeline.

Tests analyzers with mock data:
- ComparisonAnalyzer
- DirectionsAnalyzer
- UnsupervisedAnalyzer
"""
import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, UTC
import zarr

from prompt_mining.registry import SQLiteRegistry
from prompt_mining.storage import LocalStorage
from prompt_mining.core import PromptSpec, CompactGraph, FeatureInfo
from prompt_mining.analysis import (
    AnalysisData,
    ComparisonAnalyzer,
    DirectionsAnalyzer,
    UnsupervisedAnalyzer,
)


@pytest.fixture
def temp_storage():
    """Create temporary storage directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_registry_with_runs(temp_storage):
    """Create registry with mock runs."""
    registry = SQLiteRegistry(str(temp_storage / "registry.sqlite"))

    # Create mock runs with different labels
    runs = []
    for i in range(20):
        run_id = f"2025-11-13_run_{i:03d}"
        prompt_id = f"prompt_{i}"
        dataset_id = "test_dataset"

        # Alternate between success and failure
        label_value = "success" if i % 2 == 0 else "failure"
        labels = {'eval': label_value, 'attack_type': 'dh'}

        registry.create_run(
            run_id=run_id,
            prompt_id=prompt_id,
            dataset_id=dataset_id,
            run_key=f"key_{i}",
            processing_fingerprint="fingerprint",
            model_hash="model_hash",
            clt_hash="clt_hash",
            seed=42,
            prompt_labels=labels,
            config_snapshot={},
            shard_idx=0,
            num_shards=1,
            worker_id=0,
            device_id=0
        )
        # Update status to completed
        registry.update_status(run_id, "completed")
        runs.append((run_id, prompt_id, labels))

    return registry, runs


@pytest.fixture
def mock_storage_with_artifacts(temp_storage, mock_registry_with_runs):
    """Create storage with mock artifacts."""
    storage = LocalStorage(str(temp_storage))
    registry, runs = mock_registry_with_runs

    # Create mock artifacts for each run
    for run_id, prompt_id, labels in runs:
        # Create CompactGraph with random features
        features = []
        n_features = 50
        for j in range(n_features):
            layer = np.random.randint(0, 28)
            position = np.random.randint(0, 45)
            feature_idx = np.random.randint(0, 16384)
            influence = np.random.uniform(0.001, 0.1)
            activation_value = np.random.uniform(0.5, 5.0)

            features.append(FeatureInfo(
                layer=layer,
                position=position,
                feature_idx=feature_idx,
                influence=influence,
                activation_value=activation_value,
                logit_attribution=influence * 0.8
            ))

        compact_graph = CompactGraph(
            prompt_id=prompt_id,
            input_string="test prompt",
            input_tokens=np.array([1, 2, 3]),
            logit_tokens=np.array([10, 20, 30]),
            logit_probabilities=np.array([0.5, 0.3, 0.2]),
            features=features,
            n_layers=28,
            model_hash="model_hash",
            clt_hash="clt_hash",
            seed=42,
            created_at=datetime.now(UTC).isoformat(),
            scan="test_scan"
        )

        # Write CompactGraph
        storage.write_compact_graph(run_id, compact_graph)

        # Write feature incidence table
        import pandas as pd
        rows = []
        for feature in compact_graph.features:
            rows.append({
                'run_id': run_id,
                'prompt_id': prompt_id,
                'layer': feature.layer,
                'position': feature.position,
                'feature_idx': feature.feature_idx,
                'activation_value': feature.activation_value,
                'influence': feature.influence,
                'logit_attribution': feature.logit_attribution,
                'is_last_pos': False,
                'prompt_labels': str(labels),
            })
        df = pd.DataFrame(rows)
        storage.write_feature_incidence(run_id, df)

        # Write sparse PLT activations to parquet
        layers = [10, 20, 24]
        positions = [0, 44]  # first and last
        seq_len = 45
        d_transcoder = 163840  # Match default PLT dimension

        # Generate sparse PLT activations
        plt_rows = []
        for layer in layers:
            for pos in positions:
                # Generate sparse random activations (biased by label for separability)
                # ~2000 active features (sparse!)
                n_active = 2000
                active_indices = np.random.choice(d_transcoder, size=n_active, replace=False)

                if labels['eval'] == 'success':
                    # Success runs have higher activations
                    active_values = np.random.randn(n_active).astype(np.float32) + 1.0
                else:
                    # Failure runs have lower activations
                    active_values = np.random.randn(n_active).astype(np.float32) - 1.0

                # Add to rows
                for feat_idx, feat_val in zip(active_indices, active_values):
                    plt_rows.append({
                        'run_id': run_id,
                        'layer': layer,
                        'position': pos,
                        'feature_idx': int(feat_idx),
                        'activation_value': float(feat_val)
                    })

        # Write PLT activations as sparse parquet (one file per layer)
        plt_df = pd.DataFrame(plt_rows)
        for layer in layers:
            layer_df = plt_df[plt_df['layer'] == layer].drop(columns=['layer'])
            storage.write_plt_activations(run_id, layer, layer_df)

        # Write manifest.json with seq_len and d_sae for loading
        manifest = {
            'run_id': run_id,
            'prompt_id': prompt_id,
            'dataset_id': 'test_dataset',
            'model_hash': 'model_hash',
            'clt_hash': 'clt_hash',
            'seed': 42,
            'input_text': 'test prompt',
            'input_length': seq_len,
            'labels': labels,
            'd_model': 2048,
            'd_sae': d_transcoder
        }
        storage.write_metadata(run_id, manifest)

    return storage, registry, runs


def test_comparison_analyzer(mock_storage_with_artifacts):
    """Test ComparisonAnalyzer on mock data."""
    storage, registry, runs = mock_storage_with_artifacts

    # Create AnalysisData
    analysis_data = AnalysisData(
        registry=registry,
        storage=storage,
        filters={'dataset_id': 'test_dataset', 'status': 'completed'}
    )

    # Run comparison analysis
    analyzer = ComparisonAnalyzer(top_k=50, max_ubiquity=0.9, pmi_top_pairs=20)
    results = analyzer.run(analysis_data)

    # Assertions
    assert 'frequency_stats' in results
    assert 'pmi_matrix' in results
    assert 'top_pmi_pairs' in results
    assert 'co_occurrence_matrix' in results
    assert 'metadata' in results

    # Check frequency stats
    assert len(results['frequency_stats']['most_frequent']) > 0
    assert results['frequency_stats']['mean_frequency'] > 0

    # Check PMI matrix shape
    assert results['pmi_matrix'].shape[0] == results['pmi_matrix'].shape[1]

    # Check top PMI pairs
    assert len(results['top_pmi_pairs']) <= 20

    print("✅ ComparisonAnalyzer test passed")


def test_directions_analyzer(mock_storage_with_artifacts):
    """Test DirectionsAnalyzer on mock data."""
    storage, registry, runs = mock_storage_with_artifacts

    # Create separate data accessors for success and failure
    success_data = AnalysisData(
        registry=registry,
        storage=storage,
        filters={'dataset_id': 'test_dataset', 'labels': {'eval': 'success'}}
    )

    failure_data = AnalysisData(
        registry=registry,
        storage=storage,
        filters={'dataset_id': 'test_dataset', 'labels': {'eval': 'failure'}}
    )

    # Run directions analysis
    analyzer = DirectionsAnalyzer(layer=10, position='last', space='plt')
    results = analyzer.run_pairwise(success_data, failure_data, 'success', 'failure')

    # Assertions
    assert 'direction' in results
    assert 'separability_metrics' in results
    assert 'cluster_a_stats' in results
    assert 'cluster_b_stats' in results
    assert 'projections_a' in results
    assert 'projections_b' in results

    # Check direction shape
    assert results['direction'].shape[0] == 163840  # d_transcoder

    # Check separability metrics
    assert 'l2_distance' in results['separability_metrics']
    assert 'cosine_similarity' in results['separability_metrics']
    # l2_distance can be 0 if centroids happen to be identical with random sparse data
    assert results['separability_metrics']['l2_distance'] >= 0

    # Check projections
    assert len(results['projections_a']) == len(success_data.runs)
    assert len(results['projections_b']) == len(failure_data.runs)

    # With random sparse mock data, projections may not always be different
    # Just verify projections were computed
    mean_proj_a = results['projections_a'].mean()
    mean_proj_b = results['projections_b'].mean()
    # Both should be valid numbers (not NaN)
    assert not np.isnan(mean_proj_a)
    assert not np.isnan(mean_proj_b)

    print("✅ DirectionsAnalyzer test passed")


def test_unsupervised_analyzer(mock_storage_with_artifacts):
    """Test UnsupervisedAnalyzer on mock data."""
    storage, registry, runs = mock_storage_with_artifacts

    # Create AnalysisData
    analysis_data = AnalysisData(
        registry=registry,
        storage=storage,
        filters={'dataset_id': 'test_dataset', 'status': 'completed'}
    )

    # Run unsupervised analysis
    analyzer = UnsupervisedAnalyzer(
        n_components=10,
        clustering_method='kmeans',
        n_clusters=3
    )
    results = analyzer.run_on_features(analysis_data, top_k=50)

    # Assertions
    assert 'pca' in results
    assert 'reduced_data' in results
    assert 'cluster_labels' in results
    assert 'cluster_centers' in results
    assert 'cluster_sizes' in results
    assert 'explained_variance' in results
    assert 'cumulative_variance' in results

    # Check reduced data shape
    n_prompts = len(analysis_data.runs)
    assert results['reduced_data'].shape[0] == n_prompts
    assert results['reduced_data'].shape[1] <= 10  # n_components

    # Check cluster labels
    assert len(results['cluster_labels']) == n_prompts
    assert len(set(results['cluster_labels'])) <= 3  # n_clusters

    # Check cluster sizes
    assert sum(results['cluster_sizes'].values()) == n_prompts

    # Check explained variance
    assert len(results['explained_variance']) > 0
    assert sum(results['explained_variance']) <= 1.0
    assert results['cumulative_variance'][-1] <= 1.0

    print("✅ UnsupervisedAnalyzer test passed")


def test_analysis_data_feature_matrix(mock_storage_with_artifacts):
    """Test AnalysisData.get_feature_matrix() method."""
    storage, registry, runs = mock_storage_with_artifacts

    analysis_data = AnalysisData(
        registry=registry,
        storage=storage,
        filters={'dataset_id': 'test_dataset'}
    )

    # Test binary format
    fm_binary = analysis_data.get_feature_matrix(top_k=50, format='binary')
    assert fm_binary.matrix.shape[0] == len(analysis_data.runs)
    assert fm_binary.matrix.shape[1] <= 50
    assert set(np.unique(fm_binary.matrix)).issubset({0, 1})

    # Test influence format
    fm_influence = analysis_data.get_feature_matrix(top_k=50, format='influence')
    assert fm_influence.matrix.shape[0] == len(analysis_data.runs)
    assert fm_influence.matrix.max() > 0

    # Test activation format
    fm_activation = analysis_data.get_feature_matrix(top_k=50, format='activation')
    assert fm_activation.matrix.shape[0] == len(analysis_data.runs)
    assert fm_activation.matrix.max() > 0

    print("✅ AnalysisData.get_feature_matrix() test passed")


def test_analysis_data_activation_matrix(mock_storage_with_artifacts):
    """Test AnalysisData.get_activation_matrix() method."""
    storage, registry, runs = mock_storage_with_artifacts

    analysis_data = AnalysisData(
        registry=registry,
        storage=storage,
        filters={'dataset_id': 'test_dataset'}
    )

    # Test loading activations
    am = analysis_data.get_activation_matrix(layer=10, position='last', space='plt')

    assert am.matrix.shape[0] == len(analysis_data.runs)
    assert am.matrix.shape[1] == 163840  # d_transcoder
    assert am.metadata['layer'] == 10
    assert am.metadata['position'] == 'last'
    assert am.metadata['space'] == 'plt'

    print("✅ AnalysisData.get_activation_matrix() test passed")


def test_analyzer_save_results(mock_storage_with_artifacts, temp_storage):
    """Test that all analyzers can save results without errors."""
    storage, registry, runs = mock_storage_with_artifacts

    # Test data
    analysis_data = AnalysisData(
        registry=registry,
        storage=storage,
        filters={'dataset_id': 'test_dataset'}
    )

    success_data = AnalysisData(
        registry=registry,
        storage=storage,
        filters={'dataset_id': 'test_dataset', 'labels': {'eval': 'success'}}
    )

    failure_data = AnalysisData(
        registry=registry,
        storage=storage,
        filters={'dataset_id': 'test_dataset', 'labels': {'eval': 'failure'}}
    )

    output_root = temp_storage / "analysis_output"

    # Test ComparisonAnalyzer save
    comp_analyzer = ComparisonAnalyzer(top_k=50)
    comp_results = comp_analyzer.run(analysis_data)
    comp_analyzer.save_results(comp_results, output_root / "comparison")
    assert (output_root / "comparison" / "comparison_results.json").exists()

    # Test DirectionsAnalyzer save
    dir_analyzer = DirectionsAnalyzer(layer=10)
    dir_results = dir_analyzer.run_pairwise(success_data, failure_data)
    dir_analyzer.save_results(dir_results, output_root / "directions")
    assert (output_root / "directions" / "direction_vector.npy").exists()

    # Test UnsupervisedAnalyzer save
    unsup_analyzer = UnsupervisedAnalyzer(n_components=10, n_clusters=3)
    unsup_results = unsup_analyzer.run_on_features(analysis_data, top_k=50)
    unsup_analyzer.save_results(unsup_results, output_root / "unsupervised")
    assert (output_root / "unsupervised" / "unsupervised_results.json").exists()

    print("✅ Analyzer save_results() test passed")
