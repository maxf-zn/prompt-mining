"""
Unit tests for storage backends.
"""
import pytest
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

from prompt_mining.storage import LocalStorage
from prompt_mining.core import FeatureInfo, CompactGraph


@pytest.fixture
def temp_storage():
    """Create temporary storage for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = LocalStorage(tmpdir)
        yield storage


@pytest.fixture
def sample_compact_graph():
    """Create sample CompactGraph for testing."""
    features = [
        FeatureInfo(
            layer=10,
            position=31,
            feature_idx=5421,
            influence=0.034,
            activation_value=2.47,
            logit_attribution=0.021
        ),
        FeatureInfo(
            layer=20,
            position=44,
            feature_idx=8234,
            influence=0.067,
            activation_value=3.21,
            logit_attribution=0.067
        ),
    ]

    return CompactGraph(
        prompt_id="test:001",
        input_string="Hello world",
        input_tokens=np.array([1, 2, 3, 4, 5]),
        logit_tokens=np.array([42, 13, 7]),
        logit_probabilities=np.array([0.55, 0.22, 0.06]),
        features=features,
        n_layers=28,
        model_hash="test_model_hash",
        clt_hash="test_clt_hash",
        seed=42
    )


class TestLocalStorage:
    """Test LocalStorage backend."""

    def test_initialization(self, temp_storage):
        """Test storage initialization."""
        assert temp_storage.root.exists()
        assert temp_storage.root.is_dir()

    def test_initialization_with_file_uri(self):
        """Test initialization with file:// URI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(f"file://{tmpdir}")
            assert storage.root == Path(tmpdir)
            assert storage.root.exists()

    def test_get_full_path(self, temp_storage):
        """Test path resolution."""
        full_path = temp_storage.get_full_path("test/subdir/file.txt")
        assert str(temp_storage.root) in full_path
        assert "test/subdir/file.txt" in full_path

    def test_makedirs(self, temp_storage):
        """Test directory creation."""
        test_dir = "test/nested/dir"
        temp_storage.makedirs(test_dir)

        full_path = Path(temp_storage.get_full_path(test_dir))
        assert full_path.exists()
        assert full_path.is_dir()

    def test_write_read_bytes(self, temp_storage):
        """Test writing and reading bytes."""
        test_path = "test/file.bin"
        test_data = b"Hello, world!"

        # Write
        temp_storage.write_bytes(test_path, test_data)

        # Check exists
        assert temp_storage.exists(test_path)

        # Read
        read_data = temp_storage.read_bytes(test_path)
        assert read_data == test_data

    def test_write_read_compact_graph(self, temp_storage, sample_compact_graph):
        """Test CompactGraph storage."""
        run_id = "2025-11-11_test_run_001"

        # Write
        temp_storage.write_compact_graph(run_id, sample_compact_graph)

        # Verify file exists
        run_dir = temp_storage._get_run_dir(run_id)
        graph_path = Path(f"{run_dir}/compact_graph.pt")
        assert Path(graph_path).exists()

        # Read
        restored = temp_storage.read_compact_graph(run_id)

        # Verify
        assert restored.prompt_id == sample_compact_graph.prompt_id
        assert restored.input_string == sample_compact_graph.input_string
        assert np.array_equal(restored.input_tokens, sample_compact_graph.input_tokens)
        assert len(restored.features) == len(sample_compact_graph.features)

    def test_write_read_activations(self, temp_storage):
        """Test Zarr activation storage (raw only - PLT is in parquet)."""
        run_id = "2025-11-11_test_run_002"

        # Create sample activations (raw only - PLT goes to parquet now)
        activations = {
            'raw/layer10': np.random.randn(2, 2048).astype(np.float16),
            'raw/layer20': np.random.randn(2, 2048).astype(np.float16),
        }

        metadata = {
            'seq_len': 45,
            'layers_captured': [10, 20],
            'raw_positions': [0, 44],
            'plt_positions': [],
            'raw_layers': [10, 20],
            'plt_layers': []
        }

        # Write
        temp_storage.write_activations(run_id, activations, metadata)

        # Read
        zarr_group = temp_storage.read_activations(run_id)

        # Verify metadata (new format)
        assert zarr_group.attrs['seq_len'] == 45
        assert zarr_group.attrs['raw_positions'] == [0, 44]
        assert zarr_group.attrs['raw_layers'] == [10, 20]

        # Verify arrays (raw only)
        assert 'raw/layer10' in zarr_group
        assert 'raw/layer20' in zarr_group
        assert 'plt/layer10' not in zarr_group  # PLT no longer in Zarr

        layer10_raw = zarr_group['raw/layer10'][:]
        assert layer10_raw.shape == (2, 2048)

        layer20_raw = zarr_group['raw/layer20'][:]
        assert layer20_raw.shape == (2, 2048)

    def test_write_read_metadata(self, temp_storage):
        """Test JSON metadata storage."""
        run_id = "2025-11-11_test_run_003"

        metadata = {
            'run_id': run_id,
            'prompt_id': 'test:001',
            'dataset_id': 'test_dataset',
            'timestamp': '2025-11-11T12:00:00',
            'model': 'test_model',
            'config': {'seed': 42, 'top_k': 200}
        }

        # Write
        temp_storage.write_metadata(run_id, metadata)

        # Read
        restored = temp_storage.read_metadata(run_id)

        assert restored['run_id'] == run_id
        assert restored['prompt_id'] == 'test:001'
        assert restored['config']['seed'] == 42

    def test_write_feature_incidence(self, temp_storage):
        """Test feature incidence table storage."""
        run_id = "2025-11-11_test_run_004"

        # Create sample DataFrame
        data = {
            'run_id': [run_id, run_id],
            'prompt_id': ['test:001', 'test:001'],
            'layer': [10, 20],
            'position': [31, 44],
            'feature_idx': [5421, 8234],
            'activation_value': [2.47, 3.21],
            'influence': [0.034, 0.067],
        }
        df = pd.DataFrame(data)

        # Write
        temp_storage.write_feature_incidence(run_id, df)

        # Verify file exists
        tables_dir = temp_storage.get_full_path("tables/feature_incidence")
        parquet_path = Path(tables_dir) / f"run={run_id}.parquet"
        assert parquet_path.exists()

        # Read back
        df_restored = pd.read_parquet(parquet_path)
        assert len(df_restored) == 2
        assert df_restored['layer'].tolist() == [10, 20]

    def test_list_runs(self, temp_storage, sample_compact_graph):
        """Test listing runs."""
        # Initially empty
        assert len(temp_storage.list_runs()) == 0

        # Create some runs
        run_ids = [
            "2025-11-11_run_001",
            "2025-11-11_run_002",
            "2025-11-12_run_003",
        ]

        for run_id in run_ids:
            temp_storage.write_compact_graph(run_id, sample_compact_graph)

        # List all
        all_runs = temp_storage.list_runs()
        assert len(all_runs) == 3
        assert "2025-11-11_run_001" in all_runs

        # List by date
        nov11_runs = temp_storage.list_runs(date="2025-11-11")
        assert len(nov11_runs) == 2

        nov12_runs = temp_storage.list_runs(date="2025-11-12")
        assert len(nov12_runs) == 1

    def test_delete_run(self, temp_storage, sample_compact_graph):
        """Test deleting a run."""
        run_id = "2025-11-11_test_run_005"

        # Create run
        temp_storage.write_compact_graph(run_id, sample_compact_graph)
        assert run_id in temp_storage.list_runs()

        # Delete
        temp_storage.delete_run(run_id)

        # Verify deleted
        assert run_id not in temp_storage.list_runs()
        run_dir = Path(temp_storage._get_run_dir(run_id))
        assert not run_dir.exists()

    def test_get_storage_stats(self, temp_storage, sample_compact_graph):
        """Test storage statistics."""
        # Create some runs
        for i in range(3):
            run_id = f"2025-11-11_run_{i:03d}"
            temp_storage.write_compact_graph(run_id, sample_compact_graph)

        stats = temp_storage.get_storage_stats()

        assert stats['num_runs'] == 3
        assert stats['total_size_gb'] >= 0
        assert stats['root'] == str(temp_storage.root)

    def test_nonexistent_file_error(self, temp_storage):
        """Test error when reading nonexistent file."""
        with pytest.raises(FileNotFoundError):
            temp_storage.read_bytes("nonexistent/file.txt")

        with pytest.raises(FileNotFoundError):
            temp_storage.read_compact_graph("nonexistent_run")

        with pytest.raises(FileNotFoundError):
            temp_storage.read_activations("nonexistent_run")

        with pytest.raises(FileNotFoundError):
            temp_storage.read_metadata("nonexistent_run")

    def test_get_registry_path(self, temp_storage):
        """Test registry path resolution."""
        registry_path = temp_storage.get_registry_path()
        assert "registry.sqlite" in registry_path
        assert str(temp_storage.root) in registry_path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
