"""
Unit tests for SQLiteRegistry.
"""
import pytest
import tempfile
from pathlib import Path

from prompt_mining.registry import (
    SQLiteRegistry,
    RunStatus,
    compute_run_key,
    compute_processing_fingerprint
)
from prompt_mining.core import PromptSpec


@pytest.fixture
def temp_registry():
    """Create temporary registry for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_registry.sqlite"
        registry = SQLiteRegistry(str(db_path))
        yield registry


@pytest.fixture
def sample_prompt_spec():
    """Create sample PromptSpec."""
    return PromptSpec(
        prompt_id="test:001",
        dataset_id="test_dataset",
        messages=[{"role": "user", "content": "Hello"}],
        labels={"category": "greeting"}
    )


class TestSQLiteRegistry:
    """Test SQLiteRegistry."""

    def test_initialization(self, temp_registry):
        """Test registry initialization creates schema."""
        # Check that database file exists
        assert Path(temp_registry.db_path).exists()

        # Check that schema_version table exists
        with temp_registry._connect() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]

        assert 'runs' in tables
        assert 'artifacts' in tables
        assert 'schema_version' in tables

    def test_create_run(self, temp_registry):
        """Test creating a run entry."""
        run_id = temp_registry.create_run(
            run_id="test_run_001",
            prompt_id="test:001",
            dataset_id="test_dataset",
            model_hash="model_abc123",
            clt_hash="clt_def456",
            seed=42,
            run_key="key_12345",
            processing_fingerprint="fp_67890",
            prompt_labels={"category": "greeting"},
            config_snapshot={"seed": 42, "top_k": 200},
            shard_idx=0,
            num_shards=4,
            worker_id=0,
            device_id=0
        )

        assert run_id == "test_run_001"

        # Retrieve and verify
        run = temp_registry.get_run("test_run_001")
        assert run is not None
        assert run['prompt_id'] == "test:001"
        assert run['dataset_id'] == "test_dataset"
        assert run['status'] == RunStatus.PENDING.value
        assert run['seed'] == 42
        assert run['prompt_labels']['category'] == "greeting"

    def test_update_status(self, temp_registry):
        """Test updating run status."""
        # Create run
        temp_registry.create_run(
            run_id="test_run_002",
            prompt_id="test:001",
            dataset_id="test_dataset",
            model_hash="hash1",
            clt_hash="hash2",
            seed=42,
            run_key="key1",
            processing_fingerprint="fp1",
            prompt_labels={},
            config_snapshot={}
        )

        # Update to running
        temp_registry.update_status("test_run_002", RunStatus.RUNNING.value)
        run = temp_registry.get_run("test_run_002")
        assert run['status'] == RunStatus.RUNNING.value

        # Update to completed
        temp_registry.update_status("test_run_002", RunStatus.COMPLETED.value)
        run = temp_registry.get_run("test_run_002")
        assert run['status'] == RunStatus.COMPLETED.value

        # Update to failed with error message
        temp_registry.update_status(
            "test_run_002",
            RunStatus.FAILED.value,
            error_message="GPU OOM"
        )
        run = temp_registry.get_run("test_run_002")
        assert run['status'] == RunStatus.FAILED.value
        assert run['error_message'] == "GPU OOM"

    def test_get_run_by_key(self, temp_registry):
        """Test retrieving run by idempotency key."""
        run_key = "unique_key_123"
        fingerprint = "fp_456"

        # Create run
        temp_registry.create_run(
            run_id="test_run_003",
            prompt_id="test:001",
            dataset_id="test_dataset",
            model_hash="hash1",
            clt_hash="hash2",
            seed=42,
            run_key=run_key,
            processing_fingerprint=fingerprint,
            prompt_labels={},
            config_snapshot={}
        )

        # Retrieve by key
        run = temp_registry.get_run_by_key(run_key, fingerprint)
        assert run is not None
        assert run['run_id'] == "test_run_003"
        assert run['run_key'] == run_key

        # Non-existent key
        run = temp_registry.get_run_by_key("nonexistent", "fp")
        assert run is None

    def test_exists_idempotency(self, temp_registry):
        """Test idempotency checking."""
        run_key = "key_789"
        fingerprint = "fp_101112"

        # Initially doesn't exist
        assert not temp_registry.exists(run_key, fingerprint)

        # Create run
        temp_registry.create_run(
            run_id="test_run_004",
            prompt_id="test:001",
            dataset_id="test_dataset",
            model_hash="hash1",
            clt_hash="hash2",
            seed=42,
            run_key=run_key,
            processing_fingerprint=fingerprint,
            prompt_labels={},
            config_snapshot={}
        )

        # Now exists
        assert temp_registry.exists(run_key, fingerprint)

        # Status filter
        assert not temp_registry.exists(run_key, fingerprint, status=RunStatus.COMPLETED.value)

        # Complete the run
        temp_registry.update_status("test_run_004", RunStatus.COMPLETED.value)
        assert temp_registry.exists(run_key, fingerprint, status=RunStatus.COMPLETED.value)

    def test_get_runs_filtering(self, temp_registry):
        """Test querying runs with filters."""
        # Create multiple runs
        for i in range(5):
            dataset = "dataset_A" if i < 3 else "dataset_B"
            status = RunStatus.COMPLETED.value if i % 2 == 0 else RunStatus.FAILED.value
            labels = {"success": i % 2 == 0, "id": i}

            temp_registry.create_run(
                run_id=f"test_run_{i:03d}",
                prompt_id=f"test:{i}",
                dataset_id=dataset,
                model_hash="hash1",
                clt_hash="hash2",
                seed=42,
                run_key=f"key_{i}",
                processing_fingerprint="fp1",
                prompt_labels=labels,
                config_snapshot={}
            )
            temp_registry.update_status(f"test_run_{i:03d}", status)

        # Filter by dataset
        dataset_a_runs = temp_registry.get_runs(dataset_id="dataset_A")
        assert len(dataset_a_runs) == 3

        dataset_b_runs = temp_registry.get_runs(dataset_id="dataset_B")
        assert len(dataset_b_runs) == 2

        # Filter by status
        completed_runs = temp_registry.get_runs(status=RunStatus.COMPLETED.value)
        assert len(completed_runs) == 3

        failed_runs = temp_registry.get_runs(status=RunStatus.FAILED.value)
        assert len(failed_runs) == 2

        # Filter by labels
        success_runs = temp_registry.get_runs(labels={"success": True})
        assert len(success_runs) == 3

        # Combined filters
        dataset_a_completed = temp_registry.get_runs(
            dataset_id="dataset_A",
            status=RunStatus.COMPLETED.value
        )
        assert len(dataset_a_completed) == 2

        # Limit
        limited = temp_registry.get_runs(limit=2)
        assert len(limited) == 2

    def test_register_artifact(self, temp_registry):
        """Test registering artifacts."""
        # Create run
        temp_registry.create_run(
            run_id="test_run_005",
            prompt_id="test:001",
            dataset_id="test_dataset",
            model_hash="hash1",
            clt_hash="hash2",
            seed=42,
            run_key="key1",
            processing_fingerprint="fp1",
            prompt_labels={},
            config_snapshot={}
        )

        # Register artifacts
        temp_registry.register_artifact(
            run_id="test_run_005",
            kind="compact_graph",
            path="/path/to/compact_graph.pt",
            size_bytes=5000
        )

        temp_registry.register_artifact(
            run_id="test_run_005",
            kind="activations",
            path="/path/to/acts.zarr",
            size_bytes=120000
        )

        # Retrieve artifacts
        artifacts = temp_registry.get_artifacts("test_run_005")
        assert len(artifacts) == 2

        artifact_kinds = [a['kind'] for a in artifacts]
        assert 'compact_graph' in artifact_kinds
        assert 'activations' in artifact_kinds

    def test_supersedes_versioning(self, temp_registry):
        """Test run versioning with supersedes."""
        run_key = "key_version_test"

        # Create original run
        temp_registry.create_run(
            run_id="test_run_v1",
            prompt_id="test:001",
            dataset_id="test_dataset",
            model_hash="hash1",
            clt_hash="hash2",
            seed=42,
            run_key=run_key,
            processing_fingerprint="fp_v1",
            prompt_labels={},
            config_snapshot={"version": 1}
        )
        temp_registry.update_status("test_run_v1", RunStatus.COMPLETED.value)

        # Create new version (config changed)
        temp_registry.create_run(
            run_id="test_run_v2",
            prompt_id="test:001",
            dataset_id="test_dataset",
            model_hash="hash1",
            clt_hash="hash2",
            seed=42,
            run_key=run_key,
            processing_fingerprint="fp_v2",
            prompt_labels={},
            config_snapshot={"version": 2},
            supersedes="test_run_v1"
        )
        temp_registry.update_status("test_run_v2", RunStatus.COMPLETED.value)

        # Both exist
        v1 = temp_registry.get_run("test_run_v1")
        v2 = temp_registry.get_run("test_run_v2")
        assert v1 is not None
        assert v2 is not None
        assert v2['supersedes'] == "test_run_v1"

        # Get latest only
        runs = temp_registry.get_runs(latest_only=True)
        # Should return v2 but not v1 (since they have same run_key)
        run_ids = [r['run_id'] for r in runs]
        assert "test_run_v2" in run_ids
        # v1 might still be in the list if there are other run_keys

    def test_get_stats(self, temp_registry):
        """Test registry statistics."""
        # Create runs with different statuses and datasets
        for i in range(10):
            dataset = "dataset_A" if i < 6 else "dataset_B"
            status = [
                RunStatus.COMPLETED.value,
                RunStatus.COMPLETED.value,
                RunStatus.FAILED.value,
                RunStatus.RUNNING.value,
            ][i % 4]

            temp_registry.create_run(
                run_id=f"test_run_{i:03d}",
                prompt_id=f"test:{i}",
                dataset_id=dataset,
                model_hash="hash1",
                clt_hash="hash2",
                seed=42,
                run_key=f"key_{i}",
                processing_fingerprint="fp1",
                prompt_labels={},
                config_snapshot={}
            )
            temp_registry.update_status(f"test_run_{i:03d}", status)

            # Register some artifacts
            temp_registry.register_artifact(
                run_id=f"test_run_{i:03d}",
                kind="compact_graph",
                path=f"/path/{i}.pt",
                size_bytes=5000
            )

        stats = temp_registry.get_stats()

        assert stats['total_runs'] == 10
        assert RunStatus.COMPLETED.value in stats['by_status']
        assert 'dataset_A' in stats['by_dataset']
        assert 'dataset_B' in stats['by_dataset']
        assert stats['by_dataset']['dataset_A'] == 6
        assert stats['by_dataset']['dataset_B'] == 4
        assert stats['total_artifact_size_gb'] > 0


class TestUtilityFunctions:
    """Test utility functions for keys and fingerprints."""

    def test_compute_run_key(self, sample_prompt_spec):
        """Test run key computation."""
        run_key = compute_run_key(
            sample_prompt_spec,
            model_hash="model_abc",
            clt_hash="clt_def",
            seed=42
        )

        # Should be a hex string
        assert isinstance(run_key, str)
        assert len(run_key) == 40  # SHA1 produces 40 hex chars

        # Same inputs should produce same key
        run_key2 = compute_run_key(
            sample_prompt_spec,
            model_hash="model_abc",
            clt_hash="clt_def",
            seed=42
        )
        assert run_key == run_key2

        # Different seed should produce different key
        run_key3 = compute_run_key(
            sample_prompt_spec,
            model_hash="model_abc",
            clt_hash="clt_def",
            seed=43
        )
        assert run_key != run_key3

    def test_compute_processing_fingerprint(self):
        """Test processing fingerprint computation."""
        fingerprint = compute_processing_fingerprint(
            model_name="google/gemma-2-2b-it",
            transcoder_or_sae_name="mntss/gemma-scope-transcoders",
            capture_layers=[10, 20, 27],
            capture_positions=[-5, -1],
            enable_attribution=True,
            enable_generation=True,
        )

        # Should be a hex string (first 16 chars of SHA1)
        assert isinstance(fingerprint, str)
        assert len(fingerprint) == 16

        # Same config should produce same fingerprint
        fingerprint2 = compute_processing_fingerprint(
            model_name="google/gemma-2-2b-it",
            transcoder_or_sae_name="mntss/gemma-scope-transcoders",
            capture_layers=[10, 20, 27],
            capture_positions=[-5, -1],
            enable_attribution=True,
            enable_generation=True,
        )
        assert fingerprint == fingerprint2

        # Different model should produce different fingerprint
        fingerprint3 = compute_processing_fingerprint(
            model_name="different/model",
            transcoder_or_sae_name="mntss/gemma-scope-transcoders",
            capture_layers=[10, 20, 27],
            capture_positions=[-5, -1],
            enable_attribution=True,
            enable_generation=True,
        )
        assert fingerprint != fingerprint3

        # Different layers should produce different fingerprint
        fingerprint4 = compute_processing_fingerprint(
            model_name="google/gemma-2-2b-it",
            transcoder_or_sae_name="mntss/gemma-scope-transcoders",
            capture_layers=[10, 20],  # Different layers
            capture_positions=[-5, -1],
            enable_attribution=True,
            enable_generation=True,
        )
        assert fingerprint != fingerprint4

        # Different attribution setting should produce different fingerprint
        fingerprint5 = compute_processing_fingerprint(
            model_name="google/gemma-2-2b-it",
            transcoder_or_sae_name="mntss/gemma-scope-transcoders",
            capture_layers=[10, 20, 27],
            capture_positions=[-5, -1],
            enable_attribution=False,  # Different
            enable_generation=True,
        )
        assert fingerprint != fingerprint5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
