"""
Integration tests for end-to-end prompt mining pipeline.

These tests verify the complete workflow from dataset loading to artifact storage.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
import torch
import numpy as np

from prompt_mining.core.prompt_spec import PromptSpec
from prompt_mining.core.compact_graph import CompactGraph, FeatureInfo
from prompt_mining.datasets.base import Dataset
from prompt_mining.model.model_wrapper import ModelWrapper, ModelConfig
from prompt_mining.ingestion.prompt_runner import PromptRunner, RunConfig
from prompt_mining.storage.local_storage import LocalStorage
from prompt_mining.registry.sqlite_registry import SQLiteRegistry


class MockDataset(Dataset):
    """Mock dataset for testing."""

    def __init__(self, name: str, num_prompts: int = 10):
        super().__init__(name=name, dataset_id=name)
        self.num_prompts = num_prompts

    def __len__(self) -> int:
        return self.num_prompts

    def __iter__(self):
        for i in range(self.num_prompts):
            yield PromptSpec(
                prompt_id=f"{self.dataset_id}:prompt_{i}",
                dataset_id=self.dataset_id,
                text=f"Test prompt {i}",
                messages=None,
                tools=None,
                labels={"test": True, "index": i}
            )


class TestEndToEnd:
    """End-to-end integration tests."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def storage(self, temp_dir):
        """Create storage backend."""
        return LocalStorage(uri=f"file://{temp_dir}")

    @pytest.fixture
    def registry(self, temp_dir):
        """Create registry."""
        db_path = Path(temp_dir) / "registry.sqlite"
        return SQLiteRegistry(db_path=str(db_path))

    @pytest.fixture
    def mock_model_wrapper(self):
        """Create mock model wrapper."""
        mock = Mock(spec=ModelWrapper)

        # Mock model info
        mock.get_model_info.return_value = {
            "model_name": "test-model",
            "transcoder_set": "test-transcoder",
            "n_layers": 12,
            "d_model": 768,
            "d_sae": 16384,  # New: generic SAE dimension
            "vocab_size": 32000,
            "device": "cpu",
            "dtype": "bfloat16",
            "enable_attribution": False
        }

        # Mock tokenizer
        mock.tokenizer = Mock()
        mock.tokenizer.decode.return_value = "Generated text"

        # Mock tokenization
        mock.tokenize.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]])
        }

        # Mock decode method (separate from tokenizer.decode)
        mock.decode.return_value = "Generated text"

        # Mock forward pass
        mock.forward.return_value = {
            "logits": torch.randn(1, 5, 32000)
        }

        # Mock run_with_cache (new single-pass optimization)
        mock.run_with_cache.return_value = (
            torch.randn(1, 5, 32000),  # logits
            {i: torch.randn(1, 5, 768) for i in range(12)}  # cached activations for 12 layers
        )

        # Mock encode_layer_activations (new adapter method)
        mock.encode_layer_activations.return_value = torch.randn(1, 5, 16384)

        # Mock generation
        mock.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])

        # Mock chat template
        def apply_chat_template(messages, tools=None, **kwargs):
            return " ".join([m["content"] for m in messages])

        mock.apply_chat_template.side_effect = apply_chat_template

        # Config
        mock.config = Mock()
        mock.config.enable_attribution = False
        mock.config.backend = "circuit_tracer"

        # Mock adapter (new adapter pattern)
        mock.adapter = Mock()
        mock.adapter.supports_attribution.return_value = False
        mock.adapter.get_raw_model.return_value = Mock()

        # Mock the underlying model attribute (needed by PromptRunner)
        mock.model = Mock()
        mock.model.device = torch.device("cpu")
        
        # Mock model.blocks for activation capture
        # Make it subscriptable by returning a Mock with hook attributes
        def get_block_mock(idx):
            block = Mock()
            block.hook_resid_pre = Mock()
            block.hook_resid_pre.ctx = {}
            return block
        
        mock.model.blocks = Mock()
        mock.model.blocks.__getitem__ = Mock(side_effect=get_block_mock)

        return mock

    @pytest.fixture
    def run_config(self):
        """Create run configuration."""
        return RunConfig(
            topk_logits=5,
            max_new_tokens=0,  # No generation for test
            seed=42,
            features_top_k=10,
            # Use layers within mock model's 12-layer range (0-11)
            capture_acts_raw={"layers": [8, 10], "positions": [-5, -1]},
            capture_acts_plt={"layers": [8], "positions": [-5, -1]},
        )

    def test_dataset_sharding(self):
        """Test deterministic dataset sharding."""
        dataset = MockDataset(name="test", num_prompts=100)

        # Split into 4 shards
        num_shards = 4
        all_prompts = []

        for shard_idx in range(num_shards):
            shard_prompts = list(dataset.shard(shard_idx, num_shards))
            all_prompts.extend(shard_prompts)

            # Each shard should have roughly equal size
            expected_size = dataset.get_shard_size(shard_idx, num_shards)
            assert len(shard_prompts) == expected_size

        # All prompts should be present exactly once
        assert len(all_prompts) == 100
        prompt_ids = [p.prompt_id for p in all_prompts]
        assert len(set(prompt_ids)) == 100

    def test_prompt_runner_basic(
        self,
        mock_model_wrapper,
        storage,
        registry,
        run_config
    ):
        """Test PromptRunner with mock model."""
        runner = PromptRunner(
            model_wrapper=mock_model_wrapper,
            storage=storage,
            registry=registry,
            run_config=run_config
        )

        # Create test prompt
        prompt = PromptSpec(
            prompt_id="test:prompt_1",
            dataset_id="test",
            text="Hello, world!",
            messages=None,
            tools=None,
            labels={"test": True}
        )

        # Run prompt
        result = runner.run(prompt)

        # Check result (print error if failed for debugging)
        if result.status == "failed":
            print(f"\nError message: {result.error_message}")
        assert result.status == "completed"
        assert result.prompt_id == "test:prompt_1"
        assert len(result.artifacts_written) > 0
        assert "topk_logits.json" in result.artifacts_written
        assert "manifest.json" in result.artifacts_written

        # Check registry
        runs = registry.get_runs(dataset_id="test")
        assert len(runs) == 1
        assert runs[0]["status"] == "completed"

        # Check artifacts exist
        metadata = storage.read_metadata(result.run_id)
        assert metadata["prompt_id"] == "test:prompt_1"
        assert metadata["dataset_id"] == "test"

    def test_prompt_runner_idempotency(
        self,
        mock_model_wrapper,
        storage,
        registry,
        run_config
    ):
        """Test that rerunning same prompt is skipped."""
        runner = PromptRunner(
            model_wrapper=mock_model_wrapper,
            storage=storage,
            registry=registry,
            run_config=run_config
        )

        prompt = PromptSpec(
            prompt_id="test:prompt_1",
            dataset_id="test",
            text="Hello, world!",
            messages=None,
            tools=None,
            labels={"test": True}
        )

        # First run
        result1 = runner.run(prompt)
        assert result1.status == "completed"

        # Second run (should be skipped)
        result2 = runner.run(prompt)
        assert result2.status == "skipped"
        assert result2.run_id == result1.run_id

        # Registry should only have one run
        runs = registry.get_runs(dataset_id="test")
        assert len(runs) == 1

    def test_multiple_prompts(
        self,
        mock_model_wrapper,
        storage,
        registry,
        run_config
    ):
        """Test processing multiple prompts."""
        runner = PromptRunner(
            model_wrapper=mock_model_wrapper,
            storage=storage,
            registry=registry,
            run_config=run_config
        )

        dataset = MockDataset(name="test", num_prompts=10)

        # Process all prompts
        results = []
        for prompt in dataset:
            result = runner.run(prompt)
            results.append(result)

        # Check all completed
        assert all(r.status == "completed" for r in results)
        assert len(results) == 10

        # Check registry
        runs = registry.get_runs(dataset_id="test")
        assert len(runs) == 10
        assert all(r["status"] == "completed" for r in runs)

    def test_chat_template_prompt(
        self,
        mock_model_wrapper,
        storage,
        registry,
        run_config
    ):
        """Test prompt with chat template."""
        runner = PromptRunner(
            model_wrapper=mock_model_wrapper,
            storage=storage,
            registry=registry,
            run_config=run_config
        )

        # Create prompt with messages
        prompt = PromptSpec(
            prompt_id="test:chat_1",
            dataset_id="test",
            text=None,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ],
            tools=None,
            labels={"test": True}
        )

        # Run prompt
        result = runner.run(prompt)

        # Check result
        assert result.status == "completed"

        # Verify chat template was applied
        mock_model_wrapper.apply_chat_template.assert_called_once()

    def test_error_handling(
        self,
        mock_model_wrapper,
        storage,
        registry,
        run_config
    ):
        """Test error handling in PromptRunner."""
        # Configure mock to raise error (use run_with_cache since that's called now)
        mock_model_wrapper.run_with_cache.side_effect = RuntimeError("Test error")

        runner = PromptRunner(
            model_wrapper=mock_model_wrapper,
            storage=storage,
            registry=registry,
            run_config=run_config
        )

        prompt = PromptSpec(
            prompt_id="test:error_prompt",
            dataset_id="test",
            text="This will fail",
            messages=None,
            tools=None,
            labels={"test": True}
        )

        # Run prompt (should fail gracefully)
        result = runner.run(prompt)

        # Check result
        assert result.status == "failed"
        assert result.error_message is not None
        assert "Test error" in result.error_message

        # Check registry
        runs = registry.get_runs(dataset_id="test", status="failed")
        assert len(runs) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
