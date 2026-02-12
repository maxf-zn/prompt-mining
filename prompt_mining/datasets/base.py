"""
Base Dataset class for prompt mining platform.

Provides abstract interface for data sources with deterministic sharding support.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional
from prompt_mining.core.prompt_spec import PromptSpec


class Dataset(ABC):
    """
    Abstract base class for dataset providers.

    All datasets must implement:
    1. __len__() - Return total number of prompts
    2. __iter__() - Yield all PromptSpec objects
    3. shard() - Yield PromptSpec objects for a specific shard

    Sharding is deterministic: same (shard_idx, num_shards) always yields
    same prompts in same order.
    """

    def __init__(self, name: str, dataset_id: Optional[str] = None):
        """
        Initialize dataset.

        Args:
            name: Human-readable dataset name
            dataset_id: Unique identifier (defaults to name if not provided)
        """
        self.name = name
        self.dataset_id = dataset_id or name

    @abstractmethod
    def __len__(self) -> int:
        """Return total number of prompts in dataset."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[PromptSpec]:
        """Iterate over all prompts in dataset."""
        pass

    def shard(self, shard_idx: int, num_shards: int) -> Iterator[PromptSpec]:
        """
        Iterate over prompts assigned to this shard.

        Uses deterministic round-robin assignment: prompt i → shard (i % num_shards)

        Args:
            shard_idx: Shard index (0-based)
            num_shards: Total number of shards

        Yields:
            PromptSpec objects assigned to this shard

        Example:
            >>> dataset = MyDataset()
            >>> # Split across 4 GPUs
            >>> for prompt in dataset.shard(shard_idx=0, num_shards=4):
            ...     process_on_gpu_0(prompt)
        """
        if not (0 <= shard_idx < num_shards):
            raise ValueError(
                f"shard_idx must be in [0, {num_shards}), got {shard_idx}"
            )

        for i, prompt_spec in enumerate(self):
            if i % num_shards == shard_idx:
                yield prompt_spec

    def get_shard_size(self, shard_idx: int, num_shards: int) -> int:
        """
        Calculate number of prompts in a specific shard.

        Args:
            shard_idx: Shard index (0-based)
            num_shards: Total number of shards

        Returns:
            Number of prompts assigned to this shard
        """
        total = len(self)
        base_size = total // num_shards
        remainder = total % num_shards

        # First 'remainder' shards get one extra prompt
        if shard_idx < remainder:
            return base_size + 1
        else:
            return base_size

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, dataset_id={self.dataset_id!r}, size={len(self)})"
