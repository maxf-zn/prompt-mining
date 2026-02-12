"""
Base HuggingFace dataset wrapper for prompt mining platform.

Provides common infrastructure for loading and sharding HuggingFace datasets.
Subclasses implement dataset-specific conversion logic.
"""

from datasets import load_dataset
from typing import Iterator, Optional, Dict, Any
from prompt_mining.datasets.base import Dataset
from prompt_mining.core.prompt_spec import PromptSpec


class HFDataset(Dataset):
    """
    Base class for HuggingFace datasets.

    Handles loading, caching, and sharding. Subclasses implement
    dataset-specific conversion logic via _convert_to_prompt_spec().

    Attributes:
        repo_id: HuggingFace dataset repository ID
        split: Dataset split to load (train/test/validation)
        subset: Optional dataset configuration/subset name
        revision: Optional git revision/tag
        cache_dir: Optional cache directory for HF datasets
    """

    def __init__(
        self,
        repo_id: str,
        split: str = "train",
        name: Optional[str] = None,
        dataset_id: Optional[str] = None,
        subset: Optional[str] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **load_kwargs
    ):
        """
        Initialize HuggingFace dataset.

        Args:
            repo_id: HuggingFace dataset repo (e.g., "xTRam1/safe-guard-prompt-injection")
            split: Dataset split to use (default: "train")
            name: Human-readable dataset name (auto-generated if not provided)
            dataset_id: Unique dataset ID (defaults to name)
            subset: Dataset configuration/subset name (for datasets with multiple configs)
            revision: Specific git revision/tag to load
            cache_dir: Cache directory for HF datasets (uses HF default if not provided)
            **load_kwargs: Additional arguments passed to datasets.load_dataset()
        """
        self.repo_id = repo_id
        self.split = split
        self.subset = subset
        self.revision = revision
        self.cache_dir = cache_dir
        self.load_kwargs = load_kwargs

        # Generate name if not provided
        if name is None:
            name = self._generate_default_name()

        super().__init__(name=name, dataset_id=dataset_id)

        # Load HF dataset (cached automatically by HuggingFace)
        self._hf_dataset = self._load_hf_dataset()

    def _generate_default_name(self) -> str:
        """
        Generate default dataset name from repo_id.

        Examples:
            "xTRam1/safe-guard-prompt-injection" -> "safe_guard_prompt_injection"
            "username/dataset" -> "dataset"
        """
        # Extract dataset name from repo (handle org/dataset format)
        base_name = self.repo_id.split("/")[-1].replace("-", "_")

        if self.subset:
            return f"{base_name}_{self.subset}"
        return base_name

    def _load_hf_dataset(self):
        """
        Load dataset from HuggingFace.

        Returns:
            Loaded HuggingFace dataset (datasets.Dataset object)
        """
        return load_dataset(
            self.repo_id,
            name=self.subset,
            split=self.split,
            revision=self.revision,
            cache_dir=self.cache_dir,
            **self.load_kwargs
        )

    def _convert_to_prompt_spec(self, example: Dict[str, Any], index: int) -> PromptSpec:
        """
        Convert HF example to PromptSpec.

        MUST BE IMPLEMENTED BY SUBCLASS.

        This is where dataset-specific logic lives:
        - Column mapping (which HF columns map to messages/labels)
        - Label extraction and formatting
        - Message construction

        Args:
            example: Raw example dictionary from HF dataset
            index: Index in dataset (for generating prompt_id)

        Returns:
            PromptSpec object

        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _convert_to_prompt_spec()"
        )

    def _filter_example(self, example: Dict[str, Any]) -> bool:
        """
        Optional filter for examples.

        Override this to filter out unwanted examples before conversion.

        Args:
            example: Raw example dictionary from HF dataset

        Returns:
            True if example should be included, False to skip
        """
        return True

    def __len__(self) -> int:
        """Return total number of examples in dataset."""
        return len(self._hf_dataset)

    def __iter__(self) -> Iterator[PromptSpec]:
        """
        Iterate over all examples as PromptSpec objects.

        Applies filtering via _filter_example() before conversion.
        """
        for i, example in enumerate(self._hf_dataset):
            if self._filter_example(example):
                yield self._convert_to_prompt_spec(example, i)

    def get_hf_dataset(self):
        """
        Get underlying HuggingFace dataset object.

        Useful for inspecting schema, getting metadata, etc.
        """
        return self._hf_dataset
