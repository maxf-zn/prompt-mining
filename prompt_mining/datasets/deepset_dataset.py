"""
Deepset prompt injections dataset from HuggingFace.

Binary classification dataset with benign and malicious prompts.
"""

from typing import Dict, Any, Optional
from prompt_mining.datasets.hf_dataset import HFDataset
from prompt_mining.core.prompt_spec import PromptSpec


class DeepsetDataset(HFDataset):
    """
    Deepset prompt injections dataset.

    Dataset: deepset/prompt-injections

    Schema:
    - text: str (the prompt text)
    - label: int (0=benign, 1=malicious)

    Splits:
    - train: 546 rows
    - test: 116 rows

    Each prompt is converted to a single user message with malicious/benign label.
    """

    def __init__(
        self,
        split: str = "train",
        name: Optional[str] = None,
        dataset_id: Optional[str] = None,
        filter_malicious_only: bool = False,
        filter_benign_only: bool = False,
        **kwargs
    ):
        """
        Initialize Deepset dataset.

        Args:
            split: Dataset split to load ("train" or "test")
            name: Dataset name (default: "deepset")
            dataset_id: Dataset ID (default: "deepset")
            filter_malicious_only: If True, only include malicious prompts (label=1)
            filter_benign_only: If True, only include benign prompts (label=0)
            **kwargs: Additional arguments passed to HFDataset (e.g., cache_dir)
        """
        if filter_malicious_only and filter_benign_only:
            raise ValueError("Cannot set both filter_malicious_only and filter_benign_only")

        self.filter_malicious_only = filter_malicious_only
        self.filter_benign_only = filter_benign_only

        super().__init__(
            repo_id="deepset/prompt-injections",
            split=split,
            name=name or "deepset",
            dataset_id=dataset_id or "deepset",
            **kwargs
        )

    def _convert_to_prompt_spec(
        self,
        example: Dict[str, Any],
        index: int
    ) -> PromptSpec:
        """
        Convert Deepset example to PromptSpec.

        Schema:
        - text: str (prompt text)
        - label: int (0=benign, 1=malicious)

        Args:
            example: Raw HF example with 'text' and 'label' fields
            index: Index in dataset

        Returns:
            PromptSpec with single user message
        """
        # Extract fields
        text = example.get("text", "")
        label = example.get("label", 0)

        # Build simple message format (single user turn)
        messages = [
            {"role": "user", "content": text}
        ]

        # Build labels dict for analysis
        labels = {
            "malicious": bool(label == 1),
        }

        # Generate prompt_id
        prompt_id = f"{self.dataset_id}:{self.split}:{index}"

        return PromptSpec(
            prompt_id=prompt_id,
            dataset_id=self.dataset_id,
            messages=messages,
            labels=labels,
            tools=None
        )

    def _filter_example(self, example: Dict[str, Any]) -> bool:
        """
        Filter examples based on label if requested.

        Args:
            example: Raw HF example

        Returns:
            True if example should be included
        """
        if self.filter_malicious_only:
            return example.get("label", 0) == 1

        if self.filter_benign_only:
            return example.get("label", 0) == 0

        return True
