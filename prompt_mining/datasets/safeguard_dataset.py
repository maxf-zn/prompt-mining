"""
SafeGuard prompt injection dataset from HuggingFace.

Simple binary classification dataset with benign and malicious prompts.
"""

from typing import Dict, Any, Optional
from prompt_mining.datasets.hf_dataset import HFDataset
from prompt_mining.core.prompt_spec import PromptSpec


class SafeGuardDataset(HFDataset):
    """
    SafeGuard prompt injection dataset.

    Dataset: xTRam1/safe-guard-prompt-injection

    Simple schema:
    - text: str (the prompt text)
    - label: int (0=benign, 1=malicious)

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
        Initialize SafeGuard dataset.

        Args:
            split: Dataset split to load (default: "train")
            name: Dataset name (default: "safeguard")
            dataset_id: Dataset ID (default: "safeguard")
            filter_malicious_only: If True, only include malicious prompts (label=1)
            filter_benign_only: If True, only include benign prompts (label=0)
            **kwargs: Additional arguments passed to HFDataset (e.g., cache_dir)
        """
        if filter_malicious_only and filter_benign_only:
            raise ValueError("Cannot set both filter_malicious_only and filter_benign_only")

        self.filter_malicious_only = filter_malicious_only
        self.filter_benign_only = filter_benign_only

        super().__init__(
            repo_id="xTRam1/safe-guard-prompt-injection",
            split=split,
            name=name or "safeguard",
            dataset_id=dataset_id or "safeguard",
            **kwargs
        )

    def _convert_to_prompt_spec(
        self,
        example: Dict[str, Any],
        index: int
    ) -> PromptSpec:
        """
        Convert SafeGuard example to PromptSpec.

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
            "dataset_label": int(label),  # Keep original label
        }

        # Generate prompt_id
        prompt_id = f"{self.dataset_id}:{index}"

        return PromptSpec(
            prompt_id=prompt_id,
            dataset_id=self.dataset_id,
            messages=messages,
            labels=labels,
            tools=None  # No tools for this dataset
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
