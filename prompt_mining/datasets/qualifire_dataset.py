"""
Qualifire prompt injections benchmark dataset from HuggingFace.

Binary classification dataset with jailbreak and benign prompts.
"""

from typing import Dict, Any, Optional
from prompt_mining.datasets.hf_dataset import HFDataset
from prompt_mining.core.prompt_spec import PromptSpec


class QualifireDataset(HFDataset):
    """
    Qualifire prompt injections benchmark dataset.

    Dataset: qualifire/prompt-injections-benchmark

    Schema:
    - text: str (the prompt text)
    - label: str ("jailbreak" or "benign")

    Total: 5,000 samples

    Splits:
    - test: 5,000 rows (only split available)

    Each prompt is converted to a single user message with malicious/benign label.
    """

    def __init__(
        self,
        split: str = "test",
        name: Optional[str] = None,
        dataset_id: Optional[str] = None,
        filter_malicious_only: bool = False,
        filter_benign_only: bool = False,
        **kwargs
    ):
        """
        Initialize Qualifire dataset.

        Args:
            split: Dataset split to load (default: "test", only available split)
            name: Dataset name (default: "qualifire")
            dataset_id: Dataset ID (default: "qualifire")
            filter_malicious_only: If True, only include jailbreak prompts
            filter_benign_only: If True, only include benign prompts
            **kwargs: Additional arguments passed to HFDataset (e.g., cache_dir)
        """
        if filter_malicious_only and filter_benign_only:
            raise ValueError("Cannot set both filter_malicious_only and filter_benign_only")

        self.filter_malicious_only = filter_malicious_only
        self.filter_benign_only = filter_benign_only

        super().__init__(
            repo_id="qualifire/prompt-injections-benchmark",
            split=split,
            name=name or "qualifire",
            dataset_id=dataset_id or "qualifire",
            **kwargs
        )

    def _convert_to_prompt_spec(
        self,
        example: Dict[str, Any],
        index: int
    ) -> PromptSpec:
        """
        Convert Qualifire example to PromptSpec.

        Schema:
        - text: str (prompt text)
        - label: str ("jailbreak" or "benign")

        Args:
            example: Raw HF example with 'text' and 'label' fields
            index: Index in dataset

        Returns:
            PromptSpec with single user message
        """
        # Extract fields
        text = example.get("text", "")
        label_str = example.get("label", "benign")

        # Convert string label to boolean/int
        is_malicious = label_str.lower() == "jailbreak"

        # Build simple message format (single user turn)
        messages = [
            {"role": "user", "content": text}
        ]

        # Build labels dict for analysis
        labels = {
            "malicious": is_malicious,
            "label_str": label_str,
            "attack_type": "jailbreak" if is_malicious else None,
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
        label = example.get("label", "").lower()

        if self.filter_malicious_only:
            return label == "jailbreak"

        if self.filter_benign_only:
            return label == "benign"

        return True
