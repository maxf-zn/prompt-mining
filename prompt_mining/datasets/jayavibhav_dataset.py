"""
Jayavibhav prompt injection safety dataset from HuggingFace.

Multi-class classification dataset for prompt injection detection.
"""

from typing import Dict, Any, Optional, List
from prompt_mining.datasets.hf_dataset import HFDataset
from prompt_mining.core.prompt_spec import PromptSpec


class JayavibhavDataset(HFDataset):
    """
    Jayavibhav prompt injection safety dataset.

    Dataset: jayavibhav/prompt-injection-safety

    Multi-class classification dataset with three label categories.

    Schema:
    - text: str (the prompt text)
    - label: int (0, 1, or 2)

    Label meanings (based on typical safety dataset conventions):
    - 0: Safe/benign prompts
    - 1: Prompt injection attempts
    - 2: Other safety concerns (jailbreak, harmful content, etc.)

    Splits:
    - train: 50,000 rows
    - test: 10,000 rows

    Each prompt is converted to a single user message with classification labels.
    """

    def __init__(
        self,
        split: str = "train",
        name: Optional[str] = None,
        dataset_id: Optional[str] = None,
        filter_labels: Optional[List[int]] = None,
        filter_malicious_only: bool = False,
        filter_benign_only: bool = False,
        **kwargs
    ):
        """
        Initialize Jayavibhav dataset.

        Args:
            split: Dataset split to load ("train" or "test")
            name: Dataset name (default: "jayavibhav")
            dataset_id: Dataset ID (default: "jayavibhav")
            filter_labels: Only include examples with these labels (e.g., [0, 1])
            filter_malicious_only: If True, only include malicious prompts (label > 0)
            filter_benign_only: If True, only include benign prompts (label == 0)
            **kwargs: Additional arguments passed to HFDataset (e.g., cache_dir)
        """
        if filter_malicious_only and filter_benign_only:
            raise ValueError("Cannot set both filter_malicious_only and filter_benign_only")

        if filter_labels is not None and (filter_malicious_only or filter_benign_only):
            raise ValueError("Cannot use filter_labels with filter_malicious_only or filter_benign_only")

        self.filter_labels = filter_labels
        self.filter_malicious_only = filter_malicious_only
        self.filter_benign_only = filter_benign_only

        super().__init__(
            repo_id="jayavibhav/prompt-injection-safety",
            split=split,
            name=name or "jayavibhav",
            dataset_id=dataset_id or "jayavibhav",
            **kwargs
        )

    def _convert_to_prompt_spec(
        self,
        example: Dict[str, Any],
        index: int
    ) -> PromptSpec:
        """
        Convert Jayavibhav example to PromptSpec.

        Schema:
        - text: str (prompt text)
        - label: int (0, 1, or 2)

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

        # Determine if malicious (any non-zero label)
        is_malicious = label > 0

        # Build labels dict for analysis
        labels = {
            "malicious": is_malicious,
            "multiclass_label": int(label),
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
        Filter examples based on label criteria.

        Args:
            example: Raw HF example

        Returns:
            True if example should be included
        """
        label = example.get("label", 0)

        if self.filter_labels is not None:
            return label in self.filter_labels

        if self.filter_malicious_only:
            return label > 0

        if self.filter_benign_only:
            return label == 0

        return True
