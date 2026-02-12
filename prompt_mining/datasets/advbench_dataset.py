"""
AdvBench dataset from HuggingFace.

Contains malicious prompts for adversarial testing of aligned language models.
All prompts in this dataset are malicious/harmful behaviors.
"""

from typing import Dict, Any, Optional
from prompt_mining.datasets.hf_dataset import HFDataset
from prompt_mining.core.prompt_spec import PromptSpec


class AdvBenchDataset(HFDataset):
    """
    AdvBench adversarial prompt injection dataset.

    Dataset: walledai/AdvBench

    Schema:
    - prompt: str (the malicious prompt text)

    All prompts are malicious/harmful behaviors designed to test adversarial robustness.
    Each prompt is converted to a single user message with malicious=True label.
    """

    def __init__(
        self,
        split: str = "train",
        name: Optional[str] = None,
        dataset_id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize AdvBench dataset.

        Args:
            split: Dataset split to load (default: "train", only split available)
            name: Dataset name (default: "advbench")
            dataset_id: Dataset ID (default: "advbench")
            **kwargs: Additional arguments passed to HFDataset (e.g., cache_dir)
        """
        super().__init__(
            repo_id="walledai/AdvBench",
            split=split,
            name=name or "advbench",
            dataset_id=dataset_id or "advbench",
            **kwargs
        )

    def _convert_to_prompt_spec(
        self,
        example: Dict[str, Any],
        index: int
    ) -> PromptSpec:
        """
        Convert AdvBench example to PromptSpec.

        Schema:
        - prompt: str (malicious prompt text)

        Args:
            example: Raw HF example with 'prompt' field
            index: Index in dataset

        Returns:
            PromptSpec with single user message and malicious label
        """
        # Extract the prompt text
        prompt_text = example.get("prompt", "")

        # Build simple message format (single user turn)
        messages = [
            {"role": "user", "content": prompt_text}
        ]

        # Build labels dict for analysis - all prompts are malicious
        labels = {
            "malicious": True,
            "dataset_source": "advbench",
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
