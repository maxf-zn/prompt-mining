"""
Yanismiraoui prompt injections dataset from HuggingFace.

Multilingual prompt injection examples without labels (all malicious).
"""

from typing import Dict, Any, Optional
from prompt_mining.datasets.hf_dataset import HFDataset
from prompt_mining.core.prompt_spec import PromptSpec


class YanismiraouiDataset(HFDataset):
    """
    Yanismiraoui prompt injections dataset.

    Dataset: yanismiraoui/prompt_injections

    A collection of multilingual prompt injection examples including:
    - Prompt leaking attempts
    - Jailbreaking techniques
    - Mode switching requests
    - Data privacy violations
    - Requests to bypass ethical constraints

    Languages: English, French, German, Spanish, Italian, Portuguese, Romanian

    Schema:
    - prompt_injections: str (the injection prompt text)

    Split:
    - train: 1,034 rows

    All examples are malicious (prompt injection attempts) - no benign examples.
    """

    def __init__(
        self,
        split: str = "train",
        name: Optional[str] = None,
        dataset_id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Yanismiraoui dataset.

        Args:
            split: Dataset split to load (default: "train")
            name: Dataset name (default: "yanismiraoui")
            dataset_id: Dataset ID (default: "yanismiraoui")
            **kwargs: Additional arguments passed to HFDataset (e.g., cache_dir)
        """
        super().__init__(
            repo_id="yanismiraoui/prompt_injections",
            split=split,
            name=name or "yanismiraoui",
            dataset_id=dataset_id or "yanismiraoui",
            **kwargs
        )

    def _convert_to_prompt_spec(
        self,
        example: Dict[str, Any],
        index: int
    ) -> PromptSpec:
        """
        Convert Yanismiraoui example to PromptSpec.

        Schema:
        - prompt_injections: str (injection prompt)

        Args:
            example: Raw HF example with 'prompt_injections' field
            index: Index in dataset

        Returns:
            PromptSpec with single user message
        """
        # Extract the prompt text
        text = example.get("prompt_injections", "")

        # Build simple message format (single user turn)
        messages = [
            {"role": "user", "content": text}
        ]

        # Build labels dict - ALL examples are malicious
        labels = {
            "malicious": True,
            "attack_type": "prompt_injection",
            "multilingual": True,
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
