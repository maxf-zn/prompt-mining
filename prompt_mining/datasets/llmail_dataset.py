"""
LLMail-Inject Challenge dataset from Microsoft.

Email-based prompt injection attacks from the LLMail-Inject competition.
All examples are malicious (instructions hidden in email content).
"""

from typing import Dict, Any, Optional
from prompt_mining.datasets.hf_dataset import HFDataset
from prompt_mining.core.prompt_spec import PromptSpec


class LLMailDataset(HFDataset):
    """
    LLMail-Inject Challenge dataset.

    Dataset: microsoft/llmail-inject-challenge

    Email-based prompt injection attacks where malicious instructions are hidden
    in email subject lines and body content. All examples are attack attempts.

    Schema:
    - subject: str (email subject line)
    - body: str (email body content)
    - scenario: str (attack level: level1a-level3j)
    - objectives: JSON (attack success metrics)
    - output: str (system response - not used for training)

    Attack Levels:
    - Level 1 (level1a-j): Basic attacks (simple, direct)
    - Level 2 (level2a-j): Intermediate (obfuscation, system tokens)
    - Level 3 (level3a-j): Advanced (heavy obfuscation, multi-lingual)

    Splits:
    - Phase1: 371k examples (main competition data)
    - Phase2: 91k examples (extended phase)
    """

    def __init__(
        self,
        split: str = "Phase1",
        name: Optional[str] = None,
        dataset_id: Optional[str] = None,
        filter_level: Optional[int] = None,
        include_email_format: bool = True,
        **kwargs
    ):
        """
        Initialize LLMail dataset.

        Args:
            split: Dataset split ("Phase1" or "Phase2")
                  Phase1: 371k examples (recommended for training)
                  Phase2: 91k examples (recommended for testing)
            name: Dataset name (default: "llmail")
            dataset_id: Dataset ID (default: "llmail")
            filter_level: Filter by attack sophistication level (1, 2, or 3)
                         None = all levels (recommended)
            include_email_format: If True, format as "Subject: X\nBody: Y"
                                 If False, body only (no subject)
            **kwargs: Additional arguments passed to HFDataset (e.g., cache_dir)
        """
        if split not in ["Phase1", "Phase2"]:
            raise ValueError(f"Split must be 'Phase1' or 'Phase2', got: {split}")

        if filter_level is not None and filter_level not in [1, 2, 3]:
            raise ValueError(f"filter_level must be 1, 2, 3, or None, got: {filter_level}")

        self.filter_level = filter_level
        self.include_email_format = include_email_format

        super().__init__(
            repo_id="microsoft/llmail-inject-challenge",
            split=split,
            name=name or "llmail",
            dataset_id=dataset_id or "llmail",
            **kwargs
        )

    def _convert_to_prompt_spec(
        self,
        example: Dict[str, Any],
        index: int
    ) -> PromptSpec:
        """
        Convert LLMail example to PromptSpec.

        Schema:
        - subject: str (email subject)
        - body: str (email body)
        - scenario: str (e.g., "level2e")

        Args:
            example: Raw HF example with subject, body, scenario fields
            index: Index in dataset

        Returns:
            PromptSpec with email content as user message
        """
        # Extract fields
        subject = example.get("subject", "")
        body = example.get("body", "")
        scenario = example.get("scenario", "")

        # Format prompt
        if self.include_email_format:
            # Email format (more realistic for email agents)
            content = f"Subject: {subject}\n\nBody:\n{body}"
        else:
            # Body only (for comparison with other email datasets)
            content = f"{subject}\n{body}"

        # Build simple message format (single user turn)
        messages = [
            {"role": "user", "content": content}
        ]

        # Extract level from scenario (e.g., "level2e" -> 2)
        level = None
        if scenario and scenario.startswith("level"):
            try:
                level = int(scenario[5])  # Extract digit from "levelXy"
            except (IndexError, ValueError):
                pass

        # Build labels dict
        labels = {
            "malicious": True,  # ALL llmail examples are malicious
            "scenario": scenario,
            "level": level,
            "attack_type": "email_injection",
        }

        # Generate prompt_id
        prompt_id = f"{self.dataset_id}:{self.split}:{index}"

        return PromptSpec(
            prompt_id=prompt_id,
            dataset_id=self.dataset_id,
            messages=messages,
            labels=labels,
            tools=None  # No tools for this dataset
        )

    def _filter_example(self, example: Dict[str, Any]) -> bool:
        """
        Filter examples based on attack level if requested.

        Args:
            example: Raw HF example

        Returns:
            True if example should be included
        """
        if self.filter_level is None:
            return True

        scenario = example.get("scenario", "")
        if not scenario:
            return False

        # Check if scenario matches requested level
        # e.g., filter_level=2 matches "level2a", "level2b", etc.
        target_prefix = f"level{self.filter_level}"
        return scenario.startswith(target_prefix)
