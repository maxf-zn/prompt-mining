"""
OpenOrca dataset from HuggingFace.

Instruction-following dataset with system prompts, questions, and responses.
Contains high-quality data from GPT-4 and GPT-3.5 Turbo.
"""

from typing import Dict, Any, Optional
from prompt_mining.datasets.hf_dataset import HFDataset
from prompt_mining.core.prompt_spec import PromptSpec


class OpenOrcaDataset(HFDataset):
    """
    OpenOrca instruction-following dataset.

    Dataset: Open-Orca/OpenOrca

    Schema:
    - id: str (unique identifier, e.g., "niv.242684", "flan.564327")
    - system_prompt: str (instruction template/system context)
    - question: str (user's question/request)
    - response: str (model's expected response)

    Dataset Source Categories (inferred from ID prefixes):
    - niv.*: NIV (Natural Instructions V2)
    - flan.*: FLAN collection
    - t0.*: T0 dataset
    - cot.*: Chain-of-Thought reasoning

    Each example is converted to a two-turn conversation:
    1. System message with the instruction template
    2. User message with the question

    The response is stored in labels for reference but not used in training.

    Splits:
    - train: ~1M examples (main training set)
    """

    def __init__(
        self,
        split: str = "train",
        name: Optional[str] = None,
        dataset_id: Optional[str] = None,
        filter_source: Optional[str] = None,
        max_system_prompt_length: Optional[int] = None,
        max_question_length: Optional[int] = None,
        include_system_prompt: bool = True,
        **kwargs
    ):
        """
        Initialize OpenOrca dataset.

        Args:
            split: Dataset split to load (default: "train")
            name: Dataset name (default: "openorca")
            dataset_id: Dataset ID (default: "openorca")
            filter_source: Filter by source prefix (e.g., "niv", "flan", "cot", "t0")
            max_system_prompt_length: Max character length for system prompts (None = no limit)
            max_question_length: Max character length for questions (None = no limit)
            include_system_prompt: If True, include system prompt message. If False, only user question.
            **kwargs: Additional arguments passed to HFDataset (e.g., cache_dir)
        """
        self.filter_source = filter_source
        self.max_system_prompt_length = max_system_prompt_length
        self.max_question_length = max_question_length
        self.include_system_prompt = include_system_prompt

        super().__init__(
            repo_id="Open-Orca/OpenOrca",
            split=split,
            name=name or "openorca",
            dataset_id=dataset_id or "openorca",
            **kwargs
        )

    def _convert_to_prompt_spec(
        self,
        example: Dict[str, Any],
        index: int
    ) -> PromptSpec:
        """
        Convert OpenOrca example to PromptSpec.

        Schema:
        - id: str (example ID)
        - system_prompt: str (instruction template)
        - question: str (user question)
        - response: str (expected response)

        Args:
            example: Raw HF example with id, system_prompt, question, response fields
            index: Index in dataset

        Returns:
            PromptSpec with system + user message turns
        """
        # Extract fields
        example_id = example.get("id", "")
        system_prompt = example.get("system_prompt", "")
        question = example.get("question", "")
        response = example.get("response", "")

        # Build conversation
        if self.include_system_prompt and system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
        else:
            messages = [
                {"role": "user", "content": question}
            ]

        # Extract source category from ID (e.g., "niv.242684" -> "niv")
        source = None
        if example_id and "." in example_id:
            source = example_id.split(".")[0]

        # Build labels dict (not classification, but metadata)
        labels = {
            "malicious": False,  # ALL openorca examples are benign (instruction-following)
            "source": source,
            "example_id": example_id,
            "expected_response": response,  # Store for reference
            "system_prompt_length": len(system_prompt),
            "question_length": len(question),
            "response_length": len(response),
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
        Filter examples based on source and length limits.

        Args:
            example: Raw HF example

        Returns:
            True if example should be included
        """
        # Filter by source if requested
        if self.filter_source:
            example_id = example.get("id", "")
            if not example_id.startswith(f"{self.filter_source}."):
                return False

        # Filter by system prompt length
        if self.max_system_prompt_length is not None:
            system_prompt = example.get("system_prompt", "")
            if len(system_prompt) > self.max_system_prompt_length:
                return False

        # Filter by question length
        if self.max_question_length is not None:
            question = example.get("question", "")
            if len(question) > self.max_question_length:
                return False

        return True
