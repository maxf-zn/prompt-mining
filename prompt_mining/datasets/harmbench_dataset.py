"""
HarmBench dataset from HuggingFace.

Contains behavior prompts for automated red teaming and robust refusal evaluation.
All prompts in this dataset are malicious/harmful behaviors.
"""

from typing import Dict, Any, Optional, List, Iterator
from datasets import load_dataset
from prompt_mining.datasets.base import Dataset
from prompt_mining.core.prompt_spec import PromptSpec


class HarmBenchDataset(Dataset):
    """
    HarmBench adversarial prompt dataset.

    Dataset: walledai/HarmBench

    Loads all 3 configurations combined:
    - standard: prompt + category (200 examples)
    - contextual: prompt + context + category (100 examples)
    - copyright: prompt + tags (100 examples)

    Schema varies by config:
    - prompt: str (the behavior prompt text)
    - category: str (harm category, standard/contextual only)
    - context: str (additional context, contextual only)
    - tags: str (copyright tags, copyright only)

    All prompts are malicious/harmful behaviors designed for adversarial testing.
    """

    SUBSETS = ["standard", "contextual", "copyright"]

    def __init__(
        self,
        split: str = "train",
        name: Optional[str] = None,
        dataset_id: Optional[str] = None,
        filter_categories: Optional[List[str]] = None,
    ):
        """
        Initialize HarmBench dataset (loads all 3 subsets).

        Args:
            split: Dataset split to load (default: "train", only split available)
            name: Dataset name (default: "harmbench")
            dataset_id: Dataset ID (default: "harmbench")
            filter_categories: Only include prompts from these categories (standard/contextual only)
        """
        super().__init__(name=name or "harmbench", dataset_id=dataset_id or "harmbench")

        self.filter_categories = filter_categories
        self._split = split

        # Load all 3 subsets (uses HF_DATASETS_CACHE env var if set)
        self._subset_datasets = {}
        for subset_name in self.SUBSETS:
            self._subset_datasets[subset_name] = load_dataset(
                "walledai/HarmBench",
                name=subset_name,
                split=split,
            )

    def __len__(self) -> int:
        """Return total number of examples across all subsets."""
        return sum(len(ds) for ds in self._subset_datasets.values())

    def __iter__(self) -> Iterator[PromptSpec]:
        """Iterate over all examples from all subsets."""
        global_index = 0
        for subset_name in self.SUBSETS:
            dataset = self._subset_datasets[subset_name]
            for example in dataset:
                if self._filter_example(example):
                    yield self._convert_to_prompt_spec(example, global_index, subset_name)
                global_index += 1

    def _convert_to_prompt_spec(
        self,
        example: Dict[str, Any],
        index: int,
        subset_name: str
    ) -> PromptSpec:
        """Convert HarmBench example to PromptSpec."""
        prompt_text = example.get("prompt", "")
        context = example.get("context")
        category = example.get("category")
        tags = example.get("tags")

        # For contextual subset, include context in the message
        if context:
            content = f"{prompt_text}\n\n{context}"
        else:
            content = prompt_text

        messages = [{"role": "user", "content": content}]

        # Build labels - all prompts are malicious
        labels = {
            "malicious": True,
            "dataset_source": "harmbench",
            "subset": subset_name,
        }
        if category:
            labels["category"] = category
        if tags:
            labels["tags"] = tags
        if context:
            labels["has_context"] = True

        return PromptSpec(
            prompt_id=f"{self.dataset_id}:{subset_name}:{index}",
            dataset_id=self.dataset_id,
            messages=messages,
            labels=labels,
            tools=None
        )

    def _filter_example(self, example: Dict[str, Any]) -> bool:
        """Filter examples based on category."""
        if self.filter_categories is not None:
            category = example.get("category", "")
            return category in self.filter_categories
        return True
