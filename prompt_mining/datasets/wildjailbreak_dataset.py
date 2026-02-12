"""
Allen AI WildJailbreak dataset from HuggingFace.

Large-scale adversarial jailbreak dataset with vanilla and adversarial prompts.
"""

from typing import Dict, Any, Optional, List
from prompt_mining.datasets.hf_dataset import HFDataset
from prompt_mining.core.prompt_spec import PromptSpec


class WildJailbreakDataset(HFDataset):
    """
    Allen AI WildJailbreak dataset.

    Dataset: allenai/wildjailbreak

    Large-scale dataset with vanilla and adversarial prompts, both harmful and benign.

    Schema:
    - vanilla: str (original prompt)
    - adversarial: str (adversarial version, empty for vanilla data points)
    - tactics: list (jailbreak tactics used)
    - completion: str (model response)
    - data_type: str (vanilla_harmful, vanilla_benign, adversarial_harmful, adversarial_benign)

    Configs (subset):
    - train: 262k examples
    - eval: 2,210 examples (adversarial only)
    """

    VALID_DATA_TYPES = [
        "vanilla_harmful",
        "vanilla_benign",
        "adversarial_harmful",
        "adversarial_benign",
    ]

    def __init__(
        self,
        split: str = "train",
        subset: str = "train",
        name: Optional[str] = None,
        dataset_id: Optional[str] = None,
        filter_data_types: Optional[List[str]] = None,
        filter_malicious_only: bool = False,
        filter_benign_only: bool = False,
        use_adversarial: bool = True,
        **kwargs
    ):
        """
        Initialize WildJailbreak dataset.

        Args:
            split: Dataset split (default: "train")
            subset: Dataset config/subset ("train" or "eval")
            name: Dataset name (default: "wildjailbreak")
            dataset_id: Dataset ID (default: "wildjailbreak")
            filter_data_types: Only include these data types (e.g., ["adversarial_harmful"])
            filter_malicious_only: If True, only include harmful prompts
            filter_benign_only: If True, only include benign prompts
            use_adversarial: If True, prefer adversarial prompt over vanilla when available
            **kwargs: Additional arguments passed to HFDataset (e.g., cache_dir)
        """
        if filter_malicious_only and filter_benign_only:
            raise ValueError("Cannot set both filter_malicious_only and filter_benign_only")

        if filter_data_types is not None:
            for dt in filter_data_types:
                if dt not in self.VALID_DATA_TYPES:
                    raise ValueError(f"Invalid data_type: {dt}. Must be one of {self.VALID_DATA_TYPES}")

        self.filter_data_types = filter_data_types
        self.filter_malicious_only = filter_malicious_only
        self.filter_benign_only = filter_benign_only
        self.use_adversarial = use_adversarial

        super().__init__(
            repo_id="allenai/wildjailbreak",
            split=split,
            subset=subset,
            name=name or "wildjailbreak",
            dataset_id=dataset_id or "wildjailbreak",
            **kwargs
        )

    def _convert_to_prompt_spec(
        self,
        example: Dict[str, Any],
        index: int
    ) -> PromptSpec:
        """
        Convert WildJailbreak example to PromptSpec.

        Args:
            example: Raw HF example
            index: Index in dataset

        Returns:
            PromptSpec with single user message
        """
        # Extract fields
        vanilla = example.get("vanilla", "")
        adversarial = example.get("adversarial", "")
        data_type = example.get("data_type", "")
        tactics = example.get("tactics", [])

        # Choose which prompt to use
        if self.use_adversarial and adversarial:
            text = adversarial
        else:
            text = vanilla

        # Build simple message format (single user turn)
        messages = [
            {"role": "user", "content": text}
        ]

        # Determine if malicious based on data_type
        is_malicious = "harmful" in data_type
        is_adversarial = "adversarial" in data_type

        # Build labels dict
        labels = {
            "malicious": is_malicious,
            "data_type": data_type,
            "is_adversarial": is_adversarial,
            "tactics": tactics if tactics else None,
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
        Filter examples based on data type or malicious/benign.

        Args:
            example: Raw HF example

        Returns:
            True if example should be included
        """
        data_type = example.get("data_type", "")

        if self.filter_data_types is not None:
            if data_type not in self.filter_data_types:
                return False

        if self.filter_malicious_only:
            return "harmful" in data_type

        if self.filter_benign_only:
            return "benign" in data_type

        return True
