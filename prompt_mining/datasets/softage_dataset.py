"""
SoftAge prompt engineering dataset from HuggingFace.

Single-turn, benign prompts categorized by topic/type.
"""

from typing import Dict, Any, Optional

from prompt_mining.datasets.hf_dataset import HFDataset
from prompt_mining.core.prompt_spec import PromptSpec


class SoftAgeDataset(HFDataset):
    """
    SoftAge prompt engineering dataset.

    Dataset: SoftAge-AI/prompt-eng_dataset (gated)

    Schema (observed):
    - S.No.: float
    - Prompt: str
    - Category: str
    - Type: str
    """

    def __init__(
        self,
        split: str = "train",
        name: Optional[str] = None,
        dataset_id: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            repo_id="SoftAge-AI/prompt-eng_dataset",
            split=split,
            name=name or "softAge",
            dataset_id=dataset_id or "softAge",
            **kwargs,
        )

    def _convert_to_prompt_spec(self, example: Dict[str, Any], index: int) -> PromptSpec:
        prompt = example.get("Prompt", "") or ""
        category = example.get("Category", None)
        prompt_type = example.get("Type", None)
        serial = example.get("S.No.", None)

        messages = [{"role": "user", "content": prompt}]
        labels = {
            "malicious": False,
            "category": category,
            "type": prompt_type,
            "serial": serial,
        }

        prompt_id = f"{self.dataset_id}:{self.split}:{index}"
        return PromptSpec(
            prompt_id=prompt_id,
            dataset_id=self.dataset_id,
            messages=messages,
            labels=labels,
            tools=None,
        )


# Backwards-compat alias (older name used during initial rollout)
PromptEngDataset = SoftAgeDataset


