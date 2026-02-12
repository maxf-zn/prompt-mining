"""
Prompt ranking dataset from HuggingFace (data-is-better-together).

Single-turn prompts with quality ratings and topic metadata.
"""

import json
from typing import Dict, Any, Optional

from prompt_mining.datasets.hf_dataset import HFDataset
from prompt_mining.core.prompt_spec import PromptSpec


class PromptsRanked10kDataset(HFDataset):
    """
    10k prompts ranked dataset.

    Dataset: data-is-better-together/10k_prompts_ranked

    Schema (observed):
    - prompt: str
    - avg_rating: float
    - num_responses: int
    - agreement_ratio: float
    - kind: str
    - topic: str
    - cluster_description: str
    - metadata: str (JSON)
    - quality: list[dict]
    - raw_responses: list[int]
    """

    def __init__(
        self,
        split: str = "train",
        name: Optional[str] = None,
        dataset_id: Optional[str] = None,
        min_avg_rating: Optional[float] = None,
        **kwargs,
    ):
        self.min_avg_rating = min_avg_rating
        super().__init__(
            repo_id="data-is-better-together/10k_prompts_ranked",
            split=split,
            name=name or "10k_prompts_ranked",
            dataset_id=dataset_id or "10k_prompts_ranked",
            **kwargs,
        )

    def _filter_example(self, example: Dict[str, Any]) -> bool:
        if self.min_avg_rating is None:
            return True
        try:
            return float(example.get("avg_rating")) >= float(self.min_avg_rating)
        except Exception:
            return False

    def _convert_to_prompt_spec(self, example: Dict[str, Any], index: int) -> PromptSpec:
        prompt = example.get("prompt", "") or ""

        meta_raw = example.get("metadata", None)
        meta = None
        if isinstance(meta_raw, str):
            try:
                meta = json.loads(meta_raw)
            except Exception:
                meta = {"raw": meta_raw}

        messages = [{"role": "user", "content": prompt}]
        labels = {
            "malicious": False,
            "avg_rating": example.get("avg_rating", None),
            "num_responses": example.get("num_responses", None),
            "agreement_ratio": example.get("agreement_ratio", None),
            "kind": example.get("kind", None),
            "topic": example.get("topic", None),
            "cluster_description": example.get("cluster_description", None),
            "metadata": meta,
        }

        prompt_id = f"{self.dataset_id}:{self.split}:{index}"
        return PromptSpec(
            prompt_id=prompt_id,
            dataset_id=self.dataset_id,
            messages=messages,
            labels=labels,
            tools=None,
        )


