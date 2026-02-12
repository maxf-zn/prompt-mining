"""
BIPIA dataset provider for prompt mining platform.

BIPIA (Benchmark for Indirect Prompt Injection Attacks) contains prompts where
malicious instructions are embedded in external context (emails, code, tables, etc.)
that the LLM processes.

Tasks: email, code, table, qa, abstract

Reference: https://github.com/microsoft/BIPIA
Paper: https://arxiv.org/abs/2312.14197
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional, Union

from prompt_mining.datasets.base import Dataset
from prompt_mining.core.prompt_spec import PromptSpec


class BIPIADataset(Dataset):
    """
    BIPIA indirect prompt injection attack dataset.

    Uses BIPIA's native builders to construct poisoned samples, then converts
    to PromptSpec format. Supports loading multiple tasks at once.

    Example:
        >>> dataset = BIPIADataset(
        ...     bipia_root="/path/to/BIPIA",
        ...     task_names=["email", "code"],
        ...     split="test",
        ...     insert_positions=["end"]
        ... )
        >>> for prompt in dataset:
        ...     print(prompt.prompt_id, prompt.labels["malicious"])
    """

    AVAILABLE_TASKS = ["email", "code", "table", "qa", "abstract"]

    def __init__(
        self,
        bipia_root: str,
        task_names: Union[str, List[str]] = "email",
        split: str = "test",
        insert_positions: Optional[List[str]] = None,
        enable_stealth: bool = False,
        seed: int = 2023,
        include_clean: bool = False,
        name: Optional[str] = None,
        dataset_id: Optional[str] = None,
    ):
        """
        Initialize BIPIA dataset.

        Args:
            bipia_root: Path to BIPIA repository root
            task_names: Task(s) to load - single string or list of strings
                       Available: "email", "code", "table", "qa", "abstract"
            split: Data split ("train" or "test")
            insert_positions: Attack insertion positions (default: ["end", "start", "middle"])
            enable_stealth: Base64 encode attacks if True
            seed: Random seed for reproducibility
            include_clean: If True, also include samples without attack injection
            name: Dataset name (auto-generated if not provided)
            dataset_id: Unique identifier (auto-generated if not provided)
        """
        self.bipia_root = Path(bipia_root)
        self.split = split
        self.insert_positions = insert_positions or ["end", "start", "middle"]
        self.enable_stealth = enable_stealth
        self.seed = seed
        self.include_clean = include_clean

        # Normalize task_names to list
        if isinstance(task_names, str):
            task_names = [task_names]
        self.task_names = task_names

        # Validate tasks
        for task in self.task_names:
            if task not in self.AVAILABLE_TASKS:
                raise ValueError(
                    f"Unknown task: {task}. Available: {self.AVAILABLE_TASKS}"
                )

        # Validate insert positions
        valid_positions = ["end", "start", "middle"]
        for pos in self.insert_positions:
            if pos not in valid_positions:
                raise ValueError(
                    f"Unknown insert position: {pos}. Available: {valid_positions}"
                )

        # Generate name if not provided
        if name is None:
            tasks_str = "_".join(self.task_names)
            positions_str = "_".join(self.insert_positions)
            name = f"bipia_{tasks_str}_{split}_{positions_str}"

        super().__init__(name=name, dataset_id=dataset_id or f"bipia_{'_'.join(self.task_names)}")

        # Import BIPIA modules
        self._import_bipia()

        # Build samples using BIPIA builders
        self._samples = self._build_samples()

    def _import_bipia(self):
        """Import BIPIA modules by adding to sys.path."""
        bipia_str = str(self.bipia_root.resolve())
        if bipia_str not in sys.path:
            sys.path.insert(0, bipia_str)

        try:
            from bipia.data import AutoPIABuilder
            from bipia.data.utils import insert_end, insert_start, insert_middle, no_insert

            self._AutoPIABuilder = AutoPIABuilder
            self._insert_fns = {
                "end": insert_end,
                "start": insert_start,
                "middle": insert_middle,
                "none": no_insert,
            }
        except ImportError as e:
            raise ImportError(
                f"Failed to import BIPIA modules from {self.bipia_root}. "
                f"Make sure bipia_root points to BIPIA repository. Error: {e}"
            )

    def _get_attack_file(self, task_name: str) -> str:
        """Get attack file path for a task."""
        # code task uses code_attack, others use text_attack
        attack_type = "code_attack" if task_name == "code" else "text_attack"
        return str(self.bipia_root / "benchmark" / f"{attack_type}_{self.split}.json")

    def _get_context_file(self, task_name: str) -> str:
        """Get context file path for a task."""
        return str(self.bipia_root / "benchmark" / task_name / f"{self.split}.jsonl")

    def _build_samples(self) -> List[Dict[str, Any]]:
        """Build all samples using BIPIA builders."""
        samples = []

        # Build insert functions list based on config
        insert_fns = [self._insert_fns[pos] for pos in self.insert_positions]
        insert_fn_names = self.insert_positions.copy()

        for task_name in self.task_names:
            # Get builder for this task
            builder_cls = self._AutoPIABuilder.from_name(task_name)
            builder = builder_cls(seed=self.seed)

            context_file = self._get_context_file(task_name)
            attack_file = self._get_attack_file(task_name)

            # Optionally include true clean samples (original contexts only)
            if self.include_clean:
                clean_df = builder(
                    contexts=context_file,
                    attacks=attack_file,
                    insert_fns=[self._insert_fns["none"]],
                    insert_fn_names=["none"],
                    enable_stealth=False,
                )
                # Deduplicate: keep only unique contexts (first occurrence per context)
                seen_contexts = set()
                for _, row in clean_df.iterrows():
                    ctx = row.get("context", "")
                    if ctx not in seen_contexts:
                        seen_contexts.add(ctx)
                        row_dict = row.to_dict()
                        row_dict["position"] = "none"
                        row_dict["attack_name"] = None
                        row_dict["attack_str"] = None
                        system_prompt, user_prompt = builder.construct_prompt(row_dict)
                        samples.append({
                            "task_name": task_name,
                            "row": row_dict,
                            "system_prompt": system_prompt,
                            "user_prompt": user_prompt,
                        })

            # Build poisoned samples
            df = builder(
                contexts=context_file,
                attacks=attack_file,
                insert_fns=insert_fns,
                insert_fn_names=insert_fn_names,
                enable_stealth=self.enable_stealth,
            )

            for _, row in df.iterrows():
                row_dict = row.to_dict()
                system_prompt, user_prompt = builder.construct_prompt(row_dict)
                samples.append({
                    "task_name": task_name,
                    "row": row_dict,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                })

        return samples

    def _convert_to_prompt_spec(self, sample: Dict[str, Any], index: int) -> PromptSpec:
        """Convert internal sample to PromptSpec format."""
        row = sample["row"]
        task_name = sample["task_name"]
        position = row.get("position", "none")
        is_poisoned = position != "none"

        # Build messages from system and user prompts
        messages = [
            {"role": "system", "content": sample["system_prompt"]},
            {"role": "user", "content": sample["user_prompt"]},
        ]

        # Build labels
        labels = {
            "malicious": is_poisoned,
            "task_name": task_name,
            "split": self.split,
            "position": position,
            "attack_name": row.get("attack_name"),
            "ideal": row.get("ideal"),
            "enable_stealth": self.enable_stealth,
        }

        # Add question if present
        if "question" in row:
            labels["question"] = row["question"]

        # Create prompt ID
        if is_poisoned:
            attack_short = (
                str(row.get("attack_name", ""))
                .replace(" ", "_")
                .replace("/", "_")
                .replace(":", "_")[:30]
            )
            prompt_id = f"{self.dataset_id}:{task_name}:{self.split}:{position}:{attack_short}:{index}"
        else:
            prompt_id = f"{self.dataset_id}:{task_name}:{self.split}:clean:{index}"

        return PromptSpec(
            prompt_id=prompt_id,
            dataset_id=self.dataset_id,
            messages=messages,
            labels=labels,
            tools=None,
        )

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self._samples)

    def __iter__(self) -> Iterator[PromptSpec]:
        """Iterate over all samples as PromptSpec objects."""
        for i, sample in enumerate(self._samples):
            yield self._convert_to_prompt_spec(sample, i)

    # Utility methods
    def get_task_names(self) -> List[str]:
        """Return list of tasks in this dataset."""
        return self.task_names.copy()

    @staticmethod
    def available_tasks() -> List[str]:
        """Return list of all available task names."""
        return BIPIADataset.AVAILABLE_TASKS.copy()
