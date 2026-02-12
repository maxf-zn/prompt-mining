from __future__ import annotations

import importlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class IngestArgs:
    config_path: Path
    dataset_name: str
    num_prompts: Optional[int]
    force: bool


class BaseIngestionPipeline(ABC):
    """
    Base class for ingestion pipelines.

    Provides shared utilities:
    - dynamic class loading
    - dataset + evaluator construction
    - common config parsing helpers
    """

    def __init__(
        self,
        *,
        args: IngestArgs,
        model_config: Any,
        run_config: Any,
        dataset_cfg: Dict[str, Any],
        global_cfg: Dict[str, Any],
    ):
        self.args = args
        self.model_config = model_config
        self.run_config = run_config
        self.dataset_cfg = dataset_cfg
        self.global_cfg = global_cfg

        self.dataset_class_name = dataset_cfg.get("class")
        self.dataset_params = dataset_cfg.get("params", {})
        self.evaluator_class_name = dataset_cfg.get("evaluator_class")
        self.evaluator_params = dataset_cfg.get("evaluator_params", {})

        self.output_dir = Path(global_cfg.get("output_dir", "./output"))

    @staticmethod
    def load_class(class_name: str, module_prefix: str = "prompt_mining.datasets"):
        # Try direct import from module_prefix
        try:
            module = importlib.import_module(module_prefix)
            return getattr(module, class_name)
        except (ImportError, AttributeError):
            pass

        # Try with explicit submodule (CamelCase -> snake_case)
        module_name = "".join(["_" + c.lower() if c.isupper() else c for c in class_name]).lstrip("_")
        module = importlib.import_module(f"{module_prefix}.{module_name}")
        return getattr(module, class_name)

    def resolve_num_prompts(self) -> int:
        # CLI override > dataset config > global config > all (-1)
        if self.args.num_prompts is not None:
            return self.args.num_prompts
        if self.dataset_cfg.get("num_prompts") is not None:
            return int(self.dataset_cfg["num_prompts"])
        return int(self.global_cfg.get("num_prompts", -1))

    def load_dataset(self):
        if not self.dataset_class_name:
            raise ValueError("Dataset config missing 'class'")
        dataset_class = self.load_class(self.dataset_class_name, "prompt_mining.datasets")
        return dataset_class(**self.dataset_params)

    def load_evaluator(self):
        if not self.evaluator_class_name:
            return None
        evaluator_class = self.load_class(self.evaluator_class_name, "prompt_mining.evaluators")
        return evaluator_class(**self.evaluator_params)

    @staticmethod
    def pretty(obj: Any) -> str:
        try:
            return json.dumps(obj, indent=2, default=str)
        except Exception:
            return str(obj)

    @abstractmethod
    def run(self) -> int:
        raise NotImplementedError


