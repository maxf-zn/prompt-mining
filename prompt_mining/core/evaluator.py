"""
Base evaluator interface for dataset-specific evaluation logic.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from prompt_mining.core.prompt_spec import PromptSpec


class Evaluator(ABC):
    """
    Base class for evaluating prompts and/or model outputs.

    Evaluators can operate on:
    - Prompt only (e.g., PromptGuardEvaluator for malicious/benign classification)
    - Output only (e.g., checking if model refused)
    - Both (e.g., InjecAgentEvaluator checks if output executed injected instruction)

    The evaluator receives the full PromptSpec so it can:
    - Access messages/tools for its own preprocessing (e.g., apply its own chat template)
    - Access labels for context (e.g., attacker_tools for InjecAgent)
    """

    @abstractmethod
    def evaluate(
        self,
        prompt_spec: PromptSpec,
        generated_text: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate prompt and/or model output, returning additional labels.

        Args:
            prompt_spec: The prompt specification (messages, tools, labels, text)
            generated_text: The model's generated output (None if generation disabled)

        Returns:
            Dict of additional labels to merge into prompt_labels, or None if evaluation fails
        """
        pass