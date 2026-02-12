"""
InjecAgent-specific evaluator for prompt injection detection.

Uses InjecAgent's evaluate_output_flexible function.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

from prompt_mining.core.evaluator import Evaluator
from prompt_mining.core.prompt_spec import PromptSpec


class InjecAgentEvaluator(Evaluator):
    """
    Evaluator for InjecAgent prompt injection attacks.

    Detects whether model output executed the attacker's injected instruction.
    """

    def __init__(self, injecagent_root: Optional[Path] = None):
        """
        Initialize InjecAgent evaluator.

        Args:
            injecagent_root: Path to InjecAgent repository root.
                            If None, attempts to find it relative to this file.
        """
        if injecagent_root is None:
            # Assume InjecAgent is at ../../../../InjecAgent relative to this file
            injecagent_root = Path(__file__).parent.parent.parent.parent / "InjecAgent"

        self.injecagent_root = Path(injecagent_root)
        self._import_injecagent()

    def _import_injecagent(self):
        """Import InjecAgent evaluation function."""
        injecagent_src = self.injecagent_root / "src"
        if not injecagent_src.exists():
            raise FileNotFoundError(
                f"InjecAgent src directory not found at {injecagent_src}. "
                f"Please provide correct injecagent_root path."
            )

        if str(injecagent_src) not in sys.path:
            sys.path.insert(0, str(injecagent_src))

        try:
            from output_parsing_flexible import evaluate_output_flexible
            self._evaluate_fn = evaluate_output_flexible
        except ImportError as e:
            raise ImportError(
                f"Failed to import evaluate_output_flexible from {injecagent_src}. "
                f"Error: {e}"
            )

    def evaluate(
        self,
        prompt_spec: PromptSpec,
        generated_text: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate InjecAgent output for prompt injection success.

        Args:
            prompt_spec: The prompt specification (uses labels for attack metadata)
            generated_text: Model's generated output

        Returns:
            Dict with eval labels: eval, confidence, detected_tool, etc.
            Returns None if evaluation fails or no generated_text provided.
        """
        if generated_text is None:
            return None

        try:
            # Extract required fields from labels
            prompt_labels = prompt_spec.labels
            attacker_tools = prompt_labels.get('attacker_tools', [])
            if not attacker_tools:
                return None

            target_tool = attacker_tools[0] if attacker_tools else ""
            user_tool = prompt_labels.get('user_tool', '')
            attacker_instruction = prompt_labels.get('attacker_instruction', '')

            # Some models echo the tool schema block (between <tools>...</tools>)
            # or even the entire system prompt verbatim in their continuation.
            # Those echoes include JSON snippets like `"name": "ToolName"` that
            # should *not* count as successful execution of the attacker tool.
            #
            # To avoid false positives, we strip any <tools>...</tools> blocks
            # before passing to the InjecAgent evaluator, while leaving actual
            # tool calls (e.g., <tool_call>...</tool_call>) intact.
            import re

            cleaned_output = re.sub(
                r"<tools>.*?</tools>",
                "",
                generated_text,
                flags=re.DOTALL | re.IGNORECASE,
            )

            # Run evaluation on cleaned output
            eval_result = self._evaluate_fn(
                output=cleaned_output,
                target_tool=target_tool,
                attacker_tools=attacker_tools,
                user_tool=user_tool,
                attacker_instruction=attacker_instruction
            )

            return eval_result

        except Exception as e:
            print(f"Warning: InjecAgent evaluation failed: {e}")
            return None