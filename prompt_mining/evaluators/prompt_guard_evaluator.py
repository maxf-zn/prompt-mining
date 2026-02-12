"""
Prompt Guard evaluator for malicious/benign prompt classification.

Uses Meta's Llama-Prompt-Guard-2-86M model to classify prompts as:
- BENIGN: Normal, safe prompts
- MALICIOUS: Prompt injection/jailbreak attempts

Classification approach:
- Chunks input by 512 tokens (model context limit)
- Computes P(malicious) for each chunk via softmax
- Max score pooling: final score = max P(malicious) across all chunks
- Applies threshold (default 0.5) to determine label

Reference: https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M
Reference: https://github.com/meta-llama/llama-cookbook/blob/main/getting-started/responsible_ai/prompt_guard/inference.py
"""

import torch
from typing import Dict, Any, Optional

from prompt_mining.core.evaluator import Evaluator
from prompt_mining.core.prompt_spec import PromptSpec


class PromptGuardEvaluator(Evaluator):
    """
    Evaluator using Prompt Guard 2 for malicious prompt classification.

    Classifies prompts (not outputs) as benign or malicious.
    Loads the model once and reuses for all evaluations.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-Prompt-Guard-2-86M",
        device: Optional[str] = None,
        chunk_size: int = 512,
        threshold: float = 0.5,
    ):
        """
        Initialize Prompt Guard evaluator.

        Args:
            model_name: HuggingFace model ID for Prompt Guard
            device: Device to run on (auto-detects if None)
            chunk_size: Max tokens per chunk (default: 512)
            threshold: Classification threshold for max-pooled score (default: 0.5)
        """
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.chunk_size = chunk_size
        self.threshold = threshold

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Cache label mapping and find benign class id
        # PG2 uses LABEL_0/LABEL_1, convention: 0=benign, 1=malicious
        self.id2label = self.model.config.id2label
        self.benign_id = self._find_benign_id()

    def _find_benign_id(self) -> int:
        """Find the benign class id from model config, with fallbacks."""
        for label_id, label_name in self.id2label.items():
            if label_name.upper() in ("BENIGN", "SAFE", "NORMAL"):
                return label_id
        # Fallback: assume 0 is benign (standard binary classifier convention)
        return 0

    def _prepare_text(self, prompt_spec: PromptSpec) -> str:
        """
        Prepare text for classification from PromptSpec.

        Prompt Guard expects plain text (no chat template). We concatenate
        messages with their roles to preserve structure.
        """
        if prompt_spec.messages:
            parts = []
            for msg in prompt_spec.messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if content:
                    if role:
                        parts.append(f"{role}: {content}")
                    else:
                        parts.append(content)
            return "\n".join(parts)
        elif prompt_spec.text:
            return prompt_spec.text
        else:
            return ""

    def evaluate(
        self,
        prompt_spec: PromptSpec,
        generated_text: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Classify prompt as benign or malicious.

        For prompts > chunk_size tokens, chunks by chunk_size (no overlap) and
        uses max score pooling: final score = max P(malicious) across all chunks,
        then applies threshold for binary classification.

        Args:
            prompt_spec: The prompt specification (uses messages for classification)
            generated_text: Ignored - this evaluator only looks at the prompt

        Returns:
            Dict with:
                - pg_label: Predicted label ("BENIGN" or "MALICIOUS")
                - pg_malicious_score: max P(malicious) across chunks
                - pg_num_chunks: total number of chunks
                - pg_threshold: threshold used for classification
        """
        try:
            text = self._prepare_text(prompt_spec)
            if not text:
                return None

            input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"][0]

            # Chunk by chunk_size tokens (no overlap)
            chunks = [input_ids[i:i + self.chunk_size] for i in range(0, len(input_ids), self.chunk_size)]

            # Pad chunks to same length and batch
            max_len = max(len(c) for c in chunks)
            padded = torch.zeros(len(chunks), max_len, dtype=torch.long)
            attention_mask = torch.zeros(len(chunks), max_len, dtype=torch.long)
            for i, chunk in enumerate(chunks):
                padded[i, :len(chunk)] = chunk
                attention_mask[i, :len(chunk)] = 1

            inputs = {
                "input_ids": padded.to(self.device),
                "attention_mask": attention_mask.to(self.device),
            }

            with torch.no_grad():
                logits = self.model(**inputs).logits

            # Softmax to get probabilities per chunk
            probs = torch.softmax(logits, dim=-1)

            # Get P(malicious) for each chunk (1 - P(benign))
            malicious_probs = 1.0 - probs[:, self.benign_id]

            # Max score pooling: final score = max across all chunks
            max_malicious_score = malicious_probs.max().item()

            # Apply threshold for binary classification
            is_malicious = max_malicious_score >= self.threshold

            return {
                "pg_label": "MALICIOUS" if is_malicious else "BENIGN",
                "pg_malicious_score": max_malicious_score,
                "pg_num_chunks": len(chunks),
                "pg_threshold": self.threshold,
            }

        except Exception as e:
            print(f"Warning: Prompt Guard evaluation failed: {e}")
            return None