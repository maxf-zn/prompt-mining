"""
Llama Guard evaluator for safety classification.

Uses Meta's Llama-Guard-3-8B model to classify content as safe/unsafe.

This evaluator is intentionally similar in spirit to PromptGuardEvaluator:
- Loads the model once and reuses for all evaluations
- Runs a single-pass classification (no chunking)

Reference: https://huggingface.co/meta-llama/Llama-Guard-3-8B
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import torch

from prompt_mining.core.evaluator import Evaluator
from prompt_mining.core.prompt_spec import PromptSpec


class LlamaGuardEvaluator(Evaluator):
    """
    Evaluator using Llama Guard 3 for safety classification.

    Supports:
    - Prompt-only classification (generated_text=None)
    - Prompt+response classification (generated_text provided)
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-Guard-3-8B",
        device: Optional[str] = None,
        max_new_tokens: int = 64,
        hf_cache_dir: Optional[str] = None,
    ):
        """
        Args:
            model_name: HuggingFace model ID for Llama Guard
            device: Device to run on (auto-detects if None)
            max_new_tokens: Max tokens to generate for the classification answer
            hf_cache_dir: Optional Hugging Face cache directory (passed to from_pretrained).
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.hf_cache_dir = hf_cache_dir

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=hf_cache_dir)
        # Ensure pad token exists for generation
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = None
        if self.device.startswith("cuda"):
            # Prefer bf16 on modern GPUs; fall back to fp16 if needed.
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, cache_dir=hf_cache_dir)
        self.model.to(self.device)
        self.model.eval()

        # Authoritative context length (no guessing / fallbacks).
        tok_len = getattr(self.tokenizer, "model_max_length", None)
        cfg_len = getattr(self.model.config, "max_position_embeddings", None)
        if not (isinstance(tok_len, int) and tok_len > 0):
            raise ValueError(f"Tokenizer model_max_length missing/invalid: {tok_len!r}")
        if not (isinstance(cfg_len, int) and cfg_len > 0):
            raise ValueError(f"Model config max_position_embeddings missing/invalid: {cfg_len!r}")
        if tok_len != cfg_len:
            raise ValueError(
                f"Context length mismatch: tokenizer.model_max_length={tok_len} "
                f"!= config.max_position_embeddings={cfg_len}"
            )
        self.context_length = tok_len

    def _build_chat(self, prompt_spec: PromptSpec, generated_text: Optional[str]) -> List[Dict[str, str]]:
        """
        Build the chat in the format expected by Llama Guard's chat template.

        Llama Guard requires alternating user/assistant roles and does NOT support
        the 'system' role. To handle system messages, we prepend them to the first
        user message.

        - If PromptSpec has messages, extract system content and merge into first user msg.
        - Otherwise, we wrap PromptSpec.text as a single user message.
        - If generated_text is provided, append it as an assistant message to classify
          prompt+response.
        """
        if prompt_spec.messages:
            # Separate system messages from user/assistant messages
            system_parts = []
            chat = []
            for m in prompt_spec.messages:
                role = (m.get("role") or "").strip().lower()
                content = (m.get("content") or "").strip()
                if not content:
                    continue
                if role == "system":
                    system_parts.append(content)
                elif role in ("user", "assistant"):
                    chat.append({"role": role, "content": content})

            # Prepend system content to the first user message
            if system_parts and chat:
                system_prefix = "\n\n".join(system_parts)
                for i, msg in enumerate(chat):
                    if msg["role"] == "user":
                        chat[i]["content"] = f"{system_prefix}\n\n{msg['content']}"
                        break
        else:
            text = (prompt_spec.text or "").strip()
            chat = [{"role": "user", "content": text}] if text else []

        if generated_text is not None and generated_text.strip():
            chat.append({"role": "assistant", "content": generated_text.strip()})

        return chat

    def _generate_classification(self, chat: List[Dict[str, str]]) -> str:
        """
        Run a short deterministic generation to obtain the classifier output.

        Matches HF model card example:
        - tokenizer.apply_chat_template(chat, return_tensors="pt")
        - model.generate(..., max_new_tokens=..., pad_token_id=0)
        - decode(output[prompt_len:])
        """
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise RuntimeError("Tokenizer does not support apply_chat_template; cannot run Llama Guard.")

        input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.device)
        prompt_len = int(input_ids.shape[-1])

        # Ensure prompt+generation stays within context length.
        if prompt_len + int(self.max_new_tokens) > self.context_length:
            raise ValueError(
                f"Input too long for {self.model_name}: prompt_len={prompt_len} + "
                f"max_new_tokens={self.max_new_tokens} > context_length={self.context_length}."
            )

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0

        # We don't pad prompts here, so attention_mask is all-ones.
        # Passing it avoids warnings when pad_token_id == eos_token_id.
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )
        return self.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()

    @staticmethod
    def _parse_output(text: str) -> Tuple[Optional[str], List[str]]:
        """
        Parse model output into ("SAFE"/"UNSAFE", categories).

        We accept a few common variants:
        - "safe"
        - "unsafe"
        - "unsafe\nS1,S2" or "unsafe\nS1" etc.
        """
        if not text:
            return None, []

        first_line = text.strip().splitlines()[0].strip().lower()
        if "unsafe" in first_line:
            label = "UNSAFE"
        elif "safe" in first_line:
            label = "SAFE"
        else:
            label = None

        # Extract category-like tags (e.g., S1, S13) anywhere in output
        cats = sorted(set(re.findall(r"\bS(?:1[0-3]|[1-9])\b", text.upper())))
        return label, cats

    def evaluate(
        self,
        prompt_spec: PromptSpec,
        generated_text: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Classify content as safe/unsafe using Llama Guard.

        Returns:
            Dict with:
                - lg_label: "SAFE" or "UNSAFE"
        """
        try:
            chat = self._build_chat(prompt_spec, generated_text)
            if not chat:
                return None

            raw = self._generate_classification(chat)
            label, categories = self._parse_output(raw)
            if label is None:
                return None

            return {
                "lg_label": label,
                "lg_categories": categories,
            }

        except Exception as e:
            print(f"Warning: Llama Guard evaluation failed: {e}")
            return None


