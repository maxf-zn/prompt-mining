"""
ModelWrapper for multiple model backends with SAEs.

Provides a clean interface for model loading, inference, and attribution
that works with both circuit_tracer ReplacementModel and SAELens SAEs.
"""

import torch
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field

from prompt_mining.model.backend_adapter import (
    ModelBackendAdapter,
    CircuitTracerAdapter,
    SAELensAdapter,
    HuggingFaceAdapter
)


@dataclass
class ModelConfig:
    """Configuration for model loading."""
    model_name: str
    backend: str  # "circuit_tracer", "saelens", or "huggingface"

    # For circuit_tracer
    transcoder_set: Optional[str] = None

    # For SAELens and HuggingFace (multi-layer SAEs)
    sae_configs: Optional[List[Dict[str, Any]]] = None

    # For HuggingFace backend: device_map for multi-GPU
    # Examples: "auto", "balanced", {"": 0}, {"model.embed_tokens": 0, ...}
    device_map: Optional[Union[str, Dict[str, Any]]] = None

    device: Optional[torch.device] = None
    dtype: str = "float32"
    enable_attribution: bool = False


class ModelWrapper:
    """
    Unified wrapper for models with SAEs.

    Provides unified interface for:
    - Model loading with SAE integration
    - Forward passes (with/without caching)
    - Text generation
    - Activation capture

    Supports three backends:
    - circuit_tracer: ReplacementModel with transcoders (supports attribution)
    - saelens: HookedTransformer with SAEs (TransformerLens-based)
    - huggingface: HuggingFace transformers with SAEs (multi-GPU via device_map)
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize ModelWrapper.

        Args:
            config: Model configuration

        Raises:
            ValueError: If backend is unknown or config is invalid
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.adapter: Optional[ModelBackendAdapter] = None

        # Set device
        if config.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = config.device

    def load(self) -> None:
        """
        Load model and SAEs based on backend configuration.

        This should be called once per worker/process.
        """
        # Block attribution for non-circuit_tracer backends
        if self.config.backend in ("saelens", "huggingface") and self.config.enable_attribution:
            print(f"⚠ Warning: Attribution not supported with {self.config.backend}, disabling")
            self.config.enable_attribution = False

        if self.config.backend == "circuit_tracer":
            self._load_circuit_tracer()
        elif self.config.backend == "saelens":
            self._load_saelens()
        elif self.config.backend == "huggingface":
            self._load_huggingface()
        else:
            raise ValueError(
                f"Unknown backend: {self.config.backend}. "
                "Must be 'circuit_tracer', 'saelens', or 'huggingface'"
            )

    def _load_circuit_tracer(self) -> None:
        """Load circuit_tracer ReplacementModel."""
        try:
            from circuit_tracer.replacement_model import ReplacementModel
        except ImportError:
            raise ImportError(
                "circuit_tracer is not installed. "
                "Install it with: pip install -e /path/to/circuit-tracer"
            )

        if not self.config.transcoder_set:
            raise ValueError("transcoder_set required for circuit_tracer backend")

        print(f"Loading ReplacementModel: {self.config.model_name}")
        self.model = ReplacementModel.from_pretrained(
            model_name=self.config.model_name,
            transcoder_set=self.config.transcoder_set,
            device=self.device,
            dtype=self.config.dtype,
            trust_remote_code=False,
        )
        self.tokenizer = self.model.tokenizer
        self.adapter = CircuitTracerAdapter(self.model)
        print(f"✓ ReplacementModel loaded with transcoders")

    def _load_saelens(self) -> None:
        """Load SAELens HookedSAETransformer with multiple SAEs."""
        try:
            from transformer_lens import HookedTransformer
        except ImportError:
            raise ImportError(
                "transformer_lens is not installed. "
                "Install it with: pip install transformer-lens"
            )

        if not self.config.sae_configs:
            raise ValueError("sae_configs required for saelens backend")

        print(f"Loading HookedTransformer: {self.config.model_name}")
        self.model = HookedTransformer.from_pretrained(
            self.config.model_name,
            device=self.device,
            dtype=self.config.dtype,
        )
        self.tokenizer = self.model.tokenizer

        # Load and attach SAEs via adapter
        self.adapter = SAELensAdapter(self.model, self.config.sae_configs)
        print(f"✓ HookedTransformer loaded with {len(self.config.sae_configs)} SAEs")

    def _load_huggingface(self) -> None:
        """Load HuggingFace model with multi-GPU support via device_map."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is not installed. "
                "Install it with: pip install transformers"
            )

        # Note: sae_configs is optional - can use huggingface backend for raw activations only

        # Map dtype string to torch dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.config.dtype, torch.float32)

        print(f"Loading HuggingFace model: {self.config.model_name}")

        # Build load kwargs
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": False,
        }

        # Handle device placement
        if self.config.device_map is not None:
            # Multi-GPU or custom device mapping
            load_kwargs["device_map"] = self.config.device_map
            print(f"  Using device_map: {self.config.device_map}")
        else:
            # Single device
            load_kwargs["device_map"] = {"": self.device}

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **load_kwargs
        )

        # Extract model dimensions from HF config
        model_config = self.model.config

        # Create adapter (adapter stores model dimensions)
        self.adapter = HuggingFaceAdapter(
            model=self.model,
            tokenizer=self.tokenizer,
            sae_configs=self.config.sae_configs,
            n_layers=model_config.num_hidden_layers,
            d_model=model_config.hidden_size,
            d_vocab=model_config.vocab_size
        )

        n_saes = len(self.config.sae_configs) if self.config.sae_configs else 0
        print(f"✓ HuggingFace model loaded with {n_saes} SAEs")

    # ===== Common methods (backend-aware) =====

    def tokenize(
        self,
        text: str,
        return_tensors: str = "pt",
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize text using model's tokenizer.

        Args:
            text: Input text
            return_tensors: Format for returned tensors
            **kwargs: Additional tokenizer arguments

        Returns:
            Dictionary with input_ids
        """
        if self.adapter is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        tokens = self.adapter.to_tokens(text)
        return {"input_ids": tokens}

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> str:
        """
        Apply chat template to messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool schemas
            **kwargs: Additional template arguments

        Returns:
            Formatted text string
        """
        if self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        return self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            **kwargs
        )

    def forward(
        self,
        text: str,
        return_logits: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Run forward pass through model.

        NOTE: For better performance, prefer run_with_cache() which gets
        both logits and activations in a single forward pass.

        Args:
            text: Input text
            return_logits: Whether to return logits
            **kwargs: Additional forward arguments

        Returns:
            Dictionary containing:
                - logits: (batch_size, seq_len, vocab_size) if return_logits=True
        """
        if self.adapter is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        with torch.no_grad():
            tokens = self.adapter.to_tokens(text)
            logits = self.model(tokens, **kwargs)

        result = {}
        if return_logits:
            result["logits"] = logits

        return result

    def generate(
        self,
        text: str,
        max_new_tokens: int = 100,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text continuation.

        Args:
            text: Input text
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional generation arguments

        Returns:
            Generated token IDs (including input)
        """
        if self.adapter is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        with torch.no_grad():
            tokens = self.adapter.to_tokens(text)
            output = self.adapter.generate(tokens, max_new_tokens, **kwargs)

        return output

    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
        **kwargs
    ) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            **kwargs: Additional decode arguments

        Returns:
            Decoded text string
        """
        if self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        return self.tokenizer.decode(
            token_ids.squeeze(0),
            skip_special_tokens=skip_special_tokens,
            **kwargs
        )

    def to_original_device(self) -> None:
        """Move model to original device."""
        if self.model is not None:
            self.model.to(self.device)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata.

        Returns:
            Dictionary with model info (name, layers, vocab_size, backend, etc.)
        """
        if self.adapter is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        sae_layers = self.adapter.get_sae_layer_info()
        model_config = self.adapter.get_model_config()

        return {
            "model_name": self.config.model_name,
            "backend": self.config.backend,
            "transcoder_set": self.config.transcoder_set or "N/A",
            "n_layers": model_config["n_layers"],
            "d_model": model_config["d_model"],
            "d_sae": sae_layers[0].d_sae if sae_layers else None,
            "vocab_size": model_config["d_vocab"],
            "sae_layers": [info.layer_idx for info in sae_layers],
            "device": str(self.device),
            "dtype": str(self.config.dtype),
            "enable_attribution": self.config.enable_attribution,
            "supports_attribution": self.adapter.supports_attribution()
        }

    # ===== Backend-abstracted methods (delegated to adapter) =====

    def run_with_cache(
        self,
        text: str,
        layers: List[int],
        hook_points: Union[str, List[str]]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Run forward pass and cache activations (OPTIMIZATION: single pass).

        This is more efficient than calling forward() separately for logits
        and then again for activations. Supports caching multiple hook points
        per layer in a single forward pass.

        Args:
            text: Input text
            layers: List of layer indices to cache
            hook_points: Hook point name(s) - either single string or list of strings
                        Examples: "hook_resid_pre" or ["hook_resid_pre", "mlp.hook_in"]

        Returns:
            Tuple of:
                - logits: (batch, seq_len, vocab_size)
                - cache: Dict mapping full hook name -> activations
                        Keys are like "blocks.5.hook_resid_pre", "blocks.5.mlp.hook_in"
                        Values are (batch, seq_len, d_model) tensors
        """
        if self.adapter is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        tokens = self.adapter.to_tokens(text)

        with torch.no_grad():
            logits, cache = self.adapter.run_with_cache(tokens, layers, hook_points)

        return logits, cache

    def encode_layer_activations(
        self,
        activations: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """
        Encode activations through SAE (delegated to adapter).

        Works for both transcoders (circuit_tracer) and SAEs (saelens).

        Args:
            activations: (batch, seq_len, d_model) or (seq_len, d_model)
            layer_idx: Layer index

        Returns:
            (batch, seq_len, d_sae) SAE activations
        """
        if self.adapter is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        return self.adapter.encode_layer(activations, layer_idx)

    def __repr__(self) -> str:
        status = "loaded" if self.model is not None else "not loaded"
        return (
            f"ModelWrapper(backend={self.config.backend}, "
            f"model={self.config.model_name}, "
            f"device={self.device}, status={status})"
        )
