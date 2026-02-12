"""
Backend adapters for different SAE implementations.

This module provides:
- SAEManager: Shared SAE loading and encoding (used by multiple backends)
- ModelBackendAdapter: Abstract base class for model backends
- CircuitTracerAdapter: For ReplacementModel with transcoders
- SAELensAdapter: For HookedTransformer with SAEs
- HuggingFaceAdapter: For HuggingFace transformers with SAEs (multi-GPU support)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import torch


@dataclass
class SAELayerInfo:
    """Information about SAE configuration for a specific layer."""
    layer_idx: int
    hook_point: str  # "hook_resid_pre", "hook_resid_post", "hook_mlp_out", etc.
    d_sae: int


class SAEManager:
    """
    Manages SAE loading and encoding for multiple layers.

    Shared by SAELensAdapter and HuggingFaceAdapter to avoid code duplication.
    Handles loading SAEs from SAELens and encoding activations through them.
    """

    def __init__(self, sae_configs: List[Dict[str, Any]], device: str):
        """
        Initialize SAEManager and load SAEs.

        Args:
            sae_configs: List of SAE configurations:
                [{"sae_release": "...", "sae_id": "layer_19"}, ...]
            device: Device to load SAEs on (e.g., "cuda:0")
        """
        self.saes: Dict[int, Any] = {}  # layer -> SAE instance
        self.hook_points: Dict[int, str] = {}  # layer -> hook point name
        self.device = device

        if not sae_configs:
            return

        try:
            from sae_lens import SAE
        except ImportError:
            raise ImportError(
                "sae_lens is not installed. "
                "Install it with: pip install sae-lens"
            )

        for cfg in sae_configs:
            sae_release = cfg["sae_release"]
            sae_id = cfg["sae_id"]

            print(f"  Loading SAE: {sae_release}/{sae_id}...")
            sae = SAE.from_pretrained(
                release=sae_release,
                sae_id=sae_id,
                device=device
            )[0]

            # Extract hook point from SAE metadata
            hook_name = sae.cfg.metadata['hook_name']

            # Parse hook name (e.g., "blocks.19.hook_resid_post" -> layer 19, "hook_resid_post")
            parts = hook_name.split('.')
            if len(parts) >= 2 and parts[0] == 'blocks':
                layer_idx = int(parts[1])
                hook_point = '.'.join(parts[2:])
            else:
                raise ValueError(f"Unexpected hook name format: {hook_name}")

            print(f"    ✓ Layer {layer_idx} at {hook_point}")

            self.saes[layer_idx] = sae
            self.hook_points[layer_idx] = hook_point

        print(f"  ✓ Loaded {len(self.saes)} SAEs")

    def encode_layer(
        self,
        activations: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """
        Encode activations through SAE for a specific layer.

        Args:
            activations: (batch, seq_len, d_model) or (seq_len, d_model)
            layer_idx: Layer index

        Returns:
            (batch, seq_len, d_sae) SAE activations
        """
        sae = self.saes.get(layer_idx)
        if sae is None:
            raise ValueError(
                f"No SAE configured for layer {layer_idx}. "
                f"Available layers: {sorted(self.saes.keys())}"
            )

        original_shape = activations.shape
        needs_reshape = activations.ndim == 3

        if needs_reshape:
            flat_acts = activations.reshape(-1, original_shape[-1])
        else:
            flat_acts = activations

        with torch.no_grad():
            encoded = sae.encode(flat_acts)

        if needs_reshape:
            return encoded.reshape(original_shape[0], original_shape[1], -1)
        return encoded

    def get_sae_layer_info(self) -> List[SAELayerInfo]:
        """Get SAE configuration for each loaded layer."""
        return [
            SAELayerInfo(
                layer_idx=layer_idx,
                hook_point=self.hook_points[layer_idx],
                d_sae=sae.cfg.d_sae
            )
            for layer_idx, sae in sorted(self.saes.items())
        ]

    @property
    def layers(self) -> List[int]:
        """Get list of layers with loaded SAEs."""
        return sorted(self.saes.keys())


class ModelBackendAdapter(ABC):
    """
    Abstraction for different model backends with SAEs.

    Handles the operations that differ between:
    - CircuitTracer ReplacementModel (transcoders)
    - SAELens HookedTransformer (attached SAEs)
    - HuggingFace transformers (multi-GPU support)

    This adapter provides a unified interface for:
    - Tokenization and text processing
    - Forward passes with activation caching
    - Text generation
    - SAE encoding
    """

    @abstractmethod
    def encode_layer(
        self,
        activations: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """
        Encode activations through SAE for a specific layer.

        Args:
            activations: (batch, seq_len, d_model) or (seq_len, d_model)
            layer_idx: Layer index

        Returns:
            (batch, seq_len, d_sae) SAE activations
        """
        pass

    @abstractmethod
    def to_tokens(self, text: str) -> torch.Tensor:
        """
        Tokenize text to tensor.

        Args:
            text: Input text string

        Returns:
            Token IDs tensor (batch=1, seq_len)
        """
        pass

    @abstractmethod
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text continuation from tokens.

        Args:
            tokens: Input token IDs (batch, seq_len)
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional generation arguments

        Returns:
            Generated token IDs including input (batch, seq_len + generated)
        """
        pass

    @abstractmethod
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model configuration/dimensions.

        Returns:
            Dict with keys: n_layers, d_model, d_vocab
        """
        pass

    def run_with_cache(
        self,
        tokens: torch.Tensor,
        layers: List[int],
        hook_points: Union[str, List[str]]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Run forward pass and cache activations at specified layers and hook points.

        Single-pass optimization: Gets both logits and activations in one forward.
        Supports caching multiple hook points per layer in a single forward pass.

        Args:
            tokens: Input token IDs (batch, seq_len)
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
        # Normalize hook_points to list
        if isinstance(hook_points, str):
            hook_points = [hook_points]

        # Build hook names for all layer + hook_point combinations
        names_filter = [
            f"blocks.{layer_idx}.{hook_point}"
            for layer_idx in layers
            for hook_point in hook_points
        ]

        # Call underlying model's run_with_cache
        model = self.get_raw_model()
        logits, cache = model.run_with_cache(
            tokens,
            names_filter=names_filter
        )

        # Return full cache (no reorganization needed)
        return logits, cache

    @abstractmethod
    def get_sae_layer_info(self) -> List[SAELayerInfo]:
        """Get list of configured SAE layers with metadata."""
        pass

    @abstractmethod
    def supports_attribution(self) -> bool:
        """Whether this backend supports circuit_tracer attribution."""
        pass

    @abstractmethod
    def get_raw_model(self) -> Any:
        """Get underlying model object (for attribution)."""
        pass


class TransformerLensAdapterBase(ModelBackendAdapter):
    """
    Base adapter for TransformerLens-based backends.

    Provides shared implementations for to_tokens, generate, and get_model_config
    that work with any model using the TransformerLens API (HookedTransformer).

    Subclasses: CircuitTracerAdapter, SAELensAdapter
    """

    model: Any  # Subclasses must set this

    def to_tokens(self, text: str) -> torch.Tensor:
        """Tokenize text using TransformerLens API."""
        return self.model.to_tokens(text, truncate=False)

    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        **kwargs
    ) -> torch.Tensor:
        """Generate using TransformerLens API."""
        return self.model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            verbose=False,
            return_type="tokens",
            use_past_kv_cache=True,
            **kwargs
        )

    def get_model_config(self) -> Dict[str, Any]:
        """Get model dimensions from TransformerLens config."""
        return {
            "n_layers": self.model.cfg.n_layers,
            "d_model": self.model.cfg.d_model,
            "d_vocab": self.model.cfg.d_vocab,
        }

    def get_raw_model(self) -> Any:
        """Get underlying model."""
        return self.model


class CircuitTracerAdapter(TransformerLensAdapterBase):
    """
    Adapter for circuit_tracer ReplacementModel with transcoders.

    Note: Transcoders are Sparse Autoencoders (SAEs), so this adapter implements
    the generic SAE interface for circuit_tracer's transcoder system.

    Transcoders are trained on mlp.hook_in and this is fixed in their configuration.
    """

    def __init__(self, model: "ReplacementModel"):
        """
        Initialize CircuitTracer adapter.

        Args:
            model: ReplacementModel instance
        """
        self.model = model

    def encode_layer(
        self,
        activations: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """Encode through transcoders."""
        return self.model.transcoders.encode_layer(
            activations,
            layer_idx,
            apply_activation_function=True
        )

    def get_sae_layer_info(self) -> List[SAELayerInfo]:
        """Get transcoder configuration for all layers."""
        n_layers = self.model.cfg.n_layers
        d_transcoder = self.model.transcoders.d_transcoder

        # Get hook point from transcoder configuration
        # This is where the transcoders were trained (e.g., "mlp.hook_in")
        hook_point = self.model.transcoders.feature_input_hook

        return [
            SAELayerInfo(
                layer_idx=i,
                hook_point=hook_point,
                d_sae=d_transcoder
            )
            for i in range(n_layers)
        ]

    def supports_attribution(self) -> bool:
        """Circuit tracer supports attribution."""
        return True


class SAELensAdapter(TransformerLensAdapterBase):
    """Adapter for SAELens HookedTransformer with multiple SAEs."""

    def __init__(
        self,
        model: "HookedTransformer",
        sae_configs: List[Dict[str, Any]]
    ):
        """
        Initialize SAELens adapter and load SAEs.

        Args:
            model: HookedTransformer instance
            sae_configs: List of SAE configurations:
                [{"sae_release": "...", "sae_id": "layer_19"}, ...]
        """
        self.model = model
        self.sae_manager = SAEManager(sae_configs, str(model.cfg.device))

    def encode_layer(
        self,
        activations: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """Encode using SAE for specific layer."""
        return self.sae_manager.encode_layer(activations, layer_idx)

    def get_sae_layer_info(self) -> List[SAELayerInfo]:
        """Get SAE configuration for each loaded layer."""
        return self.sae_manager.get_sae_layer_info()

    def supports_attribution(self) -> bool:
        """SAELens doesn't support circuit_tracer attribution."""
        return False


class HuggingFaceAdapter(ModelBackendAdapter):
    """
    Adapter for HuggingFace transformers with SAELens SAEs.

    Key advantages over SAELensAdapter:
    - Supports device_map="auto" for multi-GPU inference
    - No TransformerLens dependency for the base model
    - Uses output_hidden_states=True for activation capture

    Limitations:
    - Only provides hook_resid_pre/hook_resid_post (residual stream)
    - Cannot capture intermediate hooks (mlp.hook_in, attn hooks, etc.)
    """

    def __init__(
        self,
        model: "AutoModelForCausalLM",
        tokenizer: "AutoTokenizer",
        sae_configs: List[Dict[str, Any]],
        n_layers: int,
        d_model: int,
        d_vocab: int,
        sae_device: Optional[str] = None
    ):
        """
        Initialize HuggingFace adapter.

        Args:
            model: HuggingFace AutoModelForCausalLM instance
            tokenizer: HuggingFace AutoTokenizer instance
            sae_configs: List of SAE configurations
            n_layers: Number of transformer layers
            d_model: Model hidden dimension
            d_vocab: Vocabulary size
            sae_device: Device for SAEs (defaults to first model device)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_vocab = d_vocab

        # Determine SAE device (use first device from model's device_map if available)
        if sae_device is None:
            if hasattr(model, 'hf_device_map') and model.hf_device_map:
                # Get device of first layer
                first_device = next(iter(model.hf_device_map.values()))
                sae_device = f"cuda:{first_device}" if isinstance(first_device, int) else str(first_device)
            else:
                sae_device = str(model.device)

        self.sae_manager = SAEManager(sae_configs, sae_device)

    def encode_layer(
        self,
        activations: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """Encode using SAE for specific layer."""
        return self.sae_manager.encode_layer(activations, layer_idx)

    def to_tokens(self, text: str) -> torch.Tensor:
        """Tokenize text using HuggingFace tokenizer."""
        tokens = self.tokenizer(text, return_tensors="pt")["input_ids"]
        # Move to model's device
        if hasattr(self.model, 'device'):
            tokens = tokens.to(self.model.device)
        return tokens

    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        **kwargs
    ) -> torch.Tensor:
        """Generate using HuggingFace API."""
        return self.model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            **kwargs
        )

    def get_model_config(self) -> Dict[str, Any]:
        """Get model dimensions from HuggingFace config."""
        return {
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "d_vocab": self.d_vocab,
        }

    def run_with_cache(
        self,
        tokens: torch.Tensor,
        layers: List[int],
        hook_points: Union[str, List[str]]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Run forward pass and cache activations using output_hidden_states.

        Note: HuggingFace only provides residual stream states between layers.
        - hook_resid_pre for layer i = hidden_states[i] (input to layer i)
        - hook_resid_post for layer i = hidden_states[i+1] (output of layer i)

        Args:
            tokens: Input token IDs (batch, seq_len)
            layers: List of layer indices to cache
            hook_points: Hook point name(s) - only "hook_resid_pre" and "hook_resid_post" supported

        Returns:
            Tuple of (logits, cache_dict)
        """
        if isinstance(hook_points, str):
            hook_points = [hook_points]

        # Validate hook points
        unsupported = set(hook_points) - {"hook_resid_pre", "hook_resid_post"}
        if unsupported:
            raise ValueError(
                f"HuggingFace backend only supports hook_resid_pre/hook_resid_post. "
                f"Unsupported: {unsupported}"
            )

        # Run forward pass with hidden states (disable KV cache to prevent memory accumulation)
        outputs = self.model(tokens, output_hidden_states=True, use_cache=False)
        logits = outputs.logits  # (batch, seq_len, vocab_size)
        hidden_states = outputs.hidden_states  # tuple of (batch, seq_len, d_model)

        # Build cache mapping TransformerLens-style keys to tensors
        # hidden_states[0] = embedding output
        # hidden_states[i] for i>0 = output after layer i-1 = input to layer i
        cache = {}
        for layer_idx in layers:
            for hook_point in hook_points:
                if hook_point == "hook_resid_pre":
                    # Input to layer = hidden_states[layer_idx]
                    cache[f"blocks.{layer_idx}.{hook_point}"] = hidden_states[layer_idx]
                elif hook_point == "hook_resid_post":
                    # Output of layer = hidden_states[layer_idx + 1]
                    cache[f"blocks.{layer_idx}.{hook_point}"] = hidden_states[layer_idx + 1]

        return logits, cache

    def get_sae_layer_info(self) -> List[SAELayerInfo]:
        """Get SAE configuration for each loaded layer."""
        return self.sae_manager.get_sae_layer_info()

    def supports_attribution(self) -> bool:
        """HuggingFace backend doesn't support circuit_tracer attribution."""
        return False

    def get_raw_model(self) -> Any:
        """Get underlying HuggingFace model."""
        return self.model
