"""
Feature extractors for the inference pipeline.

Provides pluggable feature extraction from prompts:
- RawActivationExtractor: Extract raw model activations
- SAEFeatureExtractor: Extract SAE-encoded features (extends RawActivationExtractor)

Both implement FeatureExtractorProtocol for use with InferencePipeline.
"""
import numpy as np
import torch
from typing import Protocol, Literal, Optional, Union, List, runtime_checkable

from prompt_mining.model import ModelWrapper


# Position can be a reduction strategy or specific index/indices
PositionType = Union[Literal['last', 'mean', 'max'], int, List[int]]


@runtime_checkable
class FeatureExtractorProtocol(Protocol):
    """Protocol for feature extraction from prompts."""

    def extract(self, prompt: str, model_wrapper: ModelWrapper) -> np.ndarray:
        """
        Extract features from a prompt.

        Args:
            prompt: Text prompt to extract features from
            model_wrapper: Loaded ModelWrapper instance

        Returns:
            1D numpy array of features
        """
        ...


class RawActivationExtractor:
    """
    Extract raw model activations from a prompt.

    Gets activations from a specific layer and aggregates across positions.

    Example:
        >>> extractor = RawActivationExtractor(layer=19, position='last')
        >>> features = extractor.extract("Hello world", model_wrapper)
        >>> print(features.shape)  # (d_model,)

        >>> # Specific position index
        >>> extractor = RawActivationExtractor(layer=19, position=0)  # First token
        >>> features = extractor.extract("Hello world", model_wrapper)

        >>> # Multiple positions (concatenated)
        >>> extractor = RawActivationExtractor(layer=19, position=[0, -1])
        >>> features = extractor.extract("Hello world", model_wrapper)
        >>> print(features.shape)  # (2 * d_model,)
    """

    def __init__(
        self,
        layer: int,
        position: PositionType = 'last',
        hook_point: str = "hook_resid_post",
    ):
        """
        Initialize the extractor.

        Args:
            layer: Layer index to extract activations from
            position: How to select/aggregate across sequence positions:
                - 'last': Use last token position
                - 'mean': Average across all positions
                - 'max': Max across all positions
                - int: Specific position index (supports negative indexing)
                - List[int]: Multiple positions (concatenated)
            hook_point: Hook point name (e.g., "hook_resid_post", "hook_resid_pre")
        """
        self.layer = layer
        self.position = position
        self.hook_point = hook_point

    def _get_activations(
        self,
        prompt: str,
        model_wrapper: ModelWrapper
    ) -> torch.Tensor:
        """
        Get raw activations from model.

        Args:
            prompt: Text prompt
            model_wrapper: Loaded ModelWrapper

        Returns:
            Activations tensor (batch, seq_len, d_model)
        """
        _, cache = model_wrapper.run_with_cache(
            text=prompt,
            layers=[self.layer],
            hook_points=self.hook_point,
        )

        # Get activations from cache
        cache_key = f"blocks.{self.layer}.{self.hook_point}"
        return cache[cache_key]

    def _aggregate_positions(self, activations: torch.Tensor) -> np.ndarray:
        """
        Aggregate activations across sequence positions.

        Args:
            activations: (batch, seq_len, d_model) tensor

        Returns:
            1D numpy array. Shape depends on position type:
            - Reduction ('last', 'mean', 'max') or single int: (d_model,)
            - List[int] with N positions: (N * d_model,) concatenated
        """
        # Remove batch dimension (assuming batch=1)
        acts = activations.squeeze(0)  # (seq_len, d_model)

        if self.position == 'last':
            result = acts[-1]
        elif self.position == 'mean':
            result = acts.mean(dim=0)
        elif self.position == 'max':
            result = acts.max(dim=0).values
        elif isinstance(self.position, int):
            result = acts[self.position]
        elif isinstance(self.position, list):
            # Multiple positions: concatenate
            result = torch.cat([acts[p] for p in self.position], dim=0)
        else:
            raise ValueError(f"Unknown position type: {self.position}")

        return result.float().cpu().numpy()

    def extract(self, prompt: str, model_wrapper: ModelWrapper) -> np.ndarray:
        """
        Extract raw activations from a prompt.

        Args:
            prompt: Text prompt
            model_wrapper: Loaded ModelWrapper

        Returns:
            1D numpy array (d_model,)
        """
        activations = self._get_activations(prompt, model_wrapper)
        return self._aggregate_positions(activations)

    def __repr__(self) -> str:
        return f"RawActivationExtractor(layer={self.layer}, position={self.position})"


class SAEFeatureExtractor(RawActivationExtractor):
    """
    Extract SAE-encoded features from a prompt.

    Extends RawActivationExtractor by encoding activations through an SAE.

    Example:
        >>> extractor = SAEFeatureExtractor(layer=27, position='last')
        >>> features = extractor.extract("Hello world", model_wrapper)
        >>> print(features.shape)  # (d_sae,) - sparse SAE features

        >>> # With external SAE (not from ModelWrapper)
        >>> extractor = SAEFeatureExtractor(layer=27, sae=my_sae)
        >>> features = extractor.extract("Hello world", model_wrapper)
    """

    def __init__(
        self,
        layer: int,
        position: PositionType = 'last',
        hook_point: str = "hook_resid_post",
        sae=None,
    ):
        """
        Initialize the SAE feature extractor.

        Args:
            layer: Layer index to extract activations from
            position: How to select/aggregate across sequence positions
                (same options as RawActivationExtractor)
            hook_point: Hook point name
            sae: Optional external SAE object. If None, uses ModelWrapper's SAE.
                 Must have an encode() method that takes (batch, seq_len, d_model)
                 and returns (batch, seq_len, d_sae).
        """
        super().__init__(layer=layer, position=position, hook_point=hook_point)
        self.sae = sae

    def extract(self, prompt: str, model_wrapper: ModelWrapper) -> np.ndarray:
        """
        Extract SAE features from a prompt.

        Args:
            prompt: Text prompt
            model_wrapper: Loaded ModelWrapper

        Returns:
            1D numpy array (d_sae,) - SAE feature activations
        """
        # Get raw activations
        activations = self._get_activations(prompt, model_wrapper)

        # Encode through SAE
        if self.sae is not None:
            # Use external SAE
            with torch.no_grad():
                sae_acts = self.sae.encode(activations)
        else:
            # Use ModelWrapper's SAE
            sae_acts = model_wrapper.encode_layer_activations(activations, self.layer)

        # Aggregate positions
        return self._aggregate_positions(sae_acts)

    def __repr__(self) -> str:
        sae_str = "external" if self.sae else "from_wrapper"
        return f"SAEFeatureExtractor(layer={self.layer}, position={self.position}, sae={sae_str})"
