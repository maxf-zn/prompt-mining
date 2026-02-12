"""
SAE Feature Interpreter - Fetch and generate interpretations for SAE features.

Supports multiple interpretation strategies:
- NeuronpediaStrategy: Fetch existing auto-interpretations from Neuronpedia
- LLMStrategy: Generate new interpretations using an LLM based on top activating examples

Usage:
    # From an SAE object (extracts neuronpedia_id from metadata)
    interpreter = SAEFeatureInterpreter.from_sae(sae, model_id="llama3.1-8b-it")

    # Or manually specify the neuronpedia_id
    interpreter = SAEFeatureInterpreter(
        model_id="llama3.1-8b-it",
        neuronpedia_id="27-resid-post-aa"
    )

    # Get single feature info (uses Neuronpedia auto-interp by default)
    info = interpreter.get_feature_info(feature_idx=31897)

    # Get multiple features
    features = interpreter.get_features([31897, 95512, 112114], verbose=True)
    print(interpreter.format_report(features))

    # Use LLM re-interpretation instead
    interpreter = SAEFeatureInterpreter(
        model_id="llama3.1-8b-it",
        neuronpedia_id="27-resid-post-aa",
        strategy=LLMStrategy(context="malicious prompt classifier")
    )

    # Example: interpret top classifier features (get indices from classifier first)
    top_indices = np.argsort(clf.coef_[0])[-10:][::-1]  # top 10 positive
    features = interpreter.get_features(top_indices.tolist(), verbose=True)
"""

import json
import requests
import time
from typing import Optional, List, Dict, Any, Protocol, runtime_checkable

from prompt_mining.types import FeatureInfo


# --- Interpretation Strategies ---

@runtime_checkable
class InterpretationStrategy(Protocol):
    """Protocol for interpretation strategies."""

    def interpret(
        self,
        feature_idx: int,
        neuronpedia_id: str,
        neuronpedia_data: Optional[Dict] = None,
    ) -> str:
        """
        Generate interpretation for a feature.

        Args:
            feature_idx: Feature index
            neuronpedia_id: Neuronpedia SAE path (e.g., "27-resid-post-aa")
            neuronpedia_data: Raw data from Neuronpedia API (if available)

        Returns:
            Interpretation string
        """
        ...


class NeuronpediaStrategy:
    """Strategy that uses existing Neuronpedia auto-interpretations."""

    def interpret(
        self,
        feature_idx: int,
        neuronpedia_id: str,
        neuronpedia_data: Optional[Dict] = None,
    ) -> str:
        """Extract auto-interpretation from Neuronpedia data."""
        if not neuronpedia_data:
            return ""

        explanations = neuronpedia_data.get("explanations", [])
        if explanations and isinstance(explanations, list) and len(explanations) > 0:
            return explanations[0].get("description", "")

        return neuronpedia_data.get("description", "")


class LLMInterpreter:
    """
    Generates feature interpretations using an LLM.

    Analyzes top activating examples to produce concise, accurate descriptions.
    """

    DEFAULT_PROMPT = """Analyze this SAE feature's top activating examples and describe what it detects.

Feature: {feature_idx} (SAE: {neuronpedia_id})

Top activating examples:
{examples_text}

Output a single sentence (max 20 words) describing the specific pattern this feature detects. Be precise - if it's a jailbreak technique, name it. If it's harmful content, specify the type. No preamble, just the description."""

    def __init__(
        self,
        bedrock_client=None,
        model_id: str = "us.anthropic.claude-sonnet-4-20250514-v1:0",
        region: str = "us-west-2",
        max_tokens: int = 100,
        prompt_template: Optional[str] = None,
    ):
        """
        Initialize the LLM interpreter.

        Args:
            bedrock_client: Boto3 bedrock-runtime client (created if not provided)
            model_id: Bedrock model ID
            region: AWS region for Bedrock
            max_tokens: Maximum tokens in response
            prompt_template: Custom prompt template (uses DEFAULT_PROMPT if not provided)
        """
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT

        if bedrock_client is None:
            import boto3
            self.client = boto3.client("bedrock-runtime", region_name=region)
        else:
            self.client = bedrock_client

    def interpret(
        self,
        feature_idx: int,
        neuronpedia_id: str,
        examples: List[Dict],
        context: Optional[str] = None,
    ) -> str:
        """
        Generate interpretation for a feature based on its top activating examples.

        Args:
            feature_idx: Feature index
            neuronpedia_id: Neuronpedia SAE identifier (e.g., "27-resid-post-aa")
            examples: List of activation examples from Neuronpedia
            context: Optional context about the use case

        Returns:
            LLM-generated interpretation
        """
        if not examples:
            return "No examples available for interpretation"

        # Format examples
        examples_text = self._format_examples(examples[:8])

        # Build prompt
        prompt = self.prompt_template.format(
            feature_idx=feature_idx,
            neuronpedia_id=neuronpedia_id,
            examples_text=examples_text,
            context=context or "",
        )

        # Call LLM
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            })
        )

        result = json.loads(response["body"].read())
        return result["content"][0]["text"].strip()

    def _format_examples(self, examples: List[Dict], max_chars: int = 1500) -> str:
        """Format activation examples for the prompt."""
        lines = []
        for i, ex in enumerate(examples, 1):
            # Reconstruct text from tokens
            tokens = ex.get("tokens", [])
            text = "".join(tokens)[:max_chars]
            if len("".join(tokens)) > max_chars:
                text += "..."

            activation = ex.get("maxValue", 0)
            source = ex.get("dataSource", "unknown")

            lines.append(f"--- Example {i} (activation: {activation:.2f}, source: {source}) ---")
            lines.append(text)
            lines.append("")

        return "\n".join(lines)


class LLMStrategy:
    """Strategy that generates interpretations using an LLM."""

    def __init__(
        self,
        context: str = "SAE feature analysis",
        llm_interpreter: Optional[LLMInterpreter] = None,
        **llm_kwargs,
    ):
        """
        Initialize LLM strategy.

        Args:
            context: Context description for the LLM (e.g., "malicious prompt classifier")
            llm_interpreter: Pre-configured LLMInterpreter (created if not provided)
            **llm_kwargs: Arguments passed to LLMInterpreter if creating new one
        """
        self.context = context
        self.interpreter = llm_interpreter or LLMInterpreter(**llm_kwargs)

    def interpret(
        self,
        feature_idx: int,
        neuronpedia_id: str,
        neuronpedia_data: Optional[Dict] = None,
    ) -> str:
        """Generate interpretation using LLM based on top activating examples."""
        if not neuronpedia_data:
            return "No Neuronpedia data available for LLM interpretation"

        examples = neuronpedia_data.get("activations", [])
        if not examples:
            return "No activation examples available"

        return self.interpreter.interpret(
            feature_idx=feature_idx,
            neuronpedia_id=neuronpedia_id,
            examples=examples,
            context=self.context,
        )


# --- Main Interpreter Class ---

class SAEFeatureInterpreter:
    """
    Fetch and interpret SAE features.

    Supports pluggable interpretation strategies and caches Neuronpedia API responses.
    """

    API_BASE = "https://www.neuronpedia.org/api"

    def __init__(
        self,
        model_id: str,
        neuronpedia_id: str,
        strategy: Optional[InterpretationStrategy] = None,
        cache_responses: bool = True,
        rate_limit_delay: float = 0.5,
    ):
        """
        Initialize the interpreter.

        Args:
            model_id: Neuronpedia model ID (e.g., "llama3.1-8b-it")
            neuronpedia_id: Neuronpedia SAE path (e.g., "27-resid-post-aa")
            strategy: Interpretation strategy (default: NeuronpediaStrategy)
            cache_responses: Whether to cache API responses
            rate_limit_delay: Delay between API calls in seconds
        """
        self.model_id = model_id
        self.neuronpedia_id = neuronpedia_id
        self.strategy = strategy or NeuronpediaStrategy()
        self.cache_responses = cache_responses
        self.rate_limit_delay = rate_limit_delay
        self._cache: Dict[str, Any] = {}

    @classmethod
    def from_sae(
        cls,
        sae,
        model_id: str,
        strategy: Optional[InterpretationStrategy] = None,
        **kwargs,
    ) -> "SAEFeatureInterpreter":
        """
        Create interpreter from an SAE object.

        Extracts neuronpedia_id from sae.cfg.metadata.neuronpedia_id.

        Args:
            sae: SAE object with cfg.metadata.neuronpedia_id
            model_id: Neuronpedia model ID (e.g., "llama3.1-8b-it")
            strategy: Interpretation strategy
            **kwargs: Additional arguments passed to constructor

        Raises:
            ValueError: If SAE doesn't have neuronpedia_id in metadata
        """
        neuronpedia_id = None

        if hasattr(sae.cfg, 'metadata') and sae.cfg.metadata is not None:
            neuronpedia_id = getattr(sae.cfg.metadata, 'neuronpedia_id', None)

        if not neuronpedia_id:
            raise ValueError(
                "SAE does not have a neuronpedia_id in its metadata. "
                "This SAE may not be registered on Neuronpedia. "
                "Manually specify neuronpedia_id if you know it."
            )

        return cls(model_id=model_id, neuronpedia_id=neuronpedia_id, strategy=strategy, **kwargs)

    def _feature_url(self, feature_idx: int) -> str:
        """Get Neuronpedia URL for a feature."""
        return f"https://www.neuronpedia.org/{self.model_id}/{self.neuronpedia_id}/{feature_idx}"

    def _fetch_from_neuronpedia(self, feature_idx: int) -> Optional[Dict]:
        """Fetch feature data from Neuronpedia API."""
        cache_key = f"{self.neuronpedia_id}:{feature_idx}"

        if self.cache_responses and cache_key in self._cache:
            return self._cache[cache_key]

        url = f"{self.API_BASE}/feature/{self.model_id}/{self.neuronpedia_id}/{feature_idx}"

        try:
            response = requests.get(url, timeout=30)
            time.sleep(self.rate_limit_delay)

            if response.status_code == 200:
                data = response.json()
                if self.cache_responses:
                    self._cache[cache_key] = data
                return data
            else:
                print(f"API error {response.status_code}: {response.text[:200]}")
                return None
        except Exception as e:
            print(f"Request failed: {e}")
            return None

    def get_feature_info(self, feature_idx: int) -> FeatureInfo:
        """
        Get information about a single SAE feature.

        Args:
            feature_idx: The feature index within the SAE

        Returns:
            FeatureInfo object with interpretation and metadata
        """
        info = FeatureInfo(
            feature_idx=feature_idx,
            neuronpedia_url=self._feature_url(feature_idx),
        )

        # Fetch from Neuronpedia
        data = self._fetch_from_neuronpedia(feature_idx)

        if data:
            # Store raw auto-interp
            explanations = data.get("explanations", [])
            if explanations and len(explanations) > 0:
                info.auto_interp = explanations[0].get("description", "")

            # Store top activations
            info.top_activations = data.get("activations", [])[:10]

            # Generate interpretation using strategy
            info.interpretation = self.strategy.interpret(
                feature_idx, self.neuronpedia_id, data
            )
            info.interpretation_source = (
                "llm" if isinstance(self.strategy, LLMStrategy) else "neuronpedia"
            )

        return info

    def get_features(self, feature_indices: List[int], verbose: bool = False) -> List[FeatureInfo]:
        """
        Get information for multiple features.

        Args:
            feature_indices: List of feature indices
            verbose: Print progress

        Returns:
            List of FeatureInfo objects
        """
        results = []
        for i, idx in enumerate(feature_indices, 1):
            if verbose:
                print(f"[{i}/{len(feature_indices)}] Feature {idx}...")
            info = self.get_feature_info(idx)
            results.append(info)
        return results

    def format_report(
        self,
        features: List[FeatureInfo],
        include_urls: bool = True,
        include_auto_interp: bool = False,
    ) -> str:
        """
        Format feature information as a readable report.

        Args:
            features: List of FeatureInfo objects
            include_urls: Include Neuronpedia URLs
            include_auto_interp: Show original auto-interp alongside LLM interpretation

        Returns:
            Formatted string report
        """
        lines = []
        for i, f in enumerate(features, 1):
            coef_str = f" (coef: {f.coefficient:+.4f})" if f.coefficient else ""

            lines.append(f"{i}. Feature {f.feature_idx}{coef_str}")
            lines.append(f"   {f.interpretation or 'No interpretation'}")

            if include_auto_interp and f.auto_interp and f.interpretation_source == "llm":
                lines.append(f"   [auto-interp: {f.auto_interp}]")

            if include_urls:
                lines.append(f"   {f.neuronpedia_url}")

            lines.append("")

        return "\n".join(lines)
