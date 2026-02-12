"""
PromptSpec: Dataset-agnostic prompt specification with flexible labels.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class PromptSpec:
    """
    Dataset-agnostic prompt specification.

    This is the standard format that all dataset converters produce.
    It supports both simple text prompts and structured chat templates with tool calling.

    Attributes:
        prompt_id: Unique identifier for this prompt (e.g., "injecagent:dh_0")
        dataset_id: Which dataset this prompt came from (e.g., "injecagent")
        messages: Chat template messages (list of dicts with 'role' and 'content')
                  For simple prompts, use a single user message.
        tools: Optional tool schemas in function calling format
        labels: Flexible metadata dict for analysis (e.g., {"success": true, "attack_type": "dh"})
        text: Flattened text after applying chat template (computed by tokenizer during ingestion)

    Examples:
        Simple text prompt:
            PromptSpec(
                prompt_id="custom:001",
                dataset_id="custom",
                messages=[{"role": "user", "content": "Hello world"}],
                tools=None,
                labels={"category": "greeting"}
            )

        Tool-use prompt (InjecAgent):
            PromptSpec(
                prompt_id="injecagent:dh_0",
                dataset_id="injecagent",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant..."},
                    {"role": "user", "content": "Can you fetch product details..."},
                    {"role": "assistant", "content": "<tool_call>...</tool_call>"},
                    {"role": "tool", "content": "{'product_details': {...}}"}
                ],
                tools=[{"type": "function", "function": {...}}],
                labels={"attack_type": "dh", "user_tool": "AmazonGetProductDetails"}
            )
    """
    prompt_id: str
    dataset_id: str
    messages: List[Dict[str, str]]
    labels: Dict[str, Any] = field(default_factory=dict)
    tools: Optional[List[Dict[str, Any]]] = None
    text: Optional[str] = None  # Computed during ingestion

    def __post_init__(self):
        """Validate prompt_id and dataset_id."""
        if not self.prompt_id:
            raise ValueError("prompt_id cannot be empty")
        if not self.dataset_id:
            raise ValueError("dataset_id cannot be empty")
        # Either messages or text must be provided
        if not self.messages and not self.text:
            raise ValueError("Either messages or text must be provided")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "prompt_id": self.prompt_id,
            "dataset_id": self.dataset_id,
            "messages": self.messages,
            "labels": self.labels,
            "tools": self.tools,
            "text": self.text
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptSpec":
        """Load from dictionary."""
        return cls(**data)
