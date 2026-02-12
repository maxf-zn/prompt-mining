"""
Unit tests for PromptSpec.
"""
import pytest
from prompt_mining.core import PromptSpec


class TestPromptSpec:
    """Test PromptSpec dataclass."""

    def test_simple_prompt(self):
        """Test creating simple text prompt."""
        spec = PromptSpec(
            prompt_id="test:001",
            dataset_id="test_dataset",
            messages=[{"role": "user", "content": "Hello world"}],
            labels={"category": "greeting"}
        )

        assert spec.prompt_id == "test:001"
        assert spec.dataset_id == "test_dataset"
        assert len(spec.messages) == 1
        assert spec.messages[0]["role"] == "user"
        assert spec.labels["category"] == "greeting"
        assert spec.tools is None
        assert spec.text is None

    def test_tool_use_prompt(self):
        """Test creating tool-use prompt (InjecAgent style)."""
        spec = PromptSpec(
            prompt_id="injecagent:dh_0",
            dataset_id="injecagent",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant..."},
                {"role": "user", "content": "Can you fetch product details..."},
                {"role": "assistant", "content": "<tool_call>...</tool_call>"},
                {"role": "tool", "content": "{'product_details': {...}}"}
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "AmazonGetProductDetails",
                        "description": "Fetch product details",
                        "parameters": {}
                    }
                }
            ],
            labels={
                "attack_type": "dh",
                "user_tool": "AmazonGetProductDetails"
            }
        )

        assert spec.prompt_id == "injecagent:dh_0"
        assert len(spec.messages) == 4
        assert len(spec.tools) == 1
        assert spec.labels["attack_type"] == "dh"

    def test_validation_empty_prompt_id(self):
        """Test that empty prompt_id raises error."""
        with pytest.raises(ValueError, match="prompt_id cannot be empty"):
            PromptSpec(
                prompt_id="",
                dataset_id="test",
                messages=[{"role": "user", "content": "test"}]
            )

    def test_validation_empty_dataset_id(self):
        """Test that empty dataset_id raises error."""
        with pytest.raises(ValueError, match="dataset_id cannot be empty"):
            PromptSpec(
                prompt_id="test:001",
                dataset_id="",
                messages=[{"role": "user", "content": "test"}]
            )

    def test_validation_empty_messages(self):
        """Test that empty messages without text raises error."""
        with pytest.raises(ValueError, match="Either messages or text must be provided"):
            PromptSpec(
                prompt_id="test:001",
                dataset_id="test",
                messages=None,
                text=None
            )

    def test_flexible_labels(self):
        """Test that labels dict supports flexible metadata."""
        # Simple labels
        spec1 = PromptSpec(
            prompt_id="test:001",
            dataset_id="test",
            messages=[{"role": "user", "content": "test"}],
            labels={"success": True, "category": "A"}
        )
        assert spec1.labels["success"] is True
        assert spec1.labels["category"] == "A"

        # Nested labels
        spec2 = PromptSpec(
            prompt_id="test:002",
            dataset_id="test",
            messages=[{"role": "user", "content": "test"}],
            labels={
                "attack_type": "dh",
                "metadata": {
                    "user_tool": "Tool1",
                    "attacker_tools": ["Tool2", "Tool3"]
                }
            }
        )
        assert spec2.labels["attack_type"] == "dh"
        assert isinstance(spec2.labels["metadata"], dict)

        # Empty labels (default)
        spec3 = PromptSpec(
            prompt_id="test:003",
            dataset_id="test",
            messages=[{"role": "user", "content": "test"}]
        )
        assert spec3.labels == {}

    def test_serialization(self):
        """Test to_dict and from_dict."""
        spec = PromptSpec(
            prompt_id="test:001",
            dataset_id="test_dataset",
            messages=[
                {"role": "user", "content": "Hello"}
            ],
            labels={"category": "greeting"},
            tools=[{"type": "function", "function": {"name": "test"}}],
            text="Processed text"
        )

        # Convert to dict
        d = spec.to_dict()
        assert d["prompt_id"] == "test:001"
        assert d["dataset_id"] == "test_dataset"
        assert d["text"] == "Processed text"

        # Restore from dict
        restored = PromptSpec.from_dict(d)
        assert restored.prompt_id == spec.prompt_id
        assert restored.dataset_id == spec.dataset_id
        assert restored.messages == spec.messages
        assert restored.labels == spec.labels
        assert restored.tools == spec.tools
        assert restored.text == spec.text

    def test_text_field_computed_during_ingestion(self):
        """Test that text field is initially None and set during ingestion."""
        spec = PromptSpec(
            prompt_id="test:001",
            dataset_id="test",
            messages=[{"role": "user", "content": "Hello"}]
        )

        # Initially None
        assert spec.text is None

        # In practice, this would be set by the tokenizer during ingestion
        # Here we just verify it can be set
        spec_dict = spec.to_dict()
        spec_dict["text"] = "Flattened text after chat template"
        spec_with_text = PromptSpec.from_dict(spec_dict)
        assert spec_with_text.text == "Flattened text after chat template"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
