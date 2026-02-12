"""
Unit tests for CompactGraph and FeatureInfo.
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path

from prompt_mining.core import FeatureInfo, CompactGraph


class TestFeatureInfo:
    """Test FeatureInfo dataclass."""

    def test_creation(self):
        """Test creating FeatureInfo."""
        info = FeatureInfo(
            layer=10,
            position=31,
            feature_idx=5421,
            influence=0.034,
            activation_value=2.47,
            logit_attribution=0.021
        )

        assert info.layer == 10
        assert info.position == 31
        assert info.feature_idx == 5421
        assert info.influence == 0.034
        assert info.activation_value == 2.47
        assert info.logit_attribution == 0.021

    def test_validation(self):
        """Test field validation."""
        # Negative layer
        with pytest.raises(ValueError, match="layer must be >= 0"):
            FeatureInfo(
                layer=-1,
                position=0,
                feature_idx=0,
                influence=0.0,
                activation_value=0.0,
                logit_attribution=0.0
            )

        # Negative position
        with pytest.raises(ValueError, match="position must be >= 0"):
            FeatureInfo(
                layer=0,
                position=-1,
                feature_idx=0,
                influence=0.0,
                activation_value=0.0,
                logit_attribution=0.0
            )

        # Negative feature_idx
        with pytest.raises(ValueError, match="feature_idx must be >= 0"):
            FeatureInfo(
                layer=0,
                position=0,
                feature_idx=-1,
                influence=0.0,
                activation_value=0.0,
                logit_attribution=0.0
            )

    def test_serialization(self):
        """Test to_tuple and from_tuple."""
        info = FeatureInfo(
            layer=10,
            position=31,
            feature_idx=5421,
            influence=0.034,
            activation_value=2.47,
            logit_attribution=0.021
        )

        # Round-trip
        tuple_repr = info.to_tuple()
        restored = FeatureInfo.from_tuple(tuple_repr)

        assert restored.layer == info.layer
        assert restored.position == info.position
        assert restored.feature_idx == info.feature_idx
        assert abs(restored.influence - info.influence) < 1e-6
        assert abs(restored.activation_value - info.activation_value) < 1e-6
        assert abs(restored.logit_attribution - info.logit_attribution) < 1e-6

    def test_immutability(self):
        """Test that FeatureInfo is immutable."""
        info = FeatureInfo(
            layer=10,
            position=31,
            feature_idx=5421,
            influence=0.034,
            activation_value=2.47,
            logit_attribution=0.021
        )

        with pytest.raises(AttributeError):
            info.layer = 20


class TestCompactGraph:
    """Test CompactGraph dataclass."""

    def create_sample_graph(self):
        """Create a sample CompactGraph for testing."""
        features = [
            FeatureInfo(
                layer=10,
                position=31,
                feature_idx=5421,
                influence=0.034,
                activation_value=2.47,
                logit_attribution=0.021
            ),
            FeatureInfo(
                layer=10,
                position=44,
                feature_idx=5421,
                influence=0.019,
                activation_value=1.9,
                logit_attribution=0.015
            ),
            FeatureInfo(
                layer=20,
                position=44,
                feature_idx=8234,
                influence=0.067,
                activation_value=3.21,
                logit_attribution=0.067
            ),
        ]

        return CompactGraph(
            prompt_id="test:001",
            input_string="Hello world",
            input_tokens=np.array([1, 2, 3, 4, 5]),
            logit_tokens=np.array([42, 13, 7]),
            logit_probabilities=np.array([0.55, 0.22, 0.06]),
            features=features,
            n_layers=28,
            model_hash="test_model_hash",
            clt_hash="test_clt_hash",
            seed=42,
            scan="test-scan"
        )

    def test_creation(self):
        """Test creating CompactGraph."""
        graph = self.create_sample_graph()

        assert graph.prompt_id == "test:001"
        assert graph.input_string == "Hello world"
        assert len(graph.input_tokens) == 5
        assert len(graph.features) == 3
        assert graph.n_layers == 28
        assert graph.model_hash == "test_model_hash"
        assert graph.seed == 42

    def test_validation(self):
        """Test validation."""
        features = [
            FeatureInfo(0, 0, 0, 0.0, 0.0, 0.0)
        ]

        # Empty prompt_id
        with pytest.raises(ValueError, match="prompt_id cannot be empty"):
            CompactGraph(
                prompt_id="",
                input_string="test",
                input_tokens=np.array([1]),
                logit_tokens=np.array([1]),
                logit_probabilities=np.array([1.0]),
                features=features,
                n_layers=28,
                model_hash="hash",
                clt_hash="hash",
                seed=42
            )

        # Invalid n_layers
        with pytest.raises(ValueError, match="n_layers must be > 0"):
            CompactGraph(
                prompt_id="test:001",
                input_string="test",
                input_tokens=np.array([1]),
                logit_tokens=np.array([1]),
                logit_probabilities=np.array([1.0]),
                features=features,
                n_layers=0,
                model_hash="hash",
                clt_hash="hash",
                seed=42
            )

        # Empty features (allowed for raw-only runs)
        graph_empty = CompactGraph(
            prompt_id="test:001",
            input_string="test",
            input_tokens=np.array([1]),
            logit_tokens=np.array([1]),
            logit_probabilities=np.array([1.0]),
            features=[],
            n_layers=28,
            model_hash="hash",
            clt_hash="hash",
            seed=42
        )
        assert len(graph_empty.features) == 0

    def test_pytorch_serialization(self):
        """Test to_pt and from_pt."""
        graph = self.create_sample_graph()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_graph.pt"

            # Save
            graph.to_pt(str(path))
            assert path.exists()

            # Load
            restored = CompactGraph.from_pt(str(path))

            # Verify
            assert restored.prompt_id == graph.prompt_id
            assert restored.input_string == graph.input_string
            assert np.array_equal(restored.input_tokens, graph.input_tokens)
            assert np.array_equal(restored.logit_tokens, graph.logit_tokens)
            assert len(restored.features) == len(graph.features)

            # Check first feature
            f1 = restored.features[0]
            f1_orig = graph.features[0]
            assert f1.layer == f1_orig.layer
            assert f1.position == f1_orig.position
            assert f1.feature_idx == f1_orig.feature_idx

    def test_get_features_at_layer(self):
        """Test filtering features by layer."""
        graph = self.create_sample_graph()

        layer10_features = graph.get_features_at_layer(10)
        assert len(layer10_features) == 2

        layer20_features = graph.get_features_at_layer(20)
        assert len(layer20_features) == 1

        layer15_features = graph.get_features_at_layer(15)
        assert len(layer15_features) == 0

    def test_get_features_at_position(self):
        """Test filtering features by position."""
        graph = self.create_sample_graph()

        pos31_features = graph.get_features_at_position(31)
        assert len(pos31_features) == 1
        assert pos31_features[0].layer == 10

        pos44_features = graph.get_features_at_position(44)
        assert len(pos44_features) == 2

    def test_get_top_features(self):
        """Test getting top features by influence."""
        graph = self.create_sample_graph()

        top2 = graph.get_top_features(k=2)
        assert len(top2) == 2
        # Should be sorted by influence descending
        assert top2[0].influence >= top2[1].influence
        # Top feature should be the one with influence=0.067
        assert top2[0].influence == 0.067

    def test_to_dict(self):
        """Test dictionary serialization."""
        graph = self.create_sample_graph()
        d = graph.to_dict()

        assert d['prompt_id'] == "test:001"
        assert d['n_layers'] == 28
        assert len(d['features']) == 3
        assert isinstance(d['input_tokens'], list)

    def test_immutability(self):
        """Test that CompactGraph is immutable."""
        graph = self.create_sample_graph()

        with pytest.raises(AttributeError):
            graph.prompt_id = "new_id"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
