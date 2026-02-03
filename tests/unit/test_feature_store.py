"""
Unit Tests for Feature Store
============================

Tests for the FeatureStore class.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.store import FeatureStore


class TestFeatureStoreInitialization:
    """Tests for FeatureStore initialization."""

    def test_create_with_default_path(self):
        """Test creating Feature Store with default path."""
        store = FeatureStore()
        assert store.db_path == "feature_store.db"
        assert store.is_postgres is False

    def test_create_with_custom_path(self):
        """Test creating Feature Store with custom path."""
        store = FeatureStore(db_path="/tmp/custom.db")
        assert store.db_path == "/tmp/custom.db"

    def test_initialize_creates_tables(self):
        """Test that initialize creates required tables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = FeatureStore(db_path=str(db_path))
            store.initialize()

            assert store._is_initialized is True
            assert db_path.exists()


class TestFeatureVersionManagement:
    """Tests for feature version management."""

    @pytest.fixture
    def store(self):
        """Create a temporary Feature Store for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = FeatureStore(db_path=str(db_path))
            store.initialize()
            yield store

    def test_register_version(self, store):
        """Test registering a new feature version."""
        store.register_version(
            version="v1.0.0",
            feature_names=["feature_a", "feature_b", "feature_c"],
            experiment_id="exp_001",
            model_name="test_model",
        )

        info = store.get_version_info("v1.0.0")
        assert info is not None
        assert info["version"] == "v1.0.0"
        assert info["feature_count"] == 3
        assert info["experiment_id"] == "exp_001"
        assert info["model_name"] == "test_model"
        assert info["feature_names"] == ["feature_a", "feature_b", "feature_c"]

    def test_register_version_updates_existing(self, store):
        """Test that registering same version updates it."""
        store.register_version(
            version="v1.0.0",
            feature_names=["a", "b"],
        )

        store.register_version(
            version="v1.0.0",
            feature_names=["x", "y", "z"],
            experiment_id="new_exp",
        )

        info = store.get_version_info("v1.0.0")
        assert info["feature_count"] == 3
        assert info["experiment_id"] == "new_exp"

    def test_list_versions(self, store):
        """Test listing all versions."""
        store.register_version(version="v1.0.0", feature_names=["a"])
        store.register_version(version="v2.0.0", feature_names=["a", "b"])

        versions = store.list_versions()
        assert len(versions) == 2
        version_names = [v["version"] for v in versions]
        assert "v1.0.0" in version_names
        assert "v2.0.0" in version_names

    def test_list_versions_active_only(self, store):
        """Test listing only active versions."""
        store.register_version(version="v1.0.0", feature_names=["a"], set_active=False)
        store.register_version(version="v2.0.0", feature_names=["b"], set_active=True)

        active_versions = store.list_versions(active_only=True)
        assert len(active_versions) == 1
        assert active_versions[0]["version"] == "v2.0.0"

    def test_set_active_version(self, store):
        """Test setting active version."""
        store.register_version(version="v1.0.0", feature_names=["a"])
        store.register_version(version="v2.0.0", feature_names=["b"])

        store.set_active_version("v1.0.0")

        versions = store.list_versions()
        v1 = next(v for v in versions if v["version"] == "v1.0.0")
        v2 = next(v for v in versions if v["version"] == "v2.0.0")

        assert v1["is_active"] is True
        assert v2["is_active"] is False


class TestFeatureStorage:
    """Tests for storing features."""

    @pytest.fixture
    def store(self):
        """Create a temporary Feature Store for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = FeatureStore(db_path=str(db_path))
            store.initialize()
            store.register_version(
                version="v1.0.0",
                feature_names=["feature_a", "feature_b"],
            )
            yield store

    def test_store_features_list(self, store):
        """Test storing features from list."""
        count = store.store_features(
            entity_ids=["E001", "E002", "E003"],
            feature_vectors=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            version="v1.0.0",
        )

        assert count == 3

    def test_store_features_numpy(self, store):
        """Test storing features from numpy array."""
        features = np.array([[1.0, 2.0], [3.0, 4.0]])

        count = store.store_features(
            entity_ids=["E001", "E002"],
            feature_vectors=features,
            version="v1.0.0",
        )

        assert count == 2

    def test_store_features_dataframe(self, store):
        """Test storing features from DataFrame."""
        df = pd.DataFrame(
            {"feature_a": [1.0, 3.0], "feature_b": [2.0, 4.0]},
        )

        count = store.store_features(
            entity_ids=["E001", "E002"],
            feature_vectors=df,
            version="v1.0.0",
        )

        assert count == 2

    def test_store_features_mismatched_length(self, store):
        """Test that mismatched lengths raise error."""
        with pytest.raises(ValueError, match="must have same length"):
            store.store_features(
                entity_ids=["E001", "E002"],
                feature_vectors=[[1.0, 2.0]],  # Only 1 vector for 2 IDs
                version="v1.0.0",
            )

    def test_store_features_updates_existing(self, store):
        """Test that storing same entity updates values."""
        store.store_features(
            entity_ids=["E001"],
            feature_vectors=[[1.0, 2.0]],
            version="v1.0.0",
        )

        store.store_features(
            entity_ids=["E001"],
            feature_vectors=[[10.0, 20.0]],
            version="v1.0.0",
        )

        result = store.get_features("E001", version="v1.0.0")
        assert result["feature_values"] == [10.0, 20.0]


class TestFeatureRetrieval:
    """Tests for retrieving features."""

    @pytest.fixture
    def store(self):
        """Create a populated Feature Store for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = FeatureStore(db_path=str(db_path))
            store.initialize()
            store.register_version(
                version="v1.0.0",
                feature_names=["feature_a", "feature_b"],
            )
            store.store_features(
                entity_ids=["E001", "E002", "E003"],
                feature_vectors=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                version="v1.0.0",
            )
            yield store

    def test_get_features_single(self, store):
        """Test getting features for single entity."""
        result = store.get_features("E001", version="v1.0.0")

        assert result is not None
        assert result["entity_id"] == "E001"
        assert result["version"] == "v1.0.0"
        assert result["feature_values"] == [1.0, 2.0]
        assert result["feature_names"] == ["feature_a", "feature_b"]

    def test_get_features_not_found(self, store):
        """Test getting features for non-existent entity."""
        result = store.get_features("NONEXISTENT", version="v1.0.0")
        assert result is None

    def test_get_features_uses_active_version(self, store):
        """Test that get_features uses active version when not specified."""
        result = store.get_features("E001")  # No version specified
        assert result is not None
        assert result["version"] == "v1.0.0"

    def test_get_features_batch(self, store):
        """Test getting features for multiple entities."""
        df = store.get_features_batch(["E001", "E002"], version="v1.0.0")

        assert len(df) == 2
        assert "entity_id" in df.columns
        assert "feature_a" in df.columns
        assert "feature_b" in df.columns

        e001 = df[df["entity_id"] == "E001"].iloc[0]
        assert e001["feature_a"] == 1.0
        assert e001["feature_b"] == 2.0

    def test_get_features_batch_partial(self, store):
        """Test batch retrieval with some missing entities."""
        df = store.get_features_batch(
            ["E001", "NONEXISTENT", "E003"],
            version="v1.0.0",
        )

        # Only E001 and E003 should be returned
        assert len(df) == 2
        entity_ids = df["entity_id"].tolist()
        assert "E001" in entity_ids
        assert "E003" in entity_ids
        assert "NONEXISTENT" not in entity_ids

    def test_get_missing_entities(self, store):
        """Test finding entities without stored features."""
        missing = store.get_missing_entities(
            ["E001", "E002", "NEW001", "NEW002"],
            version="v1.0.0",
        )

        assert len(missing) == 2
        assert "NEW001" in missing
        assert "NEW002" in missing
        assert "E001" not in missing


class TestFeatureStoreStats:
    """Tests for Feature Store statistics."""

    def test_get_stats_empty(self):
        """Test stats for empty store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = FeatureStore(db_path=str(db_path))
            store.initialize()

            stats = store.get_stats()
            assert stats["total_versions"] == 0
            assert stats["total_vectors"] == 0
            assert stats["active_version"] is None
            assert stats["backend"] == "SQLite"

    def test_get_stats_populated(self):
        """Test stats for populated store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = FeatureStore(db_path=str(db_path))
            store.initialize()
            store.register_version(version="v1.0.0", feature_names=["a"])
            store.store_features(
                entity_ids=["E1", "E2", "E3"],
                feature_vectors=[[1.0], [2.0], [3.0]],
                version="v1.0.0",
            )

            stats = store.get_stats()
            assert stats["total_versions"] == 1
            assert stats["total_vectors"] == 3
            assert stats["active_version"] == "v1.0.0"


class TestFeatureStoreDeletion:
    """Tests for deleting feature versions."""

    def test_delete_version(self):
        """Test deleting a feature version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = FeatureStore(db_path=str(db_path))
            store.initialize()
            store.register_version(version="v1.0.0", feature_names=["a"])
            store.store_features(
                entity_ids=["E1", "E2"],
                feature_vectors=[[1.0], [2.0]],
                version="v1.0.0",
            )

            deleted = store.delete_version("v1.0.0")
            assert deleted == 2

            # Verify deletion
            info = store.get_version_info("v1.0.0")
            assert info is None

            stats = store.get_stats()
            assert stats["total_versions"] == 0
            assert stats["total_vectors"] == 0
