"""
Feature Store Module
====================

Provides storage and retrieval of computed features with versioning support.
Supports both online (single lookup) and offline (batch) retrieval.
Uses the same database backend as MLflow (Supabase PostgreSQL or SQLite).
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Feature Store for storing and retrieving computed features.

    Supports:
    - Feature versioning (track which features belong to which model/experiment)
    - Online retrieval (single entity lookup)
    - Offline retrieval (batch retrieval for training)
    - Reuse of DatabaseConfig connection (Supabase/SQLite)

    Usage:
        # Initialize from settings
        from src.config import get_settings
        store = FeatureStore.from_settings(get_settings())

        # Register a feature version
        store.register_version(
            version="v1.0.0",
            feature_names=["total_nights", "total_guests", ...],
            experiment_id="exp_001"
        )

        # Store features
        store.store_features(
            entity_ids=["INN001", "INN002"],
            feature_vectors=[[1.0, 2.0, ...], [1.5, 2.5, ...]],
            version="v1.0.0"
        )

        # Online retrieval (single)
        features = store.get_features("INN001", version="v1.0.0")

        # Offline retrieval (batch)
        features_df = store.get_features_batch(["INN001", "INN002"])
    """

    # SQL for table creation (SQLite compatible)
    SQL_CREATE_METADATA_TABLE = """
        CREATE TABLE IF NOT EXISTS feature_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version TEXT UNIQUE NOT NULL,
            experiment_id TEXT,
            model_name TEXT,
            feature_names TEXT NOT NULL,
            feature_count INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active INTEGER DEFAULT 1
        )
    """

    SQL_CREATE_VECTORS_TABLE = """
        CREATE TABLE IF NOT EXISTS feature_vectors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id TEXT NOT NULL,
            version TEXT NOT NULL,
            feature_values TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(entity_id, version)
        )
    """

    SQL_CREATE_INDEXES = [
        "CREATE INDEX IF NOT EXISTS idx_fv_entity ON feature_vectors(entity_id)",
        "CREATE INDEX IF NOT EXISTS idx_fv_version ON feature_vectors(version)",
        "CREATE INDEX IF NOT EXISTS idx_fm_active ON feature_metadata(is_active)",
    ]

    def __init__(
        self,
        db_path: str = "feature_store.db",
        is_postgres: bool = False,
    ):
        """
        Initialize the Feature Store.

        Args:
            db_path: Path to SQLite database or PostgreSQL connection string.
            is_postgres: Whether using PostgreSQL (True) or SQLite (False).
        """
        self.db_path = db_path
        self.is_postgres = is_postgres
        self._is_initialized = False

    @classmethod
    def from_settings(cls, settings: Any) -> "FeatureStore":
        """
        Factory method to create FeatureStore from settings.

        Args:
            settings: Settings instance with database and feature_store config.

        Returns:
            Configured FeatureStore instance.
        """
        # Check if feature store is enabled
        if hasattr(settings, "feature_store") and not settings.feature_store.enabled:
            logger.info("Feature Store is disabled in settings")

        # Use dedicated feature store path or default
        if hasattr(settings, "feature_store") and settings.feature_store.db_path:
            db_path = settings.feature_store.db_path
        else:
            # Default to feature_store.db in the same location as MLflow db
            db_path = "feature_store.db"

        is_postgres = settings.database.is_cloud_db

        if is_postgres:
            # For PostgreSQL, use the Supabase connection string
            db_path = settings.database.supabase_url

        return cls(db_path=db_path, is_postgres=is_postgres)

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Get database connection as context manager.

        Yields:
            Database connection object.
        """
        if self.is_postgres:
            # PostgreSQL connection
            try:
                import psycopg2

                conn = psycopg2.connect(self.db_path)
                try:
                    yield conn
                finally:
                    conn.close()
            except ImportError:
                raise ImportError(
                    "psycopg2 is required for PostgreSQL. "
                    "Install with: pip install psycopg2-binary"
                )
        else:
            # SQLite connection
            # Ensure directory exists
            db_file = Path(self.db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)

            conn = sqlite3.connect(str(db_file))
            try:
                yield conn
            finally:
                conn.close()

    def _get_placeholder(self) -> str:
        """Get SQL placeholder for current database type."""
        return "%s" if self.is_postgres else "?"

    def initialize(self) -> None:
        """
        Initialize the Feature Store tables.

        Creates the necessary tables if they don't exist.
        """
        if self._is_initialized:
            return

        with self._get_connection() as conn:
            cursor = conn.cursor()

            if self.is_postgres:
                # PostgreSQL version with SERIAL
                metadata_sql = """
                    CREATE TABLE IF NOT EXISTS feature_metadata (
                        id SERIAL PRIMARY KEY,
                        version TEXT UNIQUE NOT NULL,
                        experiment_id TEXT,
                        model_name TEXT,
                        feature_names TEXT NOT NULL,
                        feature_count INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT TRUE
                    )
                """
                vectors_sql = """
                    CREATE TABLE IF NOT EXISTS feature_vectors (
                        id SERIAL PRIMARY KEY,
                        entity_id TEXT NOT NULL,
                        version TEXT NOT NULL,
                        feature_values TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(entity_id, version)
                    )
                """
                cursor.execute(metadata_sql)
                cursor.execute(vectors_sql)
            else:
                # SQLite version
                cursor.execute(self.SQL_CREATE_METADATA_TABLE)
                cursor.execute(self.SQL_CREATE_VECTORS_TABLE)

            # Create indexes
            for index_sql in self.SQL_CREATE_INDEXES:
                cursor.execute(index_sql)

            conn.commit()

        self._is_initialized = True
        logger.info(
            f"Feature Store initialized (backend: "
            f"{'PostgreSQL' if self.is_postgres else 'SQLite'})"
        )

    def register_version(
        self,
        version: str,
        feature_names: List[str],
        experiment_id: Optional[str] = None,
        model_name: Optional[str] = None,
        set_active: bool = True,
    ) -> None:
        """
        Register a new feature version.

        Args:
            version: Unique version identifier (e.g., "v1.0.0").
            feature_names: List of feature column names.
            experiment_id: Optional MLflow experiment ID.
            model_name: Optional model name.
            set_active: If True, set this as the active version.
        """
        self.initialize()

        feature_names_json = json.dumps(feature_names)
        ph = self._get_placeholder()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            if self.is_postgres:
                sql = f"""
                    INSERT INTO feature_metadata
                    (version, experiment_id, model_name, feature_names, feature_count)
                    VALUES ({ph}, {ph}, {ph}, {ph}, {ph})
                    ON CONFLICT (version) DO UPDATE SET
                        experiment_id = EXCLUDED.experiment_id,
                        model_name = EXCLUDED.model_name,
                        feature_names = EXCLUDED.feature_names,
                        feature_count = EXCLUDED.feature_count
                """
            else:
                sql = f"""
                    INSERT OR REPLACE INTO feature_metadata
                    (version, experiment_id, model_name, feature_names, feature_count)
                    VALUES ({ph}, {ph}, {ph}, {ph}, {ph})
                """

            cursor.execute(
                sql,
                (version, experiment_id, model_name, feature_names_json, len(feature_names)),
            )
            conn.commit()

        if set_active:
            self.set_active_version(version)

        logger.info(f"Registered feature version: {version} ({len(feature_names)} features)")

    def store_features(
        self,
        entity_ids: List[str],
        feature_vectors: Union[np.ndarray, List[List[float]], pd.DataFrame],
        version: str,
    ) -> int:
        """
        Store computed feature vectors.

        Args:
            entity_ids: List of entity identifiers (e.g., Booking_IDs).
            feature_vectors: 2D array/list of feature values.
            version: Feature version to associate with.

        Returns:
            Number of features stored.
        """
        self.initialize()

        # Convert DataFrame to numpy array if needed
        if isinstance(feature_vectors, pd.DataFrame):
            feature_vectors = feature_vectors.values

        if len(entity_ids) != len(feature_vectors):
            raise ValueError(
                f"entity_ids ({len(entity_ids)}) and feature_vectors "
                f"({len(feature_vectors)}) must have same length"
            )

        ph = self._get_placeholder()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            if self.is_postgres:
                sql = f"""
                    INSERT INTO feature_vectors (entity_id, version, feature_values)
                    VALUES ({ph}, {ph}, {ph})
                    ON CONFLICT (entity_id, version) DO UPDATE SET
                        feature_values = EXCLUDED.feature_values
                """
            else:
                sql = f"""
                    INSERT OR REPLACE INTO feature_vectors
                    (entity_id, version, feature_values)
                    VALUES ({ph}, {ph}, {ph})
                """

            count = 0
            for entity_id, features in zip(entity_ids, feature_vectors):
                # Convert to list if numpy array
                if hasattr(features, "tolist"):
                    features = features.tolist()

                features_json = json.dumps(features)
                cursor.execute(sql, (str(entity_id), version, features_json))
                count += 1

            conn.commit()

        logger.info(f"Stored {count} feature vectors for version {version}")
        return count

    def get_features(
        self,
        entity_id: str,
        version: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get features for a single entity (online retrieval).

        Args:
            entity_id: Entity identifier (e.g., Booking_ID).
            version: Feature version. If None, uses latest active version.

        Returns:
            Dictionary with entity_id, version, feature_values, and feature_names.
            Returns None if not found.
        """
        self.initialize()

        if version is None:
            version = self._get_active_version()
            if version is None:
                logger.warning("No active feature version found")
                return None

        ph = self._get_placeholder()
        sql = f"""
            SELECT fv.entity_id, fv.version, fv.feature_values, fm.feature_names
            FROM feature_vectors fv
            JOIN feature_metadata fm ON fv.version = fm.version
            WHERE fv.entity_id = {ph} AND fv.version = {ph}
        """

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (str(entity_id), version))
            row = cursor.fetchone()

        if row is None:
            return None

        return {
            "entity_id": row[0],
            "version": row[1],
            "feature_values": json.loads(row[2]),
            "feature_names": json.loads(row[3]),
        }

    def get_features_batch(
        self,
        entity_ids: List[str],
        version: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get features for multiple entities (offline retrieval).

        Args:
            entity_ids: List of entity identifiers.
            version: Feature version. If None, uses latest active version.

        Returns:
            DataFrame with entity_id and feature columns.
        """
        self.initialize()

        if version is None:
            version = self._get_active_version()
            if version is None:
                logger.warning("No active feature version found")
                return pd.DataFrame()

        feature_names = self._get_feature_names(version)
        if feature_names is None:
            return pd.DataFrame()

        ph = self._get_placeholder()
        placeholders = ", ".join([ph] * len(entity_ids))
        sql = f"""
            SELECT entity_id, feature_values
            FROM feature_vectors
            WHERE entity_id IN ({placeholders}) AND version = {ph}
        """

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (*[str(eid) for eid in entity_ids], version))
            rows = cursor.fetchall()

        # Build DataFrame
        data = []
        for row in rows:
            feature_values = json.loads(row[1])
            record = {"entity_id": row[0]}
            for name, value in zip(feature_names, feature_values):
                record[name] = value
            data.append(record)

        return pd.DataFrame(data)

    def get_missing_entities(
        self,
        entity_ids: List[str],
        version: Optional[str] = None,
    ) -> List[str]:
        """
        Find entities that don't have stored features.

        Useful for determining which entities need feature computation.

        Args:
            entity_ids: List of entity identifiers to check.
            version: Feature version. If None, uses latest active version.

        Returns:
            List of entity IDs that don't have stored features.
        """
        self.initialize()

        if version is None:
            version = self._get_active_version()
            if version is None:
                return list(entity_ids)

        ph = self._get_placeholder()
        placeholders = ", ".join([ph] * len(entity_ids))
        sql = f"""
            SELECT entity_id FROM feature_vectors
            WHERE entity_id IN ({placeholders}) AND version = {ph}
        """

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (*[str(eid) for eid in entity_ids], version))
            existing = {row[0] for row in cursor.fetchall()}

        return [eid for eid in entity_ids if str(eid) not in existing]

    def _get_active_version(self) -> Optional[str]:
        """Get the latest active feature version."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if self.is_postgres:
                sql = """
                    SELECT version FROM feature_metadata
                    WHERE is_active = TRUE
                    ORDER BY created_at DESC
                    LIMIT 1
                """
            else:
                sql = """
                    SELECT version FROM feature_metadata
                    WHERE is_active = 1
                    ORDER BY created_at DESC
                    LIMIT 1
                """

            cursor.execute(sql)
            row = cursor.fetchone()

        return row[0] if row else None

    def _get_feature_names(self, version: str) -> Optional[List[str]]:
        """Get feature names for a specific version."""
        ph = self._get_placeholder()
        sql = f"SELECT feature_names FROM feature_metadata WHERE version = {ph}"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (version,))
            row = cursor.fetchone()

        if row is None:
            return None

        return json.loads(row[0])

    def get_version_info(self, version: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific version.

        Args:
            version: Version identifier.

        Returns:
            Dictionary with version metadata, or None if not found.
        """
        self.initialize()

        ph = self._get_placeholder()
        sql = f"""
            SELECT version, experiment_id, model_name, feature_names,
                   feature_count, created_at, is_active
            FROM feature_metadata
            WHERE version = {ph}
        """

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (version,))
            row = cursor.fetchone()

        if row is None:
            return None

        return {
            "version": row[0],
            "experiment_id": row[1],
            "model_name": row[2],
            "feature_names": json.loads(row[3]),
            "feature_count": row[4],
            "created_at": row[5],
            "is_active": bool(row[6]),
        }

    def list_versions(self, active_only: bool = False) -> List[Dict[str, Any]]:
        """
        List all feature versions.

        Args:
            active_only: If True, only return active versions.

        Returns:
            List of version metadata dictionaries.
        """
        self.initialize()

        sql = """
            SELECT version, experiment_id, model_name, feature_count,
                   created_at, is_active
            FROM feature_metadata
        """

        if active_only:
            if self.is_postgres:
                sql += " WHERE is_active = TRUE"
            else:
                sql += " WHERE is_active = 1"

        sql += " ORDER BY created_at DESC"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()

        return [
            {
                "version": row[0],
                "experiment_id": row[1],
                "model_name": row[2],
                "feature_count": row[3],
                "created_at": row[4],
                "is_active": bool(row[5]),
            }
            for row in rows
        ]

    def set_active_version(self, version: str) -> None:
        """
        Set a version as the active version.

        Args:
            version: Version to activate.
        """
        self.initialize()

        ph = self._get_placeholder()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Deactivate all versions
            if self.is_postgres:
                cursor.execute("UPDATE feature_metadata SET is_active = FALSE")
            else:
                cursor.execute("UPDATE feature_metadata SET is_active = 0")

            # Activate the specified version
            if self.is_postgres:
                sql = f"UPDATE feature_metadata SET is_active = TRUE WHERE version = {ph}"
            else:
                sql = f"UPDATE feature_metadata SET is_active = 1 WHERE version = {ph}"

            cursor.execute(sql, (version,))
            conn.commit()

        logger.info(f"Set active feature version: {version}")

    def delete_version(self, version: str) -> int:
        """
        Delete a feature version and all its vectors.

        Args:
            version: Version to delete.

        Returns:
            Number of feature vectors deleted.
        """
        self.initialize()

        ph = self._get_placeholder()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Delete vectors
            cursor.execute(
                f"DELETE FROM feature_vectors WHERE version = {ph}",
                (version,),
            )
            deleted_count = cursor.rowcount

            # Delete metadata
            cursor.execute(
                f"DELETE FROM feature_metadata WHERE version = {ph}",
                (version,),
            )

            conn.commit()

        logger.info(f"Deleted version {version} ({deleted_count} vectors)")
        return deleted_count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get Feature Store statistics.

        Returns:
            Dictionary with stats (total_versions, total_vectors, active_version).
        """
        self.initialize()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM feature_metadata")
            total_versions = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM feature_vectors")
            total_vectors = cursor.fetchone()[0]

        active_version = self._get_active_version()

        return {
            "total_versions": total_versions,
            "total_vectors": total_vectors,
            "active_version": active_version,
            "backend": "PostgreSQL" if self.is_postgres else "SQLite",
        }
