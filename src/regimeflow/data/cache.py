"""
Robust disk cache with SHA-256 verification and atomic writes.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
from regimeflow.data.schema import DatasetManifest
from regimeflow.logging_utils import logger


class CacheManager:
    """
    Manages caching of datasets with cryptographic hash verification and metadata manifests.
    """

    def __init__(self, cache_dir: str | Path = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_paths(self, dataset_id: str) -> Tuple[Path, Path]:
        data_path = self.cache_dir / f"{dataset_id}.parquet"
        manifest_path = self.cache_dir / f"{dataset_id}.manifest.json"
        return data_path, manifest_path

    def save(self, dataset_id: str, df: pd.DataFrame, manifest: DatasetManifest) -> Path:
        """
        Atomically save DataFrame and manifest.
        """
        data_path, manifest_path = self.get_paths(dataset_id)
        temp_data_path = self.cache_dir / f"{dataset_id}.tmp.parquet"
        temp_manifest_path = self.cache_dir / f"{dataset_id}.tmp.manifest.json"

        # Ensure index has name
        if df.index.name is None:
            df.index.name = "date"

        # Atomic write for parquet
        df.to_parquet(temp_data_path, engine="pyarrow", index=True)
        sha256 = self.compute_sha256(temp_data_path)
        manifest.sha256_hash = sha256
        manifest.row_count = len(df)
        if len(df) > 0:
            manifest.start_date = str(df.index.min().date())
            manifest.end_date = str(df.index.max().date())

        with open(temp_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        # Atomic rename
        temp_data_path.replace(data_path)
        temp_manifest_path.replace(manifest_path)
        logger.info(f"Cached dataset '{dataset_id}' ({len(df)} rows, hash={sha256[:10]}...)")
        return data_path

    def load(self, dataset_id: str) -> Tuple[Optional[pd.DataFrame], Optional[DatasetManifest]]:
        """
        Load dataset with SHA-256 verification.
        Returns (None, None) if cache is missing or corrupt.
        """
        data_path, manifest_path = self.get_paths(dataset_id)
        if not data_path.exists() or not manifest_path.exists():
            return None, None

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_dict = json.load(f)
            manifest = DatasetManifest.from_dict(manifest_dict)

            # Check SHA-256
            current_hash = self.compute_sha256(data_path)
            if current_hash != manifest.sha256_hash:
                logger.warning(
                    f"Cache corruption detected for '{dataset_id}': "
                    f"expected {manifest.sha256_hash}, found {current_hash}."
                )
                return None, None

            df = pd.read_parquet(data_path, engine="pyarrow")
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            return df, manifest
        except Exception as e:
            logger.warning(f"Failed to read cache for '{dataset_id}': {e}")
            return None, None

    def invalidate(self, dataset_id: str) -> None:
        data_path, manifest_path = self.get_paths(dataset_id)
        if data_path.exists():
            data_path.unlink()
        if manifest_path.exists():
            manifest_path.unlink()
        logger.info(f"Invalidated cache for '{dataset_id}'")
