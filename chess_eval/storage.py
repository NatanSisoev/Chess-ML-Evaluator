import json
import os
import pickle
from datetime import datetime

import pandas as pd

from chess_eval.managers import DataManager, ModelManager


class FeatureDatasetStore:
    """
    Stores feature datasets along with full metadata.
    Default storage format: Pickle (fast, preserves types)
    """

    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)

    def _get_paths(self, name: str, version: str = None):
        version = version or datetime.now().strftime("%Y%m%d%H%M%S")
        data_path = os.path.join(self.directory, f"{name}_{version}.pkl")
        meta_path = os.path.join(self.directory, f"{name}_{version}_meta.json")
        return data_path, meta_path

    def save(self, df: pd.DataFrame, name: str, transformers: list, dm: DataManager = None, version: str = None,
             notes: str = ""):
        data_path, meta_path = self._get_paths(name, version)

        # Save dataframe
        df.to_pickle(data_path)

        # Save metadata
        metadata = {
            "name": name,
            "version": version or datetime.now().strftime("%Y%m%d%H%M%S"),
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "transformers": [t.name for t in transformers],
            "created_at": datetime.now().isoformat(),
            "notes": notes,
        }
        if dm:
            metadata.update({
                "read_size": dm.read_size,
                "sample_size": dm.sample_size,
                "test_size": dm.test_size,
                "features": list(dm.features),
            })

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return data_path, meta_path

    def load(self, data_path: str, meta_path: str = None):
        df = pd.read_pickle(data_path)
        metadata = None
        if meta_path and os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)
        return df, metadata


class FittedModelStore:
    """
    Stores fitted models with metadata.
    Storage format: Pickle
    """

    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)

    def _get_paths(self, name: str, version: str = None):
        version = version or datetime.now().strftime("%Y%m%d%H%M%S")
        model_path = os.path.join(self.directory, f"{name}_{version}.pkl")
        meta_path = os.path.join(self.directory, f"{name}_{version}_meta.json")
        return model_path, meta_path

    def save(self, mm: ModelManager, name: str, version: str = None, notes: str = ""):
        model_path, meta_path = self._get_paths(name, version)

        # Save the model object
        with open(model_path, "wb") as f:
            pickle.dump(mm.model, f)

        # Save metadata
        metadata = {
            "name": name,
            "version": version or datetime.now().strftime("%Y%m%d%H%M%S"),
            "model_class": mm.model.__class__.__name__,
            "parameters": getattr(mm.model, "get_params", lambda: {})(),
            "training_dataset": getattr(mm.dm, "filepath", None),
            "features": list(mm.dm.features) if mm.dm else [],
            "transformers": [t.name for t in mm.dm.transformers] if mm.dm and mm.dm.transformers else [],
            "created_at": datetime.now().isoformat(),
            "notes": notes,
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return model_path, meta_path

    def load(self, model_path: str, meta_path: str = None):
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        metadata = None
        if meta_path and os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)

        return model, metadata
