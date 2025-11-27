import os
import json
import pickle
from datetime import datetime
from typing import Tuple

from chess_eval.managers import DataManager, ModelManager


class StorageManager:
    """
    Unified storage manager for:
    - Feature datasets (full DataManager objects)
    - Fitted models (full ModelManager objects)

    Default paths:
        datasets -> "../data/features"
        models   -> "../models"
    """

    def __init__(self,
                 dataset_dir: str = r"..\data\features",
                 model_dir: str = r"..\models"):
        # Directories
        self.dataset_dir = dataset_dir
        self.model_dir = model_dir
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        # Metadata files
        self.dataset_meta_file = os.path.join(self.dataset_dir, "metadata.json")
        self.model_meta_file = os.path.join(self.model_dir, "metadata.json")

        # Initialize metadata files if missing
        for file in [self.dataset_meta_file, self.model_meta_file]:
            if not os.path.exists(file):
                with open(file, "w") as f:
                    json.dump({}, f)

    # -------------------- Dataset Methods --------------------

    def save_dataset(self, dm: DataManager, name: str = None, notes: str = "") -> str:
        name = name or datetime.now().strftime("%Y%m%d%H%M%S")
        file_path = os.path.join(self.dataset_dir, f"{name}.pkl")

        # Pickle the entire DataManager
        with open(file_path, "wb") as f:
            pickle.dump(dm.df, f, protocol=pickle.HIGHEST_PROTOCOL) # Store only df, not the dm

        # Update metadata
        with open(self.dataset_meta_file, "r") as f:
            all_metadata = json.load(f)

        all_metadata[name] = {
            "name": name,
            "path": file_path,
            "read_size": dm.read_size,
            "sample_size": dm.sample_size,
            "test_size": dm.test_size,
            "transformers": [t.name for t in dm.transformers] if dm.transformers else [],
            "random_state": dm.random_state,
            "df_shape": dm.df.shape if dm.df is not None else None,
            "columns": dm.df.columns.tolist() if dm.df is not None else None,
            "created_at": datetime.now().isoformat(),
            "notes": notes,
        }

        with open(self.dataset_meta_file, "w") as f:
            json.dump(all_metadata, f, indent=2)

        return file_path

    def load_dataset(self, name: str) -> Tuple[DataManager, dict]:
        with open(self.dataset_meta_file, "r") as f:
            all_metadata = json.load(f)

        if name not in all_metadata:
            raise KeyError(f"No dataset found for '{name}'")

        meta = all_metadata[name]
        with open(meta["path"], "rb") as f:
            df = pickle.load(f) # Load only df, not the dm

        return df, meta

    def list_datasets(self) -> list[str]:
        with open(self.dataset_meta_file, "r") as f:
            all_metadata = json.load(f)
        return list(all_metadata.keys())

    # -------------------- Model Methods --------------------

    def save_model(self, mm: ModelManager, name: str = None, notes: str = "") -> str:
        name = name or datetime.now().strftime("%Y%m%d%H%M%S")
        file_path = os.path.join(self.model_dir, f"{name}.pkl")

        # Pickle the entire ModelManager
        with open(file_path, "wb") as f:
            pickle.dump(mm.model, f, protocol=pickle.HIGHEST_PROTOCOL) # Store only model, not the mm

        # Update metadata
        with open(self.model_meta_file, "r") as f:
            all_metadata = json.load(f)

        all_metadata[name] = {
            "name": name,
            "path": file_path,
            "model_class": mm.model.__class__.__name__,
            "parameters": getattr(mm.model, "get_params", lambda: {})(),
            "training_dataset": getattr(mm.dm, "filepath", None) if mm.dm else None,
            "features": list(mm.dm.X.columns) if mm.dm else [],
            "transformers": [t.name for t in mm.dm.transformers] if mm.dm and mm.dm.transformers else [],
            "created_at": datetime.now().isoformat(),
            "notes": notes,
        }

        with open(self.model_meta_file, "w") as f:
            json.dump(all_metadata, f, indent=2)

        return file_path

    def load_model(self, name: str) -> Tuple[ModelManager, dict]:
        with open(self.model_meta_file, "r") as f:
            all_metadata = json.load(f)

        if name not in all_metadata:
            raise KeyError(f"No model found for '{name}'")

        meta = all_metadata[name]
        with open(meta["path"], "rb") as f:
            model = pickle.load(f)

        return model, meta

    def list_models(self) -> list[str]:
        with open(self.model_meta_file, "r") as f:
            all_metadata = json.load(f)
        return list(all_metadata.keys())
