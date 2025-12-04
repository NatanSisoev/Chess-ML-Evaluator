from __future__ import annotations

from pathlib import Path
from datetime import datetime  #type: ignore
from typing import Any, Type, Dict, Tuple, Optional, List, Union  #type: ignore
import pickle  #type: ignore

import chess
import numpy as np  #type: ignore
import pandas as pd  #type: ignore
from matplotlib import pyplot as plt
from tqdm import tqdm  #type: ignore

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
PLOTS_DIR: Path = ROOT_DIR / "plots"
DATASET_FILE: Path = DATA_DIR / "raw" / "chessData.csv"
SAVED_DATASETS_DIR: Path = DATA_DIR / "features"
SAVED_MODELS_DIR: Path = ROOT_DIR / "models"

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

FEN: str = "FEN"
EVAL: str = "Evaluation"

READ_SIZE: int = 100_000
SAMPLE_SIZE: int = 1_000
EVAL_THRESHOLD: int = 1000
FIG_DPI: int = 300
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 99
EXAMPLE_GAME_ID: int = 50

PIECE_VALUES: Dict[int, float] = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}

CENTRAL_SQUARES: set[int] = {chess.E4, chess.D4, chess.E5, chess.D5}

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------


def apply_custom_style() -> None:
    plt.style.use("ggplot")
    plt.rcParams.update({
        "text.usetex": True,
        "figure.figsize": (12, 6),
        "font.size": 20,
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
        "font.weight": "bold",
    })


apply_custom_style()


def numbered_path(base: Path) -> Path:
    stem = base.stem
    ext = base.suffix
    parent = base.parent
    candidate = base
    k = 1
    while candidate.exists():
        candidate = parent / f"{stem}_{k}{ext}"
        k += 1
    return candidate

# ----------------------------------------------------------------------
# Model registry
# ----------------------------------------------------------------------

MODELS: Dict[str, Type] = {
    "KNeighborsRegressor": KNeighborsRegressor,
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "ElasticNet": ElasticNet,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
    "ExtraTreesRegressor": ExtraTreesRegressor,
    "XGBRegressor": XGBRegressor,
    "LGBMRegressor": LGBMRegressor,
    "SVR": SVR,
    "MLPRegressor": MLPRegressor,
}

OPTIMAL_MODELS: Dict[str, Type] = {
    "KNeighborsRegressor": KNeighborsRegressor,
    "Ridge": Ridge,
    "XGBRegressor": XGBRegressor,
}

OPTIMAL_PARAMS: Dict[str, Dict[str, Any]] = {
    "Ridge": {
        "alpha": 100,
    },
    "XGBRegressor": {
        "max_depth": 7,
        "n_estimators": 3000,
        "learning_rate": 0.05,
    },
    "KNeighborsRegressor": {
        "n_neighbors": 50,
    },
}

ABBREVIATIONS: Dict[str, str] = {
    "KNeighborsRegressor": "knn",
    "Ridge": "rdg",
    "XGBRegressor": "xgb",
}
