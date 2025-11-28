from pathlib import Path

import chess
from lightgbm import LGBMRegressor
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

from tqdm import tqdm  # type: ignore
import numpy as np     # type: ignore
import pandas as pd    # type: ignore
from matplotlib import pyplot as plt

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PLOTS_DIR = ROOT_DIR / "plots"
DATASET_FILE = DATA_DIR / "raw" / "chessData.csv"
SAVED_DATASETS_DIR = DATA_DIR / "features"
SAVED_MODELS_DIR = ROOT_DIR / "models"

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

FEN = "FEN"
EVAL = "Evaluation"

READ_SIZE = 100_000
SAMPLE_SIZE = 1_000
EVAL_THRESHOLD = 1000
FIG_DPI = 300
TEST_SIZE = 0.2
RANDOM_STATE = 99
EXAMPLE_GAME_ID = 50

PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}

CENTRAL_SQUARES = {chess.E4, chess.D4, chess.E5, chess.D5}


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------


def apply_custom_style():
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

# ----------------------------------------------------------------------
# Model registry
# ----------------------------------------------------------------------

MODELS = {
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
