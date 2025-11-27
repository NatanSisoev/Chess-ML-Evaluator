from pathlib import Path

import chess
from IPython.core.display import Math

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PLOTS_DIR = ROOT_DIR / "plots"

DATASET_FILE = DATA_DIR / "raw" / "chessData.csv"

FEN = "FEN"
EVAL = "Evaluation"

READ_SIZE = 100_000  # None
SAMPLE_SIZE = 1_000  # 100_000

EVAL_THRESHOLD = 1000  # 1000
FIG_DPI = 300

TEST_SIZE = 0.2

RANDOM_STATE = 99

EXAMPLE_GAME_ID = 50
PARTS = lambda fen: [
    rf"\underbrace{{ \text{{{field}}} }}_{{ {label} }}"
    for field, label in zip(fen.split(" "), ["A", "B", "C", "D", "E", "F"])
]
FEN_LATEX = lambda fen: r" \quad ".join(PARTS(fen))

PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}

CENTRAL_SQUARES = {chess.E4, chess.D4, chess.E5, chess.D5}


def custom_style():
    import matplotlib.pyplot as plt

    plt.style.use("ggplot")

    plt.rcParams["text.usetex"] = True
    plt.rcParams["figure.figsize"] = (12, 6)

    plt.rcParams["font.size"] = 20
    plt.rcParams["axes.titlesize"] = 22
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16

    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"
    plt.rcParams["text.color"] = "black"
    plt.rcParams["font.weight"] = "bold"

custom_style()

