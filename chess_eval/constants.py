import chess
from IPython.core.display import Math

DATASET_FILE = "../data/raw/chessData.csv"

FEN = "FEN"
EVAL = "Evaluation"

READ_SIZE = 1_000  # None
SAMPLE_SIZE = 100  # 100_000

RANDOM_STATE = 99

EXAMPLE_GAME_ID = 50
PARTS = lambda fen: [rf"\underbrace{{ \text{{{field}}} }}_{{ {label} }}" for field, label in
                     zip(fen.split(" "), ["A", "B", "C", "D", "E", "F"])]
FEN_LATEX = lambda fen: Math(r" \quad ".join(PARTS(fen)))

PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0
}

CENTRAL_SQUARES = {
    chess.E4, chess.D4,
    chess.E5, chess.D5
}
