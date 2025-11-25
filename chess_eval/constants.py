import chess

DATASET_FILE = "../data/raw/chessData.csv"
DATASET_SIZE = 10_000
FEN = "FEN"
EVAL = "Evaluation"
RANDOM_STATE = 99
PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0
}

CENTRAL_SQUARES = {chess.E4, chess.D4, chess.E5, chess.D5}
