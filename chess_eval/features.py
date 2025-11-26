import abc

import pandas as pd

from chess_eval.constants import *


class FeatureBundle(abc.ABC):
    name: str
    features: set

    @staticmethod
    @abc.abstractmethod
    def compute(board: chess.Board) -> dict:
        pass

    @classmethod
    def transform(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.drop(columns=cls.features, inplace=True, errors="ignore")
        results = [cls.compute(chess.Board(fen)) for fen in df["FEN"].values]
        return pd.concat([df, pd.DataFrame(results, index=df.index)], axis=1)


# -------------------- Piece Info --------------------

class PieceInfo(FeatureBundle):
    name = "Piece Information"
    features = {
        "White_Pieces", "Black_Pieces",
        "White_Piece_Value", "Black_Piece_Value",
        "White_Pawn", "Black_Pawn",
        "White_Knight", "Black_Knight",
        "White_Bishop", "Black_Bishop",
        "White_Rook", "Black_Rook",
        "White_Queen", "Black_Queen",
        "White_Central_Pieces", "Black_Central_Pieces",
    }

    @staticmethod
    def compute(board):
        piece_map = board.piece_map()

        w_total = b_total = 0
        w_value = b_value = 0.0
        w_central = b_central = 0
        counts = {f"{c}_{t}": 0 for c in ("White", "Black")
                  for t in ("Pawn", "Knight", "Bishop", "Rook", "Queen")}

        for sq, pc in piece_map.items():
            color = pc.color
            ptype = pc.piece_type

            if color:
                w_total += 1
                w_value += PIECE_VALUES[ptype]
            else:
                b_total += 1
                b_value += PIECE_VALUES[ptype]

            if ptype != chess.KING:
                key = f"{'White' if color else 'Black'}_{chess.piece_name(ptype).capitalize()}"
                counts[key] += 1

            if sq in CENTRAL_SQUARES:
                if color:
                    w_central += 1
                else:
                    b_central += 1

        out = {
            "White_Pieces": w_total,
            "Black_Pieces": b_total,
            "White_Piece_Value": w_value,
            "Black_Piece_Value": b_value,
            "White_Central_Pieces": w_central,
            "Black_Central_Pieces": b_central,
        }
        out.update(counts)
        return out


# -------------------- Pawn Structure --------------------

class PawnStructure(FeatureBundle):
    name = "Pawn Structure"
    features = {
        "Isolated_Pawns_White", "Isolated_Pawns_Black",
        "Doubled_Pawns_White", "Doubled_Pawns_Black",
        "Passed_Pawns_White", "Passed_Pawns_Black",
        "Pawn_Islands_White", "Pawn_Islands_Black"
    }

    @staticmethod
    def fen_to_file_counts(board):
        w = [0] * 8
        b = [0] * 8
        for sq, p in board.piece_map().items():
            if p.piece_type == chess.PAWN:
                if p.color:
                    w[chess.square_file(sq)] += 1
                else:
                    b[chess.square_file(sq)] += 1
        return w, b

    @staticmethod
    def isolated(counts):
        return sum(c for i, c in enumerate(counts) if c > 0 and
                   (counts[i - 1] if i > 0 else 0) == 0 and
                   (counts[i + 1] if i < 7 else 0) == 0)

    @staticmethod
    def doubled(counts):
        return sum(c - 1 for c in counts if c > 1)

    @staticmethod
    def passed(own, opp):
        total = 0
        for i, c in enumerate(own):
            if c == 0:
                continue
            front = [i]
            if i > 0:
                front.append(i - 1)
            if i < 7:
                front.append(i + 1)
            if all(opp[f] == 0 for f in front):
                total += c
        return total

    @staticmethod
    def pawn_islands(board, color):
        files = sorted({chess.square_file(sq) for sq, p in board.piece_map().items() if
                        p.piece_type == chess.PAWN and p.color == color})
        if not files:
            return 0
        islands = 1
        for i in range(1, len(files)):
            if files[i] != files[i - 1] + 1:
                islands += 1
        return islands

    @classmethod
    def compute(cls, board):
        w, b = cls.fen_to_file_counts(board)
        return {
            "Isolated_Pawns_White": cls.isolated(w),
            "Isolated_Pawns_Black": cls.isolated(b),
            "Doubled_Pawns_White": cls.doubled(w),
            "Doubled_Pawns_Black": cls.doubled(b),
            "Passed_Pawns_White": cls.passed(w, b),
            "Passed_Pawns_Black": cls.passed(b, w),
            "Pawn_Islands_White": cls.pawn_islands(board, chess.WHITE),
            "Pawn_Islands_Black": cls.pawn_islands(board, chess.BLACK)
        }


# -------------------- King Safety --------------------

class KingSafety(FeatureBundle):
    name = "King Safety"
    features = {
        "King_Castled_White", "King_Castled_Black",
        "King_Pawns_White", "King_Pawns_Black",
        "Open_Files",
        "King_Distance_to_Last_Rank_White", "King_Distance_to_Last_Rank_Black",
        "King_Castling_Rights_White", "King_Castling_Rights_Black",
        "King_Front_Pawns_White", "King_Front_Pawns_Black",
        "King_In_Check_White", "King_In_Check_Black",
        "King_Attacked_Neighbours_White", "King_Attacked_Neighbours_Black"
    }

    @staticmethod
    def king_castled(board, color):
        k = board.king(color)
        if color == chess.WHITE:
            return int((k == chess.G1 and board.piece_at(chess.H1) == chess.Piece(chess.ROOK, chess.WHITE))
                       or (k == chess.C1 and board.piece_at(chess.A1) == chess.Piece(chess.ROOK, chess.WHITE)))
        return int((k == chess.G8 and board.piece_at(chess.H8) == chess.Piece(chess.ROOK, chess.BLACK))
                   or (k == chess.C8 and board.piece_at(chess.A8) == chess.Piece(chess.ROOK, chess.BLACK)))

    @staticmethod
    def king_pawns(board, color):
        sq = board.king(color)
        sf, sr = chess.square_file(sq), chess.square_rank(sq)
        cnt = 0
        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if df == 0 and dr == 0: continue
                f, r = sf + df, sr + dr
                if 0 <= f <= 7 and 0 <= r <= 7:
                    p = board.piece_at(chess.square(f, r))
                    if p and p.piece_type == chess.PAWN and p.color == color:
                        cnt += 1
        return cnt

    @staticmethod
    def open_files(board):
        return sum(1 for f in range(8)
                   if
                   not any(board.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, chess.WHITE) for r in range(8)
                           ) or not any(
                       board.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, chess.BLACK) for r in range(8)))

    @staticmethod
    def king_distance(board, color):
        sq = board.king(color)
        r = chess.square_rank(sq)
        return (7 - r) if color == chess.WHITE else r

    @staticmethod
    def king_castling_rights(board, color):
        return int(board.has_kingside_castling_rights(color) or board.has_queenside_castling_rights(color))

    @staticmethod
    def king_front_pawns(board, color):
        sq = board.king(color)
        sf, sr = chess.square_file(sq), chess.square_rank(sq)
        dr = 1 if color == chess.WHITE else -1
        cnt = 0
        for df in (-1, 0, 1):
            f = sf + df
            r = sr + dr
            if 0 <= f <= 7 and 0 <= r <= 7:
                p = board.piece_at(chess.square(f, r))
                if p and p.piece_type == chess.PAWN and p.color == color:
                    cnt += 1
        return cnt

    @staticmethod
    def king_in_check(board, color):
        return int(board.is_check() and board.turn == color)

    @staticmethod
    def king_attacked_neighbours(board, color):
        sq = board.king(color)
        sf, sr = chess.square_file(sq), chess.square_rank(sq)
        enemy = not color
        cnt = 0
        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if df == 0 and dr == 0: continue
                f, r = sf + df, sr + dr
                if 0 <= f <= 7 and 0 <= r <= 7 and board.is_attacked_by(enemy, chess.square(f, r)):
                    cnt += 1
        return cnt

    @classmethod
    def compute(cls, board):
        return {
            "King_Castled_White": cls.king_castled(board, chess.WHITE),
            "King_Castled_Black": cls.king_castled(board, chess.BLACK),
            "King_Pawns_White": cls.king_pawns(board, chess.WHITE),
            "King_Pawns_Black": cls.king_pawns(board, chess.BLACK),
            "Open_Files": cls.open_files(board),
            "King_Distance_to_Last_Rank_White": cls.king_distance(board, chess.WHITE),
            "King_Distance_to_Last_Rank_Black": cls.king_distance(board, chess.BLACK),
            "King_Castling_Rights_White": cls.king_castling_rights(board, chess.WHITE),
            "King_Castling_Rights_Black": cls.king_castling_rights(board, chess.BLACK),
            "King_Front_Pawns_White": cls.king_front_pawns(board, chess.WHITE),
            "King_Front_Pawns_Black": cls.king_front_pawns(board, chess.BLACK),
            "King_In_Check_White": cls.king_in_check(board, chess.WHITE),
            "King_In_Check_Black": cls.king_in_check(board, chess.BLACK),
            "King_Attacked_Neighbours_White": cls.king_attacked_neighbours(board, chess.WHITE),
            "King_Attacked_Neighbours_Black": cls.king_attacked_neighbours(board, chess.BLACK),
        }


# -------------------- Mobility --------------------

class Mobility(FeatureBundle):
    name = "Mobility"
    features = {"Mobility_White", "Mobility_Black", "Mobility_To_Move"}

    @staticmethod
    def mobility_side(board, color):
        b = board.copy()
        b.turn = color
        return len(list(b.legal_moves))

    @classmethod
    def compute(cls, board):
        return {
            "Mobility_White": cls.mobility_side(board, chess.WHITE),
            "Mobility_Black": cls.mobility_side(board, chess.BLACK),
            "Mobility_To_Move": len(list(board.legal_moves))
        }


# -------------------- Attackers --------------------

class Attackers(FeatureBundle):
    name = "Attackers"
    features = {"White_King_Attackers", "Black_King_Attackers", "White_King_Zone_Attackers",
                "Black_King_Zone_Attackers"}

    @staticmethod
    def king_attackers(board, color):
        k = board.king(color)
        if k is None: return 0
        return len(board.attackers(not color, k))

    @staticmethod
    def king_zone_attackers(board, color):
        k = board.king(color)
        if k is None: return 0
        zone = [k] + [sq for sq in chess.SQUARES if chess.square_distance(sq, k) == 1]
        return sum(len(board.attackers(not color, sq)) for sq in zone)

    @classmethod
    def compute(cls, board):
        return {
            "White_King_Attackers": cls.king_attackers(board, chess.WHITE),
            "Black_King_Attackers": cls.king_attackers(board, chess.BLACK),
            "White_King_Zone_Attackers": cls.king_zone_attackers(board, chess.WHITE),
            "Black_King_Zone_Attackers": cls.king_zone_attackers(board, chess.BLACK),
        }


# -------------------- Positional Control --------------------

class PositionalControl(FeatureBundle):
    name = "Positional Control"
    features = {"Center_Control_White", "Center_Control_Black"}

    @staticmethod
    def center_control(board, color):
        return sum(1 for sq in CENTRAL_SQUARES if board.is_attacked_by(color, sq))

    @classmethod
    def compute(cls, board):
        return {
            "Center_Control_White": cls.center_control(board, chess.WHITE),
            "Center_Control_Black": cls.center_control(board, chess.BLACK)
        }


# -------------------- Game Info --------------------

class GameInfo(FeatureBundle):
    name = "Game Info"
    features = {"Halfmove_Clock", "Fullmove_Number", "Phase", "Side_To_Move_White", "Has_En_Passant"}

    @staticmethod
    def phase(board):
        total = sum(PIECE_VALUES[pc.piece_type] for pc in board.piece_map().values())
        initial = 1 * 16 + 3 * 4 + 3 * 4 + 5 * 4 + 9 * 2
        return total / initial

    @classmethod
    def compute(cls, board):
        return {
            "Halfmove_Clock": board.halfmove_clock,
            "Fullmove_Number": board.fullmove_number,
            "Phase": cls.phase(board),
            "Side_To_Move_White": int(board.turn),
            "Has_En_Passant": int(board.ep_square is not None)
        }


FEATURE_TRANSFORMERS = [
    PieceInfo,
    PawnStructure,
    KingSafety,
    Mobility,
    Attackers,
    PositionalControl,
    GameInfo
]
