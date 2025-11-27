import abc

from chess_eval.config import *


###################### Feature Bundle ######################


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
        df.drop(columns=list(cls.features), inplace=True, errors="ignore")
        results = [cls.compute(chess.Board(fen)) for fen in df[FEN].values]
        return pd.concat([df, pd.DataFrame(results, index=df.index)], axis=1)


######################## Piece Info ########################


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


###################### Pawn Structure ######################


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


####################### King Safety ########################


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


######################### Mobility #########################


class Mobility(FeatureBundle):
    name = "Piece Mobility"
    features = {
        "Mobility_King_White", "Mobility_Queen_White", "Mobility_Rook_White",
        "Mobility_Bishop_White", "Mobility_Knight_White", "Mobility_Pawn_White",
        "Mobility_King_Black", "Mobility_Queen_Black", "Mobility_Rook_Black",
        "Mobility_Bishop_Black", "Mobility_Knight_Black", "Mobility_Pawn_Black"
    }

    @staticmethod
    def compute(board):
        # Pre-index legal moves by origin square (8x8 → fast lookup)
        move_map = {sq: 0 for sq in chess.SQUARES}
        for mv in board.legal_moves:
            move_map[mv.from_square] += 1

        out = {}
        pieces = [
            (chess.KING, "King"),
            (chess.QUEEN, "Queen"),
            (chess.ROOK, "Rook"),
            (chess.BISHOP, "Bishop"),
            (chess.KNIGHT, "Knight"),
            (chess.PAWN, "Pawn")
        ]

        for color, cname in [(chess.WHITE, "White"), (chess.BLACK, "Black")]:
            for ptype, pname in pieces:
                count = 0
                for sq in board.pieces(ptype, color):
                    count += move_map[sq]
                out[f"Mobility_{pname}_{cname}"] = count

        return out


########################## Attack ##########################


class Attack(FeatureBundle):
    name = "Relations"
    features = {
        "Threats_Created_White", "Threats_Created_Black",
        "Hanging_Pieces_White", "Hanging_Pieces_Black",
        "Hanging_Points_White", "Hanging_Points_Black",
        "Undefended_Pieces_White", "Undefended_Pieces_Black",
        "Undefended_Points_White", "Undefended_Points_Black"
    }

    @staticmethod
    def compute(board):
        out = {}

        for color, cname in [(chess.WHITE, "White"), (chess.BLACK, "Black")]:
            enemy = not color

            # THREATS: enemy pieces attacked by color
            threats = 0
            for ptype in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
                for sq in board.pieces(ptype, enemy):
                    if board.is_attacked_by(color, sq):
                        threats += 1
            out[f"Threats_Created_{cname}"] = threats

            # HANGING
            h_count = 0
            h_pts = 0
            u_count = 0
            u_pts = 0

            for ptype in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
                for sq in board.pieces(ptype, color):
                    defended = board.is_attacked_by(color, sq)
                    attacked = board.is_attacked_by(enemy, sq)

                    if attacked and not defended:
                        h_count += 1
                        h_pts += PIECE_VALUES[ptype]

                    if not defended:
                        u_count += 1
                        u_pts += PIECE_VALUES[ptype]

            out[f"Hanging_Pieces_{cname}"] = h_count
            out[f"Hanging_Points_{cname}"] = h_pts
            out[f"Undefended_Pieces_{cname}"] = u_count
            out[f"Undefended_Points_{cname}"] = u_pts

        return out


###################### Board Control #######################


class BoardControl(FeatureBundle):
    name = "Board Control"
    features = {
        "Central_Squares_Control_White", "Central_Squares_Control_Black",
        "Open_Columns_White", "Open_Columns_Black",
        "SemiOpen_Columns_White", "SemiOpen_Columns_Black",
        "Rook_on_Open_Column_White", "Rook_on_Open_Column_Black",
        "Protected_Advanced_Pawn_White", "Protected_Advanced_Pawn_Black",
        "Rook_Queen_Aligned_White", "Rook_Queen_Aligned_Black",
        "Rooks_Aligned_White", "Rooks_Aligned_Black",
        "Controlled_Squares_White", "Controlled_Squares_Black"
    }

    central = [chess.D4, chess.D5, chess.E4, chess.E5]

    @staticmethod
    def _open_columns(board):
        cols = []
        for f in range(8):
            has_w = any(board.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, chess.WHITE) for r in range(8))
            has_b = any(board.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, chess.BLACK) for r in range(8))
            cols.append((has_w, has_b))
        return cols

    @staticmethod
    def _aligned(board, color):
        rooks = list(board.pieces(chess.ROOK, color))
        queens = list(board.pieces(chess.QUEEN, color))

        rq = 0
        rr = 0

        # rook + queen
        for r in rooks:
            rf = chess.square_file(r)
            rrk = chess.square_rank(r)
            for q in queens:
                if rf == chess.square_file(q) or rrk == chess.square_rank(q):
                    rq = 1

        # rook + rook
        for i in range(len(rooks)):
            r1 = rooks[i]
            f1 = chess.square_file(r1)
            r1r = chess.square_rank(r1)
            for j in range(i + 1, len(rooks)):
                r2 = rooks[j]
                if f1 == chess.square_file(r2) or r1r == chess.square_rank(r2):
                    rr = 1

        return rq, rr

    @staticmethod
    def _protected_advanced_pawn(board, color):
        if color == chess.WHITE:
            ranks = range(3, 6)
        else:
            ranks = range(2, 5)

        for r in ranks:
            for f in range(8):
                sq = chess.square(f, r)
                piece = board.piece_at(sq)
                if piece and piece.color == color and piece.piece_type == chess.PAWN:
                    # check if any friendly piece attacks sq (excluding pawns)
                    if any(board.is_attacked_by(color, sq2)
                           for sq2 in board.pieces(chess.KNIGHT, color)
                                      | board.pieces(chess.BISHOP, color)
                                      | board.pieces(chess.ROOK, color)
                                      | board.pieces(chess.QUEEN, color)):
                        return 1
        return 0

    @staticmethod
    def compute(board):
        cols = BoardControl._open_columns(board)
        out = {}

        for color, cname in [(chess.WHITE, "White"), (chess.BLACK, "Black")]:
            # central control
            out[f"Central_Squares_Control_{cname}"] = sum(
                1 for sq in BoardControl.central if board.is_attacked_by(color, sq)
            )

            # open + semi-open
            open_c = 0
            semi_c = 0
            for f, (has_w, has_b) in enumerate(cols):
                if not has_w and not has_b:
                    open_c += 1
                else:
                    if color == chess.WHITE:
                        if not has_w:
                            semi_c += 1
                    else:
                        if not has_b:
                            semi_c += 1

            out[f"Open_Columns_{cname}"] = open_c
            out[f"SemiOpen_Columns_{cname}"] = semi_c

            # rook on open column
            rook_flag = 0
            for sq in board.pieces(chess.ROOK, color):
                f = chess.square_file(sq)
                has_w, has_b = cols[f]
                if not has_w and not has_b:
                    rook_flag = 1
                    break
            out[f"Rook_on_Open_Column_{cname}"] = rook_flag

            # protected advanced pawn
            out[f"Protected_Advanced_Pawn_{cname}"] = BoardControl._protected_advanced_pawn(board, color)

            # alignments
            rq, rr = BoardControl._aligned(board, color)
            out[f"Rook_Queen_Aligned_{cname}"] = rq
            out[f"Rooks_Aligned_{cname}"] = rr

            # total controlled squares
            count_sq = 0
            for sq in chess.SQUARES:
                if board.is_attacked_by(color, sq):
                    count_sq += 1
            out[f"Controlled_Squares_{cname}"] = count_sq

        return out


######################## Game Info #########################


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


#################### Transformers List #####################


FEATURE_TRANSFORMERS = [
    PieceInfo,
    PawnStructure,
    KingSafety,
    Mobility,
    Attack,
    BoardControl,
    GameInfo
]
