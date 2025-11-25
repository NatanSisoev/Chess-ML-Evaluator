import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from chess_eval.constants import *


class EvaluationCleaner:
    name = "Evaluation Cleaner"
    features = []

    @staticmethod
    def clean(df: pd.DataFrame, mode: str = "remove", threshold: int = 1000) -> pd.DataFrame:
        df = df[~df[EVAL].astype(str).str.contains("#")]
        df.loc[:, EVAL] = pd.to_numeric(df[EVAL], errors="coerce").astype(int)
        if mode == "clip":
            df.loc[:, EVAL] = np.clip(df[EVAL], -threshold, threshold)
        elif mode == "remove":
            df = df[df[EVAL].abs() <= threshold]
        else:
            raise ValueError("mode must be 'clip' or 'remove'")
        return df


class DataManager:
    def __init__(self, filepath: str, read_size: int = 5_000_000, sample_size: int = 100_000, test_size: float = 0.2,
                 random_state: int = 99, features: list = None, cleaner=EvaluationCleaner.clean,
                 transformers: list = None):
        self.filepath = filepath
        self.read_size = read_size
        self.sample_size = sample_size
        self.test_size = test_size
        self.random_state = random_state
        self.features = features or set()
        self.cleaner = cleaner
        self.transformers = transformers or []  # each element is a class with a 'transform' method and; a 'name','features' and 'methods' attributes corresponding to the transformer identification.

        assert sample_size <= read_size, f"Sample size ({sample_size}) must be smaller than read size ({read_size})."

        self.df_all = pd.read_csv(filepath, nrows=read_size)
        self.df = None

        self.X = None
        self.y = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.df_all = self.cleaner(self.df_all)
        self.sample()

        if len(self.transformers) > 0:
            self.apply_transformers()
            self.train_test_split()

    def sample(self, sample_size: int = None):
        """Sample random rows."""
        if sample_size is not None:
            self.sample_size = sample_size
        self.df = self.df_all.sample(n=self.sample_size, random_state=self.random_state).reset_index(drop=True)
        self.X = self.df[[FEN]]
        self.y = self.df[EVAL]
        return self.df

    def apply_transformers(self, transformers: list = None):
        """Apply transformers to dataframe."""
        if transformers is not None:
            self.transformers = transformers
        elif self.transformers is None:
            return self.X
        with tqdm(self.transformers, desc="Applying transformations", leave=False) as pbar:
            for transformer in pbar:
                if not transformer.features.issubset(self.features):
                    pbar.set_postfix({"transformer": transformer.name})
                    self.X = transformer.transform(self.X)
                    self.features.update(transformer.features)
        self.train_test_split()
        return self.X

    def train_test_split(self, features: list = None, test_size: float = None, random_state: int = None):
        """First sample then split."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X[list(features or self.features)],
            self.y,
            test_size=test_size or self.test_size,
            random_state=random_state or self.random_state
        )

        return self.X_train, self.X_test, self.y_train, self.y_test


class ChessManager:
    def __init__(self, dm: DataManager):
        self.dm = dm

    def show_board(self, idx: int):
        fen = self.dm.df.loc[idx, FEN]
        display(chess.Board(fen))

    def show_features(self, idx: int):
        display(self.dm.df.loc[idx])

    def show(self, idx: int):
        self.show_features(idx)
        self.show_board(idx)


class ModelManager:
    def __init__(self, model, dm: DataManager):
        self.model = model
        self.dm = dm
        self.y_pred = None

    def fit(self):
        self.model.fit(self.dm.X_train, self.dm.y_train)

    def predict(self):
        self.y_pred = self.model.predict(self.dm.X_test)
        return self.y_pred

    def predict_fen(self, fen: str):
        df_fen = pd.DataFrame({FEN: [fen]})
        if self.dm.transformers:
            for transformer in self.dm.transformers:
                df_fen = transformer.transform(df_fen)

        # Prepare X_new with the same columns as X_train
        X_new = df_fen.drop(FEN, axis=1)
        for col in self.dm.X_train.columns:
            if col not in X_new.columns:
                X_new[col] = 0  # Add the missing column with default value 0
        X_new = X_new[self.dm.X_train.columns]  # rearrange columns to match

        return self.model.predict(X_new)


class MetricsManager:
    def __init__(self, mm: ModelManager):
        self.mm = mm

    @property
    def y_true(self):
        return self.mm.dm.y_test

    @property
    def y_pred(self):
        return self.mm.y_pred

    def sign_accuracy(self):
        true_sign = np.sign(self.y_true)
        pred_sign = np.sign(self.y_pred)
        return np.mean(true_sign == pred_sign)

    def sign_recall(self, player: str = "white"):
        true_sign = np.sign(self.y_true)
        pred_sign = np.sign(self.y_pred)

        if player == "white":
            mask = true_sign > 0
        elif player == "black":
            mask = true_sign < 0
        elif player == "draw":
            mask = true_sign == 0
        else:
            raise ValueError("player must be 'white' or 'black'")

        if mask.sum() == 0:
            return np.nan

        return np.sum(pred_sign[mask] == true_sign[mask]) / np.sum(mask)

    def centipawn_accuracy(self, tol=200):
        return np.mean(np.abs(self.y_true - self.y_pred) <= tol)

    def r2(self):
        return r2_score(self.y_true, self.y_pred)

    def r2_adjusted(self):
        n = len(self.y_true)
        p = self.mm.dm.X_test.shape[1]
        r2 = r2_score(self.y_true, self.y_pred)
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)

    def spearman_rank_correlation(self):
        return spearmanr(self.y_true, self.y_pred)[0]

    def report(self, tol=200):
        print("MSE:", mean_squared_error(self.y_true, self.y_pred))
        print("Sign accuracy:", self.sign_accuracy())
        print("White winning recall:", self.sign_recall("white"))
        print("Black winning recall:", self.sign_recall("black"))
        print("Draw recall:", self.sign_recall("draw"))
        print(f"Centipawn ±{tol} accuracy:", self.centipawn_accuracy(tol))
        print("R2:", self.r2())
        print("R2 adjusted:", self.r2_adjusted())
        print("Spearman rank correlation:", self.spearman_rank_correlation())

    def plot_scatter(self):
        plt.figure(figsize=(10, 8))
        plt.scatter(self.y_true, self.y_pred, alpha=0.5)
        plt.xlim(-1000, 1000)
        plt.ylim(-1000, 1000)
        plt.xlabel("True Eval")
        plt.ylabel("Predicted Eval")
        plt.show()

    def plot_eval(self, f1, f2):
        w_vals = np.linspace(self.mm.dm.df[f1].min(), self.mm.dm.df[f1].max(), 100)
        b_vals = np.linspace(self.mm.dm.df[f2].min(), self.mm.dm.df[f2].max(), 100)
        W, B = np.meshgrid(w_vals, b_vals)

        grid_df = pd.DataFrame({f1: W.ravel(), f2: B.ravel()})
        Z = self.mm.model.predict(grid_df).reshape(W.shape)

        plt.figure(figsize=(8, 10))
        contour = plt.contourf(W, B, Z, levels=50, cmap="viridis")
        plt.colorbar(contour, label="Predicted Evaluation (centipawns)")

        zero_line = plt.contour(W, B, Z, levels=[0], colors="white")
        plt.clabel(zero_line, fmt="0", fontsize=10)

        plt.xlabel(f1)
        plt.ylabel(f2)
        plt.title("Predicted Evaluation")

        plt.show()

    def plot_tol_acc(self, max_tol=1000):
        tols = np.linspace(0, max_tol, 1000)
        accs = [self.centipawn_accuracy(tol) for tol in tols]

        auc = np.trapezoid(accs, tols) / max_tol

        plt.figure(figsize=(10, 8))
        plt.plot(tols, accs)
        plt.title(f"Tolerance-Accuracy Curve (AUC = {auc:.2f})")
        plt.xlabel("Centipawn Tolerance")
        plt.ylabel("Accuracy")
        plt.show()
