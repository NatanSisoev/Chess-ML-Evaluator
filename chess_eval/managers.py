import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from chess_eval.constants import *


def clean(
        df: pd.DataFrame, mode: str = "remove", threshold: int = EVAL_THRESHOLD
) -> pd.DataFrame:
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
    """
    Handles loading, cleaning, sampling, feature transformation, and train-test splitting
    for a chess evaluation dataset.

    Transformers are applied on-demand; only missing features are computed.
    """

    def __init__(
            self,
            filepath: str = DATASET_FILE,
            read_size: int = READ_SIZE,
            sample_size: int = SAMPLE_SIZE,
            test_size: float = TEST_SIZE,
            random_state: int = RANDOM_STATE,
            cleaner=clean,
            transformers: list = None,
    ):
        self.filepath = filepath
        self.read_size = read_size
        self.sample_size = sample_size
        self.test_size = test_size
        self.random_state = random_state
        self.cleaner = cleaner
        self.transformers = transformers or []

        # Load and clean dataset
        self.df_all = self.cleaner(pd.read_csv(filepath, nrows=read_size))

        # Sampled indices and train/test indices
        self.sample_idx = None
        self.train_idx = None
        self.test_idx = None

        # Sample and optionally apply transformers
        self.sample()
        if self.transformers:
            self.apply_transformers()
            self.train_test_split()

    @property
    def df(self):
        """Return the sampled dataframe (subset of df_all)."""
        return self.df_all.loc[self.sample_idx].reset_index(drop=True)

    @property
    def X(self):
        return self.df.drop(columns=[EVAL, FEN])

    @property
    def y(self):
        return self.df[EVAL]

    @property
    def X_train(self):
        return self.X.iloc[self.train_idx]

    @property
    def X_test(self):
        return self.X.iloc[self.test_idx]

    @property
    def y_train(self):
        return self.y.iloc[self.train_idx]

    @property
    def y_test(self):
        return self.y.iloc[self.test_idx]

    def sample(self, sample_size: int = None) -> pd.DataFrame:
        """
        Sample `sample_size` random rows from the cleaned dataset.
        Updates X (features) and y (target) accordingly.
        """
        if sample_size:
            self.sample_size = sample_size
        self.sample_idx = self.df_all.sample(
            n=self.sample_size, random_state=self.random_state
        ).index.to_numpy()
        return self.df

    def apply_transformers(self, transformers=None) -> pd.DataFrame:
        """
        Apply a list of feature transformers to the dataset.
        Only applies features that are not already present.
        Updates X and train-test splits.
        """
        if transformers is None:
            transformers = []
        self.transformers.extend(transformers)

        with tqdm(
                self.transformers, desc="Applying transformations", leave=False
        ) as pbar:
            for transformer in pbar:
                if transformer.features - set(self.X.columns):
                    pbar.set_postfix({"transformer": transformer.name})
                    df_transformed = transformer.transform(self.df)
                    self.df_all.loc[self.sample_idx, df_transformed.columns] = (
                        df_transformed.values
                    )

        return self.X

    def train_test_split(
            self, features: list = None, test_size: float = None, random_state: int = None
    ):
        """
        Split the dataset into train and test sets.
        Defaults to class attributes if parameters are not provided.
        """
        self.train_idx, self.test_idx = train_test_split(
            np.arange(self.sample_size),
            test_size=self.test_size,
            random_state=self.random_state,
        )
        return self.X_train, self.X_test, self.y_train, self.y_test


class ModelManager:
    def __init__(self, model, dm: DataManager):
        self.model = model
        self.dm = dm
        self.y_pred = None

    def fit(self):
        self.model.fit(self.dm.X_train, self.dm.y_train)

    def predict(self, X: pd.DataFrame = None):
        self.y_pred = self.model.predict(X or self.dm.X_test)
        return self.y_pred

    def predict_fen(self, fen: str):
        df_fen = pd.DataFrame({FEN: [fen]})

        for transformer in getattr(self.dm, "transformers", []):
            df_fen = transformer.transform(df_fen)

        X_new = df_fen.drop(FEN, axis=1, errors="ignore")
        missing_cols = set(self.dm.X_train.columns) - set(X_new.columns)
        for col in missing_cols:
            X_new[col] = 0

        X_new = X_new[self.dm.X_train.columns]

        return self.model.predict(X_new)


class MetricsManager:
    def __init__(self, mm, plots_dir=PLOTS_DIR):
        self.mm = mm

        self.plots_dir = plots_dir
        self.plots_dir.mkdir(exist_ok=True)

        plt.style.use("ggplot")
        plt.rcParams["text.usetex"] = True
        plt.rcParams["text.color"] = "black"
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelcolor"] = "black"
        plt.rcParams["xtick.color"] = "black"
        plt.rcParams["ytick.color"] = "black"

    # EVALUATIONS

    @property
    def y_true(self):
        return self.mm.dm.y_test

    @property
    def y_pred(self):
        return self.mm.y_pred

    # METRICS

    def sign_accuracy(self):
        return np.mean(np.sign(self.y_true) == np.sign(self.y_pred))

    def sign_recall(self, player="white"):
        true_sign = np.sign(self.y_true)
        pred_sign = np.sign(self.y_pred)

        if player == "white":
            mask = true_sign > 0
        elif player == "black":
            mask = true_sign < 0
        elif player == "draw":
            mask = true_sign == 0
        else:
            raise ValueError("player must be white/black/draw")

        if mask.sum() == 0:
            return np.nan

        return np.mean(pred_sign[mask] == true_sign[mask])

    def centipawn_accuracy(self, tol=200):
        return np.mean(np.abs(self.y_true - self.y_pred) <= tol)

    def r2(self):
        return r2_score(self.y_true, self.y_pred)

    def r2_adjusted(self):
        n = len(self.y_true)
        p = self.mm.dm.X_test.shape[1]
        r2 = self.r2()
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)

    def spearman_rank_correlation(self):
        return spearmanr(self.y_true, self.y_pred)[0]

    def compute_metrics(self, tol=200):
        return {
            "mse": mean_squared_error(self.y_true, self.y_pred),
            "sign_accuracy": self.sign_accuracy(),
            "white_recall": self.sign_recall("white"),
            "black_recall": self.sign_recall("black"),
            "draw_recall": self.sign_recall("draw"),
            f"acc_{tol}_tol": self.centipawn_accuracy(tol),
            "r2": self.r2(),
            "r2_adjusted": self.r2_adjusted(),
            "spearman": self.spearman_rank_correlation(),
        }

    # PLOTS

    def plot_scatter(self, save=False):
        plt.scatter(self.y_true, self.y_pred, alpha=0.5)
        plt.xlim(-EVAL_THRESHOLD, EVAL_THRESHOLD)
        plt.ylim(-EVAL_THRESHOLD, EVAL_THRESHOLD)
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title("True vs Predicted Evaluation")

        if save:
            fname = f"scatter_{time.strftime("%Y%m%d_%H%M%S")}.png"
            fpath = self.plots_dir / fname
            plt.savefig(fpath, dpi=FIG_DPI)

    def plot_tol_acc(self, max_tol=EVAL_THRESHOLD, save=False):
        tols = np.linspace(0, max_tol, max_tol)
        accs = [self.centipawn_accuracy(tol) for tol in tols]
        auc = np.trapezoid(accs, tols) / max_tol

        plt.plot(tols, accs)
        plt.title(f"Tolerance-Accuracy Curve (AUC={auc:.3f})")
        plt.xlabel("Tolerance")
        plt.ylabel("Accuracy")

        if save:
            fname = f"tol_acc_{time.strftime("%Y%m%d_%H%M%S")}.png"
            fpath = self.plots_dir / fname
            plt.savefig(fpath, dpi=FIG_DPI)
