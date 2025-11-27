import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from chess_eval.constants import *


def clean(df: pd.DataFrame, mode: str = "remove", threshold: int = EVAL_THRESHOLD) -> pd.DataFrame:
    """
    Clean a chess evaluation dataframe by removing forced checkmates
    and clipping/removing extreme evaluation values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with at least the EVAL column.
    mode : str
        'clip' to clip values to ±threshold, 'remove' to drop rows exceeding threshold.
    threshold : int
        Maximum allowed absolute evaluation.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with numeric evaluations only.
    """
    df = df[~df[EVAL].astype(str).str.contains("#")].copy()
    df[EVAL] = pd.to_numeric(df[EVAL], errors="coerce").astype(int)
    if mode == "clip":
        df.loc[:, EVAL] = np.clip(df[EVAL], -threshold, threshold)
    elif mode == "remove":
        df = df[df[EVAL].abs() <= threshold]
    else:
        raise ValueError("mode must be 'clip' or 'remove'")
    return df


class DataManager:
    """
    Handles loading, cleaning, random sampling, feature transformation,
    and train-test splitting for a chess evaluation dataset.

    Transformers are applied on-demand; only missing features are computed.
    """

    def __init__(
        self,
        filepath: str = DATASET_FILE,
        read_size: int | None = READ_SIZE,
        sample_size: int = SAMPLE_SIZE,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
        cleaner=clean,
        transformers: list = None,
    ):
        """
        Parameters
        ----------
        filepath : str
            Path to the CSV dataset.
        read_size : int
            Number of rows to read from CSV.
        sample_size : int
            Number of rows to sample for train/test split.
        test_size : float
            Fraction of sampled rows for test set.
        random_state : int
            Random seed for reproducibility.
        cleaner : Callable
            Function to clean the dataframe.
        transformers : list
            List of feature transformer objects.
        """
        self.filepath = filepath
        self.read_size = read_size
        self.sample_size = sample_size
        self.test_size = test_size
        self.random_state = random_state
        self.cleaner = cleaner if cleaner is not None else lambda x: x
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
        """Return features dataframe (excluding FEN and EVAL)."""
        return self.df.drop(columns=[EVAL, FEN])

    @property
    def y(self):
        """Return target series (EVAL column)."""
        return self.df[EVAL]

    @property
    def X_train(self):
        """Return training features."""
        return self.X.iloc[self.train_idx]

    @property
    def X_test(self):
        """Return testing features."""
        return self.X.iloc[self.test_idx]

    @property
    def y_train(self):
        """Return training targets."""
        return self.y.iloc[self.train_idx]

    @property
    def y_test(self):
        """Return testing targets."""
        return self.y.iloc[self.test_idx]

    def sample(self, sample_size: int = None) -> pd.DataFrame:
        """
        Sample random rows from the cleaned dataset.

        Parameters
        ----------
        sample_size : int, optional
            Number of rows to sample. Uses class default if None.

        Returns
        -------
        pd.DataFrame
            Sampled dataframe.
        """
        if sample_size:
            self.sample_size = sample_size
        self.sample_idx = self.df_all.sample(
            n=self.sample_size, random_state=self.random_state
        ).index.to_numpy()
        return self.df

    def apply_transformers(self, transformers=None) -> pd.DataFrame:
        """
        Apply feature transformers to the dataset.

        Only applies features not already present. Updates X and underlying df_all.

        Parameters
        ----------
        transformers : list, optional
            Additional transformers to apply.

        Returns
        -------
        pd.DataFrame
            Transformed features dataframe (X).
        """
        if transformers is None:
            transformers = []
        self.transformers.extend(transformers)

        with tqdm(self.transformers, desc="Applying transformations", leave=False) as pbar:
            for transformer in pbar:
                if transformer.features - set(self.X.columns):
                    pbar.set_postfix({"transformer": transformer.name})
                    df_transformed = transformer.transform(self.df)
                    self.df_all.loc[self.sample_idx, df_transformed.columns] = df_transformed.values

        return self.X

    def train_test_split(self, features: list = None, test_size: float = None, random_state: int = None):
        """
        Split sampled dataset into training and testing sets.

        Parameters
        ----------
        features : list, optional
            Columns to use for splitting (defaults to all features).
        test_size : float, optional
            Fraction of test set.
        random_state : int, optional
            Random seed.

        Returns
        -------
        tuple
            X_train, X_test, y_train, y_test
        """
        self.train_idx, self.test_idx = train_test_split(
            np.arange(self.sample_size),
            test_size=self.test_size,
            random_state=self.random_state,
        )
        return self.X_train, self.X_test, self.y_train, self.y_test


class ModelManager:
    """
    Wrapper around a scikit-learn model to handle fitting, predicting,
    and predicting from FEN strings.
    """

    def __init__(self, model, dm: DataManager):
        """
        Parameters
        ----------
        model : sklearn-like estimator
            Model with fit() and predict() methods.
        dm : DataManager
            DataManager instance containing train/test splits.
        """
        self.model = model
        self.dm = dm
        self.y_pred = None

    def fit(self):
        """Fit the model on the training dataset."""
        self.model.fit(self.dm.X_train, self.dm.y_train)

    def predict(self, X: pd.DataFrame = None):
        """
        Predict target values for provided features or the test set.

        Parameters
        ----------
        X : pd.DataFrame, optional
            Input features. Defaults to X_test.

        Returns
        -------
        np.ndarray
            Predictions.
        """
        self.y_pred = self.model.predict(X or self.dm.X_test)
        return self.y_pred

    def predict_fen(self, fen: str):
        """
        Predict evaluation for a single FEN string.

        Parameters
        ----------
        fen : str
            Chess position in FEN notation.

        Returns
        -------
        np.ndarray
            Predicted evaluation.
        """
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
    """
    Compute evaluation metrics and provide plotting utilities for predictions.
    """

    def __init__(self, mm, plots_dir=PLOTS_DIR):
        """
        Parameters
        ----------
        mm : ModelManager
            The model manager instance.
        plots_dir : pathlib.Path
            Directory to save plots.
        """
        self.mm = mm
        self.plots_dir = plots_dir
        self.plots_dir.mkdir(exist_ok=True)

        custom_style()

    # --- Properties ---

    @property
    def y_true(self):
        """Return ground truth target values (test set)."""
        return self.mm.dm.y_test

    @property
    def y_pred(self):
        """Return predicted values from the model."""
        return self.mm.y_pred

    # --- Metric Methods ---

    def sign_accuracy(self):
        """Compute fraction of predictions with correct sign."""
        return np.mean(np.sign(self.y_true) == np.sign(self.y_pred))

    def sign_recall(self, player="white"):
        """
        Compute recall of sign predictions for a specific player or draw.

        Parameters
        ----------
        player : str
            'white', 'black', or 'draw'.

        Returns
        -------
        float
            Recall value.
        """
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
        """
        Compute fraction of predictions within ±tol centipawns of true values.

        Parameters
        ----------
        tol : int
            Tolerance in centipawns.

        Returns
        -------
        float
        """
        return np.mean(np.abs(self.y_true - self.y_pred) <= tol)

    def r2(self):
        """Compute R² score."""
        return r2_score(self.y_true, self.y_pred)

    def r2_adjusted(self):
        """Compute adjusted R² score."""
        n = len(self.y_true)
        p = self.mm.dm.X_test.shape[1]
        r2 = self.r2()
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)

    def spearman_rank_correlation(self):
        """Compute Spearman rank correlation between predictions and true values."""
        return spearmanr(self.y_true, self.y_pred)[0]

    def compute_metrics(self, tol=200):
        """
        Compute all relevant evaluation metrics.

        Parameters
        ----------
        tol : int
            Tolerance for centipawn accuracy.

        Returns
        -------
        dict
            Dictionary of metrics including MSE, R², recalls, and Spearman correlation.
        """
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

    # --- Plotting Methods ---

    def plot_scatter(self, save=False):
        """
        Scatter plot of true vs predicted values.

        Parameters
        ----------
        save : bool
            Whether to save the plot as PNG.
        """
        plt.scatter(self.y_true, self.y_pred, alpha=0.5)
        plt.xlim(-EVAL_THRESHOLD, EVAL_THRESHOLD)
        plt.ylim(-EVAL_THRESHOLD, EVAL_THRESHOLD)
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title("True vs Predicted Evaluation")

        if save:
            fname = f"scatter_{time.strftime('%Y%m%d_%H%M%S')}.png"
            fpath = self.plots_dir / fname
            plt.savefig(fpath, dpi=FIG_DPI)

    def plot_tol_acc(self, max_tol=EVAL_THRESHOLD, save=False):
        """
        Plot accuracy as a function of tolerance and compute AUC.

        Parameters
        ----------
        max_tol : int
            Maximum tolerance to compute.
        save : bool
            Whether to save the plot.
        """
        tols = np.linspace(0, max_tol, max_tol)
        accs = [self.centipawn_accuracy(tol) for tol in tols]
        auc = np.trapezoid(accs, tols) / max_tol

        plt.plot(tols, accs)
        plt.title(f"Tolerance-Accuracy Curve (AUC={auc:.3f})")
        plt.xlabel("Tolerance")
        plt.ylabel("Accuracy")

        if save:
            fname = f"tol_acc_{time.strftime('%Y%m%d_%H%M%S')}.png"
            fpath = self.plots_dir / fname
            plt.savefig(fpath, dpi=FIG_DPI)
