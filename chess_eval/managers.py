import time
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pandas import DataFrame
from scipy.stats import spearmanr
from sklearn import clone
from sklearn.metrics import r2_score, root_mean_squared_error, make_scorer
from sklearn.model_selection import train_test_split, GroupKFold, cross_validate

from chess_eval.config import *
from chess_eval.features import FEATURE_TRANSFORMERS


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
            df: pd.DataFrame = None,
            filepath: str = DATASET_FILE,
            read_size: int | None = READ_SIZE,
            sample_size: int | None = SAMPLE_SIZE,
            frac: float = None,
            test_size: float = TEST_SIZE,
            random_state: int = RANDOM_STATE,
            cleaner=clean,
            transformers: list | None = FEATURE_TRANSFORMERS,
            features: list | None = None,
            meta: dict = None,
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
        features : list
            List of features for training and testing.
        """
        if meta is None:
            meta = {}

        self.filepath = meta.get("filepath", filepath)
        self.read_size = meta.get("read_size", read_size)
        self.test_size = meta.get("test_size", test_size)
        self.random_state = meta.get("random_state", random_state)
        self.transformers = transformers
        self.features = features
        self.cleaner = cleaner if cleaner is not None else lambda x: x

        # Load and clean dataset
        if df is None:
            self.df_all = self.cleaner(pd.read_csv(filepath, nrows=read_size))
        else:
            self.df_all = df

        if meta.get("frac", frac) is not None:
            self.sample_size = int(meta.get("frac", frac) * len(self.df_all))
        else:
            self.sample_size = meta.get("sample_size", sample_size)

        # Sampled indices and train/test indices
        self.sample_idx = None
        self.train_idx = None
        self.test_idx = None

        # Sample and optionally apply transformers
        self.sample()
        if self.transformers:
            self.apply_transformers_parallel()
        self.train_test_split()

    @property
    def df(self):
        """Return the sampled dataframe (subset of df_all)."""
        return self.df_all.loc[self.sample_idx].reset_index(drop=True)

    @property
    def X(self):
        """Return features dataframe (excluding FEN and EVAL)."""
        if self.features is not None:
            return self.df[self.features]
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

    def sample(self, sample_size: int = None, frac: float = None) -> pd.DataFrame:
        """
        Sample random rows from the cleaned dataset.

        Parameters
        ----------
        sample_size : int, optional
            Number of rows to sample. Uses class default if None.
        frac: float, optional
            Fraction of rows to sample. Uses sample_size if None.

        Returns
        -------
        pd.DataFrame
            Sampled dataframe.
        """
        if sample_size is not None:
            self.sample_size = sample_size
        elif frac is not None:
            self.sample_size = int(frac * len(self.df_all))

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

    def apply_transformers_parallel(self, transformers=None, n_jobs=-1) -> pd.DataFrame:
        if transformers is None:
            transformers = []
        self.transformers.extend(transformers)

        def apply_one(transformer):
            missing_features = transformer.features - set(self.X.columns)
            if missing_features:
                df_trans = transformer.transform(self.df)
                return transformer.name, df_trans
            return transformer.name, None

        results = Parallel(n_jobs=n_jobs)(
            delayed(apply_one)(t) for t in self.transformers
        )

        for name, df_transformed in results:
            if df_transformed is not None:
                self.df_all.loc[self.sample_idx, df_transformed.columns] = df_transformed.values

        return self.X

    def train_test_split(self, features: list = None, test_size: float = None, random_state: int = None):
        """
        Split sampled dataset into training and testing sets.

        Parameters
        ----------
        features : list, optional  # TODO: add features
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
            test_size=test_size or self.test_size,
            random_state=random_state or self.random_state,
        )
        self.features = features
        return self.X_train, self.X_test, self.y_train, self.y_test


class ModelManager:
    """
    Wrapper around a scikit-learn model to handle fitting, predicting,
    and predicting from FEN strings.
    """

    def __init__(self, model):
        """
        Parameters
        ----------
        model : sklearn-like estimator
            Model with fit() and predict() methods.
        """
        self.model = model
        self.y_pred = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        """
        Fit the model on the training dataset.

        Parameters
        ----------
        :param X_train:
        :param y_train:
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame):
        """
        Predict target values for provided features or the test set.

        Parameters
        ----------
        dm : DataManager
            DataManager instance containing test splits.

        Returns
        -------
        :param X_test:
        """
        self.y_pred = self.model.predict(X_test)
        return self.y_pred

    def predict_fen(self, fen: str, transformers: list = FEATURE_TRANSFORMERS):
        """
        Predict evaluation for a single FEN string.

        Parameters
        ----------
        fen : str
            Chess position in FEN notation.
        transformers : list, optional
            List of transformers to apply.

        Returns
        -------
        np.ndarray
            Predicted evaluation.
        """
        df_fen = pd.DataFrame({FEN: [fen]})
        for transformer in transformers:
            df_fen = transformer.transform(df_fen)

        df_fen_feats = df_fen.drop(FEN, axis=1, errors="ignore")

        return self.model.predict(df_fen_feats)

    def cross_validate(
            self,
            dm: DataManager,
            game_len: int = 100,
            games_per_group: int = 10,
            scoring: str = "spearmanr",
            cv: int = 5
    ) -> tuple[Any, DataFrame]:
        """
        Perform cross-validation while keeping entire games in the same fold.

        Parameters
        ----------
        dm : DataManager
            DataManager instance providing X, y.
        game_len : int
            Number of consecutive rows corresponding to a single game.
        games_per_group : int
            Number of consecutive games to sample for each chunk.
        scoring : str or callable
            Metric for evaluation. Default is Spearman rank correlation.
        cv : int
            Number of folds.

        Returns
        -------
        pd.DataFrame
            Data frame with mean/std of train/test scores and mean fit/score times.
        """

        if dm.features is not None:
            X = dm.df_all[dm.features]
        else:
            X = dm.df_all.drop(columns=[EVAL, FEN])
        y = dm.df_all[EVAL]

        groups = np.arange(X.shape[0]) // (game_len * games_per_group)

        def spearmanr_scorer(y_true, y_pred):
            return spearmanr(y_true, y_pred).correlation

        if scoring == "spearmanr":
            scoring = make_scorer(spearmanr_scorer)

        gkf = GroupKFold(
            n_splits=cv,
            shuffle=True,
            random_state=dm.random_state,
        )

        results = cross_validate(
            clone(self.model),
            X,
            y,
            cv=gkf.split(X, y, groups=groups),
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
        )

        summary = {
            "train_score_mean": [np.mean(results["train_score"])],
            "train_score_std": [np.std(results["train_score"])],
            "test_score_mean": [np.mean(results["test_score"])],
            "test_score_std": [np.std(results["test_score"])],
            "train_time_mean": [np.mean(results["fit_time"])],
            "test_time_mean": [np.mean(results["score_time"])]
        }

        return results, pd.DataFrame(summary)


class MetricsManager:
    """
    Compute evaluation metrics and provide plotting utilities for predictions.
    """

    def __init__(self, mm: ModelManager, dm: DataManager, plots_dir=PLOTS_DIR):
        """
        Parameters
        ----------
        mm : ModelManager
            The model manager instance.
        plots_dir : pathlib.Path
            Directory to save plots.
        """
        self.mm = mm
        self.dm = dm
        self.plots_dir = plots_dir
        self.plots_dir.mkdir(exist_ok=True)

        apply_custom_style()

    # --- Properties ---

    @property
    def y_true(self):
        """Return ground truth target values (test set)."""
        return self.dm.y_test

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
        else:
            raise ValueError("player must be white/black")

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
            "Spearman Rank": self.spearman_rank_correlation(),
            f"Accuracy (±{tol})": self.centipawn_accuracy(tol),
            "RMSE": root_mean_squared_error(self.y_true, self.y_pred),
            "Accuracy (sign)": self.sign_accuracy(),
            "Recall (white)": self.sign_recall("white"),
            "Recall (black)": self.sign_recall("black"),
            "R^2": self.r2(),
        }

    # --- Plotting Methods ---

    def plot_scatter(self, save=False, file=None):
        """
        Scatter plot of true vs predicted values.

        Parameters
        ----------
        save : bool
            Whether to save the plot as PNG.
        file : str
            Custom file name, prefix `scatter_` will be added.
        """
        plt.scatter(self.y_true, self.y_pred, alpha=0.5)
        plt.xlim(-EVAL_THRESHOLD, EVAL_THRESHOLD)
        plt.ylim(-EVAL_THRESHOLD, EVAL_THRESHOLD)
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title("Evaluation Predictions")

        if save:
            if file is None:
                fname = f"scatter_{time.strftime('%Y%m%d_%H%M%S')}.png"
            else:
                fname = f"scatter_{file}.png"
            fpath = self.plots_dir / fname
            plt.savefig(fpath, dpi=FIG_DPI)

        plt.show()

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

        plt.show()


def evaluate(mm: ModelManager, dm: DataManager, save=True, file=None):
    if mm.y_pred is None:
        raise ValueError("The model has not predicted any data.")
    metm = MetricsManager(mm, dm)
    metm.plot_scatter(save=save, file=file)
    data = metm.compute_metrics().items()
    df = pd.DataFrame(data, columns=["Metric", "Value"])
    df.style.hide(axis="index")
    return df
