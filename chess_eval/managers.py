import time
import json
from typing import Callable, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn import clone
from sklearn.metrics import r2_score, root_mean_squared_error, make_scorer
from sklearn.model_selection import train_test_split, GroupKFold, cross_validate, GridSearchCV

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
    mode : str, default "remove"
        'clip' to clip values to ±threshold, 'remove' to drop rows exceeding threshold.
    threshold : int, default EVAL_THRESHOLD
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
    Handles loading, cleaning, sampling, feature transformation,
    and train-test splitting for chess evaluation datasets.

    Attributes
    ----------
    df_all : pd.DataFrame
        Full cleaned dataset.
    sample_idx : np.ndarray
        Indices of sampled rows.
    train_idx : np.ndarray
        Training set indices.
    test_idx : np.ndarray
        Test set indices.
    features : list[str]
        List of feature column names.
    transformers : list
        List of feature transformer classes.
    """

    def __init__(
            self,
            df: Optional[pd.DataFrame] = None,
            filepath: str = DATASET_FILE,
            read_size: Optional[int] = READ_SIZE,
            skiprows: Optional[int] = SKIP_ROWS,
            sample_size: Optional[int] = SAMPLE_SIZE,
            frac: Optional[float] = None,
            test_size: float = TEST_SIZE,
            downcast: bool = True,
            random_state: int = RANDOM_STATE,
            cleaner: Callable[[pd.DataFrame], pd.DataFrame] | None = clean,
            transformers=None,
            features: Optional[List[str]] = None,
            meta: Optional[dict] = None,
    ) -> None:
        """
        Parameters
        ----------
        df : pd.DataFrame, optional
            Preloaded dataset.
        filepath : str, default DATASET_FILE
            Path to CSV dataset.
        read_size : int, optional
            Number of rows to read from CSV.
        skiprows : int, optional
            Number of rows to skip before reading from CSV.
        sample_size : int, optional
            Number of rows to sample for train/test split.
        frac : float, optional
            Fraction of dataset to sample.
        test_size : float, default TEST_SIZE
            Fraction of test set.
        downcast : bool, default True
            Whether to downcast all features to integer or not.
        random_state : int, default RANDOM_STATE
            Random seed.
        cleaner : Callable, default clean
            Function to clean the dataframe.
        transformers : list, optional
            List of feature transformers.
        features : list[str], optional
            List of features to use.
        meta : dict, optional
            Metadata to override default parameters.
        """
        if transformers is None:
            transformers = FEATURE_TRANSFORMERS
        if meta is None:
            meta = {}

        self.meta = meta
        self.filepath = meta.get("filepath", filepath)
        self.read_size = meta.get("read_size", read_size)
        self.skiprows = meta.get("skiprows", skiprows)
        self.test_size = meta.get("test_size", test_size)
        self.downcast = meta.get("downcast", downcast)
        self.random_state = meta.get("random_state", random_state)
        self.transformers = transformers
        self.features = features
        self.cleaner = cleaner if cleaner is not None else lambda x: x

        if df is None:
            self.df_all = self.cleaner(pd.read_csv(filepath, nrows=read_size, skiprows=list(range(1, skiprows))))
        else:
            self.df_all = df

        if meta.get("frac", frac) is not None:
            self.sample_size = int(meta.get("frac", frac) * len(self.df_all))
        else:
            self.sample_size = meta.get("sample_size", sample_size)

        self.sample_idx: Optional[np.ndarray] = None
        self.train_idx: Optional[np.ndarray] = None
        self.test_idx: Optional[np.ndarray] = None

        self.sample()
        if self.transformers:
            self.apply_transformers_parallel()
        self.train_test_split()

    @property
    def df(self) -> pd.DataFrame:
        """Return sampled dataframe."""
        return self.df_all.loc[self.sample_idx].reset_index(drop=True)

    @df.setter
    def df(self, new_df: pd.DataFrame):
        """Update the sampled rows in df_all with new_df."""
        self.df_all.loc[self.sample_idx, new_df.columns] = new_df.values

    @property
    def X(self) -> pd.DataFrame:
        """Return features dataframe (excluding EVAL and FEN)."""
        if self.features is not None:
            return self.df[self.features]
        return self.df.drop(columns=[EVAL, FEN, MOVE], errors="ignore")

    @property
    def y(self) -> pd.Series:
        """Return target series (EVAL column)."""
        return self.df[EVAL]

    @property
    def X_train(self) -> pd.DataFrame:
        """Return training features."""
        return self.X.iloc[self.train_idx]

    @property
    def X_test(self) -> pd.DataFrame:
        """Return testing features."""
        return self.X.iloc[self.test_idx]

    @property
    def y_train(self) -> pd.Series:
        """Return training targets."""
        return self.y.iloc[self.train_idx]

    @property
    def y_test(self) -> pd.Series:
        """Return testing targets."""
        return self.y.iloc[self.test_idx]

    def sample(self, sample_size: Optional[int] = None, frac: Optional[float] = None) -> pd.DataFrame:
        """
        Randomly sample rows from the cleaned dataset.

        Parameters
        ----------
        sample_size : int, optional
            Number of rows to sample.
        frac : float, optional
            Fraction of rows to sample.

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

    def apply_transformers(self, transformers: Optional[List] = None) -> pd.DataFrame:
        """
        Apply feature transformers to the dataset.

        Parameters
        ----------
        transformers : list, optional
            Additional transformers to apply.

        Returns
        -------
        pd.DataFrame
            Transformed features dataframe.
        """
        if transformers is None:
            transformers = []
        self.transformers.extend(transformers)

        for transformer in self.transformers:
            missing_features = transformer.features - set(self.X.columns)
            if missing_features:
                df_transformed = transformer.transform(self.df)
                for col in df_transformed.columns: # assigns each column individually instead of all at once, which avoids the dtype compatibility warning
                    self.df_all.loc[self.sample_idx, col] = df_transformed[col].values

        self.df_all.fillna(0, inplace=True)  # rows that are not sampled
        for col in self.df_all.columns:
            if col not in [FEN, EVAL, MOVE, "Phase"]:
                col_vals = self.df_all[col].values
                if np.min(col_vals) >= -128 and np.max(col_vals) <= 127:
                    self.df_all[col] = col_vals.astype("int8")
                else:
                    self.df_all[col] = col_vals.astype("int16")

        return self.X

    def apply_transformers_parallel(self, transformers: Optional[List] = None, n_jobs: int = -1) -> pd.DataFrame:
        """
        Apply feature transformers in parallel.

        Parameters
        ----------
        transformers : list, optional
            Additional transformers to apply.
        n_jobs : int, default -1
            Number of parallel jobs.

        Returns
        -------
        pd.DataFrame
            Transformed features dataframe.
        """
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
                for col in df_transformed.columns: # assigns each column individually instead of all at once, which avoids the dtype compatibility warning
                    self.df_all.loc[self.sample_idx, col] = df_transformed[col].values

        self.df_all.fillna(0, inplace=True)  # rows that are not sampled

        if self.downcast:
            for col in self.df_all.columns:
                if col not in [FEN, EVAL, MOVE, "Phase"]:
                    col_vals = self.df_all[col].values
                    if np.min(col_vals) >= -128 and np.max(col_vals) <= 127:
                        self.df_all[col] = col_vals.astype("int8")
                    else:
                        self.df_all[col] = col_vals.astype("int16")

        return self.X

    def train_test_split(
            self,
            features: Optional[List[str]] = None,
            test_size: Optional[float] = None,
            random_state: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split sampled dataset into training and testing sets.

        Parameters
        ----------
        features : list[str], optional
            Columns to use for splitting.
        test_size : float, optional
            Fraction of test set.
        random_state : int, optional
            Random seed.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
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

    Attributes
    ----------
    model : sklearn-like estimator
        The scikit-learn model.
    y_pred : Optional[np.ndarray]
        Last predictions made by the model.
    """

    def __init__(self, model: Any) -> None:
        """
        Parameters
        ----------
        model : sklearn-like estimator
            Model with fit() and predict() methods.
        """
        self.model = model
        self.y_pred: Optional[np.ndarray] = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Fit the model on training data.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training target values.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Predict target values for given features.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test features.

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        self.y_pred = self.model.predict(X_test)
        return self.y_pred

    def predict_fen(self, fen: str, transformers: Optional[List] | None = None) -> np.ndarray:
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
        if transformers is None:
            transformers = FEATURE_TRANSFORMERS
        df_fen = pd.DataFrame({FEN: [fen]})
        for transformer in transformers:
            df_fen = transformer.transform(df_fen)
        return self.model.predict(df_fen.drop(FEN, axis=1, errors="ignore"))

    def cross_validate(
            self,
            dm: DataManager,
            game_len: int = 100,
            games_per_group: int = 10,
            scoring: Union[str, Callable] = "spearmanr",
            n_splits: int = 5
    ) -> Tuple[Any, pd.DataFrame]:
        """
        Perform GroupKFold cross-validation keeping entire games together.

        Parameters
        ----------
        dm : DataManager
            DataManager providing X and y.
        game_len : int, default 100
            Rows per game.
        games_per_group : int, default 10
            Number of consecutive games per group.
        scoring : str or callable, default "spearmanr"
            Metric to evaluate.
        n_splits : int, default 5
            Number of folds.

        Returns
        -------
        tuple[Any, pd.DataFrame]
            Full cross-validation results and summary DataFrame.
        """
        X = dm.df_all[dm.features] if dm.features else dm.df_all.drop(columns=[EVAL, FEN])
        y = dm.df_all[EVAL]
        groups = np.arange(X.shape[0]) // (game_len * games_per_group)

        if scoring == "spearmanr":
            scoring = make_scorer(lambda y_true, y_pred: spearmanr(y_true, y_pred).correlation)

        gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=dm.random_state)
        results = cross_validate(clone(self.model), X, y, cv=gkf.split(X, y, groups=groups),
                                 scoring=scoring, return_train_score=True, n_jobs=-1)

        summary = pd.DataFrame({
            "train_score_mean": [np.mean(results["train_score"])],
            "train_score_std": [np.std(results["train_score"])],
            "test_score_mean": [np.mean(results["test_score"])],
            "test_score_std": [np.std(results["test_score"])],
            "train_time_mean": [np.mean(results["fit_time"])],
            "test_time_mean": [np.mean(results["score_time"])]
        })

        return results, summary

    def optimize_hyperparameters(
            self,
            dm: DataManager,
            param_grid: dict,
            game_len: int = 100,
            games_per_group: int = 10,
            scoring: Union[str, Callable] = "spearmanr",
            n_splits: int = 5
    ) -> Tuple[GridSearchCV, dict, float]:
        """
        Perform hyperparameter optimization using GroupKFold cross-validation.

        Parameters
        ----------
        dm : DataManager
            DataManager providing X and y.
        param_grid : dict
            Hyperparameters to search.
        game_len : int, default 100
            Rows per game.
        games_per_group : int, default 10
            Consecutive games per group.
        scoring : str or callable, default "spearmanr"
            Metric for evaluation.
        n_splits : int, default 5
            Number of folds.

        Returns
        -------
        tuple[GridSearchCV, dict, float]
            Fitted GridSearchCV object, best parameters, best score.
        """
        X = dm.df_all[dm.features] if dm.features else dm.df_all.drop(columns=[EVAL, FEN])
        y = dm.df_all[EVAL]
        groups = np.arange(X.shape[0]) // (game_len * games_per_group)

        if scoring == "spearmanr":
            scoring = make_scorer(lambda y_true, y_pred: spearmanr(y_true, y_pred).correlation)

        gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=dm.random_state)
        grid = GridSearchCV(clone(self.model), param_grid=param_grid, scoring=scoring, cv=gkf, n_jobs=-1,
                            return_train_score=True)
        grid.fit(X, y, groups=groups)

        return grid, grid.best_params_, grid.best_score_


class MetricsManager:
    """
    Compute evaluation metrics and plotting utilities for predictions.

    Attributes
    ----------
    mm : ModelManager
        Model manager instance.
    dm : DataManager
        Data manager instance.
    plots_dir : Path
        Directory for saving plots.
    """
    plots_dir = PLOTS_DIR

    def __init__(self, mm: ModelManager, dm: DataManager) -> None:
        """
        Parameters
        ----------
        mm : ModelManager
            Model manager.
        dm : DataManager
            Data manager.
        """
        self.mm = mm
        self.dm = dm
        apply_custom_style()

    @property
    def y_true(self) -> pd.Series:
        """Ground truth values (test set)."""
        return self.dm.y_test

    @property
    def y_pred(self) -> np.ndarray:
        """Predicted values from the model."""
        return self.mm.y_pred

    def sign_accuracy(self) -> float:
        """Fraction of predictions with correct sign."""
        return np.mean(np.sign(self.y_true) == np.sign(self.y_pred))  # type: ignore

    def sign_recall(self, player: str = "white") -> float:
        """
        Recall of sign predictions for a specific player.

        Parameters
        ----------
        player : str, default "white"
            "white" or "black".

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
            raise ValueError("player must be 'white' or 'black'")
        return np.mean(pred_sign[mask] == true_sign[mask]) if mask.sum() else np.nan

    def centipawn_accuracy(self, tol: int = 200) -> float:
        """Fraction of predictions within ±tol centipawns."""
        return np.mean(np.abs(self.y_true - self.y_pred) <= tol)  # type: ignore

    def r2(self) -> float:
        """R² score."""
        return r2_score(self.y_true, self.y_pred)

    def spearman_rank_correlation(self) -> float:
        """Spearman rank correlation."""
        return spearmanr(self.y_true, self.y_pred)[0]  # type: ignore

    def compute_metrics(self, tol: int = 200) -> dict[str, float]:
        """
        Compute all evaluation metrics.

        Parameters
        ----------
        tol : int, default 200
            Tolerance for centipawn accuracy.

        Returns
        -------
        dict[str, float]
            Metrics dictionary.
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

    def plot_scatter(self, save: bool = False, file: str | None = None) -> None:
        """
        Scatter plot of true vs predicted values.

        Parameters
        ----------
        save : bool, default False
            Whether to save the plot as PNG.
        file : str or None, default None
            Custom file name; prefix `scatter_` will be added if provided.
        """
        plt.scatter(self.y_true, self.y_pred, alpha=0.5)
        plt.xlim(-EVAL_THRESHOLD, EVAL_THRESHOLD)
        plt.ylim(-EVAL_THRESHOLD, EVAL_THRESHOLD)
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title("Evaluation Predictions")

        if save:
            fname = f"scatter_{file or time.strftime('%Y%m%d_%H%M%S')}.png"
            fpath = numbered_path(self.plots_dir / fname)
            plt.savefig(fpath, dpi=FIG_DPI)

        plt.show()

    def plot_tol_acc(self, max_tol: int = EVAL_THRESHOLD, save: bool = False) -> None:
        """
        Plot centipawn accuracy as a function of tolerance and compute AUC.

        Parameters
        ----------
        max_tol : int, default EVAL_THRESHOLD
            Maximum tolerance to compute.
        save : bool, default False
            Whether to save the plot as PNG.
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
            fpath = numbered_path(self.plots_dir / fname)
            plt.savefig(fpath, dpi=FIG_DPI)

        plt.show()

    @classmethod
    def plot_gridsearch_results(
            cls,
            grid: GridSearchCV,
            log: bool = False,
            save: bool = False,
            file: str | None = None,
            params: dict[str, float | None] | None = None,
            show_train: bool = True,
    ) -> None:
        """
        Plot results from a GridSearchCV object.

        Parameters
        ----------
        grid : GridSearchCV
            Fitted grid search object.
        log : bool, default False
            Whether to use logarithmic x-scale for 1D plots.
        save : bool, default False
            Whether to save the plot as PNG.
        file : str or None, default None
            Custom file name; prefix `grid_` will be added.
        params : dict[str, float or None], optional
            Fixed or free parameter values to filter; None means all free.
        show_train : bool, default True
            Whether to show training score curve for 1D plots.
        """
        df = pd.DataFrame(grid.cv_results_)
        all_params = list(grid.param_grid.keys())

        if params is None:
            params = {p: None for p in all_params}
        else:
            missing = set(all_params) - set(params.keys())
            if missing:
                raise ValueError(f"params missing keys: {missing}")

        free_params = [p for p, v in params.items() if v is None]
        fixed_params = {p: v for p, v in params.items() if v is not None}

        # Filter rows by fixed parameters
        for p, val in fixed_params.items():
            df = df[df[f"param_{p}"] == val]
        if df.empty:
            raise ValueError("No grid rows match the fixed parameter values.")

        # 1D plot
        if len(free_params) == 1:
            p = free_params[0]
            x = df[f"param_{p}"]
            y_test = df["mean_test_score"]
            y_train = df.get("mean_train_score", None)

            if log:
                plt.xscale("log")

            plt.plot(x, y_test, marker="o", label="Test Score")
            if y_train is not None and show_train:
                plt.plot(x, y_train, marker="x", label="Train Score")

            plt.xlabel(p)
            plt.ylabel("Score")
            plt.title(f"Score vs {p}")
            plt.legend()

        # 2D heatmap
        elif len(free_params) == 2:
            p1, p2 = free_params
            pivot = df.pivot(index=f"param_{p1}", columns=f"param_{p2}", values="mean_test_score")
            sns.heatmap(pivot, annot=True, fmt=".4f", cmap="viridis")
            plt.xlabel(p2)
            plt.ylabel(p1)
            plt.title("Mean Test Score Heatmap")

        else:
            raise ValueError("Exactly one or two free parameters required.")

        if save:
            fname = f"grid_{file or time.strftime('%Y%m%d_%H%M%S')}.png"
            fpath = numbered_path(cls.plots_dir / fname)
            plt.savefig(fpath, dpi=FIG_DPI)

        plt.show()


class StorageManager:
    """
    Unified storage manager for:
    - Feature datasets (DataManager objects)
    - Fitted models (ModelManager objects)

    Default paths:
        datasets -> "../data/features"
        models   -> "../models"
    """

    def __init__(
            self,
            dataset_dir: str = SAVED_DATASETS_DIR,
            model_dir: str = SAVED_MODELS_DIR,
    ) -> None:
        """
        Initialize storage directories and metadata files.

        Parameters
        ----------
        dataset_dir : str, default SAVED_DATASETS_DIR
            Path to store datasets.
        model_dir : str, default SAVED_MODELS_DIR
            Path to store models.
        """
        self.dataset_dir: Path = Path(dataset_dir)
        self.model_dir: Path = Path(model_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_meta_file: Path = self.dataset_dir / "metadata.json"
        self.model_meta_file: Path = self.model_dir / "metadata.json"

        # Initialize metadata files if missing
        for file in [self.dataset_meta_file, self.model_meta_file]:
            if not file.exists():
                file.write_text(json.dumps({}))

    # -------------------- Dataset Methods --------------------

    def save_dataset(
            self,
            dm: DataManager,
            name: str | None = None,
            notes: str = "",
            force: bool = False,
    ) -> str:
        """
        Save a DataManager dataset with metadata.

        Parameters
        ----------
        dm : DataManager
            DataManager object to save.
        name : str or None, default None
            Name for the dataset; uses timestamp if None.
        notes : str, default ""
            Notes to include in metadata.
        force : bool, default False
            Overwrite existing dataset if True.

        Returns
        -------
        str
            Path to the saved dataset file.
        """
        name = name or datetime.now().strftime("%Y%m%d%H%M%S")
        file_path: Path = self.dataset_dir / f"{name}.pkl"

        all_metadata: dict = json.loads(self.dataset_meta_file.read_text())

        if (name in all_metadata) and not force:
            raise ValueError(f"Dataset {name} already exists.")

        # Save the DataFrame
        with file_path.open("wb") as f:
            pickle.dump(dm.df, f, protocol=pickle.HIGHEST_PROTOCOL)  # type: ignore[arg-type]

        all_metadata[name] = {
            "name": name,
            "filepath": str(file_path),
            "read_size": dm.read_size,
            "sample_size": dm.sample_size,
            "test_size": dm.test_size,
            "random_state": dm.random_state,
            "transformers": [t.name for t in dm.transformers] if dm.transformers else [],
            "df_shape": dm.df.shape if dm.df is not None else None,
            "columns": dm.df.columns.tolist() if dm.df is not None else None,
            "created_at": datetime.now().isoformat(),
            "notes": notes,
        }

        self.dataset_meta_file.write_text(json.dumps(all_metadata, indent=2))
        return str(file_path)

    def load_dataset(self, name: str) -> Tuple[pd.DataFrame, dict]:
        """
        Load a dataset by name.

        Parameters
        ----------
        name : str
            Name of the dataset to load.

        Returns
        -------
        tuple[pd.DataFrame, dict]
            The dataset dataframe and its metadata.
        """
        all_metadata: dict = json.loads(self.dataset_meta_file.read_text())
        if name not in all_metadata:
            raise KeyError(f"No dataset found for '{name}'")

        meta: dict = all_metadata[name]
        with Path(meta["filepath"]).open("rb") as f:
            df: pd.DataFrame = pickle.load(f)

        return df, meta

    def list_datasets(self) -> pd.DataFrame:
        """
        List all saved datasets with basic metadata.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['Name', 'Shape', 'Notes'].
        """
        all_metadata: dict = json.loads(self.dataset_meta_file.read_text())
        data = [(name, meta.get("df_shape"), meta.get("notes")) for name, meta in all_metadata.items()]
        df = pd.DataFrame(data, columns=["Name", "Shape", "Notes"])
        df.style.hide(axis="index")
        return df

    # -------------------- Model Methods --------------------

    def save_model(
            self,
            mm: ModelManager,
            dm: DataManager,
            name: str | None = None,
            notes: str = "",
            force: bool = False,
            skip: bool = False,
    ) -> str:
        """
        Save a ModelManager object with metadata.

        Parameters
        ----------
        mm : ModelManager
            ModelManager object to save.
        dm : DataManager
            Associated dataset manager.
        name : str or None, default None
            Name for the model; uses timestamp if None.
        notes : str, default ""
            Notes to include in metadata.
        force : bool, default False
            Overwrite existing model if True.
        skip : bool, default False
            Skip saving if model already exists.

        Returns
        -------
        str
            Path to the saved model file.
        """
        name = name or datetime.now().strftime("%Y%m%d%H%M%S")
        file_path: Path = self.model_dir / f"{name}.pkl"

        all_metadata: dict = json.loads(self.model_meta_file.read_text())

        if (name in all_metadata) and not force:
            if skip:
                return "Skipped"
            raise ValueError(f"Model '{name}' already exists.")

        with file_path.open("wb") as f:
            pickle.dump(mm.model, f, protocol=pickle.HIGHEST_PROTOCOL)  # type: ignore[arg-type]

        all_metadata[name] = {
            "name": name,
            "filepath": str(file_path),
            "model_class": mm.model.__class__.__name__,
            "parameters": getattr(mm.model, "get_params", lambda: {})(),
            "training_dataset": str(getattr(dm, "filepath", None)) if dm else None,
            "features": list(dm.X.columns) if dm else [],
            "transformers": [t.name for t in dm.transformers] if dm and dm.transformers else [],
            "created_at": datetime.now().isoformat(),
            "notes": notes,
        }

        self.model_meta_file.write_text(json.dumps(all_metadata, indent=2))
        return str(file_path)

    def load_model(self, name: str) -> Tuple[object, dict]:
        """
        Load a model by name.

        Parameters
        ----------
        name : str
            Name of the model to load.

        Returns
        -------
        tuple[object, dict]
            The model object and its metadata.
        """
        all_metadata: dict = json.loads(self.model_meta_file.read_text())
        if name not in all_metadata:
            raise KeyError(f"No model found for '{name}'")

        meta: dict = all_metadata[name]
        with Path(meta["filepath"]).open("rb") as f:
            model = pickle.load(f)

        return model, meta

    def list_models(self) -> pd.DataFrame:
        """
        List all saved models with metadata.

        Returns
        -------
        pd.DataFrame
            Columns: ['Name', 'Model', 'Dataset', 'Notes'].
        """
        all_metadata: dict = json.loads(self.model_meta_file.read_text())
        data = [
            (name, meta.get("model_class"), Path(meta.get("training_dataset")).stem, meta.get("notes"))
            for name, meta in all_metadata.items()
        ]
        df = pd.DataFrame(data, columns=["Name", "Model", "Dataset", "Notes"])
        df.style.hide(axis="index")
        return df


def load_dataset(name: str, downcast: bool = True) -> DataManager:
    """
    Load a dataset from storage into a DataManager object.

    Parameters
    ----------
    name : str
        Name of the dataset to load.
    downcast : bool, default True
        Whether to downcast all features to integer or not.

    Returns
    -------
    DataManager
        Loaded DataManager instance.
    """
    sm = StorageManager()
    df, meta = sm.load_dataset(name)
    dm = DataManager(
        df=df,
        meta=meta,
        transformers=None,
        downcast=downcast,
    )
    return dm


def load_model(name: str, dm: bool = True) -> Tuple[ModelManager, DataManager] | Tuple[ModelManager, None]:
    """
    Load a model and its associated dataset from storage.

    Parameters
    ----------
    name : str
        Name of the model to load.
    dm : bool, default True
        Whether to load the related dataset or not.

    Returns
    -------
    tuple[ModelManager, DataManager]] | Tuple[ModelManager, None]
        Loaded ModelManager and associated DataManager (if applicable).
    """
    sm = StorageManager()
    model, meta = sm.load_model(name)
    mm = ModelManager(model=model)
    if dm:
        df_path = Path(meta["training_dataset"])
        dm = load_dataset(df_path.stem)
        return mm, dm
    return mm, None


def evaluate(mm: ModelManager, dm: DataManager, save: bool = True, file: Optional[str] = None) -> pd.DataFrame:
    """
    Evaluate a model using the provided DataManager and plot results.

    Parameters
    ----------
    mm : ModelManager
        The model manager instance with predictions.
    dm : DataManager
        The dataset manager.
    save : bool, default True
        Whether to save the scatter plot.
    file : str, optional
        Custom file name for the plot.

    Returns
    -------
    pd.DataFrame
        DataFrame of computed metrics.
    """
    if mm.y_pred is None:
        raise ValueError("The model has not predicted any data.")

    metm = MetricsManager(mm, dm)
    metm.plot_scatter(save=save, file=file)

    data = metm.compute_metrics().items()
    df = pd.DataFrame(data, columns=["Metric", "Value"])
    df.style.hide(axis="index")
    return df
