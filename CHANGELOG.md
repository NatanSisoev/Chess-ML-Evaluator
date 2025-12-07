# Changelog

All notable changes to this project will be documented in this file.

---

## [Unreleased]

- TODO: run datasets (with downcast) and train and save models


[2025-12-07] - Ferran Villarta

## [2025-12-07] - Ferran Villarta
- modified `chess_eval/managers.py`:
  - `StorageManager.save_model` and `load_model` updated to use `"filename"` instead of full `"filepath"`
  - `"training_dataset"` now stores dataset name only instead of absolute path
  - adjusted metadata handling for compatibility with updated `DataManager` and `ModelManager`  
- new file `models/import_json.py`:
  - script to clean and convert old `metadata.json` entries to new format
  - removes absolute paths, preserves other metadata fields  
- modified `models/metadata.json`:
  - all models updated to use new `"filename"` and dataset name format
  - removed absolute paths for portability

---

## [2025-12-06] - Natan Sisoev

- finished `save_datasets.ipynb` and `save_models.ipynb` with optimized versions
- ran storage notebooks and saved fitted models
- ran final notebooks `8_testing.ipynb`

---

## [2025-12-05] - Natan Sisoev

- optimized dataset storage: `float64` (8 bytes) -> `int8` (1 byte), so around 8 times smaller pickled datasets
- implemented default downcast
- updated storage files and managers
- feature importance analysis for `8_testing.ipynb`

---

## [2025-12-04] - Natan Sisoev

- documented whole `chess_eval` scripts with docstrings and type-hinting
- improved `README.md` and created this `CHANGELOG.md`
- tested the best model on `random_evals.csv`
- implemented incremental learning on all data from `chessData.csv`

---

## [2025-12-03] - Natan Sisoev

- improved hyperparameter optimization: removed staged optimization for XGBR and refined values for Ridge and KNN
- improved grid search plotting
- saved hyperparameter optimization results to `config.py`
- filtered, renamed, organized, and saved result plots
- created for loop to save all optimized models for all dataframes
- created `four_full_rd` dataframe and fitted XGBRegressor with best params on it -> `xgb_four_full_rd`
- saved and timed the dataframes from `save_datasets.ipynb`
- saved and timed the model from `save_models.ipynb`

---

## [2025-12-02] - Natan Sisoev

- finished cross-validation
  - finished notebook `5_cv.ipynb`
  - fixed some spearman rank scoring bugs
  - adapted `6_models.ipynb` to use cross-validation for comparison
  - ran model comparison with cross-validation: 1 hour and 20 minutes
- started hyperparameter optimization
  - started the `7_hyperparameters.ipynb` notebook
  - created `optimize_hyperparameters` method for optimizing hyperparameters
  - created `plot_gridsearch_results` method for visualizing optimization
  - created the hyperparameter search grids for al 3 models: Ridge, XGBRegressor and KNN
  - ran Ridge grid-search
  - ran KNN grid-search
- created structure for `save_models.ipynb` (moved all storage-related notebooks to `storage` directory)

---

## [2025-12-01] - Natan Sisoev

- started `5_cv.ipynb`
- considered the stratified version of group k-fold: not really worth it, just group different games
- finished `1_eda.ipynb`, last section referencing chunked sampling

---

## [2025-11-28] - Natan Sisoev

- changed ModelManager: removed dm attribute, too much duplicated data (passed by param for fit and test, still saves y_true and y_pred for the metrics manager)
- updated DataManager: added features attribute to train or test only on a couple of features
- added a parallelized version to the transformer application method
- added a check before overwriting a saved dataset/model
- organized datasets and models storage notebooks
- compared all models in `6_models.ipynb` and chose 3 winners: Ridge, XGBRegressor and KNN
- trained and pickled a multilayer perceptron regressor with only the piece values at each square: model `MLPR_100k_rd`
- created the cross-validation method for the model manager, TODO: document it in notebook 5

---

## [2025-11-27] - Natan Sisoev

- merge main with improvements
- updated TODO
- updated storage manager to new DataManager (removed attribute features -> X.columns)
- gathered a bunch of models to test
- improved storage system: created easy access `load_dataset` and `load_model` functions
- created `save_datasets.ipynb` and `save_models.ipynb` notebooks and created script to run overnight
- add `frac` option for dataframe sampling
- adapted `DataManager` to accept dictionary with metadata (for the storage system)
- improved `config.py` file
- added easy use `evaluate` function
- started KNN hyperparameter optimization and plots
- improved storage listing and display
- fixed small project errors and warnings

---

## [2025-11-26] - Natan Sisoev

- merged feature transformers with `FeatureBundle` class
- separated `main.ipynb` notebook into
  1. EDA
  2. Preprocessing
  3. Feature Engineering
  4. Metric Selection
  5. Cross-Validation
  6. Model Selection
  7. Hyperparameters
- improved README
- improved DataManager to indexes instead of redundant copies
- improved MetricsManager to save plots
- done 1_eda.ipynb and 2_preprocessing.ipynb, finished both notebooks

---

## [2025-11-25] - Natan Sisoev

- started `refactor` branch
- refactored the gigantic main.ipynb into:
  - managers
  - constants (uppercase values)
  - features (all class feature transformers)
  - storage (way to store computed features and fitted models)
  - `mwe.ipynb`: minimal working example of a Gradient Boosting model
- re-organized the structure of the project into the python package `chess_eval` and separate data, models and notebooks folders
- migrated to using `uv` for package managing and project metadata logging
- created `Mobility`, `Attackers`, `PositionalControl` and `GameInfo` feature transformers
- added `storage.ipynb` notebook showing usage of the `StorageManager` class
- improved README: added folder structure and getting started

---

## [2025-11-24] - Ferran Villarta

- Another way to measure accuracy is to model our data as a normal distribution `N(μ, σ)`, and then accept the prediction if it falls within the range `[evidence ± z_0.05]`.
- commented the code
- created transformers:
  - `Relations`: Threats_Created_White, Threats_Created_Black, Hanging_Pieces_White, Hanging_Pieces_Black, Hanging_Points_White, Hanging_Points_Black, Undefended_Pieces_White, Undefended_Pieces_Black, Undefended_Points_White, Undefended_Points_Black
  - `BoardControl`: Central_Squares_Control_White, Central_Squares_Control_Black, Open_Columns_White, Open_Columns_Black, SemiOpen_Columns_White, SemiOpen_Columns_Black, Rook_on_Open_Column_White, Rook_on_Open_Column_Black, Protected_Advanced_Pawn_White, Protected_Advanced_Pawn_Black, Rook_Queen_Aligned_White, Rook_Queen_Aligned_Black, Rooks_Aligned_White, Rooks_Aligned_Black, Controlled_Squares_White, Controlled_Squares_Black
  - `PieceMobility`: Mobility_King_White, Mobility_Queen_White, Mobility_Rook_White, Mobility_Bishop_White, Mobility_Knight_White, Mobility_Pawn_White, Mobility_King_Black, Mobility_Queen_Black, Mobility_Rook_Black, Mobility_Bishop_Black, Mobility_Knight_Black, Mobility_Pawn_Black
  - `MiscFeatures`: Player_to_Move, Number_of_Moves, Halfmove_Clock

---

## [2025-11-24] - Natan Sisoev

- finished king safety features
- explored R^2 and R^2 adjusted metrics
- improved managers' communications
- tried different read/sample sizes to see the effect

---

## [2025-11-19] - Natan Sisoev

- created managers: 
  - `DataManager`:
    - read a lot of rows
    - sample a random subset
    - apply transformers
    - train and test split (TODO: cross-validation)
  - `ChessManager`: show board and features
  - `ModelManager`:
    - save `y_pred` for metrics
    - predict a single FEN
  - `MetricsManager`:
    - calculate accuracy, recall (white, black and draw), and centipawn accuracy (accuracy with tolerance)
    - scatter plot `y_pred` vs `y_true`
    - plot evaluation heatmap on 2 features plane
    - plot tolerance - accuracy line w/ AUC as title
- created transformers workflow and transformers:
  - `PieceInfo`: white and black piece count, white and black separate types piece count, white and black total piece value
  - `PawnStructure`: white and black isolated pawns (count), white and black doubled pawns (count), white and black passed pawns (count)
  - `KingSafety`: white and black castled (True or False: king and rook in the castled position), white and black pawns near king (count of pawns in the 9x9 grid around the king), number of open files on the table
- full execution `GradientBoostingRegressor`:
  - 500k rows sampled from 2M
  - `PieceInfo` + `PawnStructure` + `KingSafety`
  - results:
    - MSE: 50950.35639631619
    - Sign accuracy: 0.62622
    - White winning recall: 0.923241397897763
    - Black winning recall: 0.3329284910243498
    - Draw recall: 0.0
    - Centipawn ±200 accuracy: 0.77775
    - AUC TA: 0.85
- full execution `LinearRegression`:
  - 500k rows sampled from 2M
  - `PieceInfo` + `PawnStructure` + `KingSafety`
  - results:
    - MSE: 52599.53217916981
    - Sign accuracy: 0.61593
    - White winning recall: 0.8443625909621777
    - Black winning recall: 0.4324900764263286
    - Draw recall: 0.0
    - Centipawn ±200 accuracy: 0.76108
    - AUC TA: 0.85

---

## [2025-11-17] - Ferran Villarta & Natan Sisoev

- learned about implementations and the model itself
- searched information (https://www.chessprogramming.org/Evaluation) (https://hxim.github.io/Stockfish-Evaluation-Guide/) (https://chess.stackexchange.com/questions/347/what-is-an-accurate-way-to-evaluate-chess-positions)
- added potential features to develop

## [2025-11-17] - Natan Sisoev

- documentation about FEN
- cleaned evaluation (removed forced checkmates)
- basic metrics and plots
- implemented different piece values (`extract_piece_values`)
- results (`GradientBoostingRegressor`, `max_depth=16`):
  - `WHITE_PIECES`, `BLACK_PIECES`
    - Mean Squared Error: 69516.93986923502
    - Sign accuracy: 0.6264454499748617
    - Positive sign recall: 0.9408960915157293
    - Negative sign recall: 0.3462566844919786
  - `WHITE_PIECES`, `BLACK_PIECES`, `WHITE_PIECES_VALUE`, `BLACK_PIECES_VALUE`
    - Mean Squared Error: 46940.63704518973
    - Sign accuracy: 0.6767219708396179
    - Positive sign recall: 0.8960915157292659
    - Negative sign recall: 0.5427807486631016
- new metric: `centipawn_accuracy`, measures the accuracy of the prediction being in +- tolerance rango of the real evaluation
- predicted piece values: predicts the value of each piece (the coefficients of the linear regressor)
- new graph: AUC TOL-ACC (maybe a good metric to keep an eye on)

---

## [2025-11-12] - Natan Sisoev

- explored the chessData.csv dataset
- extracted `WhitePieces` and `BlackPieces` from `FEN`
- tried basic RandomForestClassifier and LinearRegression with these two features
- dataset not totally balanced so had to check if the model always predicted one label
- analyzed predicted sign only: accuracy, recall and scatter plot to see correlation
- around 0.67 accuracy and the scatter plot looked ok for only two features
- decided to continue with this dataset

---

