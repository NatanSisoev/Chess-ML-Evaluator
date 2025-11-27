# Chess Evaluations

A machine-learning-based chess position evaluator. This project extracts features from chess positions in FEN format and trains models to predict position evaluations.

## Folder Structure

```
📁 Chess-ML-Evaluator/
├── 📁 chess_eval/
│   ├── 📄 __init__.py
│   ├── 📄 constants.py
│   ├── 📄 features.py
│   ├── 📄 managers.py
│   └── 📄 storage.py
├── 📁 data/
│   ├── 📁 raw/
│   │   ├── 📄 chessData.csv
│   │   ├── 📄 random_evals.csv
│   │   └── 📄 tactic_evals.csv
│   └── 📁 features/
│       ├── 📄 example.pkl
│       └── 🔢 metadata.json
├── 📁 models/
│   ├── 📄 example.pkl
│   └── 🔢 metadata.json
├── 📁 notebooks/
│   ├── 📄 1_eda.ipynb
│   ├── 📄 2_preprocessing.ipynb
│   ├── 📄 3_features.ipynb
│   ├── 📄 4_metric.ipynb
│   ├── 📄 5_cv.ipynb
│   ├── 📄 6_models.ipynb
│   ├── 📄 7_hyperparameters.ipynb
│   ├── 📄 mwe.ipynb
│   └── 📄 storage.ipynb
├── 📄 .gitignore
├── 📄 .python-version
├── 📄 pyproject.toml
├── 📄 README.md
└── 📄 uv.lock
```

## Getting Started

This project uses __[uv](https://uv.rst.sh/)__ for dependency and project management.

1. Install `uv` globally if not already installed:

```bash
pip install uv
```

2. Install the project and its dependencies (from `uv.lock`):

```bash
uv install
```

This will create a `.venv` (if not existing) and install all locked dependencies.

3. Activate the virtual environment managed by uv:

```bash
uv shell
```

4. Run scripts or notebooks using the environment:

```bash
uv run python notebooks/short.ipynb
```

5. Add a new dependency:

```bash
uv pip install <package_name>
uv lock  # update uv.lock
```

## Dataset

Original Dataset: https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations

Download and unzip (see [official documentation](https://www.kaggle.com/docs/api)):

```bash
kaggle datasets download -d ronakbadhe/chess-evaluations -p data/raw --unzip
```

## TODO

- [ ] notebooks
  - [x] 1_eda.ipynb (1.6 chunked sampling pending)
  - [x] 2_preprocessing.ipynb
  - [ ] 3_features.ipynb
  - [ ] 4_metric.ipynb
  - [ ] 5_cv.ipynb
  - [ ] 6_models.ipynb
  - [ ] 7_hyperparameters.ipynb
- [ ] transform the whole dataset with all features and save it to pkl for easier future access

## Features

### PieceInfo

- White_Pieces
- Black_Pieces
- White_Piece_Value
- Black_Piece_Value
- White_Pawn
- Black_Pawn
- White_Knight
- Black_Knight
- White_Bishop
- Black_Bishop
- White_Rook
- Black_Rook
- White_Queen
- Black_Queen
- White_Central_Pieces
- Black_Central_Pieces

### PawnStructure

- Isolated_Pawns_White
- Isolated_Pawns_Black
- Doubled_Pawns_White
- Doubled_Pawns_Black
- Passed_Pawns_White
- Passed_Pawns_Black
- Pawn_Islands_White
- Pawn_Islands_Black

### KingSafety

- King_Castled_White
- King_Castled_Black
- King_Pawns_White
- King_Pawns_Black
- Open_Files
- King_Distance_to_Last_Rank_White
- King_Distance_to_Last_Rank_Black
- King_Castling_Rights_White
- King_Castling_Rights_Black
- King_Front_Pawns_White
- King_Front_Pawns_Black
- King_In_Check_White
- King_In_Check_Black
- King_Attacked_Neighbours_White
- King_Attacked_Neighbours_Black

### Mobility

- Mobility_King_White
- Mobility_Queen_White
- Mobility_Rook_White
- Mobility_Bishop_White
- Mobility_Knight_White
- Mobility_Pawn_White
- Mobility_King_Black
- Mobility_Queen_Black
- Mobility_Rook_Black
- Mobility_Bishop_Black
- Mobility_Knight_Black
- Mobility_Pawn_Black

### Attack

- Threats_Created_White
- Threats_Created_Black
- Hanging_Pieces_White
- Hanging_Pieces_Black
- Hanging_Points_White
- Hanging_Points_Black
- Undefended_Pieces_White
- Undefended_Pieces_Black
- Undefended_Points_White
- Undefended_Points_Black

### BoardControl

- Central_Squares_Control_White
- Central_Squares_Control_Black
- Open_Columns_White
- Open_Columns_Black
- SemiOpen_Columns_White
- SemiOpen_Columns_Black
- Rook_on_Open_Column_White
- Rook_on_Open_Column_Black
- Protected_Advanced_Pawn_White
- Protected_Advanced_Pawn_Black
- Rook_Queen_Aligned_White
- Rook_Queen_Aligned_Black
- Rooks_Aligned_White
- Rooks_Aligned_Black
- Controlled_Squares_White
- Controlled_Squares_Black

### GameInfo

- Halfmove_Clock
- Fullmove_Number
- Phase
- Side_To_Move_White
- Has_En_Passant


## Changelog

##### 2025-11-27 - Natan Sisoev

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

##### 2025-11-26 - Natan Sisoev

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

##### 2025-11-25 - Natan Sisoev

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

##### 2025-11-24 - Ferran Villarta
- created `Relations` transformer:
  - `Relations`:
    - Threats_Created_White
    - Threats_Created_Black
    - Hanging_Pieces_White
    - Hanging_Pieces_Black
    - Hanging_Points_White
    - Hanging_Points_Black
    - Undefended_Pieces_White
    - Undefended_Pieces_Black
    - Undefended_Points_White
    - Undefended_Points_Black
    
##### 2025-11-24 - Ferran Villarta
- Another way to measure accuracy is to model our data as a normal distribution `N(μ, σ)`, and then accept the prediction if it falls within the range `[evidence ± z_0.05]`.

- created transformers:
  - `BoardControl`:
    - Central_Squares_Control_White
    - Central_Squares_Control_Black
    - Open_Columns_White
    - Open_Columns_Black
    - SemiOpen_Columns_White
    - SemiOpen_Columns_Black
    - Rook_on_Open_Column_White
    - Rook_on_Open_Column_Black
    - Protected_Advanced_Pawn_White
    - Protected_Advanced_Pawn_Black
    - Rook_Queen_Aligned_White
    - Rook_Queen_Aligned_Black
    - Rooks_Aligned_White
    - Rooks_Aligned_Black
    - Controlled_Squares_White
    - Controlled_Squares_Black

  - `PieceMobility`:
    - Mobility_King_White
    - Mobility_Queen_White
    - Mobility_Rook_White
    - Mobility_Bishop_White
    - Mobility_Knight_White
    - Mobility_Pawn_White
    - Mobility_King_Black
    - Mobility_Queen_Black
    - Mobility_Rook_Black
    - Mobility_Bishop_Black
    - Mobility_Knight_Black
    - Mobility_Pawn_Black

  - `MiscFeatures`:
    - Player_to_Move
    - Number_of_Moves
    - Halfmove_Clock
- commented the code

##### 2025-11-24 - Natan Sisoev

- finished king safety features
- explored R^2 and R^2 adjusted metrics
- improved managers' communications
- tried different read/sample sizes to see the effect

##### 2025-11-19 - Natan Sisoev

- created managers: 
  - `DataManager`:
    - read a lot of rows
    - sample a random subset
    - apply transformers
    - train and test split (TODO: cross-validation)
  - `ChessManager`: show board & features
  - `ModelManager`:
    - save `y_pred` for metrics
    - predict a single FEN
  - `MetricsManager`:
    - calculate accuracy, recall (white, black and draw), and centipawn accuracy (accuracy with tolerance)
    - scatter plot `y_pred` vs `y_true`
    - plot evaluation heatmap on 2 features plane
    - plot tolerance - accuracy line w/ AUC as title
- created transformers workflow and transformers:
  - `PieceInfo`:
    - white and black piece count
    - white and black separate types piece count
    - white and black total piece value
  - `PawnStructure`:
    - white and black isolated pawns (count)
    - white and black doubled pawns (count)
    - white and black passed pawns (count)
  - `KingSafety`:
    - white and black castled (True or False: king and rook in the castled position)
    - white and black pawns near king (count of pawns in the 9x9 grid around the king)
    - number of open files on the table
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

##### 2025-11-17 - Ferran Villarta

- learned about implementations and the model itself
- searched information (https://www.chessprogramming.org/Evaluation) (https://hxim.github.io/Stockfish-Evaluation-Guide/) (https://chess.stackexchange.com/questions/347/what-is-an-accurate-way-to-evaluate-chess-positions)
- added potential features to develop

##### 2025-11-17 - Natan Sisoev

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

##### 2025-11-12 - Natan Sisoev

- explored the chessData.csv dataset
- extracted `WhitePieces` and `BlackPieces` from `FEN`
- tried basic RandomForestClassifier and LinearRegression with these two features
- dataset not totally balanced so had to check if the model always predicted one label
- analyzed predicted sign only: accuracy, recall and scatter plot to see correlation
- around 0.67 accuracy and the scatter plot looked ok for only two features
- decided to continue with this dataset

## References

- Dataset: https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations/data
- Python-chess: https://www.kaggle.com/code/wlifferth/part-1-understanding-python-chess-and-fen
- Chess FEN: https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
- Piece Values: https://chess.fandom.com/wiki/Value

## Authors

- Natan Sisoev
- Ferran Villarta
