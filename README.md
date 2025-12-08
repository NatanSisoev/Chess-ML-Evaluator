# Chess-ML-Evaluator

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/NatanSisoev/Chess-ML-Evaluator)

A machine-learning-based chess position evaluator. This project extracts features from chess positions in FEN format and trains models to predict position evaluations.

## Table of Contents

- [Overview](#overview)
- [Folder Structure](#folder-structure)
- [Getting Started](#getting-started)
- [Features](#features)
- [Changelog](#changelog)
- [References](#references)
- [Authors](#authors)

## Overview

## Overview

Chess-ML-Evaluator is a machine-learning framework designed to predict chess position evaluations from FEN notation. The primary motivation is to provide a fast and accurate evaluation function for chess engines or analytical tools, focusing on **relative position quality** rather than absolute centipawn values. This makes it particularly useful for engines or algorithms that rely on comparing positions in a minimax or search-based strategy.

The project workflow is structured around **modular managers** and **feature transformers** in the `chess_eval` library:

- **DataManager**: Handles loading, cleaning, sampling, applying feature transformations, and train-test splitting.
- **ModelManager**: Manages model creation, fitting, predicting, cross-validation, and hyperparameter optimization.
- **MetricsManager**: Provides scoring, evaluation metrics, plotting, and ranking of models.
- **Transformers**: Extract structured features from FEN strings, including piece counts, pawn structures, king safety, mobility, attack potential, and board control.

The library is complemented by **Jupyter notebooks**, which demonstrate the full workflow:

1. Exploratory data analysis
2. Preprocessing and cleaning
3. Feature engineering
4. Metric evaluation and selection
5. Cross-validation
6. Model selection
7. Hyperparameter optimization
8. Testing and final evaluation

Additionally, the project includes **pre-trained models**, along with scripts to generate and save these resources and calculated feature datasets for fast reuse:

- `save_datasets.ipynb`: generates and pickles datasets with all calculated features.
- `save_models.ipynb`: trains and stores optimized models for later evaluation or deployment.
- `storage.ipynb`: allows inspection, loading, and testing of saved datasets and models.

This modular design allows for flexible experimentation with different models, features, and evaluation metrics while maintaining reproducibility and scalability for large chess datasets.

## Folder Structure

```
📁 Chess-ML-Evaluator/                   # repo folder
├── 📁 chess_eval/                       # main library
│   ├── 📄 __init__.py                   # imports
│   ├── 📄 config.py                     # constants, styles, and models
│   ├── 📄 features.py                   # feature transformers
│   └── 📄 managers.py                   # data, model, metrics, and storage managers
├── 📁 data/                             # datasets
│   ├── 📁 features/                     # saved pre-calculated datasets
│   │   ├── 🔢 metadata.json             # metadata information about the datasets
│   │   └── 📄 ...                       # pickled datasets 
│   └── 📁 raw/                          # kaggle datasets
│       └── 📄 ...                       # comma separated files 
├── 📁 models/                           # saved pre-fitted models
│   ├── 🔢 metadata.json                 # metadata information about the models
│   └── 📄 ...                           # pickled models 
├── 📁 notebooks/                        # notebooks working through the project
│   ├── 📄 1_eda.ipynb                   # exploratory data analysis
│   ├── 📄 2_preprocessing.ipynb         # data preprocessing
│   ├── 📄 3_features.ipynb              # feature engineering
│   ├── 📄 4_metric.ipynb                # metric selection
│   ├── 📄 5_cv.ipynb                    # cross-validation
│   ├── 📄 6_models.ipynb                # model selection
│   ├── 📄 7_hyperparameters.ipynb       # hyperparameter optimization
│   └── 📄 8_testing.ipynb               # model testing
├── 📁 plots/                            # saved plots
│   └── 📄 ...                           # names figures with the results
├── 📁 storage/                          # scripts to save all models and datasets
│   ├── 📄 save_datasets.ipynb           # generate and pickle datasets with features
│   ├── 📄 save_models.ipynb             # train and pickle optimized models
│   └── 📄 storage.ipynb                 # see stored datasets and models, and test them
├── 📄 .gitignore                        # files ignored by git
├── 📄 .python-version                   # development python version
├── 📄 CHANGELOG.md                      # record of all changes
├── 📄 pyproject.toml                    # configuration file
├── 📄 README.md                         # README
└── 📄 uv.lock                           # package versions
```

## Getting Started

This project uses **[uv](https://uv.rst.sh/)** for dependency and environment management.

1. Clone the repository.

```bash
git clone https://github.com/NatanSisoev/Chess-ML-Evaluator.git
cd Chess-ML-Evaluator
```

2. Initialize the virtual environment using `uv`.

```bash
uv init
```

This will create a virtual environment in the project folder (typically under `.venv`).

3. Synchronize dependencies.

```bash
uv sync
```

This installs all dependencies listed in `pyproject.toml`.

4. Download dataset (see [official documentation](https://www.kaggle.com/docs/api)).

```bash
kaggle datasets download -d ronakbadhe/chess-evaluations -p data/raw --unzip
```

## Features

- **PieceInfo**: White_Pieces, Black_Pieces, White_Piece_Value, Black_Piece_Value, White_Pawn, Black_Pawn, White_Knight, Black_Knight, White_Bishop, Black_Bishop, White_Rook, Black_Rook, White_Queen, Black_Queen, White_Central_Pieces, Black_Central_Pieces
- **PawnStructure**: Isolated_Pawns_White, Isolated_Pawns_Black, Doubled_Pawns_White, Doubled_Pawns_Black, Passed_Pawns_White, Passed_Pawns_Black, Pawn_Islands_White, Pawn_Islands_Black
- **KingSafety**: King_Castled_White, King_Castled_Black, King_Pawns_White, King_Pawns_Black, Open_Files, King_Distance_to_Last_Rank_White, King_Distance_to_Last_Rank_Black, King_Castling_Rights_White, King_Castling_Rights_Black, King_Front_Pawns_White, King_Front_Pawns_Black, King_In_Check_White, King_In_Check_Black, King_Attacked_Neighbours_White, King_Attacked_Neighbours_Black
- **Mobility**: Mobility_King_White, Mobility_Queen_White, Mobility_Rook_White, Mobility_Bishop_White, Mobility_Knight_White, Mobility_Pawn_White, Mobility_King_Black, Mobility_Queen_Black, Mobility_Rook_Black, Mobility_Bishop_Black, Mobility_Knight_Black, Mobility_Pawn_Black
- **Attack**: Threats_Created_White, Threats_Created_Black, Hanging_Pieces_White, Hanging_Pieces_Black, Hanging_Points_White, Hanging_Points_Black, Undefended_Pieces_White, Undefended_Pieces_Black, Undefended_Points_White, Undefended_Points_Black
- **BoardControl**: Central_Squares_Control_White, Central_Squares_Control_Black, Open_Columns, SemiOpen_Columns_White, SemiOpen_Columns_Black, Rook_on_Open_Column_White, Rook_on_Open_Column_Black, Protected_Advanced_Pawn_White, Protected_Advanced_Pawn_Black, Rook_Queen_Aligned_White, Rook_Queen_Aligned_Black, Rooks_Aligned_White, Rooks_Aligned_Black, Controlled_Squares_White, Controlled_Squares_Black
- **GameInfo**: Halfmove_Clock, Fullmove_Number, Phase, Side_To_Move_White, Has_En_Passant

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a complete record of changes.

## References

- Dataset: https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations/data
- Python-chess: https://www.kaggle.com/code/wlifferth/part-1-understanding-python-chess-and-fen
- Chess FEN: https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
- Piece Values: https://chess.fandom.com/wiki/Value

## Authors

- **Natan Sisoev** | [natan.sisoev@gmail.com](mailto:natan.sisoev@gmail.com) | [GitHub](https://github.com/NatanSisoev)

- **Ferran Villarta** | [villartaferran@gmail.com](mailto:villartaferran@gmail.com) | [GitHub](https://github.com/Ferran-Villarta)
