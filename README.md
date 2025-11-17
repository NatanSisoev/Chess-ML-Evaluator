# Chess Evaluations

ML based chess positions evaluator.

# Dataset

Dataset: https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations

To download run (see [official documentation](https://www.kaggle.com/docs/api)):

```bash
kaggle datasets download -d ronakbadhe/chess-evaluations -p data --unzip
```

# Changelog

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

##### 2025-11-12 - Natan Sisoev

- explored the chessData.csv dataset
- extracted `WhitePieces` and `BlackPieces` from `FEN`
- tried basic RandomForestClassifier and LinearRegression with these two features
- dataset not totally balanced so had to check if the model always predicted one label
- analyzed predicted sign only: accuracy, recall and scatter plot to see correlation
- around 0.67 accuracy and the scatter plot looked ok for only two features
- decided to continue with this dataset

# Authors

- Natan Sisoev
- Ferran Villarta
