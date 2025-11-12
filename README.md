# Chess Evaluations

ML based chess positions evaluator.

# Dataset

Dataset: https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations

To download run (see [official documentation](https://www.kaggle.com/docs/api)):

```bash
kaggle datasets download -d ronakbadhe/chess-evaluations -p data --unzip
```

# Changelog

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
