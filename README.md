# Chess Evaluations

ML based chess positions evaluator.

# Dataset

Dataset: https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations

To download run (see [official documentation](https://www.kaggle.com/docs/api)):

```bash
kaggle datasets download -d ronakbadhe/chess-evaluations -p data --unzip
```

# TODO

- [ ] narrate the full code (flowy from start to end)
- [ ] count trivial threats by pawns: calculate the value of all pieces threatened by opposite pawns
- [ ] add more features
- [ ] remove useless features, find best feature combinations (high `read_size`, low `sample_size`)

# Features

1. Material
Nombre de peons blanc/negre
Nombre de cavalls blanc/negre
Nombre d’alfils blanc/negre
Nombre de torres blanc/negre
Nombre de dames blanc/negre
Material total blanc
Material total negre
2. Mobilitat (per cada peça)
Mobilitat per peça (cavalls, alfils, torres, dames)
3. Seguretat del rei
Distància del rei a la línia de fons
Línia de peons defensant rei (linia superior directament, o també podríem fer la següent)
Enroc fet (sí/no)
Enroc disponible (sí/no)
Nombre de peces enemigues atacant caselles al voltant del rei
4. Control del tauler
Nombre de caselles centrals controlades (d4, d5, e4, e5) (també podríem valorar com si fos una convolució, tot el taulell, així tenim en compte totes les caselles, i hi assignem el valor en funció del que creiem) o, bé podríem crear una altra feature que sigui una total i una altra  més específica
Control de columnes obertes i semiobertes
Torre en columna oberta (sí/no)
Peó avançat protegit (sí/no)
Torre + dama alineades
Torres aliniades
Caselles controlades per cada peça  (igual que la llista que he fet abans del material)
5. Estructura peons
Files sense peons
Nombre d’illetes de peons
Nombre de peons en caselles del color de l’alfil ja que tenim alfil en blanc i alfil en negre
Peons passats
Peons doblats
Peons aïllats
Peons endarrerits
Peons candidats a promoció
6. Miscellaneous
Fase del joc (obertura, mig joc, final) (binari per normalització) MOLT IMPORTANT !!! (crec)
Halfmove clock
Player to move
7. Relacions (totals, amb punts)
Nombre d’amenaces creades
Nombre de peces penjades (atacada & no defensada)
Nombre de peces indefenses
8. Features derivades (però crec que tot això ja ho gestiona bastant la fase del joc)
Mobilitat relativa (blanc/negre)
Peons passats protegits − peons passats enemics
Control central relatiu


# Changelog
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

# Authors

- Natan Sisoev
- Ferran Villarta
