# Chess Evaluations

ML based chess positions evaluator.

# Dataset

Dataset: https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations

To download run (see [official documentation](https://www.kaggle.com/docs/api)):

```bash
kaggle datasets download -d ronakbadhe/chess-evaluations -p data --unzip
```

# Features

1. Material
Nombre de peons blanc/negre
Nombre de cavalls blanc/negre
Nombre d’alfils blanc/negre
Nombre de torres blanc/negre
Nombre de dames blanc/negre
Material total blanc
Material total negre
Diferència de material (blanc − negre)
Peons passats
Peons doblats
Peons aïllats
Peons endarrerits
Peons candidats a promoció
2. Mobilitat
Mobilitat total de les peces blanques
Mobilitat total de les peces negres
Mobilitat per peça (cavalls, alfils, torres, dames)
3. Seguretat del rei
Distància del rei a la línia de fons
Línia de peons defensant rei (linia superior directament, o també podríem fer la següent)
Torres connectades defensant el rei
Enroc fet (sí/no)
Enroc disponible (sí/no)
Nombre de peces enemigues atacant caselles al voltant del rei
4. Control del tauler
Nombre de caselles centrals controlades (d4, d5, e4, e5) (també podríem valorar com si fos una convolució, tot el taulell, així tenim en compte totes les caselles, i hi assignem el valor en funció del que creiem) o, bé podríem crear una altra feature que sigui una total i una altra  més específica
Control de columnes obertes i semiobertes
Torre en columna oberta (sí/no)
Peó avançat protegit (sí/no)
Torre + dama alineades
Caselles controlades per cada peça  (igual que la llista que he fet abans del material)
6. Estructura peons
Files sense peons
Nombre d’illetes de peons
Nombre de peons en caselles del color de l’alfil ja que tenim alfil en blanc i alfil en negre
7. Fase
Fase del joc (obertura, mig joc, final) (binari per normalització) MOLT IMPORTANT !!! (crec)
8. Relacions
Nombre d’amenaces creades
Nombre de peces penjades (atacada & no defensada)
Nombre de peces indefenses
Nombre de peces atacades menys peces defensades
Torn per jugar (blanc/negre)
9. Features derivades (però crec que tot això ja ho gestiona bastant la fase del joc)
Mobilitat relativa (blanc/negre)
Peons passats protegits − peons passats enemics
Control central relatiu


# Changelog

##### 2025-11-17 - Ferran Villarta
- learned about implementations and the model itself
- searched information (https://www.chessprogramming.org/Evaluation) (https://hxim.github.io/Stockfish-Evaluation-Guide/) (https://chess.stackexchange.com/questions/347/what-is-an-accurate-way-to-evaluate-chess-positions)


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
