# Recommendation Systems with Mixed-Type Data: An Empirical Comparison on MovieLens-1M

This repository contains the implementation accompanying my AIMS Rwanda research thesis, which compares three families of recommender systems on the MovieLens-1M dataset under a unified, leakage-free evaluation protocol. The central question is how different modelling families handle datasets that mix numerical, categorical, and binary features alongside user-item interaction data.

**Author:** Olusola Timothy Ogundepo

**Supervisor:** Professor Ernest Fokoué

**Institution:** African Institute for Mathematical Sciences (AIMS), Rwanda

## Models compared

| Model | Family | Mixed-type handling |
|---|---|---|
| Matrix Factorisation (SGD with biases) | Collaborative filtering baseline | None, ratings only |
| CatBoost (base, 22 features) | Gradient-boosted regression trees | Native categorical splits |
| CatBoost (extended, 42 features) | Gradient-boosted regression trees | Adds rating-derived statistics |
| Feature-augmented LightGCN | Graph neural network | Feature projection at layer 0 + graph propagation |

All four variants share the same time-based 80/10/10 split, the same evaluation protocol, and the same encoded feature representation, so any performance difference is attributable to the modelling family rather than the experimental setup.

## Mathematical summary

Given user features $z_u$, item features $x_i$ and the global training mean $\mu$, the prediction rules used are:

- Matrix factorisation: $\hat r_{ui} = \mu + b_u + b_i + \mathbf{p}_u \cdot \mathbf{q}_i$
- CatBoost: $\hat r_{ui} = \mu + \sum_{t=1}^{T} \gamma\, h_t([z_u, x_i])$
- LightGCN: $\hat r_{ui} = \mu + b_u + b_i + \mathbf{e}_u^{(L)} \cdot \mathbf{e}_i^{(L)}$ with layer-mean aggregation over $L = 2$ propagation rounds and $\mathbf{e}_u^{(0)} = \mathrm{emb}(u) + W_u z_u$.

Evaluation uses RMSE and MAE for rating accuracy, and Precision@10, Recall@10, NDCG@10 for ranking quality.

## Repository contents

- `RS_Code_Implementation.ipynb` — main notebook with the full pipeline: data loading, feature engineering, time-based split, training of all four models, evaluation, and interpretability diagnostics (CatBoost feature importance and PCA of LightGCN item embeddings).
- `RS_Experimentation.ipynb` — exploratory notebook with side experiments and intermediate checks.
- `ml-1m-README.txt` — original README from the MovieLens-1M release describing the raw files.

The MovieLens-1M data files are **not** included in the repository.

## Data

MovieLens-1M is a public benchmark dataset distributed by GroupLens.

- Ratings: $1{,}000{,}209$ on the integer scale $\{1, \dots, 5\}$
- Users: $6{,}040$ with gender, age band, and occupation
- Items: $3{,}706$ films with release year and 18 genre flags
- Period: April 2000 to February 2003

Download the dataset from <https://grouplens.org/datasets/movielens/1m/>, unzip it, and place the three `.dat` files inside a `datasets/` folder next to the notebooks:

```
Recommendation-System-using-Mixed-Type-Dataset-Implementation/
├── datasets/
│   ├── ratings.dat
│   ├── users.dat
│   └── movies.dat
├── RS_Code_Implementation.ipynb
└── ...
```

## Setup

The code is written in Python 3. The main dependencies are:

- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `scikit-learn` (min-max scaling, PCA, metrics)
- `catboost`
- `torch`
- `torch-geometric` (provides the `LGConv` layer)

A minimal install:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn catboost torch torch-geometric
```

Random seeds are fixed at 42 for NumPy, PyTorch, and the ranking sampler so that splits, mini-batches, and sampled evaluation users are reproducible.

## Running the experiments

Open `RS_Code_Implementation.ipynb` and run the cells top to bottom. The notebook is organised in eight parts:

1. Setup
2. Data loading and exploration
3. Feature engineering and time-based split
4. Matrix factorisation baseline
5. CatBoost (base and extended)
6. Feature-augmented LightGCN
7. Comparison table and bar plots
8. Interpretability analysis (CatBoost feature importances, LightGCN embedding PCA)

## Results on the test split

| Model | RMSE ↓ | MAE ↓ | P@10 ↑ | R@10 ↑ | NDCG@10 ↑ |
|---|---|---|---|---|---|
| Matrix factorisation | 0.9155 | **0.7170** | **0.1230** | 0.0345 | 0.1283 |
| CatBoost (base) | 1.0195 | 0.8214 | 0.0595 | 0.0146 | 0.0536 |
| CatBoost (extended) | 0.9387 | 0.7378 | 0.0675 | 0.0198 | 0.0733 |
| Feature-augmented LightGCN | **0.9087** | 0.7175 | 0.1195 | **0.0366** | **0.1350** |

Bold = best in column. Feature-augmented LightGCN wins on three of the five metrics and is essentially tied with matrix factorisation on the remaining two.

## Citing this work

If you use this code, please cite the accompanying thesis (AIMS Rwanda, 2026). A BibTeX entry will be added once the thesis is deposited.

## License

The code in this repository is released under the MIT License. The MovieLens-1M dataset is distributed separately under its own terms; see GroupLens for details.
