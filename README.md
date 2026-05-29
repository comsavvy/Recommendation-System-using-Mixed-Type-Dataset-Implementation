# Recommendation Systems with Mixed-Type Data: An Empirical Comparison on MovieLens-1M

This repository contains the implementation accompanying my AIMS Rwanda research thesis, which compares three families of recommender systems on the MovieLens-1M dataset under a unified, leakage-free evaluation protocol. Each family is trained under both a pointwise rating-prediction loss and a pairwise ranking loss, so the contribution of the loss function can be separated from that of the architecture. The central question is how different modelling families handle datasets that mix numerical, categorical, and binary features alongside user-item interaction data.

**Author:** Olusola Timothy Ogundepo

**Supervisor:** Professor Ernest Fokoué

**Institution:** African Institute for Mathematical Sciences (AIMS), Rwanda

## Models compared

Seven trained variants are compared, arranged as three pointwise/pairwise pairs plus a feature-poor CatBoost baseline:

| Model | Loss | Family | Mixed-type handling |
|---|---|---|---|
| Matrix Factorisation (MF) | Squared error (pointwise) | Collaborative filtering baseline | None, ratings only |
| BPR-MF | BPR (pairwise) | Collaborative filtering | None, ratings only |
| CatBoost (base, 22 features) | RMSE (pointwise) | Gradient-boosted regression trees | Native categorical splits |
| CatBoost (extended, 42 features) | RMSE (pointwise) | Gradient-boosted regression trees | Adds rating-derived statistics |
| CatBoost-YetiRank | YetiRank (pairwise) | Gradient-boosted regression trees | Extended feature set, ranking loss |
| Feature-augmented LightGCN | MSE (pointwise) | Graph neural network | Feature projection at layer 0 + graph propagation |
| LightGCN-BPR | BPR (pairwise) | Graph neural network | Feature projection at layer 0 + graph propagation |

All seven variants share the same time-based 80/10/10 split, the same evaluation protocol, and the same encoded feature representation, so any performance difference is attributable to the modelling family and the loss function rather than the experimental setup.

## Mathematical summary

Given user features $z_u$, item features $x_i$ and the global training mean $\mu$, the prediction rules used are:

- Matrix factorisation: $\hat r_{ui} = \mu + b_u + b_i + \mathbf{p}_u \cdot \mathbf{q}_i$
- CatBoost: $\hat r_{ui} = \mu + \sum_{t=1}^{T} \gamma\, h_t([z_u, x_i])$
- LightGCN: $\hat r_{ui} = \mu + b_u + b_i + \mathbf{e}_u^{(L)} \cdot \mathbf{e}_i^{(L)}$ with layer-mean aggregation over $L = 2$ propagation rounds and $\mathbf{e}_u^{(0)} = \mathrm{emb}(u) + W_u z_u$.

The pairwise variants (BPR-MF and LightGCN-BPR) keep the same predictor but are trained on the BPR loss

$$\mathcal{L}_{\mathrm{BPR}} = -\sum_{(u,i)\in K} \log \sigma(\hat r_{ui} - \hat r_{uj}) + \lambda\,\Omega$$

with one uniformly sampled negative item $j$ per observed pair. CatBoost-YetiRank uses CatBoost's native YetiRank pairwise objective with rated pairs grouped by user.

Evaluation uses RMSE and MAE for rating accuracy (pointwise models only) and Precision@10, Recall@10, NDCG@10 for ranking quality, averaged over five random samples of 200 test users (seeds 42, 123, 2024, 7, 2026). A non-parametric paired bootstrap with 10,000 resamples is applied to the six headline inter-model contrasts, attaching 95% confidence intervals and one-sided p-values to the reported gaps.

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

Random seeds are fixed at 42 for NumPy and PyTorch so that splits, mini-batches, model initialisation and negative sampling are reproducible. The ranking evaluation uses a separate list of five user-sampling seeds (42, 123, 2024, 7, 2026).

## Running the experiments

Open `RS_Code_Implementation.ipynb` and run the cells top to bottom. The notebook is organised in eight parts:

1. Setup
2. Data loading and exploration
3. Feature engineering and time-based split
4. Matrix factorisation (pointwise) and BPR-MF (pairwise)
5. CatBoost (base, extended, and YetiRank pairwise)
6. Feature-augmented LightGCN (pointwise) and LightGCN-BPR (pairwise)
7. Comparison tables, bar plots, and paired-bootstrap significance test
8. Interpretability analysis (CatBoost feature importances, LightGCN embedding PCA)

## Results on the test split

**Rating accuracy** (pointwise models, full test split):

| Model | RMSE ↓ | MAE ↓ |
|---|---|---|
| Matrix factorisation | 0.9155 | **0.7170** |
| CatBoost (base, 22 features) | 1.0195 | 0.8214 |
| CatBoost (extended, 42 features) | 0.9387 | 0.7378 |
| Feature-augmented LightGCN | **0.9087** | 0.7175 |

**Ranking quality** (mean ± standard deviation over five evaluation seeds):

| Model | P@10 ↑ | R@10 ↑ | NDCG@10 ↑ |
|---|---|---|---|
| Matrix factorisation (pointwise) | 0.1145 ± 0.0161 | 0.0311 ± 0.0057 | 0.1247 ± 0.0172 |
| BPR-MF (pairwise) | 0.1810 ± 0.0197 | 0.0605 ± 0.0075 | 0.1968 ± 0.0196 |
| Feature-augmented LightGCN (pointwise) | 0.1270 ± 0.0174 | 0.0415 ± 0.0060 | 0.1398 ± 0.0207 |
| LightGCN-BPR (pairwise) | **0.2026 ± 0.0255** | **0.0650 ± 0.0051** | **0.2223 ± 0.0264** |
| CatBoost extended (pointwise) | 0.0629 ± 0.0049 | 0.0173 ± 0.0025 | 0.0662 ± 0.0057 |
| CatBoost-YetiRank (pairwise) | 0.0164 ± 0.0043 | 0.0061 ± 0.0034 | 0.0183 ± 0.0060 |
| CatBoost (base, 22 features) | 0.0533 ± 0.0040 | 0.0141 ± 0.0015 | 0.0493 ± 0.0032 |

Bold = best in column. LightGCN-BPR wins on every ranking metric; on rating accuracy the feature-augmented LightGCN takes RMSE and is tied with matrix factorisation on MAE. The paired-bootstrap analysis (see notebook part 7) confirms every directional claim above at the conventional α = 0.05 threshold, with LightGCN-BPR over BPR-MF on Recall@10 as the single marginal case (p = 0.031).

## Citing this work

If you use this code, please cite the accompanying thesis (AIMS Rwanda, 2026). A BibTeX entry will be added once the thesis is deposited.

## License

The code in this repository is released under the MIT License. The MovieLens-1M dataset is distributed separately under its own terms; see GroupLens for details.
