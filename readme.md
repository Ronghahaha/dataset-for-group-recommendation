# train.py

NGRN training for **item → groups** ranking: load joint social + rating data, build k-core candidate groups, train with ranking + tag loss, pick the best checkpoint on validation, report Prec/Rec/NDCG@K on the test split.

**Depends on:** `src/data/dataset.py`, `src/models/ngrn.py`  
**Packages:** `torch`, `numpy`, `networkx` (see `requirements.txt`)

## Data

`--data_dir` needs `joint_ratings.csv`, `joint_social_edges.csv`, `joint_movies.csv`, and usually `joint_social_groups.csv` (from `scripts/build_dataset.py`). Example: `data/ml100k`.

## Run

```bash
python train.py --data_dir data/ml100k
python train.py --data_dir data/ml1m --device cuda --metrics_json out.json
```

**Useful flags:** `--k_core`, `--epochs`, `--eval_every`, `--test_ratio`, `--max_group_size`, `--max_fallback_groups`, `--seed`  
**ML-10M:** add `--tag_loss_group_chunk 512 --max_neighbors 300`

All options: `python train.py --help`
