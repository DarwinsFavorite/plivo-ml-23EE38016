# End-of-Turn Detection — Plivo Assignment (23EE38016)

Predicts `p_eot` (probability the turn is over) for each pause in a user turn,
using **only audio before the pause** (causal), no ASR and no pretrained weights.

## Results (out-of-fold, honest hidden-set estimate)
| Language | silence baseline | this model |
|----------|-----------------:|-----------:|
| English  | 1600 ms | **~1249 ms** |
| Hindi    | 850 ms  | **~824 ms**  |

Metric = mean response delay at ≤5% false-cutoff rate (lower is better). See
[SUMMARY.html](SUMMARY.html) for the full write-up and charts.

## Deliverables
- **[SUMMARY.html](SUMMARY.html)** — solution, results, charts, human-vs-agent split.
- **[predict.py](predict.py)** — `python predict.py --data_dir <folder> --out predictions.csv`
- **[predictions.csv](predictions.csv)** — predictions for both provided folders (English + Hindi rows).
- **[RUNLOG.md](RUNLOG.md)** — chronological scoring log.
- **[NOTES.md](NOTES.md)** — signal used, failure modes, next steps.

## How to run
```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy scikit-learn pandas librosa joblib
python predict.py --data_dir <folder> --out predictions.csv
# then score with the assignment's official scorer (starter/score.py from the handout):
python score.py --data_dir <folder> --pred predictions.csv
```
Retrain the saved model (`eot_model.joblib`) with `python train_model.py` (needs
the `eot_data/` folder from the handout in the working directory).

## Layout
- `eot_features.py` — causal feature extraction (graders: causality contract at top).
- `eot_pipeline.py` — data loading + per-language routing.
- `features.py` — low-level audio utilities (from the starter kit).
- `train_model.py` / `predict.py` — fit and inference.
- `eot_model.joblib` — saved per-language logistic-regression models.
