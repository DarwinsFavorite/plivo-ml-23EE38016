"""Predict p_eot for every pause in a data folder.

    python predict.py --data_dir <folder> --out predictions.csv

Loads the saved per-language models (never refits), routes each pause to its
language model by filename prefix, and writes turn_id,pause_index,p_eot.
Works on an unseen folder with the same structure/labels schema.
"""
import argparse
import csv
import os
import sys

import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eot_pipeline import featurize_dir, subset_index

MODEL = os.path.join(os.path.dirname(__file__), "eot_model.joblib")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    artifact = joblib.load(MODEL)
    rows = featurize_dir(args.data_dir)

    preds = []
    for r in rows:
        m = artifact[r["lang"]]
        idx = subset_index(m["feats"])
        x = r["feat"][idx].reshape(1, -1)
        p = float(m["clf"].predict_proba(m["scaler"].transform(x))[0, 1])
        preds.append((r["turn_id"], r["pause_index"], p))

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for tid, pi, p in preds:
            w.writerow([tid, pi, f"{p:.5f}"])
    print(f"wrote {len(preds)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
