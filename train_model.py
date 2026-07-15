"""Fit the per-language EOT models and save one artifact.

    python train_model.py

Per-language balanced logistic regression on the language-appropriate feature
subset (pitch_std helps Hindi/hurts English; energy_range_p the reverse). Chosen
by repeated randomized grouped-CV on the real delay@5% metric — see RUNLOG.md.
"""
import os
import sys

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eot_features import ENGLISH_FEATS, HINDI_FEATS
from eot_pipeline import featurize_dir, subset_index

DATA = os.path.join(os.path.dirname(__file__), "eot_data")
OUT = os.path.join(os.path.dirname(__file__), "eot_model.joblib")
SUBSETS = {"english": ENGLISH_FEATS, "hindi": HINDI_FEATS}


def main():
    artifact = {}
    for lang in ("english", "hindi"):
        rows = featurize_dir(os.path.join(DATA, lang))
        feats = SUBSETS[lang]
        idx = subset_index(feats)
        X = np.array([r["feat"] for r in rows])[:, idx]
        y = np.array([1 if r["label"] == "eot" else 0 for r in rows])
        scaler = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=4000, class_weight="balanced",
                                 solver="liblinear").fit(scaler.transform(X), y)
        artifact[lang] = {"scaler": scaler, "clf": clf, "feats": feats}
        print(f"{lang}: trained on {len(y)} pauses ({y.sum()} eot), {len(feats)} features")
    joblib.dump(artifact, OUT)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
