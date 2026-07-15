"""Shared pipeline glue for training and inference.

Keeps `eot_features.py` purely about (causal) features; this module handles
data loading, language routing, and the model artifact format.
"""
import os
import csv

import numpy as np

from features import load_wav
from eot_features import precompute, extract, FEATURE_NAMES


def language_of(audio_file):
    """Route a row to a language model by filename prefix.

    Hidden test is mostly Hindi, so unknown prefixes default to Hindi.
    """
    base = os.path.basename(audio_file).lower()
    if base.startswith("en"):
        return "english"
    if base.startswith("hi"):
        return "hindi"
    return "hindi"


def featurize_dir(data_dir):
    """Read labels.csv, load audio once per file, return per-pause rows.

    Each row: turn_id, pause_index, lang, feat (25-dim), label (or None).
    Features are causal (see eot_features): only audio before pause_start.
    """
    with open(os.path.join(data_dir, "labels.csv")) as f:
        rows = list(csv.DictReader(f))
    cache = {}
    out = []
    for r in rows:
        path = os.path.join(data_dir, r["audio_file"])
        if path not in cache:
            x, sr = load_wav(path)
            cache[path] = (x, sr, precompute(x, sr))
        x, sr, pre = cache[path]
        feat = extract(pre, x, sr, float(r["pause_start"]), int(r["pause_index"]))
        out.append({
            "turn_id": r["turn_id"],
            "pause_index": r["pause_index"],
            "lang": language_of(r["audio_file"]),
            "feat": feat,
            "label": r.get("label"),
        })
    return out


def subset_index(feats):
    return [FEATURE_NAMES.index(f) for f in feats]
