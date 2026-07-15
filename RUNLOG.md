# RUNLOG

**Metric:** mean response delay (ms) at ≤5% false-cutoff rate, from `score.py`.
Lower is better. Reported per language (hidden test is mostly Hindi).

**How scores are measured:** out-of-fold — grouped 5-fold CV on `turn_id` (a turn
is never split across folds), predictions fed to the real `score.py`. From Run 5
on, we average over 12–15 randomized fold assignments and report mean ± std,
because single splits on 100 turns/language proved too noisy to trust.

**Silence baseline to beat:** English 1600 ms, Hindi 850 ms. (Hindi is only 850 ms
because its pauses are short, so a plain silence timer is already strong.)

---

### Run 1 — Silence-only baseline
Score: **English 1600 ms · Hindi 850 ms**
Changed: ran the given baseline (`p_eot=1` for every pause). Why: establishes the
number to beat; the agent then relies purely on the swept silence timer.

### Run 2 — 15 causal prosodic features + logistic regression
Score: **English 1250 ms · Hindi 872 ms** (single split)
Changed: built the causal feature extractor (F0 slope/level, energy decay, spectral
tilt, voiced-run rhythm, turn position) and a balanced logistic-regression model.
Why: pause duration is the obvious cue but it is future info and banned, so the
model must read pre-pause prosody instead.

### Run 3 — +quadratic-F0, articulatory relaxation, syllable-rate (→22 features)
Score: **English 1197 ms · Hindi 872 ms** (single split)
Changed: added quadratic-F0 boundary tone, rolloff-drop + harmonicity, and
syllable-rate decay. Why: ablation showed these lift separation, especially for
Hindi (its end-of-turns are pitch- and voice-quality-driven).

### Run 4 — +energy_decay_ratio (→23 features)
Score: **English 1144 ms · Hindi 857 ms** (single split)
Changed: added peak-normalized trailing-off (mean last 200 ms ÷ window peak).
Why: a scale-invariant "voice trailing off" cue; biggest single English gain.

### Run 5 — Switched to repeated randomized grouped-CV (honest re-measure)
Score: **English 1291 ± 45 ms · Hindi 828 ± 26 ms**
Changed: same features, but scored over 15 randomized fold assignments instead of
one split. Why: the earlier single-split numbers were optimistic; this is the
trustworthy estimate for unseen data.

### Run 6 — +pitch_std, Hindi only
Score: **Hindi 824 ± 25 ms** (English unchanged)
Changed: added final-window pitch flatness to the Hindi model only. Why: Hindi
hold-pauses are hesitation-prolongations with flat pitch (helps Hindi); the same
feature hurt English, so it is language-specific.

### Run 7 — +energy_range_p, English only
Score: **English 1249 ± 45 ms** (Hindi unchanged)
Changed: added eGeMAPS-style energy spread (p80−p20 over last 1 s) to the English
model only. Why: it separates English eot/hold but hurt Hindi — mirror image of
pitch_std, confirming the need for per-language feature subsets.

### Run 8 — Recall-oriented / duration-weighted training (Hindi)
Score: **Hindi 824 ± 25 ms** (no change)
Changed: weighted training samples by metric impact (upweight true-ends + long
holds). Why: attempted to lift EOT recall — the binding constraint — but the gain
was absorbed by the scorer's threshold sweep. Reverted to plain balanced weights.

### Run 9 — FINAL model, in-sample `score.py` on provided folders
Score: **English 1116 ms · Hindi 736 ms** (in-sample, optimistic)
Changed: trained per-language models on all data, saved `eot_model.joblib`, ran
`predict.py` → `predictions.csv` → `score.py`. Why: produces the submitted file.
These beat the honest out-of-fold estimates because the model is scored on its own
training data; the realistic hidden-set expectation remains **English ~1249 ms,
Hindi ~824 ms** (Runs 6–7).

---

## Current status
- **English: clear win** — ~1249 ms out-of-fold vs 1600 ms baseline (~350 ms).
- **Hindi: marginal win** — ~824 ms out-of-fold vs 850 ms baseline (~26 ms, within
  seed noise on some splits). No feature or training change cracked it further; the
  decisive Hindi cue (the sentence-final verb) is semantic and needs ASR, which the
  rules forbid.
- **Explored, not adopted:** `librosa.pyin` pitch tracking for Hindi (~799 ms in a
  first test) — hurts English and is ~10× slower; left as future work in NOTES.md.
