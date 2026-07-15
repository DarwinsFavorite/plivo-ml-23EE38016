# NOTES

The model scores each pause from causal prosody in the ≤1.5 s of audio strictly
before `pause_start`: terminal F0 contour (linear slope, quadratic boundary tone,
final level/range), energy decay into the pause (slope, peak-normalized drop),
spectral tilt and harmonicity, speaking-rate deceleration and voiced-run rhythm,
plus turn position (`pause_index`, elapsed time). Two cues are language-specific —
final-window pitch flatness helps Hindi (flat = hesitation/hold) while energy
spread helps English — so we ship per-language balanced logistic-regression models
routed by filename prefix. It beats the silence-only baseline because that policy
fires on every pause and must lean entirely on the silence timer, whereas our
`p_eot` lets the scorer use a shorter global delay while holding false cutoffs at
≤5%. Honest out-of-fold results: English ~1249 ms vs a 1600 ms baseline (a clear
~350 ms win), Hindi ~824 ms vs an already-aggressive 850 ms baseline (a marginal
win). Hindi is hard because its pauses are short, so the silence timer is already
strong, and the decisive end-of-turn cue — the sentence-final verb (Hindi is
SOV) — is semantic and unavailable without ASR, which the rules forbid. The
residual failures are true ends with flat/ambiguous prosody that score low and
time out, and long hesitation-holds ("aurrr…") that occasionally score high. A paired test swapping in `librosa.pyin` pitch tracking for Hindi (motivated by
its dominant cue, terminal F0-fall, d=−0.54) gave ~799±36 ms vs ~824±25 ms for
the shipped autocorr tracker — a real-direction but modest gain (pyin won 8/12
seeds, noisier, and ~10× slower), so it was not adopted for the final model. With
one more day: revisit that pyin trade-off with more seeds/data, gather more
Hindi turns, and explore a per-language silence-timer prior.
