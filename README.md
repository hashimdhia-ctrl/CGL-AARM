[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22076271.svg)](https://doi.org/10.5281/zenodo.22076271)

# CGL-AARM: Bounded Prototype Banks for Memory-Efficient Continual Learning

Code accompanying the paper *"Bounded Prototype Banks for Memory-Efficient
Continual Learning: A Diagnostic Framework for Task Routing Under
Zero-Replay Constraints"* (Dhia Hashim).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22076271.svg)](https://doi.org/10.5281/zenodo.22076271)

## What this is

A continual learning system that retains task-specific knowledge as
**bounded statistical summaries** (per-task and per-class centroids, plus a
rank-64 PCA covariance factorisation) instead of caching raw features,
combined with a **six-signal calibrated fusion** router for identifying
which frozen task expert should handle an input at inference time.

The paper's central claims:
1. The bounded bank produces routing signals identical to a cached
   (raw-feature) configuration under the retention scheme used (Section
   4.3) — a property of the construction, not an independent compression
   result.
2. The six-signal fusion outperforms single-head Experience Replay by
   +26.41 pp at zero replay on CIFAR-100 (Table 1).
3. Task routing, not expert competence, is the dominant measured
   bottleneck: 21.71% routing accuracy against 74.82% conditional expert
   accuracy (Table 4).

See the paper for full results, limitations, and scope.

## Repository contents

| File | Produces | Paper section |
|---|---|---|
| `STEP_4_BOUNDED_BANK_OFFICIAL_TEST.py` | Main production pipeline: `Expert`, `HRouter`, `BoundedProtoBank`, `Fusion`, `calibrate`. Generates Table 1, Table 2, Table 5, Table 6. | 4.2, 4.3, 5.2, 5.3 |
| `tau_sweep_real_v53.py` | τ (LSE_TAU) sensitivity sweep, reusing the production pipeline verbatim. | 5.4, Table 7 |
| `CLOSE_OPEN_ITEMS.py` | Table 4 re-measurement (Stage 1), rank-K vs. full-rank covariance sweep (Stage 2), capacity-matched ER baselines (Stage 3). Pure NumPy except for one ResNet-18 feature-extraction call — runnable and checkable with `--smoke` (no GPU/torch required). | 4.5, 7 (limitations 1–3) |

Each script is self-contained and downloads CIFAR-100 automatically on
first run.

## Protocol (exact, as used in all reported results)

```
Backbone            ImageNet-pretrained ResNet-18, frozen after initialisation
Feature dimension   512
Dataset (primary)   CIFAR-100, 100 classes, 32x32
Dataset (secondary) TinyImageNet, 200 classes, 64x64
Task split           10 tasks x 10 classes/task (primary); 5-task variant secondary
Split seed            314159
Per-class budget      350 train / 75 calibration / 75 internal-evaluation
                       (official test set, 10,000 images, held out separately
                       and never used for tuning)
Seeds                  {42, 100, 2024, 777, 999}
Task orders            Order_A = ascending task blocks (0..9)
                        Order_B = reversed task blocks (9..0)
Memory levels (M)      {0, 50, 100, 200} total exemplars (main results)
                        {0, 200} (Stage 3 capacity-matched ER, lighter sweep)
LSE_TAU (locked)       0.05, selected on a prior calibration partition
                        before any official-test evaluation
LSE_TAU (sweep)        {0.01, 0.05, 0.10, 0.20}, post-hoc robustness check
Deployed PCA rank      64 (rank-K sweep tests {8,16,32,64,128,256,512})
Routing threshold      Chance = 10% (1/10 tasks); pre-declared practical
                        usefulness threshold = 25% (these are distinct
                        quantities — see Table 4's note in the paper)
Standard deviations    Population (ddof=0) for Table 1/Table 3 main results;
                        sample (ddof=1) for Table 4, inherited from the
                        Step 2 diagnostic source — this inconsistency is
                        disclosed in the paper's Limitations (item 4)
```

## Running the code

```bash
# Main production pipeline (Table 1, 2, 5, 6) — GPU strongly recommended
python STEP_4_BOUNDED_BANK_OFFICIAL_TEST.py

# tau sensitivity sweep (Table 7 / Section 5.4)
python tau_sweep_real_v53.py

# Table 4 re-measurement + rank-K sweep + capacity-matched ER (Section 7)
python CLOSE_OPEN_ITEMS.py --smoke   # ~30s sanity check, synthetic features, no GPU/torch needed
python CLOSE_OPEN_ITEMS.py           # full run, ~10 min feature extraction on a T4, then cached
```

If running inside a Jupyter/Colab notebook cell rather than a real command
line, note that `CLOSE_OPEN_ITEMS.py` uses `argparse.parse_known_args()`
specifically so it does not crash on Jupyter's own kernel-launcher
arguments.

## What is *not* yet in this release

An abstention / selective-prediction extension (margin-based routing
rejection with a coverage-penalised safety utility) exists as a separate,
independently-developed script but is **not part of any claim in the
current paper** and is not included in this archived release. It is
mentioned here only so the scope of what this DOI actually backs is
unambiguous.

## Citation

If you use this code, please cite both the paper and this software
release:

```
Hashim, D. Bounded Prototype Banks for Memory-Efficient Continual
Learning: A Diagnostic Framework for Task Routing Under Zero-Replay
Constraints. [venue / year TBD upon acceptance].

Hashim, D. CGL-AARM (software). Zenodo.
https://doi.org/10.5281/zenodo.22076271
```

## License

[Choose and add a LICENSE file — MIT or Apache-2.0 are standard choices
for research code. This README assumes one will be added; update this
section to name the actual license once chosen.]

## Data availability

CIFAR-100 and TinyImageNet are standard public benchmarks and are not
redistributed in this repository; both are downloaded automatically by
the provided scripts on first run. No custom or private data was used in
this work.
