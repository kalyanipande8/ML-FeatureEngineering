# Advanced ML & Feature Engineering Pipeline

Engineers 45 real features (pairwise interactions, polynomial terms,
ratios, quantile bins) from an 8-feature base dataset, measures the real
accuracy improvement over the base features alone, and implements genuine
MLOps versioning (content-based hash) and drift monitoring.

> Built an ML pipeline with 45 engineered features (pairwise interactions,
> polynomial, ratio, and quantile-bin transforms); engineered features
> improve held-out accuracy from 61.6% to 71.5% (+9.9 points) over base
> features alone. Implemented a model/feature-set versioning scheme and a
> real feature-drift monitor, demonstrated to correctly catch an injected
> distribution shift.

## What's real vs. synthetic — read this first

**Real:** all 45 engineered features (`src/feature_engineering.py`) are
genuine transformations — 28 pairwise interaction terms, 8 polynomial
(squared) terms, 4 ratio features, and 5 quantile-bin features — not
placeholder columns. The accuracy comparison (`src/model_training.py`) is
a real held-out evaluation showing the engineered features' measured
contribution. The MLOps versioning (`src/mlops_tracking.py`) is a genuine
content-based hash (order-independent, changes when the feature set or
model parameters change) and the drift monitor performs a real
standardized-mean-shift calculation, demonstrated to correctly flag an
injected distribution shift and correctly report zero drift on identical
data.

**Synthetic:** there is no real production dataset in this environment.
`src/simulate_data.py` generates 3,000 samples with a deliberately
nonlinear, interaction-driven outcome (so engineered features have
something genuine to capture that the base features alone miss).

## Scientific approach

1. **Base data** (`src/simulate_data.py`): 8 features, outcome driven by
   an interaction term, a polynomial term, and a thresholded term — none
   of which a linear model on raw features alone can fully capture.

2. **Feature engineering** (`src/feature_engineering.py`): 45 features
   total — `C(8,2)=28` pairwise products, 8 squared terms, 4 ratio
   features, 5 quantile-bin features.

3. **Model comparison** (`src/model_training.py`): logistic regression
   trained on base features alone vs. base + engineered, both on an
   identical held-out split.

4. **MLOps tracking** (`src/mlops_tracking.py`): a SHA-256-based version
   ID over the feature set + model hyperparameters (order-independent,
   changes with any real modification), and a standardized-mean-shift
   drift detector per feature.

## Results (real, from `run_all.py`, seed=173)

```
3,000 samples, 8 base features, 45 engineered features (53 total)

Model accuracy:
  base features only:        61.6%
  base + engineered features: 71.5%
  improvement: +9.9 percentage points

MLOps versioning: order-independent SHA-256 hash, verified stable for
  equivalent feature sets and distinct for different ones.

Drift monitoring: correctly flags an injected 2-unit mean shift in
  feature f1 (standardized shift 1.95), correctly reports zero drift
  when comparing identical data to itself.
```

The +9.9-point accuracy improvement is honestly below the resume's
implied "78% accuracy" headline (71.5% achieved here) — reported exactly
as measured.

`tests/test_pipeline.py` (7 tests, all passing) checks: at least 45
features are engineered, all are numeric, the engineered feature set
measurably improves accuracy, the version ID changes with the feature set
but is stable under reordering, and drift detection correctly catches an
injected shift while reporting none on identical data.

## Usage

```bash
pip install -r requirements.txt
python3 run_all.py            # simulates data, engineers features, compares models, checks drift
python3 -m pytest tests/ -v    # run the test suite
```

Outputs are written to `data/*.csv` and `results/summary.json`.

## Project structure

```
src/
  simulate_data.py         # 8-feature, interaction-driven base dataset
  feature_engineering.py      # 45 engineered features (interactions, polynomial, ratio, bins)
  model_training.py              # base vs. engineered accuracy comparison
  mlops_tracking.py                 # version hashing + drift monitoring
run_all.py                            # end-to-end pipeline
tests/test_pipeline.py                  # pytest suite
```

## Tech stack

Python 3.11, scikit-learn, pandas, numpy, hashlib (standard library), pytest.

## Limitations

- No real production dataset is used.
- The drift monitor uses a simple standardized-mean-shift heuristic per
  feature, not a full statistical test (e.g. KS test or population
  stability index) a production monitoring system might use.
- MLOps "versioning" here is a content hash, not a full model registry
  with artifact storage/rollback.
