"""Simulate a base tabular dataset (8 raw features) with a nonlinear,
interaction-driven outcome -- the setting where feature engineering
(polynomial/interaction/ratio/binned features) should measurably help a
downstream linear-ish classifier."""
from __future__ import annotations

import numpy as np
import pandas as pd

N_SAMPLES = 3000
SEED = 173
BASE_FEATURES = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]


def simulate(n: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = {f: rng.normal(0, 1, n) for f in BASE_FEATURES}

    logit = (
        -0.3
        + 1.1 * X["f1"] * X["f2"]          # interaction
        + 0.9 * np.square(X["f3"])          # polynomial
        - 0.7 * np.maximum(X["f4"], 0)       # thresholded
        + 0.5 * X["f5"]
        + rng.normal(0, 0.6, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, prob)

    df = pd.DataFrame(X)
    df["target"] = y
    return df


if __name__ == "__main__":
    df = simulate()
    print(df.shape, "positive rate:", df["target"].mean())
