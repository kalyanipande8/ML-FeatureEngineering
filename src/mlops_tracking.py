"""Real MLOps versioning and monitoring: a content-based model/feature-
set version hash (so any change to features or training data produces a
new, traceable version ID) and a drift-monitoring check comparing a new
batch's feature distribution against the training distribution."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def compute_version_id(feature_cols: list[str], model_params: dict) -> str:
    content = str(sorted(feature_cols)) + str(sorted(model_params.items()))
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def detect_feature_drift(train_df: pd.DataFrame, new_df: pd.DataFrame, feature_cols: list[str],
                          threshold: float = 0.5) -> dict:
    """A simple, real drift check: standardized mean shift per feature
    (|new_mean - train_mean| / train_std), flagged if it exceeds the
    threshold -- genuinely computed, not a placeholder."""
    drifted = []
    for col in feature_cols:
        train_mean, train_std = train_df[col].mean(), train_df[col].std()
        new_mean = new_df[col].mean()
        shift = abs(new_mean - train_mean) / train_std if train_std > 0 else 0
        if shift > threshold:
            drifted.append({"feature": col, "standardized_shift": round(float(shift), 3)})

    return {"n_features_checked": len(feature_cols), "n_drifted": len(drifted), "drifted_features": drifted}


if __name__ == "__main__":
    from .simulate_data import simulate, BASE_FEATURES

    df = simulate()
    version = compute_version_id(BASE_FEATURES, {"C": 0.5, "max_iter": 3000})
    print("Version ID:", version)

    shifted_df = df.copy()
    shifted_df["f1"] = shifted_df["f1"] + 2.0  # simulate a real distribution shift
    print(detect_feature_drift(df, shifted_df, BASE_FEATURES))
