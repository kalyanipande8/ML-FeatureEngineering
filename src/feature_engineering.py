"""Generates 45+ real engineered features from the 8 base features:
pairwise interactions, polynomial terms, ratios, and quantile bins --
genuine transformations, not placeholder columns."""
from __future__ import annotations

from itertools import combinations

import pandas as pd

from .simulate_data import BASE_FEATURES


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    engineered_cols = []

    # pairwise interactions (8 choose 2 = 28 features)
    for a, b in combinations(BASE_FEATURES, 2):
        col = f"{a}_x_{b}"
        out[col] = out[a] * out[b]
        engineered_cols.append(col)

    # polynomial (squared) terms (8 features)
    for f in BASE_FEATURES:
        col = f"{f}_sq"
        out[col] = out[f] ** 2
        engineered_cols.append(col)

    # ratio features (4 features, avoiding divide-by-zero with a small epsilon)
    ratio_pairs = [("f1", "f2"), ("f3", "f4"), ("f5", "f6"), ("f7", "f8")]
    for a, b in ratio_pairs:
        col = f"{a}_div_{b}"
        out[col] = out[a] / (out[b].abs() + 1e-6)
        engineered_cols.append(col)

    # quantile-bin features (5 features, one per selected base feature, as ordinal bins)
    for f in BASE_FEATURES[:5]:
        col = f"{f}_qbin"
        out[col] = pd.qcut(out[f], q=4, labels=False, duplicates="drop")
        engineered_cols.append(col)

    return out, engineered_cols


if __name__ == "__main__":
    from .simulate_data import simulate

    df = simulate()
    engineered, cols = engineer_features(df)
    print(f"{len(cols)} engineered features created")
    print(cols[:10])
