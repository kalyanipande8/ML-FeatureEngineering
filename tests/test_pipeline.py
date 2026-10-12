import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.simulate_data import simulate, BASE_FEATURES
from src.feature_engineering import engineer_features
from src.model_training import train_and_evaluate
from src.mlops_tracking import compute_version_id, detect_feature_drift


@pytest.fixture(scope="module")
def df():
    return simulate()


@pytest.fixture(scope="module")
def engineered(df):
    return engineer_features(df)


def test_at_least_45_features_engineered(engineered):
    _, cols = engineered
    assert len(cols) >= 45


def test_engineered_features_are_all_numeric(engineered):
    eng_df, cols = engineered
    assert eng_df[cols].select_dtypes(include="number").shape[1] == len(cols)


def test_engineered_features_improve_accuracy(df, engineered):
    eng_df, cols = engineered
    base = train_and_evaluate(df, BASE_FEATURES)
    full = train_and_evaluate(eng_df, BASE_FEATURES + cols)
    assert full["accuracy"] > base["accuracy"]


def test_version_id_changes_with_feature_set():
    v1 = compute_version_id(["a", "b"], {"C": 1.0})
    v2 = compute_version_id(["a", "b", "c"], {"C": 1.0})
    assert v1 != v2


def test_version_id_is_stable_for_same_inputs():
    v1 = compute_version_id(["a", "b"], {"C": 1.0})
    v2 = compute_version_id(["b", "a"], {"C": 1.0})
    assert v1 == v2  # order-independent


def test_drift_detection_catches_an_injected_shift(df):
    shifted = df.copy()
    shifted["f1"] = shifted["f1"] + 2.0
    result = detect_feature_drift(df, shifted, BASE_FEATURES)
    assert result["n_drifted"] >= 1
    assert any(d["feature"] == "f1" for d in result["drifted_features"])


def test_drift_detection_finds_no_drift_on_identical_data(df):
    result = detect_feature_drift(df, df, BASE_FEATURES)
    assert result["n_drifted"] == 0
