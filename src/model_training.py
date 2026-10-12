"""Trains and evaluates a classifier on base features alone vs. base +
45 engineered features -- a real, measured comparison showing the
engineered features' contribution."""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from .simulate_data import BASE_FEATURES


def train_and_evaluate(df, feature_cols: list[str], seed: int = 173) -> dict:
    X = df[feature_cols].values
    y = df["target"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=3000, C=0.5)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = round(float(accuracy_score(y_test, preds)), 4)

    return {"accuracy": accuracy, "n_features": len(feature_cols), "model": model}


if __name__ == "__main__":
    from .simulate_data import simulate
    from .feature_engineering import engineer_features

    df = simulate()
    engineered, eng_cols = engineer_features(df)

    base_result = train_and_evaluate(df, BASE_FEATURES)
    full_result = train_and_evaluate(engineered, BASE_FEATURES + eng_cols)

    print("Base features only:", base_result["accuracy"], f"({base_result['n_features']} features)")
    print("Base + engineered: ", full_result["accuracy"], f"({full_result['n_features']} features)")
