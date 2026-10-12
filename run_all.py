"""End-to-end: simulate data, engineer 45+ features, compare model
accuracy with/without them, version the model, and check for feature
drift."""
import json
import os

from src.simulate_data import simulate, BASE_FEATURES
from src.feature_engineering import engineer_features
from src.model_training import train_and_evaluate
from src.mlops_tracking import compute_version_id, detect_feature_drift

os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("=== Simulating base dataset ===")
df = simulate()
df.to_csv("data/base_data.csv", index=False)
print(df.shape)

print("\n=== Engineering 45+ features ===")
engineered, eng_cols = engineer_features(df)
print(f"{len(eng_cols)} engineered features created")
engineered.to_csv("data/engineered_data.csv", index=False)

print("\n=== Model comparison: base features vs. base + engineered ===")
base_result = train_and_evaluate(df, BASE_FEATURES)
full_result = train_and_evaluate(engineered, BASE_FEATURES + eng_cols)
print(f"Base features only ({base_result['n_features']} features): {base_result['accuracy']}")
print(f"Base + engineered ({full_result['n_features']} features):  {full_result['accuracy']}")

print("\n=== MLOps versioning ===")
version_id = compute_version_id(BASE_FEATURES + eng_cols, {"C": 0.5, "max_iter": 3000})
print("Model/feature-set version ID:", version_id)

print("\n=== Drift monitoring (simulated new batch with a real shift) ===")
new_batch = simulate(seed=174)
new_batch["f1"] = new_batch["f1"] + 2.0
drift = detect_feature_drift(df, new_batch, BASE_FEATURES)
print(drift)

summary = {
    "n_engineered_features": len(eng_cols),
    "base_feature_accuracy": base_result["accuracy"],
    "engineered_feature_accuracy": full_result["accuracy"],
    "accuracy_improvement_pct_points": round(100 * (full_result["accuracy"] - base_result["accuracy"]), 2),
    "model_version_id": version_id,
    "n_features_drifted": drift["n_drifted"],
}
with open("results/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Summary ===")
print(json.dumps(summary, indent=2))
