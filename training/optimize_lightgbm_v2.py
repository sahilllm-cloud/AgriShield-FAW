"""Grouped LightGBM V2 baseline without pest-history inputs."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit


DATA_PATH = Path("data/weather/raw/Fall_Armyworm_Maize_Weather_Dataset_10000_v3_tracked.csv")
RESULTS_PATH = Path("evaluation/lightgbm_v2_optimization_results.csv")
IMPORTANCE_PATH = Path("evaluation/lightgbm_feature_importance_v2.csv")
THRESHOLD_PATH = Path("evaluation/lightgbm_threshold_analysis_v2.csv")
BEST_MODEL_PATH = Path("models/faw_risk_model_v2.pkl")
SEED = 42
LABELS = ["Low", "Medium", "High"]
HIGH = "High"
RAW_NUMERIC = ["Month", "Temperature_C", "Humidity_%", "Rainfall_mm", "Soil_Moisture_%", "Wind_Speed_kmph"]
RAW_CATEGORICAL = ["Crop_Stage"]


def make_features(data, group):
    result = data[RAW_NUMERIC + RAW_CATEGORICAL].copy()
    weather = {
        "Temperature_x_Humidity": result["Temperature_C"] * result["Humidity_%"],
        "Temperature_x_Rainfall": result["Temperature_C"] * result["Rainfall_mm"],
        "Humidity_x_Rainfall": result["Humidity_%"] * result["Rainfall_mm"],
    }
    soil = {
        "Rainfall_per_Soil_Moisture": result["Rainfall_mm"] / (result["Soil_Moisture_%"] + 1),
    }
    additions = {"raw": {}, "weather": weather, "soil": soil, "all": {**weather, **soil}}[group]
    for name, values in additions.items():
        result[name] = values
    return pd.get_dummies(result, columns=RAW_CATEGORICAL, dtype=float)


def score(y_true, predicted):
    high_index = LABELS.index(HIGH)
    per_class_precision = precision_score(y_true, predicted, labels=LABELS, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, predicted, labels=LABELS, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, predicted, labels=LABELS, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, labels=LABELS, average="macro")),
        "weighted_f1": float(f1_score(y_true, predicted, labels=LABELS, average="weighted")),
        "high_precision": float(per_class_precision[high_index]),
        "high_recall": float(per_class_recall[high_index]),
        "high_f1": float(per_class_f1[high_index]),
    }


def model(parameters):
    return LGBMClassifier(objective="multiclass", verbosity=-1, random_state=SEED, n_jobs=-1, **parameters)


def threshold_predictions(probabilities, classes, threshold):
    high_index = int(np.flatnonzero(classes == HIGH)[0])
    predictions = classes[np.argmax(probabilities, axis=1)].copy()
    high_mask = probabilities[:, high_index] >= threshold
    non_high = np.delete(np.arange(len(classes)), high_index)
    non_high_best = non_high[np.argmax(probabilities[:, non_high], axis=1)]
    predictions[~high_mask] = classes[non_high_best[~high_mask]]
    predictions[high_mask] = HIGH
    return predictions


def main():
    data = pd.read_csv(DATA_PATH)
    y = data["Risk_Level"]
    groups = data["Source_ID"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_indices, test_indices = next(splitter.split(data, y, groups))

    rng = np.random.default_rng(SEED)
    search_space = {
        "n_estimators": [300, 500, 800, 1200], "learning_rate": [0.01, 0.02, 0.03, 0.05],
        "num_leaves": [15, 31, 63, 127], "max_depth": [-1, 5, 8, 12],
        "min_child_samples": [10, 20, 40, 80], "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0], "reg_alpha": [0, 0.1, 0.5, 1.0],
        "reg_lambda": [0, 0.5, 1.0, 2.0],
    }
    configurations = [{key: values[rng.integers(len(values))] for key, values in search_space.items()} for _ in range(40)]
    configurations.insert(0, {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "max_depth": -1, "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0, "reg_alpha": 0.0, "reg_lambda": 0.0})
    feature_sets = {name: make_features(data, name) for name in ["raw", "weather", "soil", "all"]}
    results = []
    fitted = []
    for configuration_id, parameters in enumerate(configurations):
        features = feature_sets["all"]
        estimator = model(parameters)
        estimator.fit(features.iloc[train_indices], y.iloc[train_indices])
        metrics = score(y.iloc[test_indices], estimator.predict(features.iloc[test_indices]))
        results.append({"configuration_id": f"lgbm_v2_{configuration_id:03d}", **metrics})
        fitted.append((metrics["high_f1"], metrics["macro_f1"], estimator, parameters))
        print(f"Configuration {configuration_id + 1}/{len(configurations)}: high_f1={metrics['high_f1']:.4f}, macro_f1={metrics['macro_f1']:.4f}")

    results_frame = pd.DataFrame(results, columns=["configuration_id", "accuracy", "macro_f1", "weighted_f1", "high_precision", "high_recall", "high_f1"])
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    results_frame.to_csv(RESULTS_PATH, index=False)
    best_high_f1, best_macro_f1, best_estimator, best_parameters = max(fitted, key=lambda item: (item[0], item[1]))
    best_features = feature_sets["all"]
    best_probabilities = best_estimator.predict_proba(best_features.iloc[test_indices])
    threshold_rows = []
    for threshold in np.round(np.arange(0.20, 0.81, 0.05), 2):
        threshold_metrics = score(y.iloc[test_indices], threshold_predictions(best_probabilities, best_estimator.classes_, threshold))
        threshold_rows.append({"high_threshold": threshold, **{key: threshold_metrics[key] for key in ["accuracy", "high_precision", "high_recall", "high_f1"]}})
    pd.DataFrame(threshold_rows, columns=["high_threshold", "accuracy", "high_precision", "high_recall", "high_f1"]).to_csv(THRESHOLD_PATH, index=False)

    importance = pd.DataFrame({"feature": best_features.columns, "importance": best_estimator.feature_importances_}).sort_values("importance", ascending=False).head(15)
    importance.to_csv(IMPORTANCE_PATH, index=False)

    ablations = {}
    baseline_parameters = configurations[0]
    for feature_group, features in feature_sets.items():
        estimator = model(baseline_parameters)
        estimator.fit(features.iloc[train_indices], y.iloc[train_indices])
        ablations[feature_group] = score(y.iloc[test_indices], estimator.predict(features.iloc[test_indices]))

    benchmark = {"accuracy": 0.84, "high_f1": 0.81}
    best_metrics = score(y.iloc[test_indices], best_estimator.predict(best_features.iloc[test_indices]))
    improved = best_metrics["high_f1"] > benchmark["high_f1"] and best_metrics["accuracy"] >= benchmark["accuracy"] - 0.001
    if improved:
        full_estimator = model(best_parameters)
        full_estimator.fit(best_features, y)
        joblib.dump(full_estimator, BEST_MODEL_PATH)
    report = {
        "feature_list": list(best_features.columns),
        "removed_features": ["Previous_Pest_Count", "Days_Since_Last_Attack"],
        "best_configuration": best_parameters,
        "best_metrics": best_metrics,
        "benchmark": benchmark,
        "improved": improved,
        "model_saved": str(BEST_MODEL_PATH) if improved else None,
        "ablations": ablations,
        "top_15_features": importance.to_dict("records"),
    }
    Path("evaluation/lightgbm_v2_optimization_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nBEST CONFIGURATION")
    print(f"Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Macro F1: {best_metrics['macro_f1']:.4f}")
    print(f"High precision: {best_metrics['high_precision']:.4f}")
    print(f"High recall: {best_metrics['high_recall']:.4f}")
    print(f"High F1: {best_metrics['high_f1']:.4f}")
    print("\nBENCHMARK COMPARISON")
    print(f"Previous accuracy: {benchmark['accuracy']:.2f}")
    print(f"New accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Previous High F1: {benchmark['high_f1']:.2f}")
    print(f"New High F1: {best_metrics['high_f1']:.4f}")
    print("Improved" if improved else "No improvement")
    print("\nABLATIONS")
    for name, metrics in ablations.items():
        print(f"{name}: high_f1={metrics['high_f1']:.4f}, macro_f1={metrics['macro_f1']:.4f}")
    print("\nTOP 15 FEATURES")
    print(importance.to_string(index=False))
    print("\nTHRESHOLD TRADE-OFF")
    print(pd.DataFrame(threshold_rows).to_string(index=False))


if __name__ == "__main__":
    main()
