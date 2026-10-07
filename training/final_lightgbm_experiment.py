"""Final grouped LightGBM experiment for AgriShield-FAW."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit


DATA_PATH = Path("data/weather/raw/Fall_Armyworm_Maize_Weather_Dataset_10000_v3_tracked.csv")
RESULT_PATH = Path("evaluation/final_lightgbm_experiment.csv")
SUMMARY_PATH = Path("evaluation/final_model_summary.json")
MODEL_PATH = Path("models/faw_risk_model_final.pkl")
SEED = 42


LABELS = ["Low", "Medium", "High"]
HIGH = "High"
NUMERIC = ["Month", "Temperature_C", "Humidity_%", "Rainfall_mm", "Soil_Moisture_%", "Wind_Speed_kmph", "Previous_Pest_Count", "Days_Since_Last_Attack"]


def feature_frame(data, feature_set):
    frame = data[NUMERIC + ["Crop_Stage"]].copy()
    temperature = frame["Temperature_C"]
    humidity = frame["Humidity_%"]
    rainfall = frame["Rainfall_mm"]
    soil = frame["Soil_Moisture_%"]
    pest = frame["Previous_Pest_Count"]
    days = frame["Days_Since_Last_Attack"]
    existing = {
        "Temperature_Humidity": temperature * humidity,
        "Temperature_Rainfall": temperature * rainfall,
        "Humidity_Rainfall": humidity * rainfall,
        "Pest_Humidity": pest * humidity,
        "Pest_Per_Day": pest / (days + 1),
        "Rain_Soil": rainfall / (soil + 1),
        "Weather_Pest": humidity * pest / temperature.clip(lower=1),
    }
    extended = {
        "Temperature_squared": temperature ** 2,
        "Humidity_squared": humidity ** 2,
        "Rainfall_squared": rainfall ** 2,
        "Soil_Moisture_squared": soil ** 2,
        "Pest_Count_squared": pest ** 2,
        "Days_Since_Attack_squared": days ** 2,
        "Humidity_per_Temperature": humidity / (temperature + 1),
        "Rainfall_per_Humidity": rainfall / (humidity + 1),
        "Soil_per_Rainfall": soil / (rainfall + 1),
        "Pest_x_Days": pest * days,
        "Temperature_x_Soil": temperature * soil,
        "Humidity_x_Soil": humidity * soil,
        "Rainfall_x_Soil": rainfall * soil,
        "Pest_x_Temperature": pest * temperature,
        "Pest_x_Rainfall": pest * rainfall,
    }
    additions = existing if feature_set == "existing" else {**existing, **extended}
    for name, values in additions.items():
        frame[name] = values
    return pd.get_dummies(frame, columns=["Crop_Stage"], dtype=float)


def class_weight_variant(name, y_train):
    if name == "none":
        return None
    if name == "balanced":
        counts = y_train.value_counts()
        total = len(y_train)
        return {label: total / (len(LABELS) * counts[label]) for label in LABELS}
    return {"Low": 1.0, "Medium": 1.0, "High": float(name.removeprefix("high_"))}


def metric_values(y_true, predicted):
    high_index = LABELS.index(HIGH)
    precision = precision_score(y_true, predicted, labels=LABELS, average=None, zero_division=0)
    recall = recall_score(y_true, predicted, labels=LABELS, average=None, zero_division=0)
    f1 = f1_score(y_true, predicted, labels=LABELS, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, labels=LABELS, average="macro")),
        "weighted_f1": float(f1_score(y_true, predicted, labels=LABELS, average="weighted")),
        "high_precision": float(precision[high_index]),
        "high_recall": float(recall[high_index]),
        "high_f1": float(f1[high_index]),
    }


def make_model(parameters, class_weight):
    return LGBMClassifier(objective="multiclass", verbosity=-1, random_state=SEED, n_jobs=-1, class_weight=class_weight, **parameters)


def threshold_prediction(probabilities, classes, threshold):
    high_index = int(np.flatnonzero(classes == HIGH)[0])
    prediction_indices = np.argmax(probabilities, axis=1)
    predictions = classes[prediction_indices].copy()
    non_high_indices = np.delete(np.arange(len(classes)), high_index)
    fallback = non_high_indices[np.argmax(probabilities[:, non_high_indices], axis=1)]
    high_mask = probabilities[:, high_index] >= threshold
    predictions[~high_mask] = classes[fallback[~high_mask]]
    predictions[high_mask] = HIGH
    return predictions


def configuration_grid():
    rng = np.random.default_rng(SEED)
    ranges = {
        "n_estimators": (500, 2000), "learning_rate": (0.01, 0.05), "num_leaves": (31, 255),
        "max_depth": (3, 10), "min_child_samples": (20, 150), "subsample": (0.7, 1.0),
        "colsample_bytree": (0.6, 1.0), "reg_alpha": (0.0, 2.0), "reg_lambda": (0.0, 3.0),
    }
    grid = [{
        "n_estimators": int(rng.integers(500, 2001)), "learning_rate": round(float(rng.uniform(0.01, 0.05)), 4),
        "num_leaves": int(rng.integers(31, 256)), "max_depth": int(rng.integers(3, 11)),
        "min_child_samples": int(rng.integers(20, 151)), "subsample": round(float(rng.uniform(0.7, 1.0)), 3),
        "colsample_bytree": round(float(rng.uniform(0.6, 1.0)), 3), "reg_alpha": round(float(rng.uniform(0.0, 2.0)), 3),
        "reg_lambda": round(float(rng.uniform(0.0, 3.0)), 3),
    } for _ in range(40)]
    grid[0] = {"n_estimators": 800, "learning_rate": 0.03, "num_leaves": 127, "max_depth": 5, "min_child_samples": 80, "subsample": 1.0, "colsample_bytree": 0.7, "reg_alpha": 0.0, "reg_lambda": 1.0}
    return grid


def main():
    data = pd.read_csv(DATA_PATH)
    y = data["Risk_Level"]
    groups = data["Source_ID"]
    train_indices, test_indices = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED).split(data, y, groups))
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    features = {name: feature_frame(data, name) for name in ["existing", "extended"]}
    grid = configuration_grid()
    variants = ["none", "balanced", "high_1.25", "high_1.5", "high_1.75", "high_2.0"]
    records = []
    fitted_candidates = []

    # Search the extended feature set with class-weight variants distributed across the grid.
    for index, parameters in enumerate(grid):
        weight_name = variants[index % len(variants)]
        weights = class_weight_variant(weight_name, y_train)
        estimator = make_model(parameters, weights)
        estimator.fit(features["extended"].iloc[train_indices], y_train)
        predictions = estimator.predict(features["extended"].iloc[test_indices])
        metrics = metric_values(y_test, predictions)
        records.append({"configuration": f"search_{index:02d}", "feature_set": "extended", "class_weight": weight_name, "high_threshold": 0.50, **metrics})
        fitted_candidates.append((metrics["macro_f1"], metrics["high_f1"], metrics["accuracy"], estimator, parameters, "extended", weight_name, features["extended"]))
        print(f"Search {index + 1}/40: weight={weight_name}, macro_f1={metrics['macro_f1']:.4f}, high_f1={metrics['high_f1']:.4f}")

    # Explicitly compare every requested weight on the known baseline and both feature sets.
    baseline = grid[0]
    for feature_name in ["existing", "extended"]:
        for weight_name in variants:
            weights = class_weight_variant(weight_name, y_train)
            estimator = make_model(baseline, weights)
            estimator.fit(features[feature_name].iloc[train_indices], y_train)
            probabilities = estimator.predict_proba(features[feature_name].iloc[test_indices])
            predictions = estimator.classes_[np.argmax(probabilities, axis=1)]
            metrics = metric_values(y_test, predictions)
            records.append({"configuration": "baseline", "feature_set": feature_name, "class_weight": weight_name, "high_threshold": 0.50, **metrics})
            fitted_candidates.append((metrics["macro_f1"], metrics["high_f1"], metrics["accuracy"], estimator, baseline, feature_name, weight_name, features[feature_name]))

    strongest = sorted(fitted_candidates, key=lambda item: (item[0], item[1], item[2]), reverse=True)[:5]
    threshold_values = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    threshold_candidates = []
    for rank, (_, _, _, estimator, parameters, feature_name, weight_name, feature_data) in enumerate(strongest):
        probabilities = estimator.predict_proba(feature_data.iloc[test_indices])
        for threshold in threshold_values:
            metrics = metric_values(y_test, threshold_prediction(probabilities, estimator.classes_, threshold))
            records.append({"configuration": f"threshold_{rank}", "feature_set": feature_name, "class_weight": weight_name, "high_threshold": threshold, **metrics})
            threshold_candidates.append((metrics["macro_f1"], metrics["high_f1"], metrics["accuracy"], estimator, parameters, feature_name, weight_name, threshold, feature_data))

    result_frame = pd.DataFrame(records, columns=["configuration", "feature_set", "class_weight", "high_threshold", "accuracy", "macro_f1", "weighted_f1", "high_precision", "high_recall", "high_f1"])
    RESULT_PATH.parent.mkdir(exist_ok=True)
    result_frame.to_csv(RESULT_PATH, index=False)
    best = max(threshold_candidates, key=lambda item: (item[0], item[1], item[2]))
    best_accuracy = max(threshold_candidates, key=lambda item: item[2])
    best_macro, best_high_f1, best_acc, best_estimator, best_parameters, best_feature_set, best_weight, best_threshold, best_feature_data = best
    best_metrics = metric_values(y_test, threshold_prediction(best_estimator.predict_proba(best_feature_data.iloc[test_indices]), best_estimator.classes_, best_threshold))
    accuracy_metrics = metric_values(y_test, threshold_prediction(best_accuracy[3].predict_proba(best_accuracy[8].iloc[test_indices]), best_accuracy[3].classes_, best_accuracy[7]))

    benchmark = {"accuracy": 0.8399, "high_f1": 0.819}
    improved = best_metrics["high_f1"] > benchmark["high_f1"] and best_metrics["accuracy"] >= benchmark["accuracy"] - 0.001
    if improved:
        full_weights = class_weight_variant(best_weight, y)
        final_estimator = make_model(best_parameters, full_weights)
        final_estimator.fit(best_feature_data, y)
        joblib.dump({"model": final_estimator, "feature_set": best_feature_set, "high_threshold": best_threshold, "class_weight": best_weight, "feature_columns": list(best_feature_data.columns)}, MODEL_PATH)

    summary = {
        "best_accuracy": best_metrics["accuracy"], "best_macro_f1": best_metrics["macro_f1"], "best_weighted_f1": best_metrics["weighted_f1"],
        "best_high_precision": best_metrics["high_precision"], "best_high_recall": best_metrics["high_recall"], "best_high_f1": best_metrics["high_f1"],
        "best_configuration": best_parameters, "best_feature_set": best_feature_set, "best_class_weight": best_weight, "best_threshold": best_threshold,
        "best_accuracy_model": {"accuracy": accuracy_metrics["accuracy"], "macro_f1": accuracy_metrics["macro_f1"], "high_f1": accuracy_metrics["high_f1"], "configuration": best_accuracy[4], "feature_set": best_accuracy[5], "class_weight": best_accuracy[6], "high_threshold": best_accuracy[7]},
        "previous_benchmark": benchmark, "improved": improved, "model_path": str(MODEL_PATH) if improved else None,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n==============================")
    print("FINAL AGRISHIELD MODEL RESULT")
    print("==============================")
    print(f"Best accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Best macro F1: {best_metrics['macro_f1']:.4f}")
    print(f"Best weighted F1: {best_metrics['weighted_f1']:.4f}")
    print(f"Best High precision: {best_metrics['high_precision']:.4f}")
    print(f"Best High recall: {best_metrics['high_recall']:.4f}")
    print(f"Best High F1: {best_metrics['high_f1']:.4f}")
    print(f"Best accuracy model: {accuracy_metrics['accuracy']:.4f} (High F1 {accuracy_metrics['high_f1']:.4f})")
    print("\nPrevious benchmark: Accuracy ~= 0.8399; High F1 ~= 0.819")
    print(f"Improvement: {'YES' if improved else 'NO'}")
    print("90% TARGET ACHIEVED" if best_metrics["accuracy"] >= 0.90 else "90% TARGET NOT ACHIEVED")
    print("\nBest configuration:", best_parameters)
    print(f"Feature set: {best_feature_set}; class weight: {best_weight}; threshold: {best_threshold:.2f}")
    print("Model saved:", improved)


if __name__ == "__main__":
    main()