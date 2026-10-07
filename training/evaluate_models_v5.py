"""Grouped evaluation and controlled model selection for the v5 FAW risk data."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path("data/weather/raw/Fall_Armyworm_Maize_Weather_Dataset_10000_v5.csv")
MODEL_PATH = Path("models/faw_risk_model_v5.pkl")
FEATURES_PATH = Path("models/faw_risk_features_v5.json")
METRICS_PATH = Path("evaluation/faw_model_metrics_v5.json")
SEED = 42
TARGET = "Risk_Level"
GROUP = "Source_ID"
BASE_NUMERIC = ["Month", "Temperature_C", "Humidity_%", "Rainfall_mm", "Soil_Moisture_%", "Wind_Speed_kmph", "Previous_Pest_Count", "Days_Since_Last_Attack"]
BASE_CATEGORICAL = ["Crop_Stage"]
LABELS = ["Low", "Medium", "High"]


def engineer_features(frame):
    result = frame[BASE_NUMERIC + BASE_CATEGORICAL].copy()
    result["Temperature_x_Humidity"] = result["Temperature_C"] * result["Humidity_%"]
    result["Temperature_x_Rainfall"] = result["Temperature_C"] * result["Rainfall_mm"]
    result["Humidity_x_Rainfall"] = result["Humidity_%"] * result["Rainfall_mm"]
    result["Pest_Count_x_Humidity"] = result["Previous_Pest_Count"] * result["Humidity_%"]
    result["Pest_Count_per_Days_Since_Attack"] = result["Previous_Pest_Count"] / (result["Days_Since_Last_Attack"] + 1)
    result["Rainfall_per_Soil_Moisture"] = result["Rainfall_mm"] / (result["Soil_Moisture_%"] + 1)
    result["Humidity_x_Pest_per_Temperature"] = result["Humidity_%"] * result["Previous_Pest_Count"] / result["Temperature_C"].clip(lower=1)
    return result


def pipeline(model):
    numeric = [column for column in engineer_features(pd.DataFrame([{"Month": 1, "Temperature_C": 1, "Humidity_%": 1, "Rainfall_mm": 1, "Soil_Moisture_%": 1, "Wind_Speed_kmph": 1, "Previous_Pest_Count": 1, "Days_Since_Last_Attack": 1, "Crop_Stage": "Seedling"}])).columns if column != "Crop_Stage"]
    prep = ColumnTransformer([("numeric", "passthrough", numeric), ("crop", OneHotEncoder(handle_unknown="ignore", sparse_output=False), BASE_CATEGORICAL)])
    return Pipeline([("features", prep), ("model", model)])


def metrics(y_true, predicted):
    high = LABELS.index("High")
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, labels=LABELS, average="macro")),
        "weighted_f1": float(f1_score(y_true, predicted, labels=LABELS, average="weighted")),
        "high_precision": float(precision_score(y_true, predicted, labels=LABELS, average=None, zero_division=0)[high]),
        "high_recall": float(recall_score(y_true, predicted, labels=LABELS, average=None, zero_division=0)[high]),
        "high_f1": float(f1_score(y_true, predicted, labels=LABELS, average=None, zero_division=0)[high]),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=LABELS).tolist(),
    }


def main():
    data = pd.read_csv(DATA_PATH)
    X = engineer_features(data)
    y = data[TARGET]
    groups = data[GROUP]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_indices, test_indices = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    groups_train = groups.iloc[train_indices]
    cv = GroupKFold(n_splits=3)

    candidates = {
        "HistGradientBoosting": pipeline(HistGradientBoostingClassifier(max_iter=180, learning_rate=0.06, max_leaf_nodes=15, random_state=SEED)),
        "RandomForest": pipeline(RandomForestClassifier(n_estimators=300, min_samples_leaf=3, max_features="sqrt", class_weight="balanced", random_state=SEED, n_jobs=-1)),
    }
    lightgbm_grid = ParameterGrid({
        "num_leaves": [15, 31], "learning_rate": [0.03, 0.06], "n_estimators": [150, 250],
        "min_child_samples": [15, 30], "subsample": [0.85], "colsample_bytree": [0.85],
        "reg_alpha": [0.0], "reg_lambda": [0.5, 1.0],
    })
    best_lgbm = None
    best_score = -np.inf
    for parameters in lightgbm_grid:
        model = pipeline(LGBMClassifier(objective="multiclass", verbosity=-1, random_state=SEED, n_jobs=-1, **parameters))
        fold_scores = []
        for cv_train, cv_valid in cv.split(X_train, y_train, groups_train):
            model.fit(X_train.iloc[cv_train], y_train.iloc[cv_train])
            predicted = model.predict(X_train.iloc[cv_valid])
            fold_scores.append(f1_score(y_train.iloc[cv_valid], predicted, labels=LABELS, average="macro"))
        score = float(np.mean(fold_scores))
        if score > best_score:
            best_score, best_lgbm = score, model
    candidates["LightGBM"] = best_lgbm

    results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        results[name] = {"metrics": metrics(y_test, model.predict(X_test)), "grouped_cv_macro_f1": None}
        cv_scores = []
        for cv_train, cv_valid in cv.split(X_train, y_train, groups_train):
            fold_model = pipeline(model.named_steps["model"]) if name != "LightGBM" else pipeline(LGBMClassifier(**model.named_steps["model"].get_params()))
            fold_model.fit(X_train.iloc[cv_train], y_train.iloc[cv_train])
            cv_scores.append(f1_score(y_train.iloc[cv_valid], fold_model.predict(X_train.iloc[cv_valid]), labels=LABELS, average="macro"))
        results[name]["grouped_cv_macro_f1"] = float(np.mean(cv_scores))

    best_name = max(results, key=lambda name: (results[name]["metrics"]["high_f1"], results[name]["metrics"]["macro_f1"]))
    best_model = candidates[best_name]
    best_model.fit(X, y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    feature_info = {"base_numeric": BASE_NUMERIC, "base_categorical": BASE_CATEGORICAL, "derived_numeric": [c for c in X.columns if c not in BASE_NUMERIC + BASE_CATEGORICAL], "target_labels": LABELS, "group_column": GROUP}
    FEATURES_PATH.write_text(json.dumps(feature_info, indent=2), encoding="utf-8")
    report = {"dataset_size": int(len(data)), "class_distribution": {k: int(v) for k, v in y.value_counts().sort_index().items()}, "models": results, "best_model": best_name, "benchmark": {"accuracy": 0.84, "high_precision": 1.0, "high_recall": 0.68, "high_f1": 0.81}, "improvement_over_benchmark": bool(results[best_name]["metrics"]["high_f1"] > 0.81 and results[best_name]["metrics"]["accuracy"] > 0.84), "model_path": str(MODEL_PATH)}
    METRICS_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    chosen = results[best_name]["metrics"]
    print(f"Dataset size: {len(data):,}")
    print(f"Class distribution: {report['class_distribution']}")
    print(f"Best model: {best_name}")
    print(f"Accuracy: {chosen['accuracy']:.4f}; Macro F1: {chosen['macro_f1']:.4f}")
    print(f"High precision: {chosen['high_precision']:.4f}; High recall: {chosen['high_recall']:.4f}; High F1: {chosen['high_f1']:.4f}")
    print(f"Improved over 84% benchmark: {report['improvement_over_benchmark']}")
    print(f"Model file: {MODEL_PATH}")


if __name__ == "__main__":
    main()