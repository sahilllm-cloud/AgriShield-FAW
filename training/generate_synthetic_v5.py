"""Analyze the source data and generate a risk-conditional synthetic dataset."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri
from scipy.stats import rankdata, spearmanr


SEED = 42
SOURCE_PATH = Path("data/weather/raw/Fall_Armyworm_Maize_Weather_Dataset_1000.csv")
OUTPUT_PATH = Path("data/weather/raw/Fall_Armyworm_Maize_Weather_Dataset_10000_v5.csv")
ANALYSIS_PATH = Path("training/dataset_analysis_v5.json")
METADATA_PATH = Path("training/synthetic_v5_metadata.json")
NUMERIC_FEATURES = [
    "Month", "Temperature_C", "Humidity_%", "Rainfall_mm",
    "Soil_Moisture_%", "Wind_Speed_kmph", "Previous_Pest_Count",
    "Days_Since_Last_Attack",
]
TARGET = "Risk_Level"
CROP = "Crop_Stage"


def serializable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def analyze_source(df):
    numeric = df[NUMERIC_FEATURES]
    by_risk = {}
    for risk, group in df.groupby(TARGET, sort=True):
        by_risk[risk] = {
            "count": int(len(group)),
            "mean": {k: serializable(v) for k, v in group[NUMERIC_FEATURES].mean().round(5).items()},
            "std": {k: serializable(v) for k, v in group[NUMERIC_FEATURES].std().round(5).items()},
            "min": {k: serializable(v) for k, v in group[NUMERIC_FEATURES].min().items()},
            "max": {k: serializable(v) for k, v in group[NUMERIC_FEATURES].max().items()},
        }
    stage_table = pd.crosstab(df[CROP], df[TARGET], normalize="index").round(5)
    correlations = numeric.corr(method="spearman").round(5)
    pairwise = {}
    for left_index, left in enumerate(NUMERIC_FEATURES):
        for right in NUMERIC_FEATURES[left_index + 1:]:
            rho, p_value = spearmanr(df[left], df[right])
            pairwise[f"{left}__{right}"] = {"spearman_rho": round(float(rho), 5), "p_value": round(float(p_value), 5)}

    # A probabilistic-label diagnostic: duplicate feature vectors with conflicting labels.
    feature_duplicates = df.groupby(NUMERIC_FEATURES + [CROP], dropna=False)[TARGET].nunique()
    conflicting_groups = int((feature_duplicates > 1).sum())
    return {
        "source_row_count": int(len(df)),
        "columns": list(df.columns),
        "class_distribution": {k: int(v) for k, v in df[TARGET].value_counts().sort_index().items()},
        "class_proportions": {k: round(float(v), 6) for k, v in df[TARGET].value_counts(normalize=True).sort_index().items()},
        "numerical_distributions_by_risk": by_risk,
        "spearman_correlation_matrix": correlations.to_dict(),
        "crop_stage_by_risk_row_proportions": stage_table.to_dict(),
        "pairwise_numeric_relationships": pairwise,
        "label_determinism_diagnostic": {
            "exact_duplicate_feature_groups": int(len(feature_duplicates)),
            "conflicting_duplicate_feature_groups": conflicting_groups,
            "interpretation": "Labels are treated as probabilistic when feature groups or conditional distributions overlap; no label rule is imposed by this pipeline.",
        },
    }


def gaussian_copula_sample(group, rng, count):
    values = group[NUMERIC_FEATURES].to_numpy(dtype=float)
    if len(group) < 3:
        return values[rng.integers(0, len(values), size=count)]
    uniforms = np.column_stack([
        (rankdata(values[:, index], method="average") - 0.5) / len(values)
        for index in range(values.shape[1])
    ])
    latent = ndtri(np.clip(uniforms, 0.001, 0.999))
    covariance = np.corrcoef(latent, rowvar=False)
    covariance = 0.85 * covariance + 0.15 * np.eye(covariance.shape[0])
    generated_latent = rng.multivariate_normal(np.zeros(len(NUMERIC_FEATURES)), covariance, size=count)
    generated_uniforms = ndtr(generated_latent)
    generated = np.column_stack([
        np.quantile(values[:, index], generated_uniforms[:, index], method="linear")
        for index in range(values.shape[1])
    ])
    source_min = values.min(axis=0)
    source_max = values.max(axis=0)
    return np.clip(generated, source_min, source_max)


def generate(df, generated_count=10000):
    rng = np.random.default_rng(SEED)
    class_counts = (df[TARGET].value_counts(normalize=True) * generated_count).round().astype(int)
    class_counts.iloc[-1] += generated_count - int(class_counts.sum())
    records = []
    for risk, count in class_counts.items():
        group = df[df[TARGET] == risk].reset_index(drop=True)
        numeric_values = gaussian_copula_sample(group, rng, int(count))
        stages = group[CROP].value_counts(normalize=True)
        sampled_stages = rng.choice(stages.index.to_numpy(), size=int(count), p=stages.to_numpy())
        source_indices = rng.integers(0, len(group), size=int(count))
        for row_index in range(int(count)):
            record = {feature: float(numeric_values[row_index, col]) for col, feature in enumerate(NUMERIC_FEATURES)}
            record["Month"] = int(np.clip(round(record["Month"]), 1, 12))
            record["Previous_Pest_Count"] = int(np.clip(round(record["Previous_Pest_Count"]), 0, 100))
            record["Days_Since_Last_Attack"] = int(np.clip(round(record["Days_Since_Last_Attack"]), 0, 60))
            record[CROP] = str(sampled_stages[row_index])
            record[TARGET] = str(risk)
            record["Source_ID"] = int(group.loc[source_indices[row_index], "Sample_ID"])
            records.append(record)
    result = pd.DataFrame(records)
    result.insert(0, "Sample_ID", np.arange(1, len(result) + 1))
    return result[["Sample_ID", *NUMERIC_FEATURES, CROP, TARGET, "Source_ID"]]


def main():
    source = pd.read_csv(SOURCE_PATH)
    required = {"Sample_ID", *NUMERIC_FEATURES, CROP, TARGET}
    if set(source.columns) != required:
        raise ValueError(f"Unexpected source columns: {list(source.columns)}")
    ANALYSIS_PATH.write_text(json.dumps(analyze_source(source), indent=2, default=serializable), encoding="utf-8")
    generated = generate(source)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    generated.to_csv(OUTPUT_PATH, index=False)
    metadata = {
        "source_row_count": int(len(source)),
        "generated_row_count": int(len(generated)),
        "class_distribution": {k: int(v) for k, v in generated[TARGET].value_counts().sort_index().items()},
        "feature_ranges": {feature: {"min": serializable(generated[feature].min()), "max": serializable(generated[feature].max())} for feature in NUMERIC_FEATURES},
        "random_seed": SEED,
        "generation_method": "Risk-conditional Gaussian copula over numeric ranks, conditional Crop_Stage sampling, bounded inverse empirical quantiles, and source-row traceability.",
        "source_id_count": int(generated["Source_ID"].nunique()),
        "source_id_reuse": {str(k): int(v) for k, v in generated["Source_ID"].value_counts().describe().round(3).items()},
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=serializable), encoding="utf-8")
    print(f"Generated {len(generated):,} rows at {OUTPUT_PATH}")
    print(generated[TARGET].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()