"""
PECARN TBI Modeling Module
STAT 214 Lab 1

Implements the three required models:
1) PECARN clinical decision rule
2) Logistic regression
3) XGBoost classifier
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb
except ImportError:
    xgb = None


@dataclass
class ModelMetrics:
    sensitivity: float
    specificity: float
    auc: float
    confusion: Tuple[int, int, int, int]
    threshold: float | None = None


def to_binary(series: pd.Series) -> pd.Series:
    if series.dtype.kind in "biufc":
        return series.fillna(0).astype(int)
    return series.map({"Yes": 1, "No": 0, "Suspected": 1}).fillna(0).astype(int)


def prepare_model_frames(data_clean: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    df_under2 = data_clean[data_clean["age_group"].astype(str).str.contains("<2")].copy()
    df_over2 = data_clean[~data_clean["age_group"].astype(str).str.contains("<2")].copy()

    for df in [df_under2, df_over2]:
        if len(df) == 0:
            continue
        if "gcs_total" in df.columns:
            df["pred_gcs_abnormal"] = (df["gcs_total"] < 15).astype(int)
        else:
            df["pred_gcs_abnormal"] = 0

        ams_cols = ["ams_agitated", "ams_sleep", "ams_slow", "ams_repeat"]
        existing_ams = [c for c in ams_cols if c in df.columns]
        for col in existing_ams:
            df[col + "_bin"] = to_binary(df[col])

        df["pred_ams"] = df["pred_gcs_abnormal"]
        if existing_ams:
            ams_combined = df[[c + "_bin" for c in existing_ams]].max(axis=1).fillna(0).astype(int)
            df["pred_ams"] = (df["pred_ams"] | ams_combined).astype(int)

        if "loss_of_consciousness" in df.columns:
            df["pred_loc"] = to_binary(df["loss_of_consciousness"])
        else:
            df["pred_loc"] = 0

        severe_mechs = [
            "Pedestrian struck",
            "Bike struck by auto",
            "MVC ejection",
            "Fall from elevation",
            "Object struck head",
        ]
        if "ind_mech" in df.columns:
            df["pred_mech"] = to_binary(df["ind_mech"])
        elif "injury_mech" in df.columns:
            df["pred_mech"] = df["injury_mech"].isin(severe_mechs).astype(int)
        else:
            df["pred_mech"] = 0

        if "skull_fx_palpable" in df.columns:
            df["pred_skull_palp"] = to_binary(df["skull_fx_palpable"])
        else:
            df["pred_skull_palp"] = 0

    if len(df_under2) > 0:
        if "hematoma_palpable" in df_under2.columns:
            df_under2["has_hematoma"] = to_binary(df_under2["hematoma_palpable"])
        else:
            df_under2["has_hematoma"] = 0

        if "hematoma_loc" in df_under2.columns:
            is_frontal = df_under2["hematoma_loc"] == "Frontal"
        else:
            is_frontal = False

        df_under2["pred_hematoma_nonfront"] = ((df_under2["has_hematoma"] == 1) & (~is_frontal)).astype(int)

        if "acting_normally" in df_under2.columns:
            df_under2["pred_acting_abnormal"] = (df_under2["acting_normally"] == "No").astype(int)
        else:
            df_under2["pred_acting_abnormal"] = 0

    if len(df_over2) > 0:
        if "skull_fx_basilar" in df_over2.columns:
            df_over2["pred_basilar"] = to_binary(df_over2["skull_fx_basilar"])
        else:
            df_over2["pred_basilar"] = 0

        if "vomiting_history" in df_over2.columns:
            df_over2["pred_vomit"] = to_binary(df_over2["vomiting_history"])
        else:
            df_over2["pred_vomit"] = 0

        if "headache_severity" in df_over2.columns:
            df_over2["pred_headache"] = (df_over2["headache_severity"] == "Severe").astype(int)
        elif "ind_ha" in df_over2.columns:
            df_over2["pred_headache"] = to_binary(df_over2["ind_ha"])
        else:
            df_over2["pred_headache"] = 0

    if len(df_under2) > 0 and "citbi" in df_under2.columns:
        df_under2["target"] = to_binary(df_under2["citbi"])
    if len(df_over2) > 0 and "citbi" in df_over2.columns:
        df_over2["target"] = to_binary(df_over2["citbi"])

    features_under2 = [
        "pred_ams",
        "pred_skull_palp",
        "pred_hematoma_nonfront",
        "pred_loc",
        "pred_mech",
        "pred_acting_abnormal",
    ]
    features_over2 = [
        "pred_ams",
        "pred_basilar",
        "pred_vomit",
        "pred_loc",
        "pred_mech",
        "pred_headache",
    ]

    return df_under2, df_over2, features_under2, features_over2


def pecarn_rule_under2(row: pd.Series) -> int:
    risk_factors = (
        row["pred_ams"]
        + row["pred_skull_palp"]
        + row["pred_hematoma_nonfront"]
        + row["pred_loc"]
        + row["pred_mech"]
        + row["pred_acting_abnormal"]
    )
    return 1 if risk_factors > 0 else 0


def pecarn_rule_over2(row: pd.Series) -> int:
    risk_factors = (
        row["pred_ams"]
        + row["pred_basilar"]
        + row["pred_vomit"]
        + row["pred_loc"]
        + row["pred_mech"]
        + row["pred_headache"]
    )
    return 1 if risk_factors > 0 else 0


def evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return ModelMetrics(sensitivity=sensitivity, specificity=specificity, auc=float("nan"), confusion=(tn, fp, fn, tp))


def choose_threshold_for_sensitivity(
    y_true: np.ndarray, y_prob: np.ndarray, target: float, min_specificity: float = 0.05
) -> float:
    """Choose the highest threshold that meets target sensitivity.
    
    If no threshold achieves both target sensitivity AND min_specificity,
    falls back to the threshold that maximises sensitivity + specificity sum.
    """
    thresholds = np.unique(np.round(y_prob, 6))
    thresholds = np.sort(thresholds)[::-1]

    best_t = 0.5
    best_score = -1.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        if sensitivity >= target and specificity >= min_specificity:
            return float(t)
        # Track best balanced fallback
        score = sensitivity + specificity
        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t


def run_pecarn(df_under2: pd.DataFrame, df_over2: pd.DataFrame) -> Dict[str, ModelMetrics]:
    results: Dict[str, ModelMetrics] = {}
    if len(df_under2) > 0:
        df_under2["pecarn_pred"] = df_under2.apply(pecarn_rule_under2, axis=1)
        metrics = evaluate_binary(df_under2["target"].to_numpy(), df_under2["pecarn_pred"].to_numpy())
        results["under2"] = metrics
    if len(df_over2) > 0:
        df_over2["pecarn_pred"] = df_over2.apply(pecarn_rule_over2, axis=1)
        metrics = evaluate_binary(df_over2["target"].to_numpy(), df_over2["pecarn_pred"].to_numpy())
        results["over2"] = metrics
    return results


def fit_logistic_regression(
    df: pd.DataFrame, predictors: List[str], target_sensitivity: float = 0.98
) -> Tuple[LogisticRegression, ModelMetrics]:
    model_df = df[predictors + ["target"]].dropna()
    X = model_df[predictors].to_numpy(dtype=float)
    y = model_df["target"].to_numpy(dtype=int)

    # Use larger test split for small datasets so the test set has enough positives
    n_pos = int(y.sum())
    test_size = 0.3 if n_pos < 80 else 0.2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    model.fit(X_train, y_train)

    y_train_prob = model.predict_proba(X_train)[:, 1]
    threshold = choose_threshold_for_sensitivity(y_train, y_train_prob, target_sensitivity)

    y_test_prob = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    auc = roc_auc_score(y_test, y_test_prob) if len(np.unique(y_test)) > 1 else float("nan")

    metrics = ModelMetrics(
        sensitivity=sensitivity,
        specificity=specificity,
        auc=auc,
        confusion=(tn, fp, fn, tp),
        threshold=threshold,
    )

    return model, metrics


def fit_xgboost(
    df: pd.DataFrame, predictors: List[str], target_sensitivity: float = 0.98
) -> Tuple[object | None, ModelMetrics]:
    if xgb is None:
        return None, ModelMetrics(0, 0, float("nan"), (0, 0, 0, 0), None)

    model_df = df[predictors + ["target"]].dropna()
    X = model_df[predictors].to_numpy(dtype=float)
    y = model_df["target"].to_numpy(dtype=int)

    # Use larger test split for small datasets so the test set has enough positives
    n_pos = int(y.sum())
    test_size = 0.3 if n_pos < 80 else 0.2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    n_pos = max(int(y_train.sum()), 1)
    n_neg = max(int((1 - y_train).sum()), 1)
    scale_pos_weight = n_neg / n_pos

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        max_depth=4,
        learning_rate=0.1,
        n_estimators=200,
    )
    model.fit(X_train, y_train)

    y_train_prob = model.predict_proba(X_train)[:, 1]
    threshold = choose_threshold_for_sensitivity(y_train, y_train_prob, target_sensitivity)

    y_test_prob = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    auc = roc_auc_score(y_test, y_test_prob) if len(np.unique(y_test)) > 1 else float("nan")

    metrics = ModelMetrics(
        sensitivity=sensitivity,
        specificity=specificity,
        auc=auc,
        confusion=(tn, fp, fn, tp),
        threshold=threshold,
    )

    return model, metrics


def run_all_models(data_clean: pd.DataFrame, target_sensitivity: float = 0.98) -> Dict[str, Dict[str, ModelMetrics]]:
    df_under2, df_over2, features_under2, features_over2 = prepare_model_frames(data_clean)

    pecarn_metrics = run_pecarn(df_under2, df_over2)

    _, logit_under2 = fit_logistic_regression(df_under2, features_under2, target_sensitivity)
    _, logit_over2 = fit_logistic_regression(df_over2, features_over2, target_sensitivity)

    _, xgb_under2 = fit_xgboost(df_under2, features_under2, target_sensitivity)
    _, xgb_over2 = fit_xgboost(df_over2, features_over2, target_sensitivity)

    return {
        "pecarn": pecarn_metrics,
        "logit": {"under2": logit_under2, "over2": logit_over2},
        "xgboost": {"under2": xgb_under2, "over2": xgb_over2},
    }


def format_metrics(metrics: ModelMetrics) -> str:
    tn, fp, fn, tp = metrics.confusion
    return (
        f"Sensitivity={metrics.sensitivity:.4f}, Specificity={metrics.specificity:.4f}, "
        f"AUC={metrics.auc:.4f}, Confusion=[[{tn},{fp}],[{fn},{tp}]], "
        f"Threshold={metrics.threshold if metrics.threshold is not None else 'n/a'}"
    )


def _metrics_to_dict(m: ModelMetrics) -> dict:
    tn, fp, fn, tp = m.confusion
    return {
        "sensitivity": m.sensitivity,
        "specificity": m.specificity,
        "auc": m.auc,
        "confusion": [tn, fp, fn, tp],
        "threshold": m.threshold,
    }


def save_results(results: dict, path: str = "model_results.json") -> None:
    """Save model results to JSON for use in notebook when sklearn unavailable."""
    import json

    out = {}
    for model_name, sub in results.items():
        if isinstance(sub, dict):
            out[model_name] = {k: _metrics_to_dict(v) for k, v in sub.items()}
        else:
            out[model_name] = _metrics_to_dict(sub)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {path}")


def main(data_path: str, verbose_clean: bool = False, save_results_path: str | None = None) -> None:
    """Run all models on cleaned PECARN data."""
    from clean import clean_data

    df_raw = pd.read_csv(data_path)
    data_clean = clean_data(df_raw, verbose=verbose_clean)

    results = run_all_models(data_clean)

    print("\nPECARN <2:", format_metrics(results["pecarn"]["under2"]))
    print("PECARN ≥2:", format_metrics(results["pecarn"]["over2"]))

    print("\nLogit <2:", format_metrics(results["logit"]["under2"]))
    print("Logit ≥2:", format_metrics(results["logit"]["over2"]))

    print("\nXGBoost <2:", format_metrics(results["xgboost"]["under2"]))
    print("XGBoost ≥2:", format_metrics(results["xgboost"]["over2"]))

    if save_results_path:
        save_results(results, save_results_path)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run Lab 1 models")
    default_path = Path(__file__).parent.parent / "data" / "TBI PUD 10-08-2013.csv"
    parser.add_argument("--data-path", default=str(default_path))
    parser.add_argument("--verbose-clean", action="store_true", help="Print cleaning progress")
    parser.add_argument("--save-results", metavar="PATH", help="Save results to JSON for notebook")
    args = parser.parse_args()

    main(
        args.data_path,
        verbose_clean=args.verbose_clean,
        save_results_path=args.save_results,
    )
