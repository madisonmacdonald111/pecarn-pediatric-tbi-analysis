"""
Model implementation for the PECARN TBI dataset.

Implements three classifiers: the PECARN CDR (Kuppermann et al., 2009),
an L2-regularized logistic regression, and a balanced random forest.
All models are evaluated on the primary analysis population (GCS 14-15).

Usage (from repo root):
    python code/models.py data/TBI_PUD_cleaned.csv

    This will:
        1. Load the cleaned data (output of clean.py)
        2. Preprocess for modeling
        3. Fit and evaluate all three models
        4. Print sensitivity, specificity, PPV, NPV for each model
"""

import numpy as np 
import pandas as pd

# ---------------------------------------------------------------------------
# Preprocessing - shared across all models 
# ---------------------------------------------------------------------------

def preprocess_for_modeling(df):
    """
    Apply modeling-specific preprocessing to the cleaned PECARN DataFrame 

    This is a separate step from data cleaning (clean.py). 
    Data cleaning fixes raw data quality issues. 
    Preprocessing further transforms this clean data to meet formatting requirements of the models fitted 

    This function: 
        1. Restricts to the primary analysis population (GCS 14–15, no
           pre-existing conditions that confound assessment), matching the
           population studied in Kuppermann et al.
        2. Drops the 20 rows whose outcome (has_clinically_important_tbi) is
           missing — these patients did not receive a CT scan and had no
           documented follow-up, so their true outcome is unknown.
        3. Derives the binary CDR predictor columns used directly by
           predict_cdr() (e.g., scalp_hematoma_nonfrontal, loc_any,
           severe_headache). These are not added to the clean data because
           they encode modeling-specific judgment calls about how to binarize
           multi-valued variables.
        4. Resets the index so downstream code can use positional indexing
           reliably.
    
    Parameters
    ----------
    df: pd.DataFrame 
        Cleaned DataFrame returned by clean_data() in clean.py 
    
    Returns 
    -------
    pd.DataFrame 
        Subset of df restricted to the primary analysis population, 
        with derived CDR predictor columns appended and a reset index.
        The outcome column is has_clinically_important_tbi (float, 0/1).
    """

    # --- Step 1: Restrict to primary analysis population ---
        # Kuppermann et al. define the primary analysis population as patients with a GCS of 14 or 15 (i.e., not severely impaired) 
        # and without conditions that would prevent reliable clinical assessment.
        # The clean.py pipeline marks these rows with in_primary_analysis = 1.
    df = df[df["in_primary_analysis"] == 1].copy()
    
    # --- Step 2: Drop rows with missing outcome ---
        # 20 rows have NaN for has_clinically_important_tbi
        # These patients either did not recieve a CT scan and were lost to follow-up
            # or had incomplete records 
        # We cannot use them for supervised learning. 
    df = df.dropna(subset=["has_clinically_important_tbi"])
    
    # --- Step 3: Derive binary CDR predictors ---
        # These derived columns encode the binarization judgment calls for the CDR.
        # Each decision is documented below 

    # Altered mental status: 1 if has_altered_mental_status == 1, else 0.
    # NaN -> treat as 0 (absent), consistent with conservative coding in the
         # original study (clinicians would have recorded it if present).
    df["cdr_ams"] = (df["has_altered_mental_status"] == 1).astype(int)

    # Any loss of consciousness (binary yes/no)
        # has_loss_of_consciousness_history: 1 = yes, 2=suspected/unwitnessed, 0 = no
        # The CDR counts both witnessed (1) and suspected (2) as LOC.
    df["cdr_loc_any"] = (
        df["has_loss_of_consciousness_history"].isin([1, 2])
    ).astype(int)

    # LOC > 5 seconds — used only in the >=2 branch of the CDR.
        #  loss_of_consciousness_duration: 1 = <5s, 2 = 5s–1min, 3 = 1–5min, 4 = >5min
        # Values 2, 3, 4 all represent LOC lasting at least 5 seconds
    df["cdr_loc_over5s"] = (
        df["loss_of_consciousness_duration"].isin([2, 3, 4])
    ).astype(int)

    # Palpable skull fracture — used only in the <2 branch
        # has_palpable_skull_fracture: 0 = no, 1 = yes, 2 = indeterminate/possible
        # We treat both 1 (confirmed) and 2 (suspected) as positive
            #  because in a clinical triage setting the CDR is intended to be conservative
    df["cdr_palpable_skull_fx"] = (
            df["has_palpable_skull_fracture"].isin([1, 2])
    ).astype(int)

    # Signs of basilar skull fracture - used only in the >=2 branch 
    df["cdr_basilar_skull_fx"] = (
        df["has_basilar_skull_fracture_signs"] == 1
    ).astype(int)

    # Scalp hematoma at a non-frontal location - used only in the <2 branch
    # Kuppermann specifically identifies occipital, parietal, or temporal hematomas as predictive in infants
        # frontal hematomas are not included
    # Any of the three location flags = 1 qualifies
    df["cdr_scalp_hematoma_nonfrontal"] = (
        (df["has_trauma_scalp_occipital"] == 1)
        | (df["has_trauma_scalp_parietal"] == 1)
        | (df["has_trauma_scalp_temporal"] == 1)
    ).astype(int)

    # Not acting normally per parent - used only in the <2 branch 
        # acting_normal: 1= normal, 0 = not normal. Invert so 1 = risk factor 
        # NaN -> treat as 0 (unknown, conservatively assuming acting normally) 
    df["cdr_not_acting_normally"] = (df["acting_normal"] == 0).astype(int)

    # Severe headache - used only in the >=2 branch 
        # headache_severity: 1 = mild, 2 = moderate, 3 = severe
        # has_headache_at_ED: 91 is used for patients too young/unable to report HA
            # these are in the <2 group and this predictor is not used there 
    df["cdr_severe_headache"] = (df["headache_severity"] == 3).astype(int)

    # Vomiting - used only in the >= 2 branch 
    df["cdr_vomiting"] = (df["has_vomiting_post_injury"] == 1).astype(int)

    # Severe injury mechanism - used in both branches 
        # injury_mechanism_severity: 1 = low, 2 = moderate, 3 = high (e.g., MVC, ped/bike vs. auto, fall > 5 ft, diving). Only severity=3 is high-risk per Kuppermann
    df["cdr_severe_mechanism"] = (
        df["injury_mechanism_severity"] == 3
    ).astype(int)

    df = df.reset_index(drop=True)
    return df

# ---------------------------------------------------------------------------
# Model 1: PECARN Clinical Decision Rule (Kuppermann et al., 2009)
# ---------------------------------------------------------------------------

def _cdr_under2(row):
    """
    Apply the PECARN CDR for children under 2 years old 

    The CDR for this age group uses a two-step decision tree: 

    Step 1 : Immediate CT recommended if ANY of: 
        - GCS < 15  (gcs_category == 1 in the cleaned data)
        - Altered mental status
        - Palpable or indeterminate skull fracture
    
    Step 2 : Observation vs. CT not indicated based on: 
        Observation (consider CT if ANY of):
        - Occipital, parietal, or temporal scalp hematoma
        - LOC >= 5 seconds
        - Severe injury mechanism
        - Not acting normally per parent
        Otherwise: CT not routinely indicated 

    Returns
    -------
    int 
        1 = CT recommended (high risk), 0 = CT not indicated (low risk)
        Observation cases are returned as 1 (conservative: flag is positive)
    """

    # Step 1: high-risk predictors -> CT immediately 
    if row["gcs_category"] == 1:
        return 1 
    if row["cdr_ams"] == 1: 
        return 1
    if row["cdr_palpable_skull_fx"] == 1: 
        return 1 

    # Step 2: intermediate-risk predictors -> observation (flagged as 1 here)
    if row["cdr_scalp_hematoma_nonfrontal"] == 1:
        return 1 
    # Under-2 uses LOC >= 5 seconds (not any LOC) as the intermediate predictor
    if row["cdr_loc_over5s"] == 1:
        return 1
    if row["cdr_severe_mechanism"] == 1:
        return 1
    if row["cdr_not_acting_normally"] == 1:
        return 1
    
    # No risk factors -> CT not indicated 
    return 0 

def _cdr_over2(row):
    """
    Apply the PECARN CDR for children 2 years and older. 

    Step 1 — Immediate CT recommended if ANY of:
        - GCS < 15
        - Altered mental status
        - Signs of basilar skull fracture

    Step 2 — Observation vs. CT not indicated:
        Observation (consider CT) if ANY of:
        - LOC > 5 seconds
        - Vomiting
        - Severe injury mechanism
        - Severe headache
        Otherwise: CT not routinely indicated
    
    Returns 
    -------
    int 
        1 = CT recommended (high risk), 0 = CT not indicated (low risk)
        Observation cases are flagged as 1 (conservative)
    """

    # Step 1: high-risk predictors
    if row["gcs_category"] == 1:
        return 1
    if row["cdr_ams"] == 1:
        return 1
    if row["cdr_basilar_skull_fx"] == 1:
        return 1

    # Step 2: intermediate-risk predictors
    # Over-2 uses any LOC (not just >5 seconds) as the intermediate predictor
    if row["cdr_loc_any"] == 1:
        return 1
    if row["cdr_vomiting"] == 1:
        return 1
    if row["cdr_severe_mechanism"] == 1:
        return 1
    if row["cdr_severe_headache"] == 1:
        return 1

    # No risk factors -> CT not indicated
    return 0

def predict_cdr(df):
    """
    Apply the PECARN CDR to a preprocessed DataFrame and return predictions 

    This function routes each patient through the age-appropriate branch 
    of the CDR (_cdr_under2 for patients < 2 years old, _cdr_over2 for patients >= 2 years old)
    and returns a binary prediction: 1 = CT recommended (and ciTBI predicted), 0 = CT not routinely indicated.
    
    The CDR is a rule-based classifier, not a learned model. 
    It has no free parameters and does not require a training set. 
    It is applied identically to all patients. 


    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame from preprocess_for_modeling(). Must contain
        the derived cdr_* columns and patient_age_under_2yr.

    Returns
    -------
    np.ndarray of shape (n,)
        Binary predictions: 1 = CT recommended, 0 = CT not indicated.
    """

    preds = np.empty(len(df), dtype=int)

    under2_mask = df["patient_age_under_2yr"] == 1
    over2_mask = df["patient_age_under_2yr"] == 2

    # Apply under-2 branch 
    if under2_mask.any():
        preds[under2_mask.values] = df[under2_mask].apply(
            _cdr_under2, axis=1
        ).values
    
    # Apply over-2 branch 
    if over2_mask.any(): 
        preds[over2_mask.values] = df[over2_mask].apply(
            _cdr_over2, axis=1
        ).values
    
    return preds 

def evaluate_cdr(df):
    """
    Evaluate CDR predictions against the true outcome (has_clinically_important_tbi)

    Reports sensitivity, specificity, PPV, NPV, and overall accuracy.
    Sensitivity (recall for ciTBI=1) is the most clinically important metric:
    a high-quality triage rule must not miss true positives (children who actually have a clinically important TBI).

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame from preprocess_for_modeling().

    Returns 
    -------
    dict
        Dictionary of performance metrics.
    """

    y_true = df["has_clinically_important_tbi"].astype(int).values
    y_pred = predict_cdr(df)

    tp =  int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    accuracy = (tp + tn) / len(y_true)

    return {
        "n": len(y_true),
        "n_citbi": int(y_true.sum()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "ppv": round(ppv, 4),
        "npv": round(npv, 4),
        "accuracy": round(accuracy, 4),
    }

# ---------------------------------------------------------------------------
# Model 2: Logistic Regression 
# ---------------------------------------------------------------------------

# Features used by the logistic regression and all subsequent learned models.
# These are the same 10 binary CDR predictors derived in preprocess_for_modeling(), plus a binary age flag (is_under_2). 
# We use the same predictor set as the CDR to keep comparisons apples-to-apples: any difference in performance is due to the modeling approach, not the features.
LR_FEATURES = [
    "cdr_ams",
    "cdr_loc_any",
    "cdr_loc_over5s",
    "cdr_palpable_skull_fx",
    "cdr_basilar_skull_fx",
    "cdr_scalp_hematoma_nonfrontal",
    "cdr_not_acting_normally",
    "cdr_severe_headache",
    "cdr_vomiting",
    "cdr_severe_mechanism",
    "is_under_2",
]

def _add_age_flag(df):
    """
    Add a binary is_under_2 column (1 = under 2 years old, 0 = 2 and older).

    The raw patient_age_under_2yr column uses 1 and 2 as its values rather
    than 0 and 1. This helper recodes it to a standard binary indicator so
    it can be included directly in a feature matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame from preprocess_for_modeling().

    Returns
    -------
    pd.DataFrame
        Same DataFrame with is_under_2 column added in place.
    """
    df = df.copy()
    df["is_under_2"] = (df["patient_age_under_2yr"] == 1).astype(int)
    return df

def fit_logistic_regression(df):
    """
    Fit an L2-regularized logistic regression on the full primary analysis population.

    Design choices and their justification:

    Features: The same 10 binary CDR predictors plus a binary age indicator (is_under_2).
    Rather than fitting separate models for each age group as the CDR does, 
    we train a single model on the full population with age as a feature. 
    This is appropriate because the ciTBI rates are nearly identical across age groups (0.91% vs 0.88%), 
    and the predictors behave similarly in both groups. 
    Splitting would leave only 98 positives in the under-2 group: too few for a stable separate model.

    Regularization: L2 (Ridge) with C=1.0 (sklearn default). 
    With a 1:113 class imbalance and only 376 positive cases, some regularization is prudent to prevent overfitting. 
    L2 was chosen over L1 because all 10 CDR predictors have clear domain justification for inclusion: we do not want sparsity to inadvertently drop a clinically meaningful variable.
    Cross-validation over C in {0.01, 0.1, 1.0, 10.0} showed recall was stable across all values (~0.78), confirming the model is not sensitive to this choice. 
    C=1.0 is retained as the default.

    Class weighting: class_weight='balanced' automatically upweights the minority class (ciTBI=1) by a factor proportional to the class imbalance (~113x). 
    This shifts the model's decision boundary to prioritize sensitivity : consistent with the clinical goal of minimizing missed diagnoses (without requiring manual threshold tuning).

    No imputation is needed: all CDR-derived binary predictors have zero missing values after preprocess_for_modeling().

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame from preprocess_for_modeling().

    Returns
    -------
    sklearn.linear_model.LogisticRegression
        Fitted logistic regression model.
    """
    from sklearn.linear_model import LogisticRegression

    df = _add_age_flag(df)
    X = df[LR_FEATURES].values
    y = df["has_clinically_important_tbi"].astype(int).values

    model = LogisticRegression(
        C=1.0,                    # L2 regularization strength (inverse); see docstring
        class_weight="balanced",  # upweight minority class to prioritize sensitivity
        solver="lbfgs",           # efficient for small-to-medium dense problems
        max_iter=1000,            # increase from default 100 to ensure convergence
        random_state=214,         # reproducibility
    )
    model.fit(X, y)
    return model

def predict_logistic_regression(model, df):
    """
    Generate binary predictions from a fitted logistic regression model.

    Parameters
    ----------
    model : fitted LogisticRegression
        Output of fit_logistic_regression().
    df : pd.DataFrame
        Preprocessed DataFrame from preprocess_for_modeling().

    Returns
    -------
    np.ndarray of shape (n,)
        Binary predictions: 1 = ciTBI predicted, 0 = not predicted.
    """
    df = _add_age_flag(df)
    X = df[LR_FEATURES].values
    return model.predict(X)

def evaluate_logistic_regression(model, df):
    """
    Evaluate logistic regression predictions against the true ciTBI outcome.

    Parameters
    ----------
    model : fitted LogisticRegression
        Output of fit_logistic_regression().
    df : pd.DataFrame
        Preprocessed DataFrame from preprocess_for_modeling().

    Returns
    -------
    dict
        Dictionary of performance metrics (same keys as evaluate_cdr).
    """
    y_true = df["has_clinically_important_tbi"].astype(int).values
    y_pred = predict_logistic_regression(model, df)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    accuracy = (tp + tn) / len(y_true)

    return {
        "n": len(y_true),
        "n_citbi": int(y_true.sum()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "ppv": round(ppv, 4),
        "npv": round(npv, 4),
        "accuracy": round(accuracy, 4),
    }

# ---------------------------------------------------------------------------
# Model 3: Random Forest 
# ---------------------------------------------------------------------------

# Probability threshold selected via 5-fold cross-validation. 
# In each fold, the lowest threshold on the training set that achieved >= 95% sensitivity was recorded. 
# Thresholds were stable across folds (0.26, 0.26, 0.28, 0.29, 0.29), giving a mean of 0.276. 
# We us 0.27 as the final threshold: the nearest round value to the cross-validated mean. 
# See fit_random_forest() docstring for full justification.
RF_THRESHOLD = 0.27 

def fit_random_forest(df):
    """
    Fit a balanced Random Forest with a sensitivity-targeted decision threshold. 

    Motivation: The CDR achieves 96.8% sensitivity but only 56.6% specificity, flagging over 18,000 healthy children for CT. 
    The logistic regression improves specificity to 84.1% but drops sensitivity to 76.9%, missing 87 true ciTBI cases. 
    
    This model asks whether an ensemble method can recover sensitivity closer to the CDR's level while also improving on its specificity. 
    Random forests are a natural candidate: they capture nonlinear interactions between predictors 
    (e.g., AMS combined with severe mechanism may be more predictive than either alone), which neither the rule-based CDR nor the linear logistic regression can represent.
    
    Design choices: 

    Estimators: 200 trees. This is sufficient for stable out-of-bag error estimates on a dataset of this size. 
    Increasing beyond 200 produced negligible improvement in cross-validated sensitivity.

    class_weight = 'balanced': Same rationale as the logistic regression: the 1:113 class imbalance requires upweighting ciTBI cases to prevent the model from predicting negative for everything. 

    min_samples_leaf = 10: Without a minimum leaf size, individual trees overfit to the small number of positive cases. 
    Cross-validation over min_samples_leaf in {1, 2, 5, 10, 20, 30, 50} showed that 10 gave the best balance of sensitivity and specificity (sensitivity=0.753, specificity=0.843) using the default 0.5 threshold: comparable to the logistic regression baseline. 
    The threshold tuning step (below) then recovers the sensitivity gap. 

    Decision threshold: Rather than using the default 0.5 threshold (which produced sensitivity of only ~75%), we select the threshold using 5-fold cross-validation. 
    In each training fold, we find the lowest probability threshold that achieves at least 95% sensitivity on the training data, then evaluate on the held out fold. 
    The chosen threshold was extremely stable across folds (0.26–0.29), with a mean of 0.276.
    We set the final threshold to 0.27 (nearest round value to the mean).
    This cross-validated approach guards against overfitting the threshold to the full dataset.

    Parameters 
    ----------
    df: pd.DataFrame 
        Preprocessed DataFrame from preprocess_for_modeling()

    Returns 
    -------
    sklearn.ensemble.RandomForestClassifier
        Fitted random forest model. 
        Use with RF_THRESHOLD and predict_random_forest() to generate predictions.
    """

    from sklearn.ensemble import RandomForestClassifier

    df = _add_age_flag(df)
    X = df[LR_FEATURES].values
    y = df["has_clinically_important_tbi"].astype(int).values 

    model = RandomForestClassifier(
        n_estimators=200,          # sufficient for stable estimates; see docstring
        class_weight="balanced",   # upweight minority class
        min_samples_leaf=10,       # controls overfitting; selected via cross-validation
        random_state=214,          # reproducibility
        n_jobs=-1,                 # use all available cores
    )
    model.fit(X,y)
    return model 

def predict_random_forest(model, df, threshold=RF_THRESHOLD): 
    """
    Generate binary predictions from a fitted Random Forest. 

    Applies the sensitivity-targeted probability threshold (default RF_THRESHOLD=0.27) rather than the standard 0.5 cutoff. 
    Patients whose predicted ciTBI probability meets or exceeds the threshold are classified as positive.

    Parameters 
    ----------
    model : fitted RandomForestClassifier 
        Output of fit_random_forest()
    df : pd.DataFrame 
        Preprocessed DataFrame from preprocess_for_modeling()
    threshold : float 
        Probability threshold for positive classification
        Default is RF_THRESHOLD (0.27), selected via cross-validation
    
    Returns
    -------
    np.ndarray of shape (n,)
        Binary predictions: 1 = ciTBI predicted, 0 = not predicted 
    """
    df = _add_age_flag(df)
    X = df[LR_FEATURES].values 
    proba = model.predict_proba(X)[:, 1]
    return (proba >= threshold).astype(int)

def evaluate_random_forest(model, df, threshold=RF_THRESHOLD):
    """
    Evaluate Random Forest predictions against the true ciTBI outcome. 

    Parameters 
    ----------
    model : fitted RandomForestClassifier 
        Output of fit_random_forest()
    df : pd.DataFrame
        Preprocessed DataFrame from preprocess_for_modeling()
    threshold : float 
        Probability threshold. Default is RF_THRESHOLD (0.27)
    
    Returns 
    -------
    dict 
        Dictionary of performance metrics (same keys as evaluate_cdr)
    """
    y_true = df["has_clinically_important_tbi"].astype(int).values 
    y_pred = predict_random_forest(model, df, threshold=threshold)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    accuracy = (tp + tn) / len(y_true)

    return {
        "n": len(y_true),
        "n_citbi": int(y_true.sum()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "ppv": round(ppv, 4),
        "npv": round(npv, 4),
        "accuracy": round(accuracy, 4),
    }

# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python code/models.py <path_to_cleaned_csv>")
        sys.exit(1)

    cleaned_path = sys.argv[1]
    print(f"Loading cleaned data from: {cleaned_path}")
    df_clean = pd.read_csv(cleaned_path)
    print(f"Cleaned data shape: {df_clean.shape}")

    print("\nPreprocessing for modeling...")
    df_model = preprocess_for_modeling(df_clean)
    print(f"Modeling population shape: {df_model.shape}")
    print(
        f"ciTBI positives: {df_model['has_clinically_important_tbi'].sum():.0f} "
        f"({df_model['has_clinically_important_tbi'].mean():.2%})"
    )

    print("\n--- PECARN CDR Evaluation ---")
    metrics = evaluate_cdr(df_model)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\n--- Logistic Regression Evaluation ---")
    lr_model = fit_logistic_regression(df_model)
    lr_metrics = evaluate_logistic_regression(lr_model, df_model)
    for k, v in lr_metrics.items():
        print(f"  {k}: {v}")

    print("\n--- Random Forest Evaluation ---")
    rf_model = fit_random_forest(df_model)
    rf_metrics = evaluate_random_forest(rf_model, df_model)
    for k, v in rf_metrics.items():
        print(f"  {k}: {v}")