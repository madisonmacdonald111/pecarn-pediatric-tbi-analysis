import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# Kuppermann CDR
def kupp_cdr(df: pd.DataFrame) -> pd.Series:
    """Apply the Kuppermann clinical decision rule to predict ciTBI."""

    def predict_row(row):
        gcs_total = row["GCSTotal"]

        if (gcs_total < 15) or (row["SFxPalp"] == 1):
            return 1

        if (
            row["LOCSeparate"] == 1
            or row["HASeverity"] == 3
            or row["High_impact_InjSev"] == 1
            or row["Vomit"] == 1
        ):
            return 1

        return 0

    return df.apply(predict_row, axis=1)


# Logistic Regression
def logistic_model(df: pd.DataFrame, outcome_col: str = "PosIntFinal"):
    """Fit logistic regression and return predicted probabilities."""

    predictors = [
        "AgeinYears",
        "Vomit",               
        "High_impact_InjSev",  
        "HASeverity",          
        "GCSTotal",            
        "LOCSeparate",         
        "SFxPalp",            
    ]

    model_df = df[predictors + [outcome_col]].copy()

    model_df = model_df[model_df[outcome_col].notna()]

    model_df = model_df.dropna()

    X = model_df[predictors]
    y = model_df[outcome_col].astype(int)

    model = LogisticRegression(max_iter=500)
    model.fit(X, y)

    proba = pd.Series(
        model.predict_proba(X)[:, 1],
        index=model_df.index,
        name="logit_ciTBI_prob",
    )
    return proba


# Random Forest

def random_forest_model(df: pd.DataFrame, outcome_col: str = "PosIntFinal"):
    """Fit a Random Forest classifier and return predicted probabilities."""

    predictors = [
        "AgeinYears",
        "Vomit",
        "High_impact_InjSev",
        "HASeverity",
        "GCSTotal",
        "LOCSeparate",
        "SFxPalp",
    ]

    model_df = df[predictors + [outcome_col]].copy()
    model_df = model_df[model_df[outcome_col].notna()]
    model_df = model_df.dropna()

    X = model_df[predictors]
    y = model_df[outcome_col].astype(int)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42,
    )
    model.fit(X, y)

    proba = pd.Series(
        model.predict_proba(X)[:, 1],
        index=model_df.index,
        name="rf_ciTBI_prob",
    )
    return proba