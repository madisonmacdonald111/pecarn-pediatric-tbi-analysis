from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

def load_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parents[1] / "data" / "TBI PUD 10-08-2013.csv"
    return pd.read_csv(data_path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply PECARN data dictionary rules and clinically-motivated consistency checks."""
    
    df = df.copy()

    # Replace true missing codes (93–99)
    missing_codes = [93,94,95,96,97,98,99]
    df = df.replace(missing_codes, np.nan)

    # Convert categorical variables
    categorical_vars = [
        'EmplType','Certification','InjuryMech','High_impact_InjSev',
        'Amnesia_verb','LOCSeparate','Seiz','ActNorm','HA_verb','Vomit',
        'Dizzy','Intubated','Paralyzed','Sedated','GCSGroup','AMS','SFxPalp',
        'FontBulg','SFxBas','Hema','Clav','NeuroD','OSI','Drugs','CTForm1',
        'CTDone','EDCT','PosCT','DeathTBI','HospHead','HospHeadPosCT',
        'Intub24Head','Neurosurgery','PosIntFinal'
    ]
    for col in categorical_vars:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Convert ordinal variables
    ordinal_map = {
        "LocLen": [1,2,3,4,92],
        "HASeverity": [1,2,3,92],
        "HAStart": [1,2,3,4,92],
        "VomitNbr": [1,2,3,92],
        "VomitStart": [1,2,3,4,92],
        "VomitLast": [1,2,3,92]
    }
    for col, order in ordinal_map.items():
        if col in df.columns:
            df[col] = pd.Categorical(df[col], categories=order, ordered=True)

    # Logical consistency rules
    df.loc[(df["LOCSeparate"] == 0) & (df["LocLen"].isin([1,2,3,4])),
           "LocLen"] = np.nan

    df.loc[(df["Seiz"] == 0) & (df["SeizOccur"].isin([1,2,3])),
           ["SeizOccur","SeizLen"]] = np.nan

    df.loc[(df["Hema"] == 0) & (df["HemaLoc"].isin([1,2,3])),
           ["HemaLoc","HemaSize"]] = np.nan

    clav_cols = ["ClavFace","ClavNeck","ClavFro","ClavOcc","ClavPar","ClavTem"]
    df.loc[df["Clav"] == 0, clav_cols] = np.nan

    osi_cols = ["OSIExtremity","OSICut","OSICspine","OSIFlank","OSIAbdomen","OSIPelvis","OSIOth"]
    df.loc[df["OSI"] == 0, osi_cols] = np.nan

    df.loc[df["SFxPalp"] == 0, "SFxPalpDepress"] = np.nan

    bas_cols = ["SFxBasHem","SFxBasOto","SFxBasPer","SFxBasRet","SFxBasRhi"]
    df.loc[df["SFxBas"] == 0, bas_cols] = np.nan

    finding_cols = [f"Finding{i}" for i in list(range(1,15)) + [20,21,22,23]]
    df.loc[df["CTDone"] == 0, finding_cols] = 92

    return df

if __name__ == "__main__":
    df = load_data()
    cleaned = clean_data(df)
    cleaned.to_csv("../data/cleaned.csv", index=False)
    print("Cleaning complete. Shape:", cleaned.shape)