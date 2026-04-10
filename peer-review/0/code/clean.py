"""
PECARN TBI Data Cleaning Module
STAT 214 Lab 1

Main cleaning function for the PECARN TBI dataset. The pipeline:
1. Handles special codes (90/91/92) representing missing/unknown values
2. Converts categorical codes to meaningful labels
3. Standardizes column names to snake_case
4. Derives clinically important variables (citbi, age_group)
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd


def clean_data(df_raw, documentation_path=None, verbose=True):
    """
    Clean the raw PECARN TBI dataset.
    
    This is the main data cleaning function required by Lab 1 instructions.
    Takes raw DataFrame immediately after loading with pandas and returns
    cleaned version following VDS principles.
    
    Parameters:
    -----------
    df_raw : pandas.DataFrame
        Raw data immediately after pd.read_csv('TBI PUD 10-08-2013.csv')
        
    documentation_path : str or Path, optional
        Path to 'TBI PUD Documentation 10-08-2013.xlsx' (reserved for future use).

    verbose : bool, optional
        If True (default), print progress messages. Set False to suppress output.

    Returns:
    --------
    df_clean : pandas.DataFrame
        Cleaned dataset with:
        - Special codes (90/91/92) converted to NaN
        - Categorical variables converted to meaningful labels
        - Column names standardized to snake_case
        - Derived variables added (citbi, age_group)
        
    Examples:
    ---------
    >>> import pandas as pd
    >>> from clean import clean_data
    >>> df_raw = pd.read_csv('../data/TBI PUD 10-08-2013.csv')
    >>> df_clean = clean_data(df_raw)
    >>> df_clean.head()
    """
    
    # Make a copy to avoid modifying original
    data = df_raw.copy()
    
    # Track cleaning operations for logging
    log = {
        'original_rows': len(data),
        'original_cols': len(data.columns),
        'special_codes_replaced': 0,
        'categorical_vars_converted': 0,
        'columns_renamed': 0,
        'validation_issues': [],
        'final_rows': 0,
        'final_cols': 0
    }
    
    def _log(msg):
        if verbose:
            print(msg)

    _log("=" * 80)
    _log("PECARN TBI Data Cleaning Pipeline")
    _log("=" * 80)
    _log(f"Input: {log['original_rows']:,} rows x {log['original_cols']} columns\n")
    
    # --- Define special codes ---
    # From TBI PUD Documentation:
    # 90 = Other / Unknown
    # 91 = Not done
    # 92 = Not applicable
    special_codes = [90, 91, 92]
    
    # --- Category Mappings ---
    # These map numeric codes to meaningful categorical labels
    # Generated from TBI PUD Documentation 10-08-2013.xlsx
    
    category_mappings = {
        # Demographics
        'Gender': {1: 'Male', 2: 'Female'},
        'Race': {1: 'White', 2: 'Black', 3: 'Asian', 4: 'American Indian/Alaskan Native', 5: 'Pacific Islander'},
        'Ethnicity': {1: 'Hispanic/Latino', 2: 'Non-Hispanic'},
        
        # Provider Information
        'EmplType': {1: 'Nurse Practitioner', 2: 'Physician Assistant', 3: 'Resident', 4: 'Fellow', 5: 'Faculty'},
        'Certification': {1: 'Emergency Medicine', 2: 'Pediatrics', 3: 'Pediatrics Emergency Medicine', 
                         4: 'Family Practice', 5: 'Other'},
        
        # Injury & History
        'InjuryMech': {
            1: 'MVC occupant', 2: 'Pedestrian struck', 3: 'Bike struck by auto', 4: 'Bike collision/fall', 
            5: 'Other wheeled transport', 6: 'Fall from standing', 7: 'Walked/ran into object', 
            8: 'Fall from elevation', 9: 'Fall down stairs', 10: 'Sports', 11: 'Assault', 12: 'Object struck head'
        },
        'High_impact_InjSev': {1: 'Low', 2: 'Moderate', 3: 'High'},
        'LOCSeparate': {0: 'No', 1: 'Yes', 2: 'Suspected'},
        'LocLen': {1: '<5 sec', 2: '5 sec-<1 min', 3: '1-5 min', 4: '>5 min'},
        'SeizOccur': {1: 'Immediately on contact', 2: 'Within 30 mins', 3: '>30 mins after injury'},
        'SeizLen': {1: '<1 min', 2: '1-5 min', 3: '>5-15 min', 4: '>15 min'},
        'HASeverity': {1: 'Mild', 2: 'Moderate', 3: 'Severe'},
        'HAStart': {1: 'Before injury', 2: 'Within 1 hr', 3: '1-4 hrs', 4: '>4 hrs'},
        'VomitNbr': {1: 'Once', 2: 'Twice', 3: '>2 times'},
        'VomitStart': {1: 'Before injury', 2: 'Within 1 hr', 3: '1-4 hrs', 4: '>4 hrs'},
        'VomitLast': {1: '<1 hr before ED', 2: '1-4 hrs before ED', 3: '>4 hrs before ED'},
        
        # Physical Exam - GCS
        'GCSEye': {1: 'None', 2: 'Pain', 3: 'Verbal', 4: 'Spontaneous'},
        'GCSVerbal': {1: 'None', 2: 'Incomprehensible', 3: 'Inappropriate', 4: 'Confused', 5: 'Oriented'},
        'GCSMotor': {1: 'None', 2: 'Extension', 3: 'Flexion', 4: 'Withdraws', 5: 'Localizes', 6: 'Follows commands'},
        'GCSGroup': {1: 'GCS 15', 2: 'GCS 14', 3: 'GCS 11-13', 4: 'GCS 8-10', 5: 'GCS 4-7', 6: 'GCS 3'},
        
        # Physical Exam - Specific Signs
        'SFxPalp': {0: 'No', 1: 'Yes', 2: 'Unclear exam'},
        'SFxPalpDepress': {0: 'No', 1: 'Yes', 2: 'Unclear exam'},
        'FontBulg': {0: 'No', 1: 'Yes', 2: 'Unclear exam'},
        'SFxBas': {0: 'No', 1: 'Yes', 2: 'Unclear exam'},
        'HemaLoc': {1: 'Frontal', 2: 'Occipital', 3: 'Parietal/Temporal'},
        'HemaSize': {1: 'Small (<1cm)', 2: 'Medium (1-3cm)', 3: 'Large (>3cm)'},
        'NeuroD': {0: 'No', 1: 'Yes', 2: 'Unclear exam'},
        
        # Outcomes & Interventions
        'EDDisposition': {
            1: 'Home', 2: 'OR', 3: 'Admit general', 4: 'Short-stay/observation', 
            5: 'ICU', 6: 'Transferred', 7: 'AMA', 8: 'Death in ED'
        },
    }
    
    # Binary variables (0=No, 1=Yes, with possible special codes)
    binary_vars = [
        'Amnesia_verb', 'Seiz', 'ActNorm', 'HA_verb', 'Vomit', 'Dizzy', 
        'Intubated', 'Paralyzed', 'Sedated', 'AMS', 
        'AMSAgitated', 'AMSSleep', 'AMSSlow', 'AMSRepeat', 'AMSOth',
        'SFxBasHem', 'SFxBasOto', 'SFxBasPer', 'SFxBasRet', 'SFxBasRhi',
        'Hema', 
        'Clav', 'ClavFace', 'ClavNeck', 'ClavFro', 'ClavOcc', 'ClavPar', 'ClavTem',
        'OSI', 'OSIExtremity', 'OSICut', 'OSICspine', 'OSIFlank', 'OSIAbdomen', 'OSIPelvis', 'OSIOth',
        'NeuroDMotor', 'NeuroDSensory', 'NeuroDCranial', 'NeuroDReflex', 'NeuroDOth',
        'Drugs',
        'IndAge', 'IndAmnesia', 'IndAMS', 'IndClinSFx', 'IndHA', 'IndHema', 'IndLOC', 'IndMech', 
        'IndNeuroD', 'IndRqstMD', 'IndRqstParent', 'IndRqstTrauma', 'IndSeiz', 'IndVomit', 'IndXraySFx', 'IndOth',
        'CTSed', 'CTSedAgitate', 'CTSedAge', 'CTSedRqst', 'CTSedOth',
        'Observed', 'CTDone', 'EDCT', 'PosCT', 'DeathTBI', 'HospHead', 'HospHeadPosCT', 
        'Intub24Head', 'Neurosurgery', 'PosIntFinal'
    ]
    
    # Add CT Findings (Finding1-Finding23, excluding 15-19 which don't exist)
    binary_vars += [f'Finding{i}' for i in list(range(1, 15)) + list(range(20, 24))]
    
    # ========== STEP 1: Handle Special Codes (90/91/92) ==========
    _log("Step 1: Handling special codes (90/91/92)...")

    total_replaced = 0
    for col in data.columns:
        if data[col].dtype in ("int64", "float64") and col != "PatNum":
            mask = data[col].isin(special_codes)
            count = int(mask.sum())
            if count > 0:
                data.loc[mask, col] = np.nan
                total_replaced += count

    log["special_codes_replaced"] = total_replaced
    _log(f"  Converted {total_replaced:,} special codes (90/91/92) to NA\n")
    
    # ========== STEP 2: Convert Categorical & Binary Variables ==========
    _log("Step 2: Converting categorical variables...")
    
    # 2.1 Binary variables (0 -> No, 1 -> Yes)
    binary_mapping = {0: 'No', 1: 'Yes'}
    for col in binary_vars:
        if col in data.columns:
            data[col] = data[col].map(binary_mapping)
            log['categorical_vars_converted'] += 1
            
    # 2.2 Multi-level categorical variables
    for col, mapping in category_mappings.items():
        if col in data.columns:
            data[col] = data[col].map(mapping)
            log['categorical_vars_converted'] += 1
            
    _log(f"  Converted {log['categorical_vars_converted']} categorical/binary variables\n")

    # ========== STEP 3: Standardize Column Names (Snake Case) ==========
    _log("Step 3: Standardizing column names to snake_case...")
    
    column_rename_map = {
        'PatNum': 'patient_id',
        # Demographics
        'AgeInMonth': 'age_months', 'AgeinYears': 'age_years', 'AgeTwoPlus': 'age_category', 
        'Gender': 'gender', 'Race': 'race', 'Ethnicity': 'ethnicity',
        # Provider
        'EmplType': 'provider_type', 'Certification': 'provider_credential',
        # Injury
        'InjuryMech': 'injury_mech', 'High_impact_InjSev': 'injury_severity_score',
        # History
        'Amnesia_verb': 'amnesia_history', 'LOCSeparate': 'loss_of_consciousness', 'LocLen': 'loc_duration',
        'Seiz': 'seizure_history', 'SeizOccur': 'seizure_timing', 'SeizLen': 'seizure_duration',
        'ActNorm': 'acting_normally', 
        'HA_verb': 'headache_history', 'HASeverity': 'headache_severity', 'HAStart': 'headache_start',
        'Vomit': 'vomiting_history', 'VomitNbr': 'vomit_count', 'VomitStart': 'vomit_start', 
        'VomitLast': 'vomit_last_time', 'Dizzy': 'dizziness',
        # Physical Exam
        'GCSTotal': 'gcs_total', 'GCSEye': 'gcs_eye', 'GCSVerbal': 'gcs_verbal', 
        'GCSMotor': 'gcs_motor', 'GCSGroup': 'gcs_group', 'AMS': 'altered_mental_status',
        'SFxPalp': 'skull_fx_palpable', 'SFxPalpDepress': 'skull_fx_depressed', 
        'FontBulg': 'fontanelle_bulging', 'SFxBas': 'skull_fx_basilar', 
        'Hema': 'hematoma_palpable', 'HemaLoc': 'hematoma_loc', 'HemaSize': 'hematoma_size',
        'Clav': 'clavicle_fracture', 'NeuroD': 'neuro_deficit', 'OSI': 'other_injuries', 
        'Drugs': 'intoxication_signs',
        # Outcomes
        'EDDisposition': 'ed_disposition', 'CTDone': 'ct_done', 'EDCT': 'ed_ct', 
        'PosCT': 'positive_ct', 'DeathTBI': 'death_tbi', 'HospHead': 'hospitalized_tbi', 
        'Intub24Head': 'intubated_tbi', 'Neurosurgery': 'neurosurgery', 
        'PosIntFinal': 'positive_intervention_final'
    }
    
    # Auto-convert unmapped columns using camelCase to snake_case regex
    def clean_col_name(name):
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    final_rename_map = {}
    for col in data.columns:
        if col in column_rename_map:
            final_rename_map[col] = column_rename_map[col]
        else:
            final_rename_map[col] = clean_col_name(col)
            
    data = data.rename(columns=final_rename_map)
    log["columns_renamed"] = len(final_rename_map)
    _log(f"  Renamed {len(final_rename_map)} columns (e.g., PatNum -> patient_id)\n")

    # ========== STEP 4: Derive Variables ==========
    _log("Step 4: Deriving clinically important variables...")
    
    # 4.1 Age group (<2 years vs ≥2 years) - Critical for PECARN stratification
    if "age_years" in data.columns:
        data["age_group"] = data["age_years"].apply(
            lambda x: "<2 years" if x < 2 else ">=2 years" if pd.notna(x) else None
        )
        _log("  Created age_group (<2 years vs >=2 years)")
    
    # 4.2 clinically important TBI (ciTBI) - Primary outcome
    # Definition: Death due to TBI, neurosurgery, intubation >24hrs for TBI, 
    # or hospital admission ≥2 nights for TBI in presence of TBI on CT
    if "positive_intervention_final" in data.columns:
        data["citbi"] = data["positive_intervention_final"]
        _log("  Created citbi from PosIntFinal")
    else:
        # Fallback using individual components
        required_cols = ['death_tbi', 'neurosurgery', 'intubated_tbi', 'hospitalized_tbi', 'positive_ct']
        if all(c in data.columns for c in required_cols):
            data["citbi"] = (
                (data["death_tbi"] == "Yes")
                | (data["neurosurgery"] == "Yes")
                | (data["intubated_tbi"] == "Yes")
                | ((data["hospitalized_tbi"] == "Yes") & (data["positive_ct"] == "Yes"))
            ).map({True: "Yes", False: "No"})
            _log("  Derived citbi from outcome components")
        else:
            log['validation_issues'].append("Cannot derive citbi: missing outcome columns")
    
    _log("")

    # ========== STEP 5: Validation & Quality Checks ==========
    _log("Step 5: Data validation checks...")
    
    # 5.1 GCS consistency check
    if 'gcs_total' in data.columns:
        na_gcs = data['gcs_total'].isna().sum()
        if na_gcs > 0:
            log['validation_issues'].append(f"GCS Total Missing: {na_gcs} rows")
        
        # Check GCS range (3-15)
        invalid_gcs = ((data['gcs_total'] < 3) | (data['gcs_total'] > 15)).sum()
        if invalid_gcs > 0:
            log['validation_issues'].append(f"GCS out of range: {invalid_gcs} rows")
    
    # 5.2 Check high missingness variables
    missing_summary = data.isnull().mean()
    high_missing = missing_summary[missing_summary > 0.5].index.tolist()
    if high_missing:
        _log(f"  Warning: {len(high_missing)} columns have >50% missing data")

    _log("")

    # ========== Cleaning Complete ==========
    log["final_rows"] = len(data)
    log["final_cols"] = len(data.columns)

    _log("=" * 80)
    _log("Data Cleaning Complete")
    _log("=" * 80)
    _log(f"Output: {log['final_rows']:,} rows x {log['final_cols']} columns")
    _log(f"Special codes replaced: {log['special_codes_replaced']:,}")
    _log(f"Categorical variables converted: {log['categorical_vars_converted']}")

    if log["validation_issues"]:
        _log(f"\nValidation Issues: {len(log['validation_issues'])}")
        for issue in log["validation_issues"][:5]:
            _log(f"  - {issue}")

    _log("\n" + "=" * 80 + "\n")
    
    return data


def check_reality(df):
    """
    Performs a 'Reality Check' by comparing dataset statistics against 
    known values from the original Kuppermann et al. (2009) study.
    
    Prints a comparison table for:
    - Sample Size
    - Age Group distribution (<2 vs >=2)
    - Gender distribution
    - CT Scan Rate
    - ciTBI Rate
    """
    print("\n" + "="*60)
    print("REALITY CHECK: Comparison with Kuppermann et al. (2009)")
    print("="*60)
    print(f"{'Metric':<25} | {'Our Data':<15} | {'Paper (Approx)':<15}")
    print("-" * 60)
    
    n_total = len(df)
    print(f"{'Total Patients':<25} | {n_total:<15,} | {'42,412':<15}")
    
    # Age < 2
    if 'age_group' in df.columns:
        n_under2 = (df['age_group'] == '<2 years').sum()
        pct_under2 = n_under2 / n_total * 100
        print(f"{'Age < 2 Years':<25} | {pct_under2:<14.1f}% | {'25%':<15}")
    
    # Gender (Male)
    if 'gender' in df.columns:
        n_male = (df['gender'] == 'Male').sum()
        pct_male = n_male / n_total * 100
        print(f"{'Gender (Male)':<25} | {pct_male:<14.1f}% | {'62%':<15}")
        
    # CT Rate
    if 'ct_done' in df.columns:
        n_ct = (df['ct_done'] == 'Yes').sum()
    elif 'ed_ct' in df.columns:
        n_ct = (df['ed_ct'] == 1).sum()
    else:
        n_ct = 0
    pct_ct = n_ct / n_total * 100
    print(f"{'CT Scan Rate':<25} | {pct_ct:<14.1f}% | {'35%':<15}")
    
    # ciTBI Rate
    if 'citbi' in df.columns:
        n_citbi = (df['citbi'] == 'Yes').sum()
        pct_citbi = n_citbi / n_total * 100
        print(f"{'ciTBI Rate':<25} | {pct_citbi:<14.2f}% | {'0.9%':<15}")
        
    print("-" * 60)
    print("verify: Are our values within reasonable range of the reference?")
    print("="*60 + "\n")


def check_stability(df):
    """
    Performs a 'Stability Check' by creating a perturbed version of the dataset.
    
    Perturbation:
    - Original Strategy: 'Suspected' Loss of Consciousness (LOC) treated as Risk Factor.
    - Perturbed Strategy: 'Suspected' LOC treated as 'No' (Conservative/Benign).
    
    Returns:
    --------
    df_perturbed : pandas.DataFrame
        A copy of the dataframe where 'loss_of_consciousness' has 'Suspected' 
        mapped to 'No'.
    """
    print("\n" + "="*60)
    print("STABILITY CHECK: Generating Perturbed Dataset")
    print("="*60)
    
    df_pert = df.copy()
    
    if 'loss_of_consciousness' in df_pert.columns:
        # Check original counts
        counts_orig = df['loss_of_consciousness'].value_counts()
        print("Original LOC Distribution:")
        print(counts_orig)
        
        # Perturb: Map 'Suspected' to 'No'
        # Note: We are modifying the source column so downstream 'to_binary' 
        # logic (which maps 'No'->0) will treat it as 0.
        mask_suspected = df_pert['loss_of_consciousness'] == 'Suspected'
        n_changed = mask_suspected.sum()
        
        df_pert.loc[mask_suspected, 'loss_of_consciousness'] = 'No'
        
        print(f"\nPerturbation: Remapped {n_changed} 'Suspected' cases to 'No'.")
        print("New LOC Distribution:")
        print(df_pert['loss_of_consciousness'].value_counts())
        
    else:
        print("Warning: 'loss_of_consciousness' column not found. No perturbation applied.")
        
    print("-" * 60 + "\n")
    return df_pert


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / "data" / "TBI PUD 10-08-2013.csv"
    if data_path.exists():
        df_raw = pd.read_csv(data_path)
        df_clean = clean_data(df_raw, verbose=True)
        print(f"Cleaned: {df_clean.shape}")
        print(df_clean[["patient_id", "age_years", "gender", "gcs_total"]].head())
    else:
        print(f"Data not found: {data_path}")
