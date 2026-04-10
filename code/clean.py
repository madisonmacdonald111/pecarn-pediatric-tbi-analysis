"""
Data cleaning functions for the PECARN TBI dataset 

The main entry point is 'clean_data()', which takes the raw DataFrame and returns a fully cleaned DataFrame. 
Optional parameters let you swap out judgment calls to support stability checks.

Usage: 
    Running from the terminal (after cloning the repo): 
        python code/clean.py "data/TBI PUD 10-08-2013.csv"

        This will:
            1. Load the raw data
            2. Run the full cleaning pipeline
            3. Save the cleaned data to data/TBI_PUD_cleaned.csv

    Importing in a script or notebook:
        import pandas as pd
        from clean import clean_data

        raw = pd.read_csv("data/TBI PUD 10-08-2013.csv")
        cleaned = clean_data(raw)
"""

import numpy as np 
import pandas as pd 

# ---------------------------------------------------------------------------
# Private helper: parent -> child column relationships
# ---------------------------------------------------------------------------

def _get_conditional_checks():
    """
    Return the parent -> child column mapping used by validate_data() and repair_data()

    Defined as a function (rather than a module-level variable) so that both functions can call it without duplicating the dictionary. 
    The leading underscore signals this is an internal helper.

    Each key is a "parent" column — if the parent is 0 or NaN, all of its child columns should be NaN. 
        For example, if has_posttraumatic_seizure = 0 then seizure_timing must be NaN (there was no seizure to time).
    
    Returns
    -------
    dict[str, list[str]]
        Mapping of parent column name to list of child column names.
    """
    return {
        "has_loss_of_consciousness_history": ["loss_of_consciousness_duration"],
        "has_posttraumatic_seizure": ["seizure_timing", "posttraumatic_seizure_duration"],
        "has_headache_at_ED": ["headache_severity", "headache_start_time"],
        "has_vomiting_post_injury": [
            "number_of_vomiting_episodes",
            "vomiting_start_time",
            "vomiting_last_time",
        ],
        "has_altered_mental_status": [
            "ams_agitated", "ams_sleepy", "ams_slow_to_respond",
            "ams_repetitive_questions", "ams_other",
        ],
        "has_palpable_skull_fracture": ["skull_fracture_depressed"],
        "has_basilar_skull_fracture_signs": [
            "has_basilar_hemotympanum", "has_basilar_csf_otorrhea",
            "has_basilar_raccoon_eyes", "has_basilar_battles_sign",
            "has_basilar_csf_rhinorrhea",
        ],
        "has_hematomas_or_swellings": [
            "hemotomas_or_swellings_location", "largest_hemotoma_or_swelling_size",
        ],
        "has_trauma_above_clavicles": [
            "has_trauma_face", "has_trauma_neck", "has_trauma_scalp_frontal",
            "has_trauma_scalp_occipital", "has_trauma_scalp_parietal",
            "has_trauma_scalp_temporal",
        ],
        "has_neuro_deficit": [
            "has_neuro_deficit_motor", "has_neuro_deficit_sensory",
            "has_neuro_deficit_cranial_nerve", "has_neuro_deficit_reflexes",
            "has_neuro_deficit_other",
        ],
        "has_other_substantial_injury_non_head": [
            "has_osi_extremity", "has_osi_laceration_requiring_operation",
            "has_osi_cervical_spine", "has_osi_chest_back_flank",
            "has_osi_abdominal", "has_osi_pelvis", "has_osi_other",
        ],
        "ct_head_imaging_ordered": [
            "ct_primary_indication_young_age", "ct_primary_indication_amnesia",
            "ct_primary_indication_altered_mental_status",
            "ct_primary_indication_skull_fracture", "ct_primary_indication_headache",
            "ct_primary_indication_scalp_hematoma",
            "ct_primary_indication_loss_of_consciousness",
            "ct_primary_indication_injury_mechanism",
            "ct_primary_indication_neuro_deficit", "ct_primary_indication_md_request",
            "ct_primary_indication_parental_request",
            "ct_primary_indication_trauma_team_request",
            "ct_primary_indication_seizure", "ct_primary_indication_vomiting",
            "ct_primary_indication_skull_fracture_on_xray",
            "ct_primary_indication_other",
        ],
        "ct_sedation_given": [
            "ct_sedation_reason_agitation", "ct_sedation_reason_young_age",
            "ct_sedation_reason_technician_request", "ct_sedation_reason_other",
        ],
    }

# ---------------------------------------------------------------------------
# Step 1: Rename columns to human-readable names
# ---------------------------------------------------------------------------

def rename_columns(df, col_names_path="documents/dslc_documentation/new_col_names.csv"):
    """
    Replace the original column names with descriptive, human-readable names

    The mapping is stored in a CSV file with two columns: 
        - 'original': the original column name from the raw data 
        - 'new_names': the human-readable replacement names 
    
    Parameters
    ----------
    df: pd.DataFrame 
        Raw TBI DataFrame with original column names
    col_names_path: str 
        Path to the CSV file containing the name mapping
    
    Returns
    -------
    pd.DataFrame 
        DataFrame with renamed columns (in-place copy)
    """
    df = df.copy()
    name_mapping = pd.read_csv(col_names_path)

    # zip() pairs each original name with its new name: [("SubjectID", "patient_number"), ...]
    # dict() turns those pairs into a lookup dictionary: {"SubjectID": "patient_number", ...}
    rename_dict = dict(zip(name_mapping["original"], name_mapping["new_names"]))
    
    # inplace=True means modify df directly instead of returning a new copy
    df.rename(columns=rename_dict, inplace=True)
    return df 

# ---------------------------------------------------------------------------
# Step 2: Replace coded missing values (92) with the appropriate NaN 
# ---------------------------------------------------------------------------

def replace_coded_missing(df):
    """
    Replace numeric codes that represent missing or not-applicable values with proper NaN values

    In this dataset: 
        - 92 = "Not Applicable" for most sub-detail columns (e.g., seizure duration when no seizure occurred)
        These are treated as NaN because the parent variable already captures whether the condition was present. 
        Keeping 92 would imply a meaningful category where none exists. 

    Note: 91 is NOT replaced globally here. 
    For most columns it means "Pre-verbal/Non-verbal" which is a meaningful clinical category, not missing.

    Parameters
    ----------
    df: pd.DataFrame

    Returns
    -------
    pd.DataFrame 
    """
    df = df.copy()

    # [92] is a list so pandas checks every cell in every column for this value
        # inplace=True modifies the DataFrame directly rather than returning a new one
    df.replace([92], np.nan, inplace=True)
    return df 

# ---------------------------------------------------------------------------
# Step 3: Fix GCS out-of-range entry errors
# ---------------------------------------------------------------------------

def fix_gcs_entry_errors(df): 
    """
    Correct data entry errors in GCS component scores 
    where recorded values fall outside the valid range defined in the data documentation

    Valid ranges (from data documentation):
        gcs_eye_score: 1-4 (1=None, 4=Spontaneous)
        gcs_verbal_score: 1-5 (1=None, 5=Oriented)
        gcs_motor_score: 1-6 (1=None, 6=Follows commands)

    Errors found and judgment calls applied: 
        - gcs_eye_score = 5 is corrected to 4 (max valid value).
            The score of 5 likely meant "spontaneous" (max). 
            The total is adjusted down by 1 to remain consistent.
        - gcs_verbal_score = 0 corrected to 1 (min valid value = "None").
            A score of 0 is not defined; "None" is coded as 1 in documentation. 
            The total is adjusted up by 1. 
        - gcs_verbal_score = 6 corrected to 5 (max valid value). 
            The score of 6 likely meant "oriented" (max).
            The total is adjusted down by 1.
    
    In all cases, the gcs_total_score is updated simultaneously to stay internally consistent.
        Note that all adjustments to the gcs_total_score does not change it's associated gcs_category.     
    
    Parameters
    ----------
    df: pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    # Convert to numeric to handle category dtype (same issue as in validate_data)
    eye = pd.to_numeric(df["gcs_eye_score"], errors="coerce")
    verbal = pd.to_numeric(df["gcs_verbal_score"], errors="coerce")
    total = pd.to_numeric(df["gcs_total_score"], errors="coerce")

    # gcs_eye_score = 5 is invalid (max is 4); likely meant 4 
    mask_eye5 = eye == 5
    total = total.where(~mask_eye5, total - 1)
    eye = eye.where(~mask_eye5, 4)

    # gcs_verbal_score = 0 is invalid (min is 1 = "None"); adjust total up 
    mask_verb0 = verbal == 0
    total = total.where(~mask_verb0, total + 1)
    verbal = verbal.where(~mask_verb0, 1)

    # gcs_verbal_score = 6 is invalid (max is 5); likely meant 5 
    mask_verb6 = verbal == 6
    total = total.where(~mask_verb6, total - 1)
    verbal = verbal.where(~mask_verb6, 5)

    df["gcs_eye_score"] = eye.astype(df["gcs_eye_score"].dtype)
    df["gcs_verbal_score"] = verbal.astype(df["gcs_verbal_score"].dtype)
    df["gcs_total_score"] = total.astype(df["gcs_total_score"].dtype)

    return df

# ---------------------------------------------------------------------------
# Step 4: Impute missing GCS component scores
# ---------------------------------------------------------------------------

def impute_gcs_components(df): 
    """
    Impute a single missing GCS component score 
    when the total score and the other two components are all available 

    Judgment call: If exactly one component is missing but the total and other two components are recorded, 
    the missing value can be calculated exactly as: 
        missing_component = total - sum(other two components)
    
    This is not a statistical guess, it is a mathematically determined value. 
    The total score is trusted more than any individual component because it is a higher-level summary recorded by the physician. 

    Only rows with EXACTLY one missing component are imputed. 
    Rows with two or more missing components are left unchanged (no reliable way to recover them)

    Parameters
    ----------
    df: pd.DataFrame

    Returns
    -------
    pd.DataFrame 
    """
    df = df.copy()
    components = ["gcs_eye_score", "gcs_verbal_score", "gcs_motor_score"]

    # Find rows where the total is known, at least 2 components are known, and exactly 1 component is missing 
    can_impute = df[
        df["gcs_total_score"].notna()
        & (
            (df["gcs_eye_score"].notna() & df["gcs_verbal_score"].notna())
            | (df["gcs_eye_score"].notna() & df["gcs_motor_score"].notna())
            | (df["gcs_verbal_score"].notna() & df["gcs_motor_score"].notna())
        )
        & (
            df["gcs_eye_score"].isna()
            | df["gcs_verbal_score"].isna()
            | df["gcs_motor_score"].isna()
        )
    ]

    # For each qualifying row, find which component is missing and fill it 
    for idx in can_impute.index:

        # Pull out just the GCS columns for this one row
        row = df.loc[idx, components + ["gcs_total_score"]]

        # List comprehension: build a list of column names where the value is NaN
        missing_col = [c for c in components if pd.isna(row[c])][0]

        # Same pattern: build a list of column names where the value is NOT NaN
        present_cols = [c for c in components if pd.notna(row[c])]

        # Fill the missing value: total minus the sum of the two known components
        df.loc[idx, missing_col] = row["gcs_total_score"] - row[present_cols].sum()
    
    return df 

# ---------------------------------------------------------------------------
# Step 5: Fix inconsistent GCS total scores
# ---------------------------------------------------------------------------

def fix_gcs_total_score(df): 
    """
    Correct gcs_total_score values that do not equal the sum of the three component scores, 
    when all three components are present. 

    Judgment call: When all three components are recorded and their sum does not match gcs_total_score, we trust the components over the total. 
    This is because the components are individually assessed and recorded ; the total is a derived summary that is more prone to transcription error by physician. 

    Only rows where ALL three components are present are checked. 
    Rows with any missing component are left unchanged. 

    Parameters
    ----------
    df: pd.DataFrame 

    Returns 
    -------
    pd.DataFrame  
    """
    df = df.copy()

    # Compute what the total should be from the three components 
    calculated_total = (
        df["gcs_eye_score"] + df["gcs_verbal_score"] + df["gcs_motor_score"]
    )

    # Only check rows where all three components are present 
    all_present = (
        df["gcs_eye_score"].notna()
        & df["gcs_verbal_score"].notna()
        & df["gcs_motor_score"].notna()
    )

    # Find rows where the recorded total disagrees with the computed total 
    inconsistent = all_present & (df["gcs_total_score"] != calculated_total)

    # Replace with the correct calculated total 
    df.loc[inconsistent, "gcs_total_score"] = calculated_total[inconsistent]

    return df 

# ---------------------------------------------------------------------------
# Step 6: Fix GCS category (derived from total score)
# ---------------------------------------------------------------------------

def fix_gcs_category(df): 
    """
    Re-derive gcs_category from gcs_total_score to ensure consistency.

    Category definitions from the data documentation:
        1 = GCS total score 3–13  (moderate/severe TBI)
        2 = GCS total score 14–15 (minor TBI — primary study population)
    
    Any rows where gcs_category disagrees with the total score are corrected. 
    Rows with a missing total score are left unchanged. 

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    # Assign category 1 for total scores 3-13 
    df.loc[df["gcs_total_score"].notna() & (df["gcs_total_score"] <= 13), "gcs_category"] = 1

    # Assign category 2 for total scores 14–15
    df.loc[df["gcs_total_score"].notna() & (df["gcs_total_score"] >= 14), "gcs_category"] = 2

    return df 

# ---------------------------------------------------------------------------
# Step 7: Validate and impute CT TBI findings 
# ---------------------------------------------------------------------------

def fix_ct_shows_tbi(df):
    """
    Correct ct_shows_tbi values for rows where a depressed skull fracture is recorded
    but ct_shows_tbi is incorrectly set to 0.

    Domain knowledge (from Kuppermann et al., Panel 2):
        "Skull fractures were not regarded as traumatic brain injuries on CT
        unless the fracture was depressed by at least the width of the skull."
    
    Therefore:
        - ct_finding_skull_fracture = 1 WITH skull_fracture_depressed = 0
            - ct_shows_tbi CAN remain 0 (non-depressed fracture is NOT a TBI)
        - ct_finding_skull_fracture = 1 WITH skull_fracture_depressed = 1
            - ct_shows_tbi MUST = 1 (depressed fracture IS a TBI on CT)
    
    Judgment call: 9 rows had skull_fracture_depressed = 1 but ct_shows_tbi = 0. 
    These are corrected to ct_shows_tbi = 1 based on the clinical definition above.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    # CT findings columns to check for any positive finding
    ct_finding_cols = [
        "ct_finding_cerebellar_hemorrhage",
        "ct_finding_cerebral_contusion",
        "ct_finding_cerebral_edema",
        "ct_finding_cerebral_hemorrhage",
        "ct_finding_skull_diastasis",
        "ct_finding_epidural_hematoma",
        "ct_finding_extra_axial_hematoma",
        "ct_finding_intraventricular_hemorrhage",
        "ct_finding_midline_shift",
        "ct_finding_pneumocephalus",
        "ct_finding_skull_fracture",
        "ct_finding_subarachnoid_hemorrhage",
        "ct_finding_subdural_hematoma",
        "ct_finding_traumatic_infarction",
        "ct_extra_finding_diffuse_axonal_injury",
        "ct_extra_finding_herniation",
        "ct_extra_finding_shear_injury",
        "ct_extra_finding_sigmoid_sinus_thrombosis",
    ]

    # Find rows where ct_shows_tbi is 0 or NaN but a finding is positive
        # (df[ct_finding_cols] == 1) creates a True/False table for all finding columns
        # .any(axis=1) then checks each ROW and returns True if ANY column in that row is True
        # axis=1 means "check across columns" (axis=0 would check across rows instead)
    has_positive_finding = (df[ct_finding_cols] == 1).any(axis=1)
    invalid_ct_tbi = df[
        ((df["ct_shows_tbi"] == 0) | df["ct_shows_tbi"].isna()) & has_positive_finding
    ]

    # Among those, only correct rows with a DEPRESSED skull fracture
    depressed_fracture_rows = invalid_ct_tbi[
        invalid_ct_tbi["skull_fracture_depressed"] == 1
    ].index

    df.loc[depressed_fracture_rows, "ct_shows_tbi"] = 1

    return df

# ---------------------------------------------------------------------------
# Step 8: Impute missing patient numbers 
# ---------------------------------------------------------------------------

def impute_patient_numbers(df): 
    """
    Fill in missing patient_number values by identifying gaps 
    in the expected sequential range and assigning the missing numbers. 

    Patient numbers are sequential unique identifies (1 to 43399).
    Two values were found to be missing. 
    Since the sequence is known, the missing values can be determined exactly by finding gaps in range min to max. 

    Judgment call: This is not a statistical imputation,
    it is a deterministic assignment based on known structure of the identifier. 

    Parameters 
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    # dropna() removes NaN values before doing arithmetic
        # can't do min/max on NaN
    existing_nums = df["patient_number"].dropna().astype(int)

    # set(range(...)) creates the complete set of integers we EXPECT to see
        # e.g., if min=1 and max=43399, this is {1, 2, 3, ..., 43399}
        # +1 is needed because range() is exclusive of the end value
    expected_nums = set(range(int(existing_nums.min()), int(existing_nums.max()) + 1))
    
    # Set subtraction: expected minus actual gives us exactly the missing numbers
    missing_nums = sorted(expected_nums - set(existing_nums))

    # Assign missing numbers to the NaN rows in order 
    missing_indices = df[df["patient_number"].isna()].index

    # zip() pairs up two lists element by element: (index_1, missing_num_1), (index_2, missing_num_2), ...
        # This lets us assign each missing number to the correct row in one loop
    for idx, num in zip(missing_indices, missing_nums):
        df.loc[idx, "patient_number"] = num
    
    return df

# ---------------------------------------------------------------------------
# Step 9: Impute missing age in months for newborns 
# ---------------------------------------------------------------------------

def impute_age_months(df):
    """
    Impute missing patient_age_months values as 0 ONLY when
    patient_age_years is also 0 (i.e., the patient is a newborn).

    Judgment call: If age in years is 0 and age in months is missing, 
    the most reasonable value is 0 months (less than 1 month old). 
    For patients with age_years > 0, we do NOT impute because any value within that year range would be a guess. 
    There are 12 possible correct values for each year, and we cannot determine which is correct without additional information. 

    Parameters
    ----------
    df: pd.DataFrame

    Returns 
    -------
    pd.DataFrame 
    """
    df = df.copy()

    # Only impute when years = 0 (newborn) and months is missing 
    newborn_missing = df["patient_age_months"].isna() & (df["patient_age_years"] == 0)
    df.loc[newborn_missing, "patient_age_months"] = 0

    return df 

# ---------------------------------------------------------------------------
# Step 10: Convert columns to appropriate data types 
# ---------------------------------------------------------------------------

def convert_data_types(df): 
    """
    Convert columns from float64 (pandas default when NaNs are present)
    to semantically appropriate types.

    Three type categories are used:
    
    1. 'category' — for columns where numeric values represent labels,
    not quantities. This saves memory, prevents accidental arithmetic,
    and signals to downstream code that these are discrete categories.
    Used for both multi-level categoricals and binary yes/no columns.

    2. 'Int64' (capital I, nullable integer) — for columns that are
       genuinely numeric counts or identifiers but may contain NaN.
       Regular int64 cannot hold NaN; Int64 can.
    
    3. float64 is left as-is where no conversion is appropriate.

    Parameters 
    ----------
    df : pd.DataFrame 

    Returns
    -------
    pd.DataFrame 
    """
    df = df.copy()

    # --- Multi-level categorical columns ---
    # These are columns where each number represents a distinct named category
    categorical_cols = [
        "physician_employment_type",        # 1-5 categories
        "physician_certification",          # 1-4 + 90 (other)
        "injury_mechanism",                 # 1-12 + 90 (other)
        "injury_mechanism_severity",        # 1=Low, 2=Moderate, 3=High
        "loss_of_consciousness_duration",   # 1-4 ordered categories
        "seizure_timing",                   # 1-3 ordered categories
        "posttraumatic_seizure_duration",   # 1-4 ordered categories
        "headache_severity",                # 1-3 ordered categories
        "headache_start_time",              # 1-4 ordered categories
        "number_of_vomiting_episodes",      # 1-3 ordered categories
        "vomiting_start_time",              # 1-4 ordered categories
        "vomiting_last_time",               # 1-4 ordered categories
        "gcs_eye_score",                    # 1-4 ordered
        "gcs_verbal_score",                 # 1-5 ordered
        "gcs_motor_score",                  # 1-6 ordered
        "gcs_category",                     # 1=GCS 3-13, 2=GCS 14-15
        "hemotomas_or_swellings_location",  # 1-3 categories
        "largest_hemotoma_or_swelling_size", # 1-3 ordered
        "patient_gender",                   # 1=Male, 2=Female
        "patient_ethnicity",                # 1=Hispanic, 2=Non-Hispanic
        "patient_race",                     # 1-5 + 90 (other)
        "ed_discharge_status",              # 1-8 + 90 (other)
        "patient_age_under_2yr",            # 1=Under 2yr, 2=2yr and older
    ] 

    # --- Binary yes/no columns --- 
    # These only take 0 (No) or 1 (Yes), with possible special values 
        # like 91 (preverbal/nonverbal) or 2 (suspected/unclear)
    binary_yes_no_cols = [
        "has_event_amnesia",                          # 0/1/91 (91=preverbal)
        "has_loss_of_consciousness_history",          # 0/1/2 (2=suspected)
        "has_posttraumatic_seizure",
        "acting_normal",
        "has_headache_at_ED",                         # 0/1/91 (91=preverbal)
        "has_vomiting_post_injury",
        "has_dizziness_at_ED",
        "eval_after_intubation",
        "eval_after_paralysis",
        "eval_after_sedated",
        "has_altered_mental_status",
        "ams_agitated",
        "ams_sleepy",
        "ams_slow_to_respond",
        "ams_repetitive_questions",
        "ams_other",
        "has_palpable_skull_fracture",                # 0/1/2 (2=unclear exam)
        "skull_fracture_depressed",
        "has_anterior_fontanelle_bulging",
        "has_basilar_skull_fracture_signs",
        "has_basilar_hemotympanum",
        "has_basilar_csf_otorrhea",
        "has_basilar_raccoon_eyes",
        "has_basilar_battles_sign",
        "has_basilar_csf_rhinorrhea",
        "has_hematomas_or_swellings",
        "has_trauma_above_clavicles",
        "has_trauma_face",
        "has_trauma_neck",
        "has_trauma_scalp_frontal",
        "has_trauma_scalp_occipital",
        "has_trauma_scalp_parietal",
        "has_trauma_scalp_temporal",
        "has_neuro_deficit",
        "has_neuro_deficit_motor",
        "has_neuro_deficit_sensory",
        "has_neuro_deficit_cranial_nerve",
        "has_neuro_deficit_reflexes",
        "has_neuro_deficit_other",
        "has_other_substantial_injury_non_head",
        "has_osi_extremity",
        "has_osi_laceration_requiring_operation",
        "has_osi_cervical_spine",
        "has_osi_chest_back_flank",
        "has_osi_abdominal",
        "has_osi_pelvis",
        "has_osi_other",
        "clinical_suspicion_of_intoxication",
        "ct_head_imaging_ordered",
        "ct_primary_indication_young_age",
        "ct_primary_indication_amnesia",
        "ct_primary_indication_altered_mental_status",
        "ct_primary_indication_skull_fracture",
        "ct_primary_indication_headache",
        "ct_primary_indication_scalp_hematoma",
        "ct_primary_indication_loss_of_consciousness",
        "ct_primary_indication_injury_mechanism",
        "ct_primary_indication_neuro_deficit",
        "ct_primary_indication_md_request",
        "ct_primary_indication_parental_request",
        "ct_primary_indication_trauma_team_request",
        "ct_primary_indication_seizure",
        "ct_primary_indication_vomiting",
        "ct_primary_indication_skull_fracture_on_xray",
        "ct_primary_indication_other",
        "ct_sedation_given",
        "ct_sedation_reason_agitation",
        "ct_sedation_reason_young_age",
        "ct_sedation_reason_technician_request",
        "ct_sedation_reason_other",
        "ed_observation_for_ct_decision",
        "head_ct_performed",
        "head_ct_performed_in_ed",
        "ct_shows_tbi",
        "ct_finding_cerebellar_hemorrhage",
        "ct_finding_cerebral_contusion",
        "ct_finding_cerebral_edema",
        "ct_finding_cerebral_hemorrhage",
        "ct_finding_skull_diastasis",
        "ct_finding_epidural_hematoma",
        "ct_finding_extra_axial_hematoma",
        "ct_finding_intraventricular_hemorrhage",
        "ct_finding_midline_shift",
        "ct_finding_pneumocephalus",
        "ct_finding_skull_fracture",
        "ct_finding_subarachnoid_hemorrhage",
        "ct_finding_subdural_hematoma",
        "ct_finding_traumatic_infarction",
        "ct_extra_finding_diffuse_axonal_injury",
        "ct_extra_finding_herniation",
        "ct_extra_finding_shear_injury",
        "ct_extra_finding_sigmoid_sinus_thrombosis",
        "death_due_to_tbi",
        "hospitalized_2plus_nights_head_injury",
        "hospitalized_2plus_nights_with_positive_ct",
        "intubated_24plus_hours_head_trauma",
        "neurosurgery_performed",
        "has_clinically_important_tbi",
    ]

    # --- Integer columns (nullable, may contain NaN) --- 
    # These are genuinely numeric but not continuous floats
    integer_cols = [
        "patient_number",     # unique sequential identifier
        "patient_age_months", # continuous numeric count of months
        "gcs_total_score",    # numeric sum of three GCS components (3-15)
        "patient_age_years",  # continuous numeric count of years
    ]

    # apply category conversion 
        # categorical_cols + binary_yes_no_cols concatenates the two lists into one
        # so we can loop over all columns that need category conversion in a single pass
    for col in categorical_cols + binary_yes_no_cols: 
        df[col] = df[col].astype("category")
    
    # apply nullable integer conversion (Int64 handles NaN; int64 does not)
    for col in integer_cols:
        df[col] = df[col].astype("Int64")
    
    return df 

# ---------------------------------------------------------------------------
# Step 11: Add analysis flag for primary study population 
# ---------------------------------------------------------------------------

def add_primary_analysis_flag(df, gcs_threshold=14):
    """
    Add a binary flag column indicating whether a patient belong to the primary analysis population defined in Kuppermann et al.

    The study enrolled patients with GCS scores of 14-15 as the primary population for the clinical decision rule derivation. 
    Patients with GCS 3-13 were enrolled but analyzed separately.

    Judgment call: Rather than dropping the GCS 3-13 rows (which were intentionally collected), we flag them.
    This preserves all data while allowing downstream analyses to easily filter to the primary population
    using: df[df['in_primary_analysis'] == 1]

    The gcs_threshold parameter allows this judgment call to be adjusted for the stability check 
    (e.g., setting threshold=13 would include all patients regardless of GCS)

    Parameters 
    ----------
    df : pd.DataFrame 
    gcs_threshold : int 
        Minimum GCS score to be included in the primary analysis. 
        Default is 14, matching Kuppermann et al.
    
    Returns 
    -------
    pd.DataFrame 
    """
    df = df.copy() 

    # 1 = in primary analysis population, 0 = separate analysis population
        # df["gcs_total_score"] >= gcs_threshold) produces True/False values
        #  .astype(int) converts True -> 1 and False -> 0 so the column is numeric (1/0) not boolean
    df["in_primary_analysis"] = (df["gcs_total_score"] >= gcs_threshold).astype(int)

    return df 

# ---------------------------------------------------------------------------
# Step 12: Validate data - warn about any remaining inconsistencies
# ---------------------------------------------------------------------------

def validate_data(df):
    """
    Run a suite of validation checks on the cleaned DataFrame and print warnings for any inconsistencies found. 

    This function does NOT modify the data - it only prints warnings. 
    It is intended to be run after all cleaning steps so that any issues a new dataset might have 
    (that the original data did not) are surfaced clearly rather than silently ignored. 

    All checks below were investigated during the original EDA on the PECARN TBI dataset and found 0 issues. 
    A new dataset with the same columns might not be so clean. 

    Checks performed: 
        1. Valid value ranges : all categorical columns only contain documented valid values
        2. Conditional validity : sub-detail columns are NaN when their parent condition is absent (= 0 or NaN)
        3. Age consistency : patient_age_months // 12 == patient_age_years
        4. Age category : patient_age_under_2yr matches patient_age_years
        5. Relational checks : three domain-specific cross-column rules:
            - a. depressed fracture requires palpable fracture
            - b. hospitalized_with_positive_ct requires both hospitalized and ct_shows_tbi
            - c. ct in ED requires ct performed at all
    
    Parameters
    ----------
    df : pd.DataFrame 
        Cleaned DataFrame (after all cleaning steps have been applied).
    
    Returns 
    -------
    None (prints warnings only; does not modify data)
    """

    # Running tally of total issues found across all checks
    issues_found = 0 

    # ---------------------------------------------------------------------------
    # Check 1: Valid value ranges for all categorical columns 
    # ---------------------------------------------------------------------------

    # Each column in the dataset can only take specific documented values.
        # For example, gcs_eye_score can only be 1, 2, 3, or 4.
    # Any value outside this list is a data entry error that was not caught during cleaning and needs to be investigated manually.
    valid_values = {
        "physician_employment_type":        [1, 2, 3, 4, 5],
        "physician_certification":          [1, 2, 3, 4, 90],
        "injury_mechanism":                 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 90],
        "injury_mechanism_severity":        [1, 2, 3],
        "has_event_amnesia":                [0, 1, 91],
        "has_loss_of_consciousness_history": [0, 1, 2],
        "loss_of_consciousness_duration":   [1, 2, 3, 4],
        "has_posttraumatic_seizure":        [0, 1],
        "seizure_timing":                   [1, 2, 3],
        "posttraumatic_seizure_duration":   [1, 2, 3, 4],
        "acting_normal":                    [0, 1],
        "has_headache_at_ED":               [0, 1, 91],
        "headache_severity":                [1, 2, 3],
        "headache_start_time":              [1, 2, 3, 4],
        "has_vomiting_post_injury":         [0, 1],
        "number_of_vomiting_episodes":      [1, 2, 3],
        "vomiting_start_time":              [1, 2, 3, 4],
        "vomiting_last_time":               [1, 2, 3],
        "has_dizziness_at_ED":              [0, 1],
        "eval_after_intubation":            [0, 1],
        "eval_after_paralysis":             [0, 1],
        "eval_after_sedated":               [0, 1],
        "gcs_eye_score":                    [1, 2, 3, 4],
        "gcs_verbal_score":                 [1, 2, 3, 4, 5],
        "gcs_motor_score":                  [1, 2, 3, 4, 5, 6],
        "gcs_category":                     [1, 2],
        "has_altered_mental_status":        [0, 1],
        "ams_agitated":                     [0, 1],
        "ams_sleepy":                       [0, 1],
        "ams_slow_to_respond":              [0, 1],
        "ams_repetitive_questions":         [0, 1],
        "ams_other":                        [0, 1],
        "has_palpable_skull_fracture":      [0, 1, 2],
        "skull_fracture_depressed":         [0, 1],
        "has_anterior_fontanelle_bulging":  [0, 1],
        "has_basilar_skull_fracture_signs": [0, 1],
        "has_basilar_hemotympanum":         [0, 1],
        "has_basilar_csf_otorrhea":         [0, 1],
        "has_basilar_raccoon_eyes":         [0, 1],
        "has_basilar_battles_sign":         [0, 1],
        "has_basilar_csf_rhinorrhea":       [0, 1],
        "has_hematomas_or_swellings":       [0, 1],
        "hemotomas_or_swellings_location":  [1, 2, 3],
        "largest_hemotoma_or_swelling_size": [1, 2, 3],
        "has_trauma_above_clavicles":       [0, 1],
        "has_trauma_face":                  [0, 1],
        "has_trauma_neck":                  [0, 1],
        "has_trauma_scalp_frontal":         [0, 1],
        "has_trauma_scalp_occipital":       [0, 1],
        "has_trauma_scalp_parietal":        [0, 1],
        "has_trauma_scalp_temporal":        [0, 1],
        "has_neuro_deficit":                [0, 1],
        "has_neuro_deficit_motor":          [0, 1],
        "has_neuro_deficit_sensory":        [0, 1],
        "has_neuro_deficit_cranial_nerve":  [0, 1],
        "has_neuro_deficit_reflexes":       [0, 1],
        "has_neuro_deficit_other":          [0, 1],
        "has_other_substantial_injury_non_head": [0, 1],
        "has_osi_extremity":                [0, 1],
        "has_osi_laceration_requiring_operation": [0, 1],
        "has_osi_cervical_spine":           [0, 1],
        "has_osi_chest_back_flank":         [0, 1],
        "has_osi_abdominal":                [0, 1],
        "has_osi_pelvis":                   [0, 1],
        "has_osi_other":                    [0, 1],
        "clinical_suspicion_of_intoxication": [0, 1],
        "ct_head_imaging_ordered":          [0, 1],
        "ct_primary_indication_young_age":  [0, 1],
        "ct_primary_indication_amnesia":    [0, 1],
        "ct_primary_indication_altered_mental_status": [0, 1],
        "ct_primary_indication_skull_fracture": [0, 1],
        "ct_primary_indication_headache":   [0, 1],
        "ct_primary_indication_scalp_hematoma": [0, 1],
        "ct_primary_indication_loss_of_consciousness": [0, 1],
        "ct_primary_indication_injury_mechanism": [0, 1],
        "ct_primary_indication_neuro_deficit": [0, 1],
        "ct_primary_indication_md_request": [0, 1],
        "ct_primary_indication_parental_request": [0, 1],
        "ct_primary_indication_trauma_team_request": [0, 1],
        "ct_primary_indication_seizure":    [0, 1],
        "ct_primary_indication_vomiting":   [0, 1],
        "ct_primary_indication_skull_fracture_on_xray": [0, 1],
        "ct_primary_indication_other":      [0, 1],
        "ct_sedation_given":                [0, 1],
        "ct_sedation_reason_agitation":     [0, 1],
        "ct_sedation_reason_young_age":     [0, 1],
        "ct_sedation_reason_technician_request": [0, 1],
        "ct_sedation_reason_other":         [0, 1],
        "patient_age_under_2yr":            [1, 2],
        "patient_gender":                   [1, 2],
        "patient_ethnicity":                [1, 2],
        "patient_race":                     [1, 2, 3, 4, 5, 90],
        "ed_observation_for_ct_decision":   [0, 1],
        "ed_discharge_status":              [1, 2, 3, 4, 5, 6, 7, 8, 90],
        "head_ct_performed":                [0, 1],
        "head_ct_performed_in_ed":          [0, 1],
        "ct_shows_tbi":                     [0, 1],
        "ct_finding_cerebellar_hemorrhage": [0, 1],
        "ct_finding_cerebral_contusion":    [0, 1],
        "ct_finding_cerebral_edema":        [0, 1],
        "ct_finding_cerebral_hemorrhage":   [0, 1],
        "ct_finding_skull_diastasis":       [0, 1],
        "ct_finding_epidural_hematoma":     [0, 1],
        "ct_finding_extra_axial_hematoma":  [0, 1],
        "ct_finding_intraventricular_hemorrhage": [0, 1],
        "ct_finding_midline_shift":         [0, 1],
        "ct_finding_pneumocephalus":        [0, 1],
        "ct_finding_skull_fracture":        [0, 1],
        "ct_finding_subarachnoid_hemorrhage": [0, 1],
        "ct_finding_subdural_hematoma":     [0, 1],
        "ct_finding_traumatic_infarction":  [0, 1],
        "ct_extra_finding_diffuse_axonal_injury": [0, 1],
        "ct_extra_finding_herniation":      [0, 1],
        "ct_extra_finding_shear_injury":    [0, 1],
        "ct_extra_finding_sigmoid_sinus_thrombosis": [0, 1],
        "death_due_to_tbi":                 [0, 1],
        "hospitalized_2plus_nights_head_injury": [0, 1],
        "hospitalized_2plus_nights_with_positive_ct": [0, 1],
        "intubated_24plus_hours_head_trauma": [0, 1],
        "neurosurgery_performed":           [0, 1],
        "has_clinically_important_tbi":     [0, 1],
    }

    print("=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)

    print("\n--- Check 1: Valid value ranges ---")
    for col, valid in valid_values.items():
        # Convert to numeric first because after type conversion some columns are stored as 'category' dtype
            # which doesn't support direct comparison 
        # errors="coerce" means: if a value cannot be converted to a number, replace it with NaN instead of raising an error = safe fallback
        col_numeric = pd.to_numeric(df[col], errors="coerce")

        # Flag any non-NaN value that isn't in the documented valid list
            # .isin(valid) returns True if the value IS in the valid list
            # the ~ operator is a logical NOT, flipping True -> False and False -> True
        invalid_mask = col_numeric.notna() & ~col_numeric.isin(valid)
        count = invalid_mask.sum()
        if count > 0:
            # Show exactly which invalid values were found to help with debugging
            bad_vals = sorted(col_numeric[invalid_mask].unique())
            print(f"  WARNING: {col} has {count} out-of-range values: {bad_vals}")
            issues_found += count

    if issues_found == 0:
        print("\tOK - all columns within valid ranges")
    
    # ---------------------------------------------------------------------------
    # Check 2: Conditional validity (parent variable = 0 or NaN -> child variable = NaN)
    # ---------------------------------------------------------------------------

    print("\n--- Check 2: Conditional validity (parent/child relationships) ---")
    
    # Call the shared helper so this dictionary only exists in one place
    conditional_checks = _get_conditional_checks()
    cond_issues = 0 
    for main_col, related_cols in conditional_checks.items():
        # Parent is "absent" if it equals 0 (No) or is NaN (missing)
            # In either case, the child/detail columns should all be NaN
            # errors="coerce" safely handles category dtype columns by converting to numeric
        main_absent = (
            pd.to_numeric(df[main_col], errors="coerce") == 0
        ) | df[main_col].isna()

        # Build a boolean mask: True if ANY child column has a real positive value
            # (not NaN and not 0) — meaning detail was recorded when it shouldn't be
            # pd.Series(False, index=df.index) creates a column of all False values as a starting point
                # We then OR each child column into it — if any child is True, the row becomes True
        related_has_value = pd.Series(False, index=df.index)
        for col in related_cols:
            col_numeric = pd.to_numeric(df[col], errors="coerce")
            related_has_value = related_has_value | (
                col_numeric.notna() & (col_numeric != 0)
            )
        
        # A row is invalid if the parent is absent BUT a child has a value
        count = (main_absent & related_has_value).sum()
        if count > 0:
            print(f"  WARNING: {main_col} is absent but {count} rows have child values")
            cond_issues += count
    
    if cond_issues == 0: 
        print("  OK - all parent/child relationships are consistent")
    # Add conditional issues to the running total
    issues_found += cond_issues

    # ---------------------------------------------------------------------------
    # Check 3: Age consistency (months // 12 should equal years)
    # ---------------------------------------------------------------------------

    print("\n--- Check 3: Age consistency (months vs years) ---")

    # Only check rows where both age fields are present
    has_both = df["patient_age_months"].notna() & df["patient_age_years"].notna()
    
    # errors="coerce" used throughout for safe conversion of category/mixed dtype columns
    months_numeric = pd.to_numeric(df.loc[has_both, "patient_age_months"], errors="coerce")
    years_numeric = pd.to_numeric(df.loc[has_both, "patient_age_years"], errors="coerce")
    
    # Floor division of months by 12 should equal the recorded age in years
        # e.g., 26 months // 12 = 2, so patient_age_years should be 2
    inconsistent_age = (months_numeric // 12).astype(int) != years_numeric.astype(int)
    count = inconsistent_age.sum()
    if count > 0:
        print(f"  WARNING: {count} rows where patient_age_months // 12 != patient_age_years")
        issues_found += count
    else:
        print("  OK - age in months and years are consistent")

    # ---------------------------------------------------------------------------
    # Check 4: Age category matches age in years
    # ---------------------------------------------------------------------------

    print("\n--- Check 4: Age category vs age in years ---")

    # patient_age_under_2yr should be 1 if age < 2 years, and 2 if age >= 2 years
        # This is important because the study used this cutoff to define two separate prediction rule populations (Kuppermann et al.)
    has_age = df["patient_age_years"].notna() & df["patient_age_under_2yr"].notna()
    years_num = pd.to_numeric(df.loc[has_age, "patient_age_years"], errors="coerce")
    cat_num = pd.to_numeric(df.loc[has_age, "patient_age_under_2yr"], errors="coerce")
    
    # Flag rows where the category doesn't match the age in years
    invalid_cat = (
        ((years_num < 2) & (cat_num != 1)) |    # age < 2 but not flagged as under 2
        ((years_num >= 2) & (cat_num != 2))     # age >= 2 but flagged as under 2
    )
    count = invalid_cat.sum()
    if count > 0:
        print(f"  WARNING: {count} rows where patient_age_under_2yr is inconsistent with patient_age_years")
        issues_found += count
    else:
        print("  OK - age category is consistent with age in years")

    # ---------------------------------------------------------------------------
    # Check 5a: Depressed fracture requires palpable fracture
    # ---------------------------------------------------------------------------

    print("\n--- Check 5: Domain-specific relational checks ---")

    # A depressed skull fracture physically requires a palpable skull fracture
        # You cannot have a depression without being able to feel the fracture
    dep_frac = pd.to_numeric(df["skull_fracture_depressed"], errors="coerce")
    palp_frac = pd.to_numeric(df["has_palpable_skull_fracture"], errors="coerce")
    count = ((dep_frac == 1) & (palp_frac != 1)).sum()
    if count > 0:
        print(f"  WARNING: {count} rows with depressed fracture but no palpable fracture")
        issues_found += count
    else:
        print("  OK - all depressed fractures have a corresponding palpable fracture")

    # ---------------------------------------------------------------------------
    # Check 5b: hospitalized_with_positive_ct requires both conditions
    # ---------------------------------------------------------------------------

    # hospitalized_2plus_nights_with_positive_ct is literally the intersection of hospitalized_2plus_nights_head_injury AND ct_shows_tbi — both must be 1
    hosp_ct = pd.to_numeric(df["hospitalized_2plus_nights_with_positive_ct"], errors="coerce")
    hosp = pd.to_numeric(df["hospitalized_2plus_nights_head_injury"], errors="coerce")
    ct_tbi = pd.to_numeric(df["ct_shows_tbi"], errors="coerce")
    count = ((hosp_ct == 1) & ((hosp != 1) | (ct_tbi != 1))).sum()
    if count > 0:
        print(f"  WARNING: {count} rows with hospitalized_with_positive_ct = 1 but missing hospitalization or positive CT")
        issues_found += count
    else:
        print("  OK - hospitalized_with_positive_ct is consistent with its components")
    
    # ---------------------------------------------------------------------------
    # Check 5c: CT in ED requires CT performed at all
    # ---------------------------------------------------------------------------

    # You cannot have a CT performed specifically in the ED without having had a CT performed at all
    ct_ed = pd.to_numeric(df["head_ct_performed_in_ed"], errors="coerce")
    ct_any = pd.to_numeric(df["head_ct_performed"], errors="coerce")
    count = ((ct_ed == 1) & (ct_any != 1)).sum()
    if count > 0:
        print(f"  WARNING: {count} rows with head_ct_performed_in_ed = 1 but head_ct_performed != 1")
        issues_found += count
    else:
        print("  OK - CT in ED is consistent with CT performed overall")

    # ---------------------------------------------------------------------------
    # Summary 
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 60)
    if issues_found == 0:
        print("VALIDATION PASSED - no issues found")
    else:
        print(f"VALIDATION COMPLETE - {issues_found} total issues found (see warnings above)")
    print("=" * 60)

# ---------------------------------------------------------------------------
# OPTIONAL STEP: Repair data - fix issues flagged by validate_data()
# ---------------------------------------------------------------------------

def repair_data(df):
    """
    Attempt to automatically fix issues that validate_data() flagged.

    This function is OPTIONAL and intended to be called manually after running validate_data() on a new dataset that did not fully pass.
    It should NOT be needed for the original PECARN TBI dataset, which passes all validation checks after clean_data() is applied.
    
    What each repair does:
        1. Out-of-range GCS values : re-runs fix_gcs_entry_errors() - same fix as in the main pipeline
        2. Conditional validity violations : sets child column to NaN when its parent column is 0 or NaN
        3. Age inconsistency : keeps patient_age_years as the authoritative value and sets patient_age_months to NaN 
            (judgment call: years is more reliably recorded than months)
        4. Age category mismatch : re-derives patient_age_under_2yr from patient_age_years 
        5. Relational violations : applies the same domain logic as fix_ct_shows_tbi() for the three cross-column rules 
    
    After calling repair_data(), re-run validate_data() to confirm all issues were resolved.
    Any remaining warnings indicate problems that require manual investigation.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame that failed one or more validate_data() checks.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with auto-repairable issues fixed.
    """
    df = df.copy()
    repairs_made = 0

    print("=" * 60)
    print("REPAIR REPORT")
    print("=" * 60)

    # ---------------------------------------------------------------------------
    # Repair 1: Out-of-range GCS values 
    # ---------------------------------------------------------------------------

    # Count out-of-range values before fixing so we can report what changed 
    print("\n--- Repair 1: Out-of-range GCS values ---")
    eye = pd.to_numeric(df["gcs_eye_score"],    errors="coerce")
    verbal = pd.to_numeric(df["gcs_verbal_score"], errors="coerce")
    n_eye5  = (eye == 5).sum()
    n_verb0 = (verbal == 0).sum()
    n_verb6 = (verbal == 6).sum()
    fixed = n_eye5 + n_verb0 + n_verb6

    # Call the existing fix function rather than duplicating its logic here
    df = fix_gcs_entry_errors(df)

    if fixed > 0: 
        print(f"\tFixed {n_eye5} gcs_eye_score = 5 -> 4")
        print(f"\tFixed {n_verb0} gcs_verbal_score = 0 -> 1")
        print(f"\tFixed {n_verb6} gcs_verbal_score = 6 -> 5")
        repairs_made += fixed 
    else: 
        print("\tNo out-of-range GCS values found")
    
    # ---------------------------------------------------------------------------
    # Repair 2: Conditional validity violations
    # ---------------------------------------------------------------------------

    # If a parent column is 0 or NaN, all child columns must be NaN 
    # Any child column that has a value when it shouldn't is set to NaN here
    print("\n--- Repair 2: Conditional validity violations ---")

    # Use the same mapping as used in validate_data()
        # Call the shared helper so this dictionary only exists in one place
    conditional_checks = _get_conditional_checks()
    cond_fixed = 0 
    for main_col, related_cols in conditional_checks.items():
        # Same logic as validate_data: parent is absent if 0 or NaN
        main_absent = (
            pd.to_numeric(df[main_col], errors="coerce") == 0
        ) | df[main_col].isna()
    
        for col in related_cols: 
            col_numeric = pd.to_numeric(df[col], errors="coerce")
            # Rows where parent is absent but this child has a value - set it to NaN
            bad_rows = main_absent & col_numeric.notna() & (col_numeric != 0)
            count = bad_rows.sum()
            if count > 0:
                df.loc[bad_rows, col] = np.nan
                print(f"  Set {count} values to NaN in {col} (parent {main_col} is absent)")
                cond_fixed += count

    if cond_fixed == 0:
        print("  No conditional validity violations found")
    repairs_made += cond_fixed

    # ---------------------------------------------------------------------------  
    # Repair 3: Age inconsistency (months // 12 not = years)
    # ---------------------------------------------------------------------------  

    # Judgment call: trust patient_age_years over patient_age_months 
        # Rows where they disagree get patient_age_months set to NaN 
        # We cannot determine the correct month value without additional information 
    print("\n--- Repair 3: Age inconsistency (months vs years) ---")

    has_both = df["patient_age_months"].notna() & df["patient_age_years"].notna()

    # Find rows where floor division by months by 12 doesn't match years 
    inconsistent = has_both & (
        (pd.to_numeric(df["patient_age_months"], errors="coerce") // 12).astype("Int64")
        != pd.to_numeric(df["patient_age_years"], errors="coerce").astype("Int64")
    )
    count = inconsistent.sum()
    if count > 0:
        # Set months to NaN for inconsistent rows — years is kept as-is
        df.loc[inconsistent, "patient_age_months"] = pd.NA
        print(f"  Set patient_age_months to NaN for {count} rows where months // 12 != years")
        print("  (patient_age_years was kept as the authoritative value)")
        repairs_made += count
    else:
        print("  No age inconsistencies found")
    
    # ---------------------------------------------------------------------------
    # Repair 4: Age category mismatch 
    # ---------------------------------------------------------------------------

    # Re-derive patient_age_under_2yr directly from patient_age_years.
    # 1 = under 2 years old, 2 = 2 years old or older (Kuppermann et al. cutoff)
    print("\n--- Repair 4: Age category mismatch ---")

    has_age = df["patient_age_years"].notna() & df["patient_age_under_2yr"].notna()

    # Flag rows where the recorded category doesn't match what age_years implies
    invalid_cat = has_age & (
        ((pd.to_numeric(df["patient_age_years"], errors="coerce") < 2) &
         (pd.to_numeric(df["patient_age_under_2yr"], errors="coerce") != 1)) |
        ((pd.to_numeric(df["patient_age_years"], errors="coerce") >= 2) &
         (pd.to_numeric(df["patient_age_under_2yr"], errors="coerce") != 2))
    )
    count = invalid_cat.sum()
    if count > 0:
        # Re-derive the correct category from age in years
        correct_cat = np.where(
            pd.to_numeric(df["patient_age_years"], errors="coerce") < 2, 1, 2
        )
        # Preserve dtype — patient_age_under_2yr is category dtype after clean_data()
        df.loc[invalid_cat, "patient_age_under_2yr"] = correct_cat[invalid_cat].astype(
            df["patient_age_under_2yr"].dtype
        )
        print(f"  Re-derived patient_age_under_2yr for {count} rows from patient_age_years")
        repairs_made += count
    else:
        print("  No age category mismatches found")

    # ---------------------------------------------------------------------------
    # Repair 5: Domain-specific relational violations 
    # ---------------------------------------------------------------------------

    print("\n--- Repair 5: Domain-specific relational checks ---")

    # Repair 5a: depressed fracture requires palpable fracture
        # if skull_fracture_depressed = 1 but has_palpable_skull_fracture != 1
        # set has_palpable_skull_fracture = 1 (the depression is itself evidence of a palpable fracture)
    dep_frac = pd.to_numeric(df["skull_fracture_depressed"], errors="coerce")
    palp_frac = pd.to_numeric(df["has_palpable_skull_fracture"], errors="coerce")
    bad_rows = (dep_frac == 1) & (palp_frac != 1)
    count = bad_rows.sum()
    if count > 0:
        df.loc[bad_rows, "has_palpable_skull_fracture"] = pd.Categorical(
            [1] * count, categories=df["has_palpable_skull_fracture"].cat.categories
        )
        print(f"  Set has_palpable_skull_fracture=1 for {count} rows with depressed fracture")
        repairs_made += count
    else:
        print("  OK - depressed fracture / palpable fracture relationship is consistent")

    # Repair 5b: hospitalized_with_positive_ct requires both parent conditions
        # if the compound variable is 1 but either component is not 1, set the compound to NaN
        # then we cannot determine which component is wrong, so we remove the derived value
    hosp_ct = pd.to_numeric(df["hospitalized_2plus_nights_with_positive_ct"], errors="coerce")
    hosp = pd.to_numeric(df["hospitalized_2plus_nights_head_injury"],       errors="coerce")
    ct_tbi = pd.to_numeric(df["ct_shows_tbi"],                                errors="coerce")
    bad_rows = (hosp_ct == 1) & ((hosp != 1) | (ct_tbi != 1))
    count = bad_rows.sum()
    if count > 0:
        df.loc[bad_rows, "hospitalized_2plus_nights_with_positive_ct"] = np.nan
        print(f"  Set hospitalized_2plus_nights_with_positive_ct to NaN for {count} rows")
        print("  (compound variable cannot be 1 when a required component is not 1)")
        repairs_made += count
    else:
        print("  OK - hospitalized_with_positive_ct is consistent with its components")
    
    # Repair 5c: CT in ED requires CT performed at all 
        # if head_ct_performed_in_ed = 1 but head_ct_performed != 1,
    # set head_ct_performed = 1 — the ED record is evidence that a CT was performed
    ct_ed  = pd.to_numeric(df["head_ct_performed_in_ed"], errors="coerce")
    ct_any = pd.to_numeric(df["head_ct_performed"], errors="coerce")
    bad_rows = (ct_ed == 1) & (ct_any != 1)
    count = bad_rows.sum()
    if count > 0:
        df.loc[bad_rows, "head_ct_performed"] = pd.Categorical(
            [1] * count, categories=df["head_ct_performed"].cat.categories
        )
        print(f"  Set head_ct_performed=1 for {count} rows where CT was performed in ED")
        repairs_made += count
    else:
        print("  OK - CT in ED is consistent with CT performed overall")
    
    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 60)
    if repairs_made == 0:
        print("REPAIR COMPLETE - no issues found to fix")
    else:
        print(f"REPAIR COMPLETE - {repairs_made} total fixes applied")
        print("Re-run validate_data(df) to confirm all issues are resolved.")
        print("Any remaining warnings require manual investigation.")
    print("=" * 60)

    return df

# ---------------------------------------------------------------------------
# Main Function: clean_data 
# ---------------------------------------------------------------------------

def clean_data(
    df, 
    col_names_path = "documents/dslc_documentation/new_col_names.csv", 
    gcs_threshold=14, 
    impute_gcs=True, 
    fix_ct_tbi=True,
    validate=True,
): 
    """
    Apply all cleaning steps to the raw PECARN TBI DataFrame in sequence. 

    This function is the single entry point for data cleaning. 
    It chains together all individual cleaning steps defined above. 
    Optional parameters allow judgment calls to be varied for stability checks. 

    Cleaning steps applied in order: 
        1. Rename columns to human-readable names 
        2. Replace code missing values (92: NaN)
        3. Fix out-of-range GCS component from total + other two 
        4. Impute single missing GCS component from total + other two
        5. Fix GCS total scores that don't match component sums
        6. Re-derive GCS category from corrected total score
        7. Correct ct_shows_tbi for depressed skull fracture rows
        8. Impute 2 missing patient_number values (sequential gap fill)
        9. Impute missing age_months = 0 for newborns (age_years = 0)
        10. Convert columns to appropriate data types (category / Int64)
        11. Add in_primary_analysis flag (GCS >= threshold)
        12. Validation checks for new data 

    Parameters 
    ----------
    df : pd.DataFrame 
        Raw TBI DataFrame, loaded directly from the CSV with pd.read_csv.
    
    col_names_path : str 
        Path to the column name mapping CSV.
        Default: "code/new_col_names.csv"
    
    gcs_threshold : int 
        Minimum GCS total score for a patient to be flagged as part of the primary analysis population. 
        Default is 14. 
        Stability check: try gcs_threshold=13 to include all patients.
    
    impute_gcs : bool
        Whether to impute missing GCS components from the total score.
        Default is True.
        Stability check: set to False to leave missing components as NaN.
    
    fix_ct_tbi : bool
        Whether to correct ct_shows_tbi = 0 for depressed skull fracture rows based on domain knowledge from Kuppermann et al.
        Default is True.
        Stability check: set to False to leave those 9 rows uncorrected.
    
    Returns 
    -------
    pd.DataFrame 
        Fully cleaned DataFrame ready for EDA and modeling. 
    """

    # Step 1 : Rename columns
    df = rename_columns(df, col_names_path=col_names_path)

    # Step 2: Replace 92 coded missing values with NaN
    df = replace_coded_missing(df)

    # Step 3: Skipped - fix_gcs_entry_errors() moved to Step 10b (must run after type conversion)
        # moved to later (step 10b) -> (otherwise this step will fail)

    # Step 4: Impute single missing GCS components (optional)
    if impute_gcs:
        df = impute_gcs_components(df)
    
    # Step 5: Fix GCS total scores that disagree with component sums
    df = fix_gcs_total_score(df)

    # Step 6: Re-derive GCS category from corrected total score
    df = fix_gcs_category(df)

    # Step 7: Correct ct_shows_tbi for depressed skull fractures (optional)
    if fix_ct_tbi:
        df = fix_ct_shows_tbi(df)

    # Step 8: Impute 2 missing patient_number values
    df = impute_patient_numbers(df)

    # Step 9: Impute age in months = 0 for newborns only
    df = impute_age_months(df)

    # Step 10: Convert columns to appropriate types
    df = convert_data_types(df)

    # Step 10b: Fix GCS out-of-range AFTER type conversion
    df = fix_gcs_entry_errors(df)   

    # Step 11: Add primary analysis flag
    df = add_primary_analysis_flag(df, gcs_threshold=gcs_threshold)

    # Step 12: Run validation checks and print warnings for any issues found 
        # Set validate=False to skip this step (e.g., for faster repeated runs)
    if validate: 
        validate_data(df)

    return df 

# ---------------------------------------------------------------------------
# For running this script from the command line 
# ---------------------------------------------------------------------------

# This block only runs when the script is called directly from the terminal:
    # $ python code/clean.py "data/TBI PUD 10-08-2013.csv"
# It does NOT run when clean.py is imported as a module in another script 
    # e.g., "from clean import clean_data" is what the if __name__ block essentially does 

if __name__ == "__main__": 
    import sys 

    # sys.argv is the list of words typed after "python" in the terminal
    # sys.argv[0] is always the script name itself ("clean.py")
    # sys.argv[1] is the first argument — we expect the path to the raw CSV

    if len(sys.argv) < 2: 
        print("Usage: python code/clean.py <path_to_raw_csv>")
        sys.exit(1)  # exit with error code 1 to signal something went wrong
    
    # allow for changing of the optional arguments 
    raw_path = sys.argv[1]
    gcs_threshold = int(sys.argv[2])   if len(sys.argv) > 2 else 14
    impute_gcs    = sys.argv[3].lower() != 'false' if len(sys.argv) > 3 else True
    fix_ct_tbi    = sys.argv[4].lower() != 'false' if len(sys.argv) > 4 else True

    print(f"Loading raw data from: {raw_path}")
    raw = pd.read_csv(raw_path)
    print(f"Raw data shape: {raw.shape}")

    print("\nRunning cleaning pipeline...")
    cleaned = clean_data(raw, col_names_path="documents/dslc_documentation/new_col_names.csv",
                     gcs_threshold=gcs_threshold, impute_gcs=impute_gcs, fix_ct_tbi=fix_ct_tbi)
    
    print(f"\nCleaned data shape: {cleaned.shape}")

    # save clean data as parquet (retain proper dtypes)
    # allow for different output path if other judgment calls are test (different function arguments)
    output_path = sys.argv[5] if len(sys.argv) > 5 else (
        "data/TBI_PUD_cleaned.csv" if (gcs_threshold==14 and impute_gcs and fix_ct_tbi)
        else "data/TBI_PUD_cleaned_perturbed.csv"
    )
    cleaned.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")
    print("\nDone.")