# inference_preproc.py
import numpy as np
import pandas as pd
from utils.helper import clean_html, mask_post_flare_terms, flag_any

SAFE_NUMERIC_COLS = [
    "patient_age", "has_psoriasis", "on_steroid_med", "on_biologic",
    "itch_present", "dry_skin", "plaques_present", "silvery_scale",
    "elbows_involved", "hyperpigmentation", "smoker", "alcohol_use",
    "family_melanoma"
]

TEXT_FIELDS = ['assesment','complaints','examination','patientSummary','currentmedication']

def preprocess_single(raw_note: dict, tfidf, svd, scaler):
    """
    
    raw_note: dict with raw columns same shape as original raw dataframe row.
    returns: X_final (1d numpy row), debug dict
    """

    df = pd.DataFrame([raw_note]).copy()


    text_cols = ['complaints','pastHistory','assesment','reviewofsystem',
                 'currentmedication','procedure','allergy','examination',
                 'patientSummary','diagnoses']
    for col in text_cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").apply(clean_html)

    # Recreate same flags as training
    df['diagnosis_codes'] = df['diagnoses'].str.findall(r'[A-Z]\d{2}\.\d')
    df['has_psoriasis'] = df['diagnoses'].str.contains('L40', case=False, na=False).astype(int)
    df['psoriasis_type'] = df['diagnoses'].str.extract(r'(Plaque|Arthropathic|Guttate|Pustular)', expand=False)

    flare_terms = ['flare', 'worse', 'itch', 'red', 'scaling', 'burning', 'rash', 'lesion']
    df['complaint_flare_kw'] = df['complaints'].str.lower().apply(lambda t: int(any(k in t for k in flare_terms)))
    df['complaint_no_relief'] = df['complaints'].str.contains("without relief|no improvement", case=False, na=False).astype(int)

    df['flare_in_assessment'] = df['assesment'].apply(lambda t: flag_any(t, ['flare', 'worsen', 'flare-up']))
    df['trigger_mentioned'] = df['assesment'].apply(lambda t: flag_any(t, ['stress','infection','weather','medication']))
    df['steroid_started'] = df['assesment'].apply(lambda t: flag_any(t, ['triamcinolone','steroid','ointment','cream']))

    df['has_medications'] = ~df['currentmedication'].str.contains("no active", case=False, na=False)
    df['on_steroid_med'] = df['currentmedication'].str.contains("steroid|triamcinolone|clobetasol|hydrocortisone", case=False, na=False)
    df['on_biologic'] = df['currentmedication'].str.contains("adalimumab|secukinumab|ixekizumab|etanercept", case=False, na=False)

    df['plaques_present'] = df['examination'].str.contains("plaque", case=False, na=False)
    df['silvery_scale'] = df['examination'].str.contains("silvery|scale", case=False, na=False)
    df['elbows_involved'] = df['examination'].str.contains("elbow", case=False, na=False)
    df['hyperpigmentation'] = df['examination'].str.contains("hyperpigment", case=False, na=False)

    df['itch_present'] = df['reviewofsystem'].str.contains("itch", case=False, na=False)
    df['dry_skin'] = df['reviewofsystem'].str.contains("dry skin", case=False, na=False)
    df['fever_absent'] = df['reviewofsystem'].str.contains("no fever", case=False, na=False)

    df['smoker'] = df['pastHistory'].str.contains("smoker", case=False, na=False)
    df['alcohol_use'] = df['pastHistory'].str.contains("alcohol.*yes", case=False, na=False)
    df['family_melanoma'] = df['pastHistory'].str.contains("melanoma.*yes", case=False, na=False)

    df['patient_age'] = pd.to_numeric(df['patientSummary'].str.extract(r'(\d{1,2})\s*year', expand=False), errors='coerce').fillna(0).astype(float)

    # Mask post-flare terms to avoid leakage
    for col in ['assesment','complaints','examination','patientSummary','currentmedication']:
        df[col + '_clean'] = df[col].fillna('').apply(mask_post_flare_terms)


    text_combined = (df['assesment_clean'].fillna('') + ' ' +
                     df['complaints_clean'].fillna('') + ' ' +
                     df['examination_clean'].fillna('')).astype(str)

    # TF-IDF -> SVD
    X_text_tfidf = tfidf.transform(text_combined)
    X_text_svd = svd.transform(X_text_tfidf)  

    for col in SAFE_NUMERIC_COLS:
        if col not in df.columns:
            df[col] = 0
    X_num = df[SAFE_NUMERIC_COLS].fillna(0).astype(float).values  # shape (1, n_num)
    X_num_scaled = scaler.transform(X_num)

    X_final = np.hstack([X_num_scaled, X_text_svd])

    debug = {
        "SAFE_NUMERIC_COLS": SAFE_NUMERIC_COLS,
        "X_final_shape": X_final.shape,
        "svd_components": X_text_svd.shape
    }

    return X_final, debug
