import sys
import mlflow
import asyncio
import logging
import joblib, os
import numpy as np
import pandas as pd
import nest_asyncio
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from db.db import get_db
from typing_extensions import Tuple
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from pipeline.extract_data import fetch_final_data, get_patient_ids
from utils.helper import clean_html, flag_any, mask_post_flare_terms
import warnings
warnings.filterwarnings("ignore")



log_directory = "logs"
data_directory = "data"
os.makedirs(data_directory, exist_ok=True)
os.makedirs(log_directory, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_directory, "preprocessing.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class FeatureExtraction:
    def __init__(self):
        """Preprocessing and feature extraction pipeline"""
    def extract_features(self):
        """Extract features from final data"""
        try:
            nest_asyncio.apply()        
            patients_ids = asyncio.run(get_patient_ids(next(get_db())))
            logger.info(f"Total patients IDs: {len(patients_ids)}")
            event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(event_loop)
            df = event_loop.run_until_complete(fetch_final_data(next(get_db()), patients_ids))
            logger.info(f"Total rows in extracted data: {df.shape[0]}")
            
            if df is not None:
                df.to_csv(os.path.join(data_directory, 'final_data.csv'), index=False)
                logger.info(f"Final data saved to final_data.csv")
                
            event_loop.close()
            
        except Exception as e:
            logger.error(f"An error occurred: {e}")
        
        try:
            """Feature Engineering and preprocessing of the data and preparing for the model training"""
            logger.info(f"Preprocessing the data....")
            df.info()
            df = df.drop(columns=['biopsyNotes', 'mohsNotes', 'referringPhysician', 'Physician'], axis=1)
            df.columns = df.columns.str.strip()
            df = df.loc[:, ~df.columns.duplicated(keep='first')]
            df.drop(columns=["Rendering Provider", "Referring Provider", "Billing Provider"], inplace=True)
            logger.info(f"columns after dropping the columns: {df.columns}")
            df['noteDate'] = pd.to_datetime(df['noteDate'], errors='coerce')
            df = df.sort_values(['patientId','noteDate']).reset_index(drop=True)
            text_cols = ['complaints','pastHistory','assesment','reviewofsystem',
                'currentmedication','procedure','allergy','examination',
                'patientSummary','diagnoses']
            df[text_cols] = df[text_cols].fillna("")
            for col in text_cols:
                df[col] = df[col].fillna("").apply(clean_html)
                
            logger.info(f"text columns preprocessed: {df.columns}")
            
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
            
            df['patient_age'] = df['patientSummary'].str.extract(r'(\d{1,2})\s*year', expand=False).astype(float)
            df['patient_gender'] = df['patientSummary'].str.extract(r'\b(Female|Male)\b', expand=False)
            df['follow_up_visit'] = df['patientSummary'].str.contains("follow up", case=False, na=False)
            
            
            df['has_allergy'] = ~df['allergy'].str.contains("no known", case=False, na=False)
            
            df['flare_signal'] = (
            df['complaint_flare_kw'] |
            df['flare_in_assessment'] |
            df['itch_present']
            ).astype(int)

            df['any_steroid_use'] = (
            df['steroid_started'] | df['on_steroid_med']
            ).astype(int)

            df['flare_risk_score'] = (
            df['flare_signal']*2 + df['any_steroid_use'] + df['trigger_mentioned']
            )
            logger.info(f"Preprocessing completed.")
            return df
        except Exception as e:
            logger.error(f"An error occurred in preprocessing: {e}")
            return None
    
            
    def split_data(self, df: pd.DataFrame):
        """Split data into train and test sets"""
        logger.info(f"Ready for splitting and finalizing data preparation.......")

        df["flare_label"] = np.where((df["flare_signal"] == 1) & (df["any_steroid_use"] == 1), 1, 0)
        logger.info(f"target column flare_label created.")
        df = df.sort_values(['patientId','noteDate']).reset_index(drop=True)
        df['flare_label_next'] = df.groupby('patientId')['flare_label'].shift(-1)
        df = df.dropna(subset=['flare_label_next']).reset_index(drop=True)
        df['flare_label_next'] = df['flare_label_next'].astype(int)
        target_col = 'flare_label_next'
        leak_cols = [
            'flare_label', 'flare_signal', 'flare_risk_score',
            'flare_in_assessment', 'any_steroid_use', 'steroid_started',
            'complaint_flare_kw', 'complaint_no_relief'
        ]

        for c in leak_cols:
            if c in df.columns:
                df.pop(c)

        [c for c in leak_cols if c in df.columns]
        for col in ['assesment','complaints','examination','patientSummary','currentmedication']:
            df[col + '_clean'] = df[col].fillna('').apply(mask_post_flare_terms)
        
        
        safe_numeric_cols = [
            "patient_age", "has_psoriasis", "on_steroid_med", "on_biologic",
            "itch_present", "dry_skin", "plaques_present", "silvery_scale",
            "elbows_involved", "hyperpigmentation", "smoker", "alcohol_use",
            "family_melanoma"
        ]

        safe_numeric_cols = [c for c in safe_numeric_cols if c in df.columns]
        text_inputs = ['assesment_clean', 'complaints_clean', 'examination_clean']
        text_inputs = [c for c in text_inputs if c in df.columns]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        train_idx, test_idx = next(gss.split(df, groups=df['patientId']))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df  = df.iloc[test_idx].reset_index(drop=True)
        print("Train patients:", train_df['patientId'].nunique(), "Test patients:", test_df['patientId'].nunique())
        print("Train pos rate:", train_df[target_col].mean(), "Test pos rate:", test_df[target_col].mean())
        train_text = (train_df['assesment_clean'].fillna('') + ' ' +
              train_df['complaints_clean'].fillna('') + ' ' +
              train_df['examination_clean'].fillna('')).astype(str)

        tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=5000, min_df=5, stop_words='english')
        X_text_train = tfidf.fit_transform(train_text)

        svd = TruncatedSVD(n_components=100, random_state=42)
        X_text_train_svd = svd.fit_transform(X_text_train)

        os.makedirs('/tmp/preproc', exist_ok=True)
        joblib.dump(tfidf, '/tmp/preproc/tfidf.joblib')
        joblib.dump(svd, '/tmp/preproc/svd.joblib')
        scaler = StandardScaler()
        X_num_train = scaler.fit_transform(train_df[safe_numeric_cols].fillna(0).astype(float).values)
        joblib.dump(scaler, '/tmp/preproc/scaler.joblib')

        X_text_train = tfidf.transform(train_text)
        X_text_train_svd = svd.transform(X_text_train)

        test_text = (test_df['assesment_clean'].fillna('') + ' ' +
                    test_df['complaints_clean'].fillna('') + ' ' +
                    test_df['examination_clean'].fillna('')).astype(str)
        X_text_test = tfidf.transform(test_text)
        X_text_test_svd = svd.transform(X_text_test)

        X_num_test = scaler.transform(test_df[safe_numeric_cols].fillna(0).astype(float).values)
        X_train = np.hstack([X_num_train, X_text_train_svd])
        X_test  = np.hstack([X_num_test,  X_text_test_svd])
        y_train = train_df[target_col].values
        y_test  = test_df[target_col].values
        print(X_train.shape, X_test.shape, y_train.mean(), y_test.mean())
        logger.info(f"Data split into train and test sets.")
        logger.info(f"TF-IDF vocab size: {len(tfidf.vocabulary_)}")
        logger.info(f"SVD components: {X_text_train_svd.shape}")
        logger.info(f"train test split completed.....")
        
        return X_train, X_test, y_train, y_test            


def ml_flow():
    feature_extraction = FeatureExtraction()
    df = feature_extraction.extract_features()
    if df is None:
        logger.error("Feature extraction returned None.")
        return
    X_train, X_test, y_train, y_test  = feature_extraction.split_data(df)
    return X_train, X_test, y_train, y_test




        

if __name__ == '__main__':
    ml_flow()