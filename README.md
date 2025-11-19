## Project Detail.
- This project is about classifying psoriasis flare using Machine learning from the Legend Ehr database of the patients who have psoriasis.
- The system can predict the which patient could have psoraisis flare based on the elements like itching like or the other factors that is already their in the patient psoraisis diagnosis.


## Results

[LightGBM] [Info] Total Bins 30724
[LightGBM] [Info] Number of data points in the train set: 9535, number of used features: 132
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.500000 -> initscore=-0.000000
[LightGBM] [Info] Start training from score -0.000000
              precision    recall  f1-score   support

           0       0.69      0.69      0.69       917
           1       0.77      0.78      0.78      1250

    accuracy                           0.74      2167
   macro avg       0.73      0.73      0.73      2167
weighted avg       0.74      0.74      0.74      2167

ROC-AUC: 0.819104907306434



## installation
scispacy model:
can be load through spacy.
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz 