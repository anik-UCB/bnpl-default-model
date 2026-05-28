# Add this to src/evaluate.py and run it
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score
import sys
sys.path.append('src')
from features import engineer_features, get_feature_cols, split_data

df      = pd.read_excel('data/data_bnpl.xlsx')
df_feat = engineer_features(df, is_training=True)
_, test = split_data(df_feat)

model    = joblib.load('models/xgboost_model.pkl')
features = joblib.load('models/feature_cols.pkl')

X_test = test[features]
y_test = test['target']
proba  = model.predict_proba(X_test)[:, 1]

print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'Flagged%':>10}")
print("-" * 45)
for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    preds = (proba >= t).astype(int)
    prec  = precision_score(y_test, preds, zero_division=0)
    rec   = recall_score(y_test, preds, zero_division=0)
    flagged = preds.mean()
    print(f"{t:>10.2f} {prec:>10.3f} {rec:>10.3f} {flagged:>10.1%}")