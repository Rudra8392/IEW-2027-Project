import pandas as pd
import numpy as np
from data_pipeline import load_train_csv, flag_washout, engineer_depth_features, create_well_splits, encode_categoricals, impute_and_scale
from models_baseline import train_xgboost_baseline
import scipy.signal
from metrics import LITHOLOGY_CLASSES, eval_facies

print("Loading and training full dataset...")
df = load_train_csv()
df = flag_washout(df)
df = engineer_depth_features(df)
train_df, val_df = create_well_splits(df, test_size=0.2, random_state=42)
train_df, val_df, cat_encoder = encode_categoricals(train_df, val_df)

ignore_cols = ["WELL", "DEPTH_MD", "GROUP", "FORMATION", "FORCE_2020_LITHOFACIES_LITHOLOGY", "FORCE_2020_LITHOFACIES_CONFIDENCE", "WASHOUT_FLAG"]
features = [c for c in train_df.columns if c not in ignore_cols]
final_features = features + ["GROUP", "FORMATION"]
for c in train_df.columns:
    if "ROLL" in c or "GRAD" in c or "TREND" in c:
        if c not in final_features: final_features.append(c)
            
train_df, val_df, scaler, medians = impute_and_scale(train_df, val_df, final_features)
target = "FORCE_2020_LITHOFACIES_LITHOLOGY"
train_df = train_df.dropna(subset=[target])
val_df = val_df.dropna(subset=[target])

X_train = train_df[final_features].values
y_train = train_df[target].values
X_val = val_df[final_features].values
y_val = val_df[target].values

model, metrics = train_xgboost_baseline(X_train, y_train, X_val, y_val)
print("AGGREGATE MACRO F1:", metrics["macro_f1"])
print("AGGREGATE PENALTY:", metrics["penalty_score"])
