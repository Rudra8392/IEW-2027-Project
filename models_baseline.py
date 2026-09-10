import xgboost as xgb
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from metrics import eval_facies, LITHOLOGY_CLASSES

def train_xgboost_baseline(X_train, y_train, X_val, y_val):
    """
    Trains XGBoost baseline for facies classification with inverse-frequency weighting.
    """
    print("Computing class weights...")
    # Compute class weights to handle imbalance (inverse frequency)
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    
    # Map weights to sample weights
    sample_weights = np.vectorize(class_weights.get)(y_train)
    
    # Needs to remap lithology classes to 0..11 for XGBoost
    class_map = {c: i for i, c in enumerate(LITHOLOGY_CLASSES)}
    inv_class_map = {i: c for i, c in enumerate(LITHOLOGY_CLASSES)}
    
    y_train_mapped = np.vectorize(class_map.get)(y_train)
    y_val_mapped = np.vectorize(class_map.get)(y_val)
    
    # Ensure all classes are present in y_train_mapped by appending dummy rows with weight 0
    missing_classes = set(range(len(LITHOLOGY_CLASSES))) - set(np.unique(y_train_mapped))
    if missing_classes:
        dummy_X = np.zeros((len(missing_classes), X_train.shape[1]))
        dummy_y = np.array(list(missing_classes))
        dummy_w = np.zeros(len(missing_classes))
        
        X_train = np.vstack([X_train, dummy_X])
        y_train_mapped = np.concatenate([y_train_mapped, dummy_y])
        sample_weights = np.concatenate([sample_weights, dummy_w])
        
    print("Training XGBoost Baseline...")
    clf = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=6, 
        learning_rate=0.1, 
        objective='multi:softprob',
        tree_method='hist', # fast histogram optimized
        n_jobs=-1,
        random_state=42
    )
    
    clf.fit(
        X_train, y_train_mapped,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val_mapped)],
        verbose=10
    )
    
    # Predict
    print("Evaluating XGBoost Baseline...")
    y_pred_mapped = clf.predict(X_val)
    y_pred = np.vectorize(inv_class_map.get)(y_pred_mapped)
    
    metrics = eval_facies(y_val, y_pred)
    print("Baseline Metrics:", metrics)
    
    return clf, metrics

if __name__ == "__main__":
    import pandas as pd
    from data_pipeline import load_train_csv, flag_washout, engineer_depth_features, create_well_splits, encode_categoricals, impute_and_scale
    
    print("Loading data for baseline training...")
    df = load_train_csv()
    df = flag_washout(df)
    df = engineer_depth_features(df)
    
    train_df, val_df = create_well_splits(df, test_size=0.15)
    train_df, val_df, cat_encoder = encode_categoricals(train_df, val_df)
    
    ignore_cols = ["WELL", "DEPTH_MD", "GROUP", "FORMATION", "FORCE_2020_LITHOFACIES_LITHOLOGY", "FORCE_2020_LITHOFACIES_CONFIDENCE", "WASHOUT_FLAG"]
    features = [c for c in train_df.columns if c not in ignore_cols]
    
    # Include engineered categoricals and depth features
    final_features = features + ["GROUP", "FORMATION"]
    for c in train_df.columns:
        if "ROLL" in c or "GRAD" in c or "TREND" in c:
            if c not in final_features:
                final_features.append(c)
                
    train_df, val_df, scaler, medians = impute_and_scale(train_df, val_df, final_features)
    
    # Target
    target = "FORCE_2020_LITHOFACIES_LITHOLOGY"
    
    # Filter rows with missing targets
    train_df = train_df.dropna(subset=[target])
    val_df = val_df.dropna(subset=[target])
    
    X_train = train_df[final_features].values
    y_train = train_df[target].values
    X_val = val_df[final_features].values
    y_val = val_df[target].values
    
    model, metrics = train_xgboost_baseline(X_train, y_train, X_val, y_val)
