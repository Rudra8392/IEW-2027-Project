import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import warnings
warnings.filterwarnings('ignore')

FORCE_2020_COLS = [
    "WELL", "DEPTH_MD", "X_LOC", "Y_LOC", "Z_LOC", "GROUP", "FORMATION", 
    "CALI", "RSHA", "RMED", "RDEP", "RHOB", "GR", "SGR", "NPHI", "PEF", 
    "DTC", "SP", "BS", "ROP", "DTS", "DCAL", "DRHO", "MUDWEIGHT", "RMIC", 
    "ROPA", "RXO", "FORCE_2020_LITHOFACIES_LITHOLOGY", "FORCE_2020_LITHOFACIES_CONFIDENCE"
]

def load_train_csv(filepath="CSV_train.csv"):
    """Loads the training CSV, which is missing headers in its raw form."""
    df = pd.read_csv(filepath, sep=";", skiprows=1, header=None)
    df.columns = FORCE_2020_COLS
    return df

def flag_washout(df, threshold=1.0):
    """
    Flags washed out intervals where CALI > BS + threshold.
    If CALI or BS is missing, it assumes no washout (0).
    """
    if "CALI" in df.columns and "BS" in df.columns:
        df["WASHOUT_FLAG"] = ((df["CALI"] - df["BS"]) > threshold).astype(int)
        # We can also penalize heavily washed out readings or just pass the flag to the model
    else:
        df["WASHOUT_FLAG"] = 0
    return df

def engineer_depth_features(df, window=50):
    """
    Adds rolling statistics over depth for key curves to capture sequential trends.
    Group by WELL so rolling doesn't cross well boundaries.
    """
    # Important curves to get rolling stats for
    curves_to_roll = ["GR", "RHOB", "NPHI", "DTC", "RDEP"]
    
    # Sort by depth to ensure sequential order
    df = df.sort_values(by=["WELL", "DEPTH_MD"])
    
    for curve in curves_to_roll:
        if curve in df.columns:
            # Rolling mean
            df[f"{curve}_ROLL_MEAN_{window}"] = df.groupby("WELL")[curve].transform(
                lambda x: x.rolling(window, min_periods=1, center=True).mean()
            )
            # Rolling std (texture/noise indicator)
            df[f"{curve}_ROLL_STD_{window}"] = df.groupby("WELL")[curve].transform(
                lambda x: x.rolling(window, min_periods=1, center=True).std()
            )
            # Depth gradient (pseudo-derivative)
            df[f"{curve}_GRAD"] = df.groupby("WELL")[curve].diff() / df.groupby("WELL")["DEPTH_MD"].diff()
            df[f"{curve}_GRAD"].replace([np.inf, -np.inf], np.nan, inplace=True)
            
    # Normal compaction trend residual proxy: local detrended GR
    if "GR" in df.columns:
        df["GR_DEPTH_TREND"] = df["GR"] - df.groupby("WELL")["GR"].transform(
            lambda x: x.rolling(200, min_periods=1, center=True).mean()
        )
        
    return df

def encode_categoricals(train_df, val_df, cat_cols=["GROUP", "FORMATION"]):
    """
    Encodes categorical features. Fits on train, transforms train and val.
    Handles unseen categories in validation.
    """
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    # Fill NAs in categories with a placeholder string
    for c in cat_cols:
        if c in train_df.columns:
            train_df[c] = train_df[c].fillna("UNKNOWN")
        if val_df is not None and c in val_df.columns:
            val_df[c] = val_df[c].fillna("UNKNOWN")
            
    # Only fit if columns exist
    existing_cols = [c for c in cat_cols if c in train_df.columns]
    if existing_cols:
        train_df[existing_cols] = encoder.fit_transform(train_df[existing_cols])
        if val_df is not None:
            val_df[existing_cols] = encoder.transform(val_df[existing_cols])
            
    return train_df, val_df, encoder

def impute_and_scale(train_df, val_df, features):
    """
    Fits scaler and imputer on training wells, applies to val wells.
    Avoids information leakage.
    """
    # Simple median imputation for baseline (can be enhanced with physics-based later)
    medians = train_df[features].median()
    
    train_df[features] = train_df[features].fillna(medians)
    if val_df is not None:
        val_df[features] = val_df[features].fillna(medians)
        
    scaler = StandardScaler()
    train_df[features] = scaler.fit_transform(train_df[features])
    if val_df is not None:
        val_df[features] = scaler.transform(val_df[features])
        
    return train_df, val_df, scaler, medians

def create_well_splits(df, n_splits=1, test_size=0.2, random_state=42):
    """
    Creates leakage-safe well-level train/validation splits.
    """
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(df, groups=df['WELL']))
    
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    
    return train_df, val_df

if __name__ == "__main__":
    print("Loading data...")
    df = load_train_csv()
    print(f"Data shape: {df.shape}")
    
    print("Flagging washout...")
    df = flag_washout(df)
    
    print("Engineering depth features...")
    df = engineer_depth_features(df)
    
    print("Splitting train/val at well level...")
    train_df, val_df = create_well_splits(df, test_size=0.15)
    print(f"Train wells: {train_df['WELL'].nunique()} | Val wells: {val_df['WELL'].nunique()}")
    
    print("Encoding categoricals...")
    train_df, val_df, cat_encoder = encode_categoricals(train_df, val_df)
    
    # Identify numeric features for scaling
    ignore_cols = ["WELL", "DEPTH_MD", "GROUP", "FORMATION", "FORCE_2020_LITHOFACIES_LITHOLOGY", "FORCE_2020_LITHOFACIES_CONFIDENCE", "WASHOUT_FLAG"]
    features = [c for c in train_df.columns if c not in ignore_cols]
    
    print("Imputing and scaling...")
    train_df, val_df, scaler, medians = impute_and_scale(train_df, val_df, features)
    
    print("Pipeline test complete!")
    print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}")
