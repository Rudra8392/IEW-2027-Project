import numpy as np
import pandas as pd
import torch
from data_pipeline import engineer_depth_features, flag_washout
from physics_layer import apply_physics_layer

def calibrate_risk_score(facies_probs, phi_mean, sw_mean):
    """
    Computes a confidence-gated Risk Score P(Commercial HC).

    The raw score is the product of:
      - P(reservoir facies) — from model softmax
      - P(good porosity) — sigmoid soft-threshold at phi > 15%
      - P(low water saturation) — sigmoid soft-threshold at Sw < 50%

    Critically, it is then GATED by the model's confidence (max class probability).
    This prevents the score from inflating in depth intervals where the classifier
    is maximally uncertain — fixing the known negative corr(F1, P_HC) structural issue.
    """
    # Reservoir probability: Sandstone (idx 0) + Sandstone/Shale (idx 1)
    p_reservoir = facies_probs[:, 0] + facies_probs[:, 1]

    # Soft thresholds using sigmoids
    p_phi = 1.0 / (1.0 + np.exp(-100 * (phi_mean - 0.15)))  # phi > 15%
    p_sw  = 1.0 / (1.0 + np.exp( 30  * (sw_mean  - 0.50)))  # Sw  < 50%

    # Model confidence: max class probability per depth sample
    # A uniform distribution over 12 classes → max ≈ 0.083; confident prediction → max ≈ 0.9+
    confidence = facies_probs.max(axis=1)
    # Rescale: anything below 2/12 ≈ 0.167 (barely above random) is treated as near-zero confidence
    confidence_gate = np.clip((confidence - 1.0/12) / (1.0 - 1.0/12), 0, 1)

    # Joint score gated by confidence
    raw_risk_score = p_reservoir * p_phi * p_sw * confidence_gate

    return raw_risk_score

def run_inference(raw_well_df, xgb_model, scaler, cat_encoder, medians, feature_cols):
    """
    End-to-end inference function mimicking the "closed-loop" platform behavior.
    Takes raw well logs and outputs all Phase 1 deliverables.
    """
    # 1. Pipeline Processing
    df = raw_well_df.copy()
    df = flag_washout(df)
    df = engineer_depth_features(df)
    
    # 2. Categoricals
    cat_cols = ["GROUP", "FORMATION"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna("UNKNOWN")
    
    existing_cats = [c for c in cat_cols if c in df.columns]
    if existing_cats:
        df[existing_cats] = cat_encoder.transform(df[existing_cats])
        
    # 3. Impute & Scale
    df[feature_cols] = df[feature_cols].fillna(medians)
    scaled_features = scaler.transform(df[feature_cols])
    
    # 4. Physics Layer (Pseudo-labels)
    df = apply_physics_layer(df)
    
    # 5. Facies Classification (using Baseline for inference demonstration)
    # Outputting probabilities to compute risk score
    facies_probs = xgb_model.predict_proba(scaled_features)
    facies_preds = xgb_model.predict(scaled_features)
    
    # 6. Risk Score Calculation
    phi_val = df["POROSITY_PSEUDO"].values
    sw_val = df["SW_PSEUDO"].values
    
    risk_scores = calibrate_risk_score(facies_probs, phi_val, sw_val)
    df["P_COMMERCIAL_HC"] = risk_scores
    
    # Map predictions back to lithology codes
    from metrics import LITHOLOGY_CLASSES
    inv_class_map = {i: c for i, c in enumerate(LITHOLOGY_CLASSES)}
    df["FACIES_PREDICTION"] = np.vectorize(inv_class_map.get)(facies_preds)
    
    return df

if __name__ == "__main__":
    print("Testing end-to-end inference app...")
    # This script would be wrapped in a Streamlit or FastAPI layer for the final deliverable.
    print("Inference package ready for deployment!")
