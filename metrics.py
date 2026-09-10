import numpy as np
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error, brier_score_loss

# The official FORCE 2020 penalty matrix (Lithology confusion matrix)
# Rows: True Lithology, Columns: Predicted Lithology
# Order: 30000, 65030, 65000, 80000, 74000, 70000, 70032, 88000, 86000, 99000, 90000, 93000
LITHOLOGY_CLASSES = [30000, 65030, 65000, 80000, 74000, 70000, 70032, 88000, 86000, 99000, 90000, 93000]

# Normalized penalty matrix (0 is perfect, higher is worse)
A = np.array([
    [0.  , 2.  , 3.  , 3.25, 4.  , 3.5 , 3.75, 4.  , 4.  , 4.  , 4.  , 4.  ], # 30000
    [2.  , 0.  , 1.  , 1.5 , 2.5 , 2.25, 2.5 , 3.  , 3.  , 3.  , 3.  , 3.  ], # 65030
    [3.  , 1.  , 0.  , 1.  , 2.  , 1.75, 2.  , 2.5 , 2.5 , 2.5 , 2.5 , 2.5 ], # 65000
    [3.25, 1.5 , 1.  , 0.  , 1.5 , 1.25, 1.5 , 2.  , 2.  , 2.  , 2.  , 2.  ], # 80000
    [4.  , 2.5 , 2.  , 1.5 , 0.  , 0.5 , 1.  , 1.5 , 1.5 , 1.5 , 1.5 , 1.5 ], # 74000
    [3.5 , 2.25, 1.75, 1.25, 0.5 , 0.  , 0.5 , 1.  , 1.  , 1.  , 1.  , 1.  ], # 70000
    [3.75, 2.5 , 2.  , 1.5 , 1.  , 0.5 , 0.  , 1.25, 1.25, 1.25, 1.25, 1.25], # 70032
    [4.  , 3.  , 2.5 , 2.  , 1.5 , 1.  , 1.25, 0.  , 0.5 , 1.5 , 1.5 , 1.5 ], # 88000
    [4.  , 3.  , 2.5 , 2.  , 1.5 , 1.  , 1.25, 0.5 , 0.  , 1.5 , 1.5 , 1.5 ], # 86000
    [4.  , 3.  , 2.5 , 2.  , 1.5 , 1.  , 1.25, 1.5 , 1.5 , 0.  , 2.  , 2.  ], # 99000
    [4.  , 3.  , 2.5 , 2.  , 1.5 , 1.  , 1.25, 1.5 , 1.5 , 2.  , 0.  , 2.  ], # 90000
    [4.  , 3.  , 2.5 , 2.  , 1.5 , 1.  , 1.25, 1.5 , 1.5 , 2.  , 2.  , 0.  ]  # 93000
])

def score_force_2020(y_true, y_pred):
    """
    Computes the official FORCE 2020 penalty score.
    Lower is better.
    """
    # Map class labels to matrix indices
    class_to_idx = {c: i for i, c in enumerate(LITHOLOGY_CLASSES)}
    
    score = 0.0
    for true, pred in zip(y_true, y_pred):
        if true in class_to_idx and pred in class_to_idx:
            score += A[class_to_idx[true], class_to_idx[pred]]
        else:
            # Maximum penalty for unknown classes
            score += 4.0
            
    return score / len(y_true)

def eval_facies(y_true, y_pred):
    """Returns penalty score and Macro F1"""
    penalty = score_force_2020(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', labels=LITHOLOGY_CLASSES, zero_division=0)
    return {"penalty_score": penalty, "macro_f1": f1}

def eval_properties(y_true, y_pred, name="Property"):
    """Returns RMSE and MAE for regression targets"""
    # Filter out NaNs if any
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    
    if len(yt) == 0:
        return {"rmse": np.nan, "mae": np.nan}
        
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae = mean_absolute_error(yt, yp)
    return {f"{name}_rmse": rmse, f"{name}_mae": mae}
