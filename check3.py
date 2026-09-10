import pandas as pd

print("Loading CSV_train.csv")
try:
    df_csv = pd.read_csv("CSV_train.csv", sep=";", skiprows=1, header=None)
    
    cols = [
        "WELL", "DEPTH_MD", "X_LOC", "Y_LOC", "Z_LOC", "GROUP", "FORMATION", 
        "CALI", "RSHA", "RMED", "RDEP", "RHOB", "GR", "SGR", "NPHI", "PEF", 
        "DTC", "SP", "BS", "ROP", "DTS", "DCAL", "DRHO", "MUDWEIGHT", "RMIC", 
        "ROPA", "RXO", "FORCE_2020_LITHOFACIES_LITHOLOGY", "FORCE_2020_LITHOFACIES_CONFIDENCE"
    ]
    
    if df_csv.shape[1] == len(cols):
        df_csv.columns = cols
        print("Assigned 29 FORCE 2020 headers successfully.\n")
        
        print("--- 1. Units check ---")
        for col in ["RHOB", "NPHI", "DTC"]:
            print(f"{col}: min={df_csv[col].min():.3f}, max={df_csv[col].max():.3f}, mean={df_csv[col].mean():.3f}")
                
        print("\n--- 2. Missing curves per well (sample of worst offenders) ---")
        missing = df_csv.groupby("WELL").apply(lambda w: w.isna().mean())
        print(missing.mean().sort_values(ascending=False).head(10))
        
        print("\n--- 3. Facies class balance ---")
        print(df_csv["FORCE_2020_LITHOFACIES_LITHOLOGY"].value_counts())
    else:
        print(f"Column count mismatch. Expected {len(cols)}, got {df_csv.shape[1]}")

except Exception as e:
    print("Error:", e)

