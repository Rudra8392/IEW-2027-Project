import pandas as pd
import glob
import lasio

print("Loading CSV_train.csv skipping the first row (the 'x')")
try:
    df_csv = pd.read_csv("CSV_train.csv", sep=";", skiprows=1, header=None)
    print("CSV shape:", df_csv.shape)
    # The first column is usually WELL in this dataset
    csv_wells = set(df_csv.iloc[:, 0].unique()) 
except Exception as e:
    print("Error reading CSV_train.csv:", e)
    csv_wells = set()

print("\nReading LAS files...")
las_wells = set()
las_files = glob.glob("Force_2020_all_wells_train_test_blind_hidden_final/**/*.las", recursive=True)
print(f"Found {len(las_files)} LAS files")

for f in las_files:
    try:
        l = lasio.read(f)
        well_name = l.well.WELL.value
        las_wells.add(well_name)
    except Exception as e:
        print(f"Error reading {f}: {e}")

print("\nIn LAS but NOT in CSV_train.csv (likely your real test/blind set):", las_wells - csv_wells)
print("In CSV but NOT found as a LAS file:", csv_wells - las_wells)

# Let's also parse one LAS file to get the column headers and use them for the units check.
if las_files:
    try:
        l = lasio.read(las_files[0])
        headers = [c.mnemonic for c in l.curves]
        print("\nLAS Headers:", headers)
        if len(headers) == df_csv.shape[1]:
            df_csv.columns = headers
            print("\nAssigned LAS headers to CSV successfully.")
            
            print("\n--- 1. Units check ---")
            for col in ["RHOB", "NPHI", "DTC"]:
                if col in df_csv.columns:
                    print(f"{col}: min={df_csv[col].min():.3f}, max={df_csv[col].max():.3f}, mean={df_csv[col].mean():.3f}")
                else:
                    print(f"{col}: NOT FOUND")
                    
            print("\n--- 2. Missing curves per well (sample of worst offenders) ---")
            if "WELL" in df_csv.columns:
                missing = df_csv.groupby("WELL").apply(lambda w: w.isna().mean())
                print(missing.mean().sort_values(ascending=False).head(10))
            else:
                missing = df_csv.groupby(df_csv.columns[0]).apply(lambda w: w.isna().mean())
                print(missing.mean().sort_values(ascending=False).head(10))
            
            print("\n--- 3. Facies class balance ---")
            if "FORCE_2020_LITHOFACIES_LITHOLOGY" in df_csv.columns:
                print(df_csv["FORCE_2020_LITHOFACIES_LITHOLOGY"].value_counts())
            else:
                print("FORCE_2020_LITHOFACIES_LITHOLOGY not found in columns")

    except Exception as e:
        print("Error getting headers:", e)

