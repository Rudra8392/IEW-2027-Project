import pandas as pd
import glob
import lasio

print("Loading CSV_train.csv")
try:
    df_csv = pd.read_csv("CSV_train.csv", sep=";")
    print("CSV columns:", df_csv.columns.tolist())
    csv_wells = set(df_csv["WELL"].unique()) if "WELL" in df_csv.columns else set()
except Exception as e:
    print("Error reading CSV_train.csv:", e)
    csv_wells = set()

print("\nReading LAS files...")
las_wells = set()
for f in glob.glob("Force_2020_all_wells_train_test_blind/**/*.las", recursive=True):
    try:
        l = lasio.read(f)
        las_wells.add(l.well.WELL.value)
    except Exception as e:
        print(f"Error reading {f}: {e}")

print("\nIn LAS but NOT in CSV_train.csv (likely your real test/blind set):", las_wells - csv_wells)
print("In CSV but NOT found as a LAS file:", csv_wells - las_wells)

