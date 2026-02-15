"""Convert Small LCL Data CSVs to a single Parquet file."""

import glob
import pandas as pd

DATA_DIR = "data/Small LCL Data"
OUTPUT_PATH = "data/Small_LCL_Data.parquet"

csv_files = sorted(glob.glob(f"{DATA_DIR}/LCL-June2015v2_*.csv"))
print(f"Found {len(csv_files)} CSV files")

dfs = []
for f in csv_files:
    df = pd.read_csv(f, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
print(f"Combined shape: {combined.shape}")

combined["DateTime"] = pd.to_datetime(combined["DateTime"], errors="coerce")
combined["KWH/hh (per half hour)"] = pd.to_numeric(
    combined["KWH/hh (per half hour)"].astype(str).str.strip(), errors="coerce"
)

combined.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow")
print(f"Saved to {OUTPUT_PATH}")
