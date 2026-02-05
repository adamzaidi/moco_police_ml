import pandas as pd
from pathlib import Path

# Get project root dynamically
BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "data" / "raw_incidents.csv"

print(f"Reading from: {CSV_PATH}")

df = pd.read_csv(CSV_PATH, nrows=1000)

print("\n=== COLUMN NAMES ===")
print(df.columns.tolist())

print("\n=== DATA TYPES ===")
print(df.dtypes)

print("\n=== SAMPLE ROWS ===")
print(df.head(10))

print("\n=== MISSING VALUES (TOP 10) ===")
print(df.isnull().sum().sort_values(ascending=False).head(10))