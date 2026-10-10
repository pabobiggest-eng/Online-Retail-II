
import pandas as pd
from pathlib import Path

# Load dataset
file_path = Path("dataset") / "online_retail_II.xlsx"

print("Loading dataset...")
df = pd.read_excel(file_path)

# Dataset overview
print("\n=== ONLINE RETAIL II PROJECT ===")
print("Dataset loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# Save initial report
df.head(100).to_csv("dataset_sample.csv", index=False)

print("\nSample saved: dataset_sample.csv")
print("PROJECT STEP 1 COMPLETED!")

import pandas as pd
from pathlib import Path

# Dataset path
file_path = Path("dataset/online_retail_II.xlsx")

# Check dataset exists
if not file_path.exists():
    raise FileNotFoundError(
        f"Dataset not found: {file_path.resolve()}"
    )

# Load original dataset
df = pd.read_excel(file_path)

print("=" * 55)
print("ONLINE RETAIL II — DATASET DIAGNOSTICS")
print("=" * 55)

# Basic information
print("\n1. DATASET OVERVIEW")
print("Total rows:", len(df))
print("Total columns:", len(df.columns))
print("Column names:", df.columns.tolist())
print("Date range:", df["InvoiceDate"].min(), "to", df["InvoiceDate"].max())

# Missing values
print("\n2. MISSING VALUES")
print(df.isnull().sum())

# Duplicate analysis
print("\n3. DUPLICATE ANALYSIS")
print("Exact duplicate rows:", df.duplicated().sum())
print("Rows involved in duplicates:", df.duplicated(keep=False).sum())
print("Rows after removing exact duplicates:", len(df.drop_duplicates()))

# Customer ID and description
print("\n4. CUSTOMER AND PRODUCT CHECKS")
print("Missing Customer IDs:", df["Customer ID"].isna().sum())
print("Missing descriptions:", df["Description"].isna().sum())
print("Unique customers:", df["Customer ID"].nunique())
print("Unique products:", df["StockCode"].nunique())

# Invoice analysis
print("\n5. INVOICE CHECKS")
print(
    "Invoices starting with C:",
    df["Invoice"].astype(str).str.strip().str.startswith("C").sum()
)
print("Unique invoices:", df["Invoice"].nunique())

# Quantity and price checks
print("\n6. QUANTITY AND PRICE CHECKS")
print("Zero/negative quantities:", (df["Quantity"] <= 0).sum())
print("Zero/negative prices:", (df["Price"] <= 0).sum())

# Save diagnostic report
Path("reports").mkdir(exist_ok=True)

with open("reports/dataset_diagnostic_report.txt", "w", encoding="utf-8") as f:
    f.write("ONLINE RETAIL II — DATASET DIAGNOSTIC REPORT\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Total rows: {len(df)}\n")
    f.write(f"Total columns: {len(df.columns)}\n")
    f.write(f"Columns: {df.columns.tolist()}\n")
    f.write(f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}\n")
    f.write(f"Exact duplicates: {df.duplicated().sum()}\n")
    f.write(f"Rows involved in duplicates: {df.duplicated(keep=False).sum()}\n")
    f.write(f"Missing Customer IDs: {df['Customer ID'].isna().sum()}\n")
    f.write(f"Missing descriptions: {df['Description'].isna().sum()}\n")
    f.write(f"Unique customers: {df['Customer ID'].nunique()}\n")
    f.write(f"Unique products: {df['StockCode'].nunique()}\n")
    f.write(f"Cancelled invoice rows: {df['Invoice'].astype(str).str.strip().str.startswith('C').sum()}\n")
    f.write(f"Zero/negative quantities: {(df['Quantity'] <= 0).sum()}\n")
    f.write(f"Zero/negative prices: {(df['Price'] <= 0).sum()}\n")
    f.write("\nMissing values by column:\n")
    f.write(df.isnull().sum().to_string())

print("\n" + "=" * 55)
print("STEP 1 COMPLETED SUCCESSFULLY!")
print("Report saved: reports/dataset_diagnostic_report.txt")
print("Original dataset has NOT been modified.")
print("=" * 55)
