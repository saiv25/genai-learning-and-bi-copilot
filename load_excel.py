
from pathlib import Path
import sqlite3
import pandas as pd


# --------------------------------------------------
# 1. Locate project files
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

if (SCRIPT_DIR / "data.xlsx").exists():
    BASE_DIR = SCRIPT_DIR
elif (SCRIPT_DIR.parent / "data.xlsx").exists():
    BASE_DIR = SCRIPT_DIR.parent
else:
    raise FileNotFoundError(
        "data.xlsx was not found. Place it in GENAI_Series."
    )

EXCEL_PATH = BASE_DIR / "data.xlsx"
DB_PATH = BASE_DIR / "my_tasks.db"


# --------------------------------------------------
# 2. Load Excel data
# --------------------------------------------------

print(f"Reading Excel file: {EXCEL_PATH}")

df = pd.read_excel(EXCEL_PATH)

print(f"Original rows: {len(df):,}")
print("Original columns:")
print([repr(column) for column in df.columns])


# --------------------------------------------------
# 3. Clean column names
# --------------------------------------------------

# Fixes column names such as ' Sales' -> 'Sales'
df.columns = df.columns.astype(str).str.strip()

# Prevent duplicate names after cleaning
duplicates = df.columns[df.columns.duplicated()].tolist()

if duplicates:
    raise ValueError(
        f"Duplicate columns after cleaning: {duplicates}"
    )


# --------------------------------------------------
# 4. Validate required business columns
# --------------------------------------------------

required_columns = [
    "Segment",
    "Country",
    "Product",
    "Discount Band",
    "Units Sold",
    "Manufacturing Price",
    "Sale Price",
    "Gross Sales",
    "Discounts",
    "Sales",
    "COGS",
    "Profit",
    "Date",
    "Month Number",
    "Month Name",
    "Year"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in Excel file: {missing_columns}"
    )

# Keep only the expected columns, in a consistent order
df = df[required_columns].copy()


# --------------------------------------------------
# 5. Clean text values
# --------------------------------------------------

for column in df.select_dtypes(include=["object"]).columns:
    df[column] = df[column].map(
        lambda value: value.strip()
        if isinstance(value, str)
        else value
    )


# --------------------------------------------------
# 6. Clean date values
# --------------------------------------------------

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
).dt.strftime("%Y-%m-%d")


# --------------------------------------------------
# 7. Import into SQLite
# --------------------------------------------------

print(f"\nConnecting to database: {DB_PATH}")

with sqlite3.connect(DB_PATH) as conn:

    df.to_sql(
        name="browser_data",
        con=conn,
        if_exists="replace",
        index=False
    )

    # Verify the imported table
    schema = pd.read_sql_query(
        'PRAGMA table_info("browser_data")',
        conn
    )

    imported_columns = schema["name"].tolist()

    print("\nImported database columns:")
    print(imported_columns)

    if "Sales" not in imported_columns:
        raise ValueError(
            "Sales column was not imported correctly."
        )

    # Verify the number of imported records
    imported_rows = pd.read_sql_query(
        'SELECT COUNT(*) AS total FROM "browser_data"',
        conn
    ).iloc[0]["total"]


# --------------------------------------------------
# 8. Final status
# --------------------------------------------------

print("\n" + "=" * 50)
print("EXCEL IMPORT SUCCESSFUL")
print("=" * 50)
print(f"Excel file : {EXCEL_PATH}")
print(f"Database   : {DB_PATH}")
print(f"Table      : browser_data")
print(f"Rows       : {imported_rows:,}")
print("Sales column verified.")
print("Ready to launch the dashboard!")
