import pandas as pd

# Numeric columns to clean in BMMS overview
NUMERIC_COLS = [
    "chainage",
    "km",
    "lon",
    "lat",
    "width",
    "length",
    "spans",
    "constructionYear",
    "structureNr",
]

# Bounding box for Bangladesh (lat/lon)
LAT_MIN, LAT_MAX = 20.5, 26.7
LON_MIN, LON_MAX = 88.0, 92.7

# ==========================================================
# Code logic


def clean_numeric_series(s):
    if s is None:
        return s

    # Convert to string, strip whitespace, and remove non-breaking/thin spaces
    s = (
        s.astype(str)
        .str.strip()
        .str.replace("\u00a0", "", regex=False)
        .str.replace("\u2009", "", regex=False)
        .replace({"^$": None, "^-$": None, "^--$": None}, regex=True)
    )

    # Handle decimal comma (only when no dot present)
    comma_no_dot = s.str.contains(",", na=False) & ~s.str.contains(".", na=False)
    s.loc[comma_no_dot] = s.loc[comma_no_dot].str.replace(",", ".", regex=False)

    # Remove thousands separators
    s = s.str.replace(",", "", regex=False)

    return pd.to_numeric(s, errors="coerce")


def open_xlsx(xlsx_path):
    raw = pd.read_excel(xlsx_path, sheet_name="BMMS_overview", dtype=str)
    df = raw["BMMS_overview"] if isinstance(raw, dict) else raw

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = clean_numeric_series(df[col])

    return df