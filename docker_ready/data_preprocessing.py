# data_preprocessing.py
import pandas as pd
from pathlib import Path


def descriptive_stats_pipeline():
    data_path = Path('processed')
    data_path.mkdir(parents=True, exist_ok=True)

    # --- Load raw parts (no low-level cleaning here) ---
    dim_df       = pd.read_csv(data_path / 'dim.csv')
    linktable_df = pd.read_csv(data_path / 'linktable.csv', low_memory=False)
    inst_df      = pd.read_csv(data_path / 'inst.csv')
    longlat_df   = pd.read_csv(data_path / 'LongLat.csv')

    # =====================================================================
    # Merge LongLat into dim_df via RTUNumber <-> LKey
    # LongLat.csv has columns: Longitude, Latitude, Location, LKey
    # We want each RTU (RTUNumber) in dim_df to carry its Long/Lat/Location.
    # =====================================================================
    if 'RTUNumber' not in dim_df.columns:
        raise ValueError("dim.csv is missing 'RTUNumber' column required for LongLat merge.")
    if 'LKey' not in longlat_df.columns:
        raise ValueError("LongLat.csv is missing 'LKey' column required for LongLat merge.")

    # Ensure same type for join keys
    dim_df['RTUNumber'] = dim_df['RTUNumber'].astype(str)
    longlat_df['LKey']  = longlat_df['LKey'].astype(str)

    # Left join: keep all dim rows, enrich with LongLat if available
    dim_df = pd.merge(
        dim_df,
        longlat_df,
        how='left',
        left_on='RTUNumber',
        right_on='LKey'
    )
    # Optional: drop LKey if you don't need it downstream
    # dim_df = dim_df.drop(columns=['LKey'], errors='ignore')

    # --- Merge to get Equipment / Description / DateTime / CurrValue ---
    # inst + linktable on HkeyDateTime
    df = pd.merge(inst_df, linktable_df, how='inner', on='HkeyDateTime')

    # Hkey must match between inst/linktable merge result and dim
    df['Hkey']     = df['Hkey'].astype(str)
    dim_df['Hkey'] = dim_df['Hkey'].astype(str)

    # Now bring in dimension info (including Long/Lat/Location from LongLat)
    df = pd.merge(df, dim_df, how='inner', on='Hkey')

    # --- Minimal formatting only (no heavy filtering) ---
    needed = ['Equipment', 'Description', 'DateTime', 'CurrValue']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after merge: {missing}")

    # Normalize/parse DateTime (accept many formats)
    dt_str = df['DateTime'].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
    df['DateTime'] = pd.to_datetime(dt_str, errors='coerce')

    # Coerce CurrValue numeric (invalid -> NaN)
    df['CurrValue'] = pd.to_numeric(df['CurrValue'], errors='coerce')

    # --- Build final detector input + master dataset ---
    # Always keep these core columns first
    base_cols = ['Equipment', 'Description', 'DateTime', 'CurrValue']

    # Keep ALL other enrichment columns too, including Longitude/Latitude/Location
    extra_cols = [c for c in df.columns if c not in base_cols]
    out = df[base_cols + extra_cols].copy()

    # Drop only rows the detector can't use (NaT time or NaN value)
    drop_dt = out['DateTime'].isna().sum()
    drop_cv = out['CurrValue'].isna().sum()
    out = out.dropna(subset=['DateTime', 'CurrValue']).sort_values('DateTime')

    out_path = data_path / 'data.csv'
    out.to_csv(out_path, index=False)
    print(
        f"Saved detector/master input: '{out_path.as_posix()}' "
        f"-> {len(out)} rows (dropped NaT DateTime: {drop_dt}, NaN CurrValue: {drop_cv})."
    )


if __name__ == '__main__':
    descriptive_stats_pipeline()
