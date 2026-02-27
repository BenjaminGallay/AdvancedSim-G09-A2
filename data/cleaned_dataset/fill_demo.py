import os
import xlsx_tools
import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
roads_csv = os.path.join(BASE_DIR, "data", "cleaned_dataset", "_roads3.csv")
bmms_xlsx = os.path.join(BASE_DIR, "data", "cleaned_dataset", "BMMS_overview.xlsx")
out_csv = os.path.join(BASE_DIR, "data", "roadN1.csv")
MERGE_CONSECUTIVE_LINKS = True


def bmms_aggregate(bmms):
    """Deduplicate / aggregate BMMS so there is one row per (road, lrp).

    Uses median for numeric `length` and median of mapped condition codes.
    """
    # ensure length is numeric for median
    bmms = bmms.copy()
    bmms["length"] = pd.to_numeric(bmms["length"], errors="coerce")

    # translate textual conditions to numeric codes for median aggregation
    cond_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    bmms["condition_code"] = bmms["condition"].astype(str).str.strip().str.upper().map(cond_map)

    bmms_agg = (
        bmms.groupby(["road", "lrp"], as_index=False)
        .agg({"length": "median", "condition_code": "median"})
    )

    # round condition median to nearest integer and keep as nullable Int
    bmms_agg["condition_bmms"] = bmms_agg["condition_code"].round().astype("Int64")
    bmms_sub = bmms_agg.rename(columns={"length": "length_bmms"})
    bmms_sub = bmms_sub[["road", "lrp", "length_bmms", "condition_bmms"]]

    return bmms_sub

def bmms_backfill(bmms_sub, df):
    """Backfill missing BMMS info for a segment by checking the bmms_sub entry for the segment's `lrp_next`."""
    if bmms_sub.empty:
        return

    # build lookup dicts keyed by (road, lrp)
    _len_map = { (r, l): v for r, l, v in bmms_sub[["road", "lrp", "length_bmms"]].itertuples(index=False) }
    _cond_map = { (r, l): v for r, l, v in bmms_sub[["road", "lrp", "condition_bmms"]].itertuples(index=False) }

    if "lrp_next" not in df.columns:
        return

    # fill length_bmms where missing using the bmms record for the segment's end LRP
    mask_len = df["length_bmms"].isna() & df["lrp_next"].notna()
    if mask_len.any():
        df.loc[mask_len, "length_bmms"] = df.loc[mask_len].apply(
            lambda r: _len_map.get((r["road"], r["lrp_next"])), axis=1
        )

    # fill condition_bmms where missing using the bmms record for the segment's end LRP
    mask_cond = df["condition_bmms"].isna() & df["lrp_next"].notna()
    if mask_cond.any():
        df.loc[mask_cond, "condition_bmms"] = df.loc[mask_cond].apply(
            lambda r: _cond_map.get((r["road"], r["lrp_next"])), axis=1
        )

def fill_type(df):
    
    is_bridge = (df["gap"] == "BS") & (df["gap_next"] == "BE")
    is_ferry = (df["gap"] == "FS") & (df["gap_next"] == "FE")
    df["model_type"] = "link"
    df.loc[is_bridge, "model_type"] = "bridge"
    df.loc[is_ferry, "model_type"] = "ferry"

def fill_length(df):
    """Compute segment lengths, prefer BMMS length for bridges, fallback to chainage diff."""
    # chainage is in km in the roads CSV; convert difference to meters
    df["length_calc"] = (
        pd.to_numeric(df["chainage_next"], errors="coerce") - pd.to_numeric(df["chainage"], errors="coerce")
    ) * 1000.0
    df["length"] = np.where(df["model_type"] == "bridge", df["length_bmms"], df["length_calc"])
    invalid_bridge_mask = (df["model_type"] == "bridge") & df["length"].isna()
    if invalid_bridge_mask.any():
        # if length_bmms is missing for a bridge, fallback to chainage difference
        df.loc[invalid_bridge_mask, "length"] = df.loc[invalid_bridge_mask, "length_calc"]
        
def fill_condition(segments):
    """Assign aggregated numeric condition codes to bridge segments; default 0 when missing."""
    segments["condition"] = pd.NA
    bridge_mask = segments["model_type"] == "bridge"
    if "condition_bmms" in segments.columns:
        # condition_bmms is nullable Int64 from aggregation; assign directly
        segments.loc[bridge_mask, "condition"] = segments.loc[bridge_mask, "condition_bmms"].astype("Int64")
    # if condition_bmms is missing, fill with 0
    segments.loc[bridge_mask & segments["condition"].isna(), "condition"] = 0
    
        
def build_segments(df_roads, df_bmms):
    """Vectorized construction of segments between consecutive lrps per road.

    Returns a DataFrame with columns: road,id,model_type,name,lat,lon,length,condition
    """
    # ensure ordering within each road
    df = df_roads.sort_values(["road", "chainage"]).reset_index(drop=True)

    # next-row values per road
    df["lrp_next"] = df.groupby("road")["lrp"].shift(-1)
    df["chainage_next"] = df.groupby("road")["chainage"].shift(-1)
    df["gap_next"] = df.groupby("road")["gap"].shift(-1)

    # normalize column names for both frames to make merge keys identical
    roads = df.rename(columns=lambda c: c.strip().lower())
    bmms = df_bmms.rename(columns=lambda c: c.strip().lower())

    # accept either 'lrp' or 'lrpname' in BMMS and map to 'lrp'
    if "lrpname" in bmms.columns and "lrp" not in bmms.columns:
        bmms = bmms.rename(columns={"lrpname": "lrp"})

    # require road/lrp/length, but make 'condition' optional (may be absent)
    required = {"road", "lrp", "length"}
    missing = required - set(bmms.columns)
    if missing:
        raise KeyError(f"BMMS file is missing required columns: {sorted(missing)}")

    # if 'condition' is absent in the BMMS file, create it (will be aggregated as missing)
    if "condition" not in bmms.columns:
        bmms["condition"] = pd.NA

    bmms_sub = bmms_aggregate(bmms)
    # merge with validation to catch unexpected duplicates
    df = roads.merge(bmms_sub, on=["road", "lrp"], how="left", validate="many_to_one")

    # Some BMMS values (length/condition) may be recorded on the end LRP rather than the start LRP.
    bmms_backfill(bmms_sub, df)
    # determine model types
    fill_type(df)
    # compute lengths: non-bridges use chainage difference; bridges use BMMS length
    fill_length(df)
    # only keep entries where a next lrp exists (segments between rows)    
    segments = df[df["lrp_next"].notna()].copy()
    # build ids
    segments["id"] = segments["road"].astype(str) + "_" + segments["lrp"].astype(str) + "_" + segments["lrp_next"].astype(str)

    # use aggregated numeric condition codes from BMMS (median) for bridges
    fill_condition(segments)
    
    # select and order columns for output
    # keep lrp fields for possible post-processing (merging consecutive links)
    out = segments[["road", "id", "model_type", "name", "lat", "lon", "length", "condition", "lrp", "lrp_next"]].copy()
    return out


def build_sourcesinks(df_roads):
    """Create start and end rows (sourcesink) per road."""
    df = df_roads.sort_values(["road", "chainage"]).reset_index(drop=True)
    first = df.groupby("road").first().reset_index()
    last = df.groupby("road").last().reset_index()

    starts = pd.DataFrame(
        {
            "road": first["road"],
            "id": first["road"].astype(str) + "_start",
            "model_type": "sourcesink",
            "name": first["road"],
            "lat": first["lat"],
            "lon": first["lon"],
            "length": 0,
            "condition": "",
        }
    )

    ends = pd.DataFrame(
        {
            "road": last["road"],
            "id": last["road"].astype(str) + "_end",
            "model_type": "sourcesink",
            "name": last["road"],
            "lat": last["lat"],
            "lon": last["lon"],
            "length": 0,
            "condition": "",
        }
    )

    return starts, ends



def merge_links(df_out):
    """Merge consecutive 'link' segments in display order into a single link row."""
    merged_rows = []
    for road, group in df_out.groupby("road", sort=False):
        # process rows in appearance order
        acc = None
        for _, row in group.iterrows():
            if row["model_type"] != "link":
                if acc is not None:
                    merged_rows.append(acc)
                    acc = None
                merged_rows.append(row.to_dict())
                continue

            # row is a link
            if acc is None:
                # start accumulating; ensure id/name follow requested format
                acc = row.to_dict()
                start_lrp = acc.get("lrp")
                end_lrp = acc.get("lrp_next")
                acc["id"] = f"{road}_{start_lrp}_{end_lrp}"
                acc["name"] = acc["id"]
                # coerce length to numeric
                acc["length"] = float(acc.get("length") or 0)
            else:
                # merge any consecutive link rows (sequential in appearance)
                end_lrp = row.get("lrp_next")
                acc["id"] = f"{road}_{acc.get('lrp')}_{end_lrp}"
                acc["name"] = acc["id"]
                acc["length"] = float(acc.get("length") or 0) + float(row.get("length") or 0)
                acc["lrp_next"] = end_lrp

        if acc is not None:
            merged_rows.append(acc)

    df_new = pd.DataFrame(merged_rows)
    # drop helper LRP columns if present
    for c in ["lrp", "lrp_next"]:
        if c in df_new.columns:
            df_new = df_new.drop(columns=[c])
    return df_new
                

def main():
    df_roads = pd.read_csv(roads_csv)
    df_bmms = xlsx_tools.open_xlsx(bmms_xlsx)
    
    segments = build_segments(df_roads, df_bmms)
    starts, ends = build_sourcesinks(df_roads)

    df_out = pd.concat([starts, segments, ends], ignore_index=True, sort=False)

    # Optional: merge consecutive 'link' segments into a single link
    if MERGE_CONSECUTIVE_LINKS:
        df_out = merge_links(df_out)

    #For instance keep N1
    #df_out = df_out[df_out["road"] == "N1"]
    
    #Assign road id correctly
    df_out["name"] = df_out["id"]
    road_group = df_out.groupby("road", sort=True)
    road_number = road_group.ngroup() + 1
    element_number = road_group.cumcount()
    df_out["id"] = (road_number * 1_000_000 + element_number).astype("Int64")

    df_out.to_csv(out_csv, index=False)
    print(f"Wrote {len(df_out)} rows to {out_csv}")


if __name__ == "__main__":
    main()