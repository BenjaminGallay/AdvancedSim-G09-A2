import os
import numpy as np
import pandas as pd
import xlsx_tools
import preprocess_bmms


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
roads_csv = os.path.join(BASE_DIR, "data", "input_dataset_reformatting", "_roads3.csv")
bmms_xlsx = os.path.join(BASE_DIR, "data", "input_dataset_reformatting", "BMMS_overview.xlsx")
out_csv = os.path.join(BASE_DIR, "data", "roads.csv")

def bmms_backfill(bmms_sub, df):
    bmms_next_map = {
        "length_bmms": "length_bmms_next",
        "condition_bmms": "condition_bmms_next",
        "length_l_bmms": "length_l_bmms_next",
        "length_r_bmms": "length_r_bmms_next",
        "condition_l_bmms": "condition_l_bmms_next",
        "condition_r_bmms": "condition_r_bmms_next",
    }

    # Create a column containing the next LRP for each row in df.
    bmms_next = bmms_sub[["road", "LRPName", *bmms_next_map.keys()]].rename(
        columns={"LRPName": "lrp_next", **bmms_next_map}
    )
    df_next = df.merge(bmms_next, on=["road", "lrp_next"], how="left", validate="many_to_one")

    next_cols = list(bmms_next_map.values())
    df[next_cols] = df_next[next_cols]

    df[list(bmms_next_map.keys())] = df[list(bmms_next_map.keys())].combine_first(
        df[next_cols].rename(columns={next_name: name for name, next_name in bmms_next_map.items()})
    )



######### Simple fills #########
def fill_type(df):
    # Get the type of each segment by looking at the gap and gap_next columns.
    is_bridge = (df["gap"] == "BS") & (df["gap_next"] == "BE")
    is_ferry = (df["gap"] == "FS") & (df["gap_next"] == "FE")
    
    # By default, fill links.
    df["model_type"] = "link"

    df.loc[is_bridge, "model_type"] = "bridge"
    df.loc[is_ferry, "model_type"] = "ferry"


def fill_length(df):
    # Calculate length from chainage, then override with BMMS length for bridges where available.
    df["length_calc"] = (df["chainage_next"] - df["chainage"]) * 1000.0
    # Check whether BMMS gives the length for the start or end LRP of the bridge.
    bridge_length = df["length_bmms_next"].combine_first(df["length_bmms"])
    df["length"] = np.where(df["model_type"] == "bridge", bridge_length, df["length_calc"])
    fallback = (df["model_type"] == "bridge") & df["length"].isna()
    if fallback.any():
        df.loc[fallback, "length"] = df.loc[fallback, "length_calc"]


def fill_condition(segments):
    # For non-bridge segments, condition is not defined (NA). 
    # For bridge segments, condition is taken from BMMS where available, with backfill from next segment. 
    # If still missing, default to 0.
    segments["condition"] = pd.NA
    bridge_mask = segments["model_type"] == "bridge"
    # Check whether BMMS gives the condition for the start or end LRP of the bridge.
    bridge_condition = segments["condition_bmms_next"].combine_first(segments["condition_bmms"])
    segments.loc[bridge_mask, "condition"] = bridge_condition.loc[bridge_mask]
    segments.loc[bridge_mask & segments["condition"].isna(), "condition"] = 0


def fill_bridgedual(segments):
    # For bridge segments, bridgedual is taken from BMMS where available, with backfill from next segment.
    # For non-bridge segments, bridgedual is not defined (NA).
    bridge_mask = segments["model_type"] == "bridge"
    segments["bridgedual"] = segments["bridgedual"].combine_first(segments["bridgedual_bmms"])
    # bridgedual applies only to bridge rows
    segments.loc[~bridge_mask, "bridgedual"] = pd.NA


def fill_side_metrics(segments):
    # For dual bridges, fill left/right length and condition from BMMS where available
    # If only one side is present in BMMS, mirror it to the other side. 
    # For non-bridge or single bridges, these remain NA.
    bridge_dual_mask = (segments["model_type"] == "bridge") & segments["bridgedual"].notna()

    side_from_next = segments[["length_l_bmms_next", "length_r_bmms_next", "condition_l_bmms_next", "condition_r_bmms_next"]].rename(
        columns={
            "length_l_bmms_next": "lengthL",
            "length_r_bmms_next": "lengthR",
            "condition_l_bmms_next": "conditionL",
            "condition_r_bmms_next": "conditionR",
        }
    )
    side_from_current = segments[["length_l_bmms", "length_r_bmms", "condition_l_bmms", "condition_r_bmms"]].rename(
        columns={
            "length_l_bmms": "lengthL",
            "length_r_bmms": "lengthR",
            "condition_l_bmms": "conditionL",
            "condition_r_bmms": "conditionR",
        }
    )

    # If a value is available only from next, backfill it to current.
    side_values = side_from_next.combine_first(side_from_current)
    
    segments[["lengthL", "lengthR", "conditionL", "conditionR"]] = side_values.where(bridge_dual_mask, pd.NA)

    # If only one side exists in BMMS, mirror values to the other side.
    segments.loc[bridge_dual_mask, "lengthL"] = segments.loc[bridge_dual_mask, "lengthL"].combine_first(segments.loc[bridge_dual_mask, "lengthR"])
    segments.loc[bridge_dual_mask, "lengthR"] = segments.loc[bridge_dual_mask, "lengthR"].combine_first(segments.loc[bridge_dual_mask, "lengthL"])
    segments.loc[bridge_dual_mask, "conditionL"] = segments.loc[bridge_dual_mask, "conditionL"].combine_first(segments.loc[bridge_dual_mask, "conditionR"])
    segments.loc[bridge_dual_mask, "conditionR"] = segments.loc[bridge_dual_mask, "conditionR"].combine_first(segments.loc[bridge_dual_mask, "conditionL"])
    

def assign_numeric_ids(df_out):
    group = df_out.groupby("road", sort=True)
    road_number = group.ngroup() + 1
    element_number = group.cumcount()
    df_out["id"] = road_number * 1_000_000 + element_number
    return df_out

########### Main processing #########
def build_segments(df_roads, bmms_sub):
    # Build segment rows from consecutive road points.
    # Each row represents one directed segment: (road, lrp) -> (road, lrp_next).
    df = df_roads.sort_values(["road", "chainage"], kind="mergesort").reset_index(drop=True)

    # 1) Compute next-point fields per road.
    #    The last point of each road has no next point, so it cannot form a segment.
    df["lrp_next"] = df.groupby("road")["lrp"].shift(-1)
    df["chainage_next"] = df.groupby("road")["chainage"].shift(-1)
    df["gap_next"] = df.groupby("road")["gap"].shift(-1)

    # 2) Join BMMS metadata on (road, current lrp), then backfill from next lrp.
    #    This handles cases where BMMS data is recorded on the bridge end LRP.
    df = df.merge(bmms_sub, left_on=["road", "lrp"], right_on=["road", "LRPName"], how="left", validate="many_to_one")

    bmms_backfill(bmms_sub, df)

    # 3) Derive core segment properties.
    #    - type from gap pattern
    #    - length from chainage (or BMMS for bridges)
    fill_type(df)
    fill_length(df)

    # 4) Keep only rows with a valid next point (= valid segment starts).
    segments = df[df["lrp_next"].notna()].copy()
    segments["id"] = segments["road"] + "_" + segments["lrp"] + "_" + segments["lrp_next"]

    # 5) Fill bridge-only fields.
    fill_condition(segments)
    fill_bridgedual(segments)
    fill_side_metrics(segments)

    segments["_chainage_order"] = segments["chainage"]

    return segments[["road","id","model_type","name","lat","lon","length","condition","lengthR","lengthL","conditionR","conditionL","bridgedual","lrp","lrp_next","_chainage_order"]].copy()


def build_sourcesinks(df_roads):
    df = df_roads.sort_values(["road", "chainage"], kind="mergesort").reset_index(drop=True)
    
    # Get first and last row for each road.
    first = df.groupby("road").first().reset_index()
    last = df.groupby("road").last().reset_index()

    # Create sourcesink rows for the start and end of each road.
    starts = pd.DataFrame(
        {
            "road": first["road"],
            "id": first["road"] + "_start",
            "model_type": "sourcesink",
            "name": first["road"],
            "lat": first["lat"],
            "lon": first["lon"],
            "length": 0,
            "condition": "",
            "lengthR": pd.NA,
            "lengthL": pd.NA,
            "conditionR": pd.NA,
            "conditionL": pd.NA,
            "bridgedual": pd.NA,
            "_chainage_order": first["chainage"] - 1e-6,
        }
    )

    ends = pd.DataFrame(
        {
            "road": last["road"],
            "id": last["road"] + "_end",
            "model_type": "sourcesink",
            "name": last["road"],
            "lat": last["lat"],
            "lon": last["lon"],
            "length": 0,
            "condition": "",
            "lengthR": pd.NA,
            "lengthL": pd.NA,
            "conditionR": pd.NA,
            "conditionL": pd.NA,
            "bridgedual": pd.NA,
            "_chainage_order": last["chainage"] + 1e-6,
        }
    )

    return starts, ends


def merge_links(df_out):
    # Merge consecutive "link" segments on the same road into longer links.
    # Bridges/ferries/sourcesinks are kept as-is.
    work = df_out.copy()
    # Preserve original global order so merged output can be restored consistently.
    work["_order"] = np.arange(len(work))

    # Identify link rows only.
    is_link = work["model_type"].eq("link")

    # Start a new run whenever a link row does not directly follow another link row on the same road.
    run_start = is_link & (~is_link.groupby(work["road"]).shift(fill_value=False))
    work["_run_id"] = run_start.groupby(work["road"]).cumsum()
    work.loc[~is_link, "_run_id"] = pd.NA

    # Keep non-links untouched; only process link runs.
    non_links = work.loc[~is_link].copy()
    link_rows = work.loc[is_link].copy()

    # Merge each run to one row: keep first row metadata, sum length, use last lrp_next.
    if not link_rows.empty:
        grouped = link_rows.groupby(["road", "_run_id"], sort=False, as_index=False)
        first = grouped.first()
        run_len = grouped.size().rename(columns={"size": "_run_len"})
        length_sum = grouped["length"].sum(min_count=1).rename(columns={"length": "_length_sum"})
        last_next = grouped.last()[["road", "_run_id", "lrp_next"]].rename(columns={"lrp_next": "_last_lrp_next"})

        merged_links = first.merge(run_len, on=["road", "_run_id"], how="left")
        merged_links = merged_links.merge(length_sum, on=["road", "_run_id"], how="left")
        merged_links = merged_links.merge(last_next, on=["road", "_run_id"], how="left")
        merged_links["length"] = merged_links["_length_sum"]
        merged_links["lrp_next"] = np.where(merged_links["_run_len"] > 1, merged_links["_last_lrp_next"], merged_links["lrp_next"])

        # Rebuild id/name only for rows that actually merged 2+ links.
        multi = merged_links["_run_len"] > 1
        merged_links.loc[multi, "id"] = merged_links.loc[multi, "road"] + "_" + merged_links.loc[multi, "lrp"] + "_" + merged_links.loc[multi, "lrp_next"]
        merged_links.loc[multi, "name"] = merged_links.loc[multi, "id"]

        merged_links = merged_links[work.columns]
        out = pd.concat([non_links, merged_links], ignore_index=True, sort=False)
    else:
        out = work

    # Restore original ordering and remove temporary helper columns.
    out = out.sort_values("_order", kind="mergesort").reset_index(drop=True)
    out = out.drop(columns=["_order", "_run_id", "lrp", "lrp_next"])
    return out


def main():
    # Load raw roads3 and BMMS inputs.
    roads_raw = pd.read_csv(roads_csv)
    bmms_raw = xlsx_tools.open_xlsx(bmms_xlsx)
    print(f"Opened {roads_csv} and {bmms_xlsx}")

    # Preprocess BMMS into roads3-like points, resolve duplicates, and prepare BMMS merge table.
    roads_preprocessed, bmms_for_merge = preprocess_bmms.preprocess(roads_raw, bmms_raw)
    print(f'Preprocessed roads and BMMS data')
    
    # Build simulation rows: start, segments, end.
    segments = build_segments(roads_preprocessed, bmms_for_merge)
    starts, ends = build_sourcesinks(roads_preprocessed)

    df_out = pd.concat([starts, segments, ends], ignore_index=True, sort=False)
    df_out = df_out.sort_values(["road", "_chainage_order"], kind="mergesort").reset_index(drop=True)
    
    # Merge links for efficiency.
    df_out = merge_links(df_out)
    print(f'Merged links')

    # Finalize output IDs.
    df_out["name"] = df_out["id"]
    df_out = assign_numeric_ids(df_out)

    df_out = df_out.drop(columns=["_chainage_order"])

    df_out.to_csv(out_csv, index=False)
    print(f"Wrote {len(df_out)} rows to {out_csv}")


if __name__ == "__main__":
    main()
