import re
import numpy as np
import pandas as pd


CONDITION_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}
NAME_SIDE_REGEX = r"(?:\((L|R|LEFT|RIGHT)\)|[-_\s](L|R|LEFT|RIGHT)|\b(L|R|LEFT|RIGHT)\b)\s*\.?$"


def aggregate_bmms_for_merge(df_bmms):
    """
    Aggregate BMMS data for merging with road data.
    Detect left/right sides, extract condition and length, resolve duplicates by median, and mark bridgedual.
    """
    bmms = df_bmms.copy()

    # Detect side flags (L/R) from bridge name.
    extracted = bmms["name"].str.extract(NAME_SIDE_REGEX, flags=re.IGNORECASE)
    side_raw = extracted.bfill(axis=1).iloc[:, 0].str.upper()
    bmms["side_tag"] = side_raw.map({"L": "L", "LEFT": "L", "R": "R", "RIGHT": "R"})

    bmms["condition_code"] = bmms["condition"].map(CONDITION_MAP)

    # 3) Keep valid rows and aggregate duplicates per (road, LRPName).

    # Merge duplicates by median
    grouped = bmms.groupby(["road", "LRPName"], as_index=False)[["length", "condition_code", "lat", "lon", "chainage"]].median()

    side_rows = bmms[bmms["side_tag"].isin(["L", "R"])].copy()
    side_agg = (
        side_rows.groupby(["road", "LRPName", "side_tag"], as_index=False)[["length", "condition_code"]].median()
        .pivot(index=["road", "LRPName"], columns="side_tag", values=["length", "condition_code"])
    )
    if not side_agg.empty:
        side_agg.columns = [
            "length_l_bmms" if (metric == "length" and side == "L") else
            "length_r_bmms" if (metric == "length" and side == "R") else
            "condition_l_bmms" if (metric == "condition_code" and side == "L") else
            "condition_r_bmms"
            for metric, side in side_agg.columns
        ]
        side_agg = side_agg.reset_index()
        grouped = grouped.merge(side_agg, on=["road", "LRPName"], how="left")

    for col in ["length_l_bmms", "length_r_bmms", "condition_l_bmms", "condition_r_bmms"]:
        if col not in grouped.columns:
            grouped[col] = np.nan

    # Mark bridgedual if any side tag is present 
    side_keys = side_rows[["road", "LRPName"]].drop_duplicates()
    side_keys["bridgedual"] = "1"
    grouped = grouped.merge(side_keys, on=["road", "LRPName"], how="left")

    # Convert median output to int
    grouped["condition"] = grouped["condition_code"].round().astype("Int64")
    grouped["condition_l_bmms"] = grouped["condition_l_bmms"].round().astype("Int64")
    grouped["condition_r_bmms"] = grouped["condition_r_bmms"].round().astype("Int64")

    merge_bmms = grouped[["road", "LRPName", "length", "condition", "bridgedual", "length_l_bmms", "length_r_bmms", "condition_l_bmms", "condition_r_bmms"]].copy()
    merge_bmms = merge_bmms.rename(
        columns={
            "length": "length_bmms",
            "condition": "condition_bmms",
            "bridgedual": "bridgedual_bmms",
        }
    )

    return grouped, merge_bmms


def resolve_duplicates(df_roads, bmms_grouped):
    """
    Resolve duplicates between roads and BMMS:
        If a duplicate (road, LRP) key exists in both datasets
        keep the BMMS row if its gap is non-empty, otherwise keep the road row.
    """
    roads = df_roads.copy()
    bmms = bmms_grouped.copy()

    # Create a composite key for duplicate detection.
    roads["_dup_key"] = roads["road"] + "|" + roads["lrp"]
    bmms["_dup_key"] = bmms["road"] + "|" + bmms["LRPName"]

    # Identify overlapping keys.
    overlap = set(roads["_dup_key"]).intersection(set(bmms["_dup_key"]))
    if not overlap:
        # If no duplicates, simply return original datasets without the temporary keys.
        return roads.drop(columns=["_dup_key"]), bmms.drop(columns=["_dup_key"])

    # Filter roads and BMMS to identify which rows to keep based on the gap rule.
    roads_overlap = roads[roads["_dup_key"].isin(overlap)].copy()
    non_empty_gap = roads_overlap["gap"].notna() & (roads_overlap["gap"] != "")
    keys_keep_roads = set(roads_overlap.loc[non_empty_gap, "_dup_key"])

    keys_drop_roads = overlap - keys_keep_roads

    roads_filtered = roads[~roads["_dup_key"].isin(keys_drop_roads)].copy()
    bmms_filtered = bmms[~bmms["_dup_key"].isin(keys_keep_roads)].copy()

    return roads_filtered.drop(columns=["_dup_key"]), bmms_filtered.drop(columns=["_dup_key"])


def synthesize_roads_like_points_from_bmms(roads_df, bmms_grouped):
    """
    Generate BS/BE LRP points from BMMS rows for merging with roads:
    For each BMMS LRP, create two points:
        - Start point (BS) at the original chainage with the original lat/lon.
        - End point (BE) at the chainage + length with lat/lon interpolated along the road geometry.
    
    """
    columns = ["road", "lrp", "chainage", "lat", "lon", "gap", "name", "condition", "bridgedual"]
    if bmms_grouped.empty:
        return pd.DataFrame(columns=columns)

    roads_by_road = {}
    for road, group in roads_df.groupby("road", sort=False):
        geom = group[["road", "lrp", "chainage", "lat", "lon"]].copy()
        geom = geom.sort_values("chainage")
        roads_by_road[road] = geom

    generated_parts = []
    
    # Iterate over BMMS roads to generate BS/BE points.
    for road, bmms_part in bmms_grouped.groupby("road", sort=False):
        road_geom = roads_by_road.get(road)
        if road_geom is None or road_geom.empty:
            continue

        part = bmms_part.copy()

        part["end_chainage"] = part["chainage"] + part["length"] / 1000.0

        ch = road_geom["chainage"].to_numpy(dtype=float)
        lat = road_geom["lat"].to_numpy(dtype=float)
        lon = road_geom["lon"].to_numpy(dtype=float)

        end_clip = np.clip(part["end_chainage"].to_numpy(dtype=float), ch[0], ch[-1])

        # Interpolate BE coordinates along roads geometry
        part["end_lat"] = np.interp(end_clip, ch, lat)
        part["end_lon"] = np.interp(end_clip, ch, lon)

        lrp_start = part["LRPName"]
        lrp_end = lrp_start + "_E"
        condition_vals = part["condition"].fillna(0)
    
        # Create start and end points with appropriate attributes for merging with roads.
        start_df = pd.DataFrame(
            {
                "road": road,
                "lrp": lrp_start,
                "chainage": part["chainage"],
                "lat": part["lat"],
                "lon": part["lon"],
                "gap": "BS",
                "name": "BMMS_" + lrp_start + "_S",
                "condition": condition_vals.values,
                "bridgedual": part["bridgedual"].values,
            }
        )

        end_df = pd.DataFrame(
            {
                "road": road,
                "lrp": lrp_end,
                "chainage": part["end_chainage"],
                "lat": part["end_lat"],
                "lon": part["end_lon"],
                "gap": "BE",
                "name": "BMMS_" + lrp_start + "_E",
                "condition": condition_vals.values,
                "bridgedual": part["bridgedual"].values,
            }
        )
        generated_parts.extend([start_df, end_df])

    # If no valid BMMS rows or no matching roads, return empty DataFrame with correct columns.
    if not generated_parts:
        return pd.DataFrame(columns=columns)

    return pd.concat(generated_parts, ignore_index=True, sort=False)[columns]


def preprocess(df_roads, df_bmms):
    # Merge duplicate and detect left/right in BMMS
    bmms_grouped, bmms_merge_all = aggregate_bmms_for_merge(df_bmms)
    
    # Resolve duplicates between roads and BMMS
    roads_filtered, bmms_filtered = resolve_duplicates(df_roads, bmms_grouped)
    
    # Generate BS/BE points from BMMS for merging with roads
    bmms_generated_points = synthesize_roads_like_points_from_bmms(roads_filtered, bmms_filtered)
    
    # Final merge and sort by road and chainage.
    roads_augmented = pd.concat([roads_filtered, bmms_generated_points], ignore_index=True, sort=False)
    roads_augmented = roads_augmented.sort_values(["road", "chainage"], kind="mergesort").reset_index(drop=True)

    return roads_augmented, bmms_merge_all
