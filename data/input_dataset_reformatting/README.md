# Files

- the ```/data/input_dataset_reformatting/``` folder contains the cleaned dataset from the teachers and the code relative to its reformatting into the ```roads.csv``` file.

The new formatting of the ```roads.csv``` file is the following :

|     Column | Description                                              |
| ---------: | :------------------------------------------------------- |
|       road | On which road does the component belong to               |
|         id | **Unique ID** of the component                           |
| model_type | Type (i.e. class) of the model component to be generated |
|       name | Name of the object                                       |
|        lat | Latitude in Decimal Degrees                              |
|        lon | Longitude in Decimal Degrees                             |
|     length | Length of the object in meters                           |
|  condition | 0 for condition A, 1 for condition B...                  |
|     lengthR| Length of the R component of a bridge                    |
|     lengthL| Length of the L component of a bridge                    |
|  conditionR| Condition of the R component of a bridge                 |
|  conditionL| Condition of the L component of a bridge                 |
|  bridgedual| 1 if bridge has L&R component, empty otherwise           |

- the ```N1road.csv``` file contains all the truncated part of the N1 road from the ```roads.csv``` file, corresponding to the portion between Chittagong and Dhaka



This folder builds `data/roadN1.csv` from:
- road data: `_roads3.csv`
- BMMS bridge data: `BMMS_overview.xlsx`

# Methods

Main entrypoint: `data/cleaned_dataset/fill_demo.py`.

## BMMS preprocessing details (`preprocess_bmms.py`)

### `aggregate_bmms_for_merge`
- Detect side tag (`L`/`R`) from BMMS `name`
- Map condition letters with `CONDITION_MAP` (`A/B/C/D -> 0/1/2/3`).
- Aggregate duplicates by `(road, LRPName)` using median for:
  - `length`, `condition_code`, `lat`, `lon`, `chainage`.
- Build side-specific median fields via pivot:
  - `length_l_bmms`, `length_r_bmms`
  - `condition_l_bmms`, `condition_r_bmms`
- Mark `bridgedual = "1"` when any side tag is detected in `name`.
- Convert condition fields to nullable integer after rounding median values:
  - `condition`, `condition_l_bmms`, `condition_r_bmms` as `Int64`.

### `resolve_duplicates`
- Build duplicate keys:
  - roads: `road + "|" + lrp`
  - BMMS: `road + "|" + LRPName`
- For overlapping keys, apply gap rule on roads rows:
  - if roads `gap` is non-empty, keep roads row and drop BMMS row
  - otherwise keep BMMS row and drop roads row

### `synthesize_roads_like_points_from_bmms`
- For retained BMMS rows, generate two LRPs:
  - start point at `chainage`, `gap = "BS"`
  - end point at `chainage + length/1000`, `gap = "BE"`
- Use BMMS `lat/lon` for the start point, and interpolate end `lat/lon` from roads geometry along chainage.
- Carry bridge attributes (`condition`, `bridgedual`, etc) into generated points.

## Segment construction (`fill_demo.py`)

### `build_segments`
- Sort by `(road, chainage)` and generate `lrp_next`, `chainage_next`, `gap_next` with `shift(-1)`.
- Merge BMMS on `(road, lrp)` vs `(road, LRPName)`.
- Some bridges are only referenced by their ending LRP, so BMMS values are backfilled from that next LRP.
- Assign `model_type`:
  - `BS -> BE` => `bridge`
  - `FS -> FE` => `ferry`
  - else => `link`
- Compute length:
  - default: chainage difference in meters
  - for bridges: BMMS length is preferred; if length is not available, use chainage difference
- Keep only rows with valid `lrp_next`.
- Fill bridge-specific fields:
  - `condition`
  - `bridgedual`
  - side metrics `lengthL`, `lengthR`, `conditionL`, `conditionR`

### Side-metric behavior (`fill_side_metrics`)
- Only applied to rows where `model_type == "bridge"` and `bridgedual` is present.
- Use BMMS values from next LRP first, then current LRP (`combine_first`).
- If only one side exists (only L or only R), mirror it to the missing side, assuming both sides have the same characteristics.

## Source-sinks and link merge

### `build_sourcesinks`
- Adds one `sourcesink` row at the first and last point of each road.
- Uses `_chainage_order` offset (`-1e-6`, `+1e-6`) so start/end stay around segments after sorting.

### `merge_links`
- Keeps non-link rows unchanged.
- Detects consecutive `link` runs per road.
- Merges each run to one row:
  - take first-row metadata
  - set `length` to run sum
  - set `lrp_next` to last row in run
- Rebuild `id`/`name` only for runs with more than one merged link.

### `assign_numeric_ids`
- Replace `id` with numeric id for model use:
  - `id = road_index * 1_000_000 + element_index`

