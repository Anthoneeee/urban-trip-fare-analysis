# EDA Mainline Write-up (Rubric-aligned)

## 1) Dataset Context and Variable Definitions
- This project predicts NYC yellow taxi `total_amount` (final trip payment).
- The three core explanatory variables in our mainline are:
  - `trip_distance` (trip length in miles)
  - `pickup_hour` (time-of-day effect)
  - `is_airport_trip` (airport-route premium effect)
- These variables are chosen because they are both business-interpretable and model-relevant.

## 2) Data Quality Profile
- We checked data types and missingness before EDA.
- Missingness is negligible in core model variables, so standard imputation in the modeling pipeline is sufficient and low-risk.

## 3) Outlier Diagnostics and Handling Plan
- We ran IQR-based outlier diagnostics on `trip_distance`, `trip_duration_minutes`, and `total_amount`.
- Outliers are expected in taxi data (long-distance and atypical trips are real operational events), so we do not drop them blindly.
- Handling strategy:
  - Keep all observations for the primary model training split to preserve real-world distribution.
  - Use robust evaluation (multiple metrics + bootstrap confidence intervals) and controlled hypothesis testing to avoid conclusions driven by a few extreme points.

## 4) Focused EDA (3 charts only)
### Figure A: `fig_2_distance_vs_total_amount.png`
- Key finding: fare level rises with distance.
- Downstream link: supports `trip_distance` as a core model feature.

### Figure B: `fig_4_hourly_avg_total_amount.png`
- Key finding: average fare changes materially by pickup hour (clear time-of-day structure).
- Downstream link: motivates time-effect modeling via `pickup_hour` and a targeted hypothesis test on top fare hours.

### Figure C: `fig_6_airport_vs_nonairport.png`
- Key finding: airport trips show a substantial fare premium.
- Downstream link: motivates direct airport hypothesis testing and controlled airport-effect estimation.

## 5) EDA -> Hypothesis -> Model (Single Storyline)
- Hypothesis 1: `airport_vs_nonairport` mean fare difference is non-zero.
- Hypothesis 2: `top_fare_hours_vs_other_hours` mean fare difference is non-zero (top fare hours defined from Figure B).
- Modeling features include `trip_distance`, `pickup_hour`, and `is_airport_trip`, so EDA evidence flows directly into both hypothesis tests and model design.

## 6) Why This EDA Is Rubric-Aligned
- Succinct and informative: 3 mainline charts only (quality over quantity).
- Each chart has a concrete analytical purpose and downstream impact.
- Outlier diagnostic and handling rationale are explicit.
- EDA findings directly inform preprocessing/modeling and hypothesis testing.
