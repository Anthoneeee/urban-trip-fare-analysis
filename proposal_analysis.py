import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib

# 使用无界面后端，避免 PyCharm 交互后端兼容性问题。
# Use a non-interactive backend to avoid compatibility issues with the PyCharm interactive backend.
MPL_CONFIG_DIR = Path(".mplconfig")
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR.resolve()))
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Proposal Analysis Script for NYC Taxi 2024 Full-Year Sample
# ------------------------------------------------------------
# 这个脚本做的事情：
# What this script does:
# 1. 读取你已经做好的 2024 全年样本集 CSV
# 1. Read your prepared 2024 full-year sample CSV
# 2. 查看数据结构、数据类型、缺失值、描述统计
# 2. Inspect data structure, data types, missing values, and descriptive statistics
# 3. 生成 proposal 阶段最关键的 EDA 图
# 3. Generate the most important EDA plots for the proposal stage
# 4. 做两个简单的 permutation test（假设检验）
# 4. Run two simple permutation tests (hypothesis tests)
# 5. 读取独立模型脚本输出的建模结果（不在此脚本重复训练）
# 5. Load modeling results produced by the separate model script (no retraining here)
# 6. 把图和部分结果保存到本地文件夹
# 6. Save figures and selected results to a local folder
# ============================================================


# =========================
# 0. 路径设置
# 0. Path setup
# =========================
DATA_PATH = "nyc_taxi_2024_cleaned_sample.csv"

# 输出文件夹：图和结果会保存在这里
# Output folder: figures and results will be saved here.
OUTPUT_DIR = Path("proposal_outputs_2024")
OUTPUT_DIR.mkdir(exist_ok=True)

# 清理已弃用的 EDA 输出，避免旧文件干扰最终提交。
# Remove deprecated EDA outputs so stale files do not confuse final deliverables.
DEPRECATED_EDA_FILES = [
    "fig_1_total_amount_distribution.png",
    "fig_3_duration_vs_total_amount.png",
    "fig_5_borough_avg_total_amount.png",
    "fig_7_weekday_vs_weekend.png",
    "borough_avg_total_amount.csv",
    "weekend_avg_total_amount.csv",
    "hypothesis_test_controlled_results.csv",
    "fig_14_controlled_hypothesis_permutation.png",
    "fig_8_hourly_cost_efficiency.png",
    "fig_9_airport_gap_by_distance_band.png",
    "fig_10_hourly_price_risk_quantiles.png",
    "fig_11_top_routes_mean_total_amount.png",
    "fig_12_monthly_decomposition.png",
    "hour_efficiency_summary.csv",
    "airport_by_distance_band.csv",
    "airport_controlled_effect.csv",
    "bootstrap_controlled_airport_ci.csv",
    "hourly_risk_quantiles.csv",
    "route_structure_summary.csv",
    "monthly_decomposition.csv",
    "monthly_weekend_gap.csv",
]
for filename in DEPRECATED_EDA_FILES:
    stale_path = OUTPUT_DIR / filename
    if stale_path.exists():
        stale_path.unlink()


# =========================
# 0.5 Bootstrap 工具函数
# 0.5 Bootstrap utility function
# =========================
def bootstrap_mean_diff_ci(values_a, values_b, n_boot=3000, random_state=42):
    """
    对两组均值差（a - b）做 bootstrap，返回 95% CI。
    Run bootstrap for the mean difference (a - b) and return a 95% confidence interval.
    """
    values_a = np.asarray(values_a)
    values_b = np.asarray(values_b)

    n_a = len(values_a)
    n_b = len(values_b)
    rng = np.random.default_rng(random_state)
    boot_diffs = np.empty(n_boot)

    for i in range(n_boot):
        sample_a = values_a[rng.integers(0, n_a, n_a)]
        sample_b = values_b[rng.integers(0, n_b, n_b)]
        boot_diffs[i] = sample_a.mean() - sample_b.mean()

    return {
        "bootstrap_mean": float(boot_diffs.mean()),
        "bootstrap_std": float(boot_diffs.std(ddof=1)),
        "ci_lower_95": float(np.quantile(boot_diffs, 0.025)),
        "ci_upper_95": float(np.quantile(boot_diffs, 0.975)),
    }


def summarize_iqr_outliers(df, cols):
    """
    使用 IQR 规则对关键数值变量做异常值诊断。
    Diagnose outliers for key numeric variables using the IQR rule.
    """
    rows = []
    for col in cols:
        s = df[col].dropna()
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask = (s < lower) | (s > upper)
        rows.append({
            "column": col,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": int(outlier_mask.sum()),
            "outlier_pct": float((outlier_mask.mean() * 100.0)),
        })
    return pd.DataFrame(rows)


# =========================
# 1. 读取数据
# 1. Load data
# =========================
print("=" * 60)
print("1. Reading data...")
print("=" * 60)

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"]
)

print("Data loaded successfully.")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nHead:")
print(df.head())


# =========================
# 2. 数据整体情况
# 2. Overall data profile
# =========================
print("\n" + "=" * 60)
print("2. Data overview and quality check")
print("=" * 60)

print("\nData types:")
print(df.dtypes)

print("\nMissing values count:")
print(df.isna().sum())

print("\nMissing values percentage:")
print((df.isna().mean() * 100).round(2))

# 保存缺失值信息
# Save missing-value summary.
missing_df = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_pct": (df.isna().mean() * 100).round(2)
}).reset_index().rename(columns={"index": "column"})
missing_df.to_csv(OUTPUT_DIR / "missing_summary.csv", index=False)

# 查看月份分布
# Check month distribution.
print("\nPickup month distribution:")
print(df["pickup_month"].value_counts().sort_index())

# 查看周末/工作日比例
# Check weekend/weekday proportion.
print("\nis_weekend distribution:")
print(df["is_weekend"].value_counts(dropna=False))

# 查看时间段分布
# Check time-block distribution.
print("\nhour_block distribution:")
print(df["hour_block"].value_counts(dropna=False))


# =========================
# 3. 数值变量描述统计
# 3. Descriptive statistics for numeric variables
# =========================
print("\n" + "=" * 60)
print("3. Descriptive statistics")
print("=" * 60)

numeric_cols = [
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "tip_amount",
    "total_amount",
    "trip_duration_minutes",
    "Airport_fee"
]

desc = df[numeric_cols].describe()
print(desc)

desc.to_csv(OUTPUT_DIR / "numeric_describe.csv")


# =========================
# 3.5 异常值诊断（IQR）
# 3.5 Outlier diagnostics (IQR)
# =========================
print("\n" + "=" * 60)
print("3.5 Outlier diagnostics by IQR")
print("=" * 60)

outlier_cols = ["trip_distance", "trip_duration_minutes", "total_amount"]
outlier_df = summarize_iqr_outliers(df, outlier_cols)
print(outlier_df)
outlier_df.to_csv(OUTPUT_DIR / "outlier_iqr_summary.csv", index=False)


# =========================
# 4. 聚焦 EDA（只保留 3 张主线图）
# 4. Focused EDA (keep only 3 core plots for model storyline)
# =========================
print("\n" + "=" * 60)
print("4. Focused EDA: keep three model-aligned figures only")
print("=" * 60)

# 图 A：trip_distance vs total_amount（对应模型中的 trip_distance）
# Figure A: trip_distance vs total_amount (aligned with model feature trip_distance).
plot_df = df.sample(min(5000, len(df)), random_state=42)

plt.figure(figsize=(8, 5))
plt.scatter(plot_df["trip_distance"], plot_df["total_amount"], s=8, alpha=0.35)
plt.xlabel("trip_distance")
plt.ylabel("total_amount")
plt.title("trip_distance vs total_amount")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_2_distance_vs_total_amount.png", dpi=180)
plt.close()

# 图 B：按小时平均 total_amount（保留老师认可的主图）
# Figure B: average total_amount by hour (the figure specifically recommended by feedback).
hour_avg = (
    df.groupby("pickup_hour", as_index=False)["total_amount"]
      .mean()
      .sort_values("pickup_hour")
)
print("\nHourly mean total_amount:")
print(hour_avg)
hour_avg.to_csv(OUTPUT_DIR / "hour_avg_total_amount.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(hour_avg["pickup_hour"], hour_avg["total_amount"], marker="o")
plt.xlabel("pickup_hour")
plt.ylabel("mean total_amount")
plt.title("Average total_amount by pickup_hour")
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_4_hourly_avg_total_amount.png", dpi=180)
plt.close()

# 图 C：机场 vs 非机场（对应模型中的 is_airport_trip）
# Figure C: airport vs non-airport (aligned with model feature is_airport_trip).
airport_avg = (
    df.groupby("is_airport_trip", as_index=False)
      .agg(
          mean_total_amount=("total_amount", "mean"),
          trip_count=("total_amount", "size")
      )
)
print("\nAirport vs non-airport mean total_amount:")
print(airport_avg)
airport_avg.to_csv(OUTPUT_DIR / "airport_avg_total_amount.csv", index=False)

plt.figure(figsize=(6, 5))
plt.bar(airport_avg["is_airport_trip"].astype(str), airport_avg["mean_total_amount"])
plt.xlabel("is_airport_trip")
plt.ylabel("mean total_amount")
plt.title("Average total_amount: airport vs non-airport")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_6_airport_vs_nonairport.png", dpi=180)
plt.close()


# =========================
# 4.5 EDA 主线定义（用于后续假设检验与建模叙事）
# 4.5 Define EDA storyline links for downstream hypothesis/model narrative
# =========================
TOP_HOUR_COUNT = 3
top_fare_hour_df = (
    hour_avg.sort_values("total_amount", ascending=False)
    .head(TOP_HOUR_COUNT)
    .copy()
)
top_fare_hours = top_fare_hour_df["pickup_hour"].astype(int).tolist()
df["is_top_fare_hour"] = df["pickup_hour"].isin(top_fare_hours)

print("\nTop fare hours derived from EDA fig_4:", top_fare_hours)
top_fare_hour_df.to_csv(OUTPUT_DIR / "top_fare_hours_from_eda.csv", index=False)

lineage_df = pd.DataFrame([
    {
        "eda_figure": "fig_2_distance_vs_total_amount.png",
        "eda_key_variable": "trip_distance",
        "linked_model_feature": "trip_distance",
        "linked_hypothesis_test": "used as supporting covariate in both hypothesis narratives",
        "purpose": "fare level grows with distance, supports distance as core predictor"
    },
    {
        "eda_figure": "fig_4_hourly_avg_total_amount.png",
        "eda_key_variable": "pickup_hour",
        "linked_model_feature": "pickup_hour",
        "linked_hypothesis_test": "top_fare_hours_vs_other_hours",
        "purpose": "hourly fare pattern motivates explicit time effect testing"
    },
    {
        "eda_figure": "fig_6_airport_vs_nonairport.png",
        "eda_key_variable": "is_airport_trip",
        "linked_model_feature": "is_airport_trip",
        "linked_hypothesis_test": "airport_vs_nonairport",
        "purpose": "airport premium pattern motivates a direct hypothesis test"
    },
])
lineage_df.to_csv(OUTPUT_DIR / "eda_model_hypothesis_lineage.csv", index=False)


# =========================
# 11. 相关系数分析
# 11. Correlation analysis
# =========================
print("\n" + "=" * 60)
print("11. Correlation analysis")
print("=" * 60)

corr_cols = [
    "trip_distance",
    "fare_amount",
    "tip_amount",
    "total_amount",
    "trip_duration_minutes",
    "pickup_hour",
    "pickup_weekday"
]

corr_matrix = df[corr_cols].corr()
print(corr_matrix)

corr_matrix.to_csv(OUTPUT_DIR / "correlation_matrix.csv")


# =========================
# 12. 假设检验 1：机场 vs 非机场
# 12. Hypothesis test 1: airport vs non-airport
# =========================
print("\n" + "=" * 60)
print("12. Permutation test: airport vs non-airport")
print("=" * 60)

observed_diff_airport = (
    df[df["is_airport_trip"] == True]["total_amount"].mean()
    - df[df["is_airport_trip"] == False]["total_amount"].mean()
)

n_perm = 10000
count = 0
rng = np.random.default_rng(42)

values = df["total_amount"].to_numpy()
labels = df["is_airport_trip"].to_numpy()

for _ in range(n_perm):
    shuffled = rng.permutation(labels)
    diff = values[shuffled == True].mean() - values[shuffled == False].mean()
    if abs(diff) >= abs(observed_diff_airport):
        count += 1

p_value_airport = (count + 1) / (n_perm + 1)

print("Observed difference:", observed_diff_airport)
print("Permutation p-value:", p_value_airport)


# =========================
# 13. 假设检验 2：EDA 高价小时段 vs 其他小时段
# 13. Hypothesis test 2: top fare hours from EDA vs other hours
# =========================
print("\n" + "=" * 60)
print("13. Permutation test: top fare hours vs other hours")
print("=" * 60)

observed_diff_top_hours = (
    df[df["is_top_fare_hour"] == True]["total_amount"].mean()
    - df[df["is_top_fare_hour"] == False]["total_amount"].mean()
)

count = 0
labels = df["is_top_fare_hour"].to_numpy()

for _ in range(n_perm):
    shuffled = rng.permutation(labels)
    diff = values[shuffled == True].mean() - values[shuffled == False].mean()
    if abs(diff) >= abs(observed_diff_top_hours):
        count += 1

p_value_top_hours = (count + 1) / (n_perm + 1)

print("Top fare hours (from fig_4):", top_fare_hours)
print("Observed difference:", observed_diff_top_hours)
print("Permutation p-value:", p_value_top_hours)

# 保存 hypothesis test 结果
# Save hypothesis test results.
airport_true_mask = df["is_airport_trip"] == True
airport_false_mask = df["is_airport_trip"] == False
top_hour_true_mask = df["is_top_fare_hour"] == True
top_hour_false_mask = df["is_top_fare_hour"] == False

hypothesis_df = pd.DataFrame([
    {
        "test": "airport_vs_nonairport",
        "null_hypothesis": "mean(total_amount | airport) = mean(total_amount | non-airport)",
        "alternative_hypothesis": "mean(total_amount | airport) != mean(total_amount | non-airport)",
        "group_a_label": "airport",
        "group_b_label": "non_airport",
        "group_a_n": int(airport_true_mask.sum()),
        "group_b_n": int(airport_false_mask.sum()),
        "group_a_mean": float(df.loc[airport_true_mask, "total_amount"].mean()),
        "group_b_mean": float(df.loc[airport_false_mask, "total_amount"].mean()),
        "observed_difference_a_minus_b": observed_diff_airport,
        "p_value": p_value_airport,
        "n_perm": n_perm,
    },
    {
        "test": "top_fare_hours_vs_other_hours",
        "null_hypothesis": "mean(total_amount | top fare hours) = mean(total_amount | other hours)",
        "alternative_hypothesis": "mean(total_amount | top fare hours) != mean(total_amount | other hours)",
        "group_a_label": "top_fare_hours",
        "group_b_label": "other_hours",
        "group_a_n": int(top_hour_true_mask.sum()),
        "group_b_n": int(top_hour_false_mask.sum()),
        "group_a_mean": float(df.loc[top_hour_true_mask, "total_amount"].mean()),
        "group_b_mean": float(df.loc[top_hour_false_mask, "total_amount"].mean()),
        "observed_difference_a_minus_b": observed_diff_top_hours,
        "p_value": p_value_top_hours,
        "n_perm": n_perm,
    },
])
hypothesis_df.to_csv(OUTPUT_DIR / "hypothesis_test_results.csv", index=False)


# =========================
# 14. 读取模型结果（模型训练已拆分到独立脚本）
# 14. Load model outputs (training has been split into a separate script)
# =========================
print("\n" + "=" * 60)
print("14. Load modeling outputs from separate script")
print("=" * 60)

model_result_path = OUTPUT_DIR / "model_results_detailed.csv"
model_result_simple_path = OUTPUT_DIR / "model_results.csv"
model_bootstrap_path = OUTPUT_DIR / "bootstrap_model_metric_ci.csv"

if model_result_path.exists():
    results_df = pd.read_csv(model_result_path)
    print("Loaded:", model_result_path)
    print(results_df[["model", "RMSE", "MAE", "R2"]].head())
else:
    print("Model results not found.")
    print("Please run `python proposal_model_training.py` first.")

if model_result_simple_path.exists():
    print("Loaded:", model_result_simple_path)

if model_bootstrap_path.exists():
    model_bootstrap_df = pd.read_csv(model_bootstrap_path)
    print("Loaded:", model_bootstrap_path)
    print(model_bootstrap_df.head(6))


# =========================
# 14.5 Bootstrap：差值区间 + 指标区间
# 14.5 Bootstrap: difference intervals + metric intervals
# =========================
print("\n" + "=" * 60)
print("14.5 Bootstrapping for uncertainty intervals")
print("=" * 60)

N_BOOT_DIFF = 3000

# 对假设检验中的均值差做 bootstrap CI
# Compute bootstrap CIs for mean differences in the hypothesis tests.
airport_true_vals = df.loc[df["is_airport_trip"] == True, "total_amount"].to_numpy()
airport_false_vals = df.loc[df["is_airport_trip"] == False, "total_amount"].to_numpy()
airport_ci = bootstrap_mean_diff_ci(
    airport_true_vals,
    airport_false_vals,
    n_boot=N_BOOT_DIFF,
    random_state=42
)

top_hour_true_vals = df.loc[df["is_top_fare_hour"] == True, "total_amount"].to_numpy()
top_hour_false_vals = df.loc[df["is_top_fare_hour"] == False, "total_amount"].to_numpy()
top_hour_ci = bootstrap_mean_diff_ci(
    top_hour_true_vals,
    top_hour_false_vals,
    n_boot=N_BOOT_DIFF,
    random_state=42
)

bootstrap_hypothesis_df = pd.DataFrame([
    {
        "test": "airport_vs_nonairport",
        "group_a_label": "airport",
        "group_b_label": "non_airport",
        "observed_difference": observed_diff_airport,
        "bootstrap_mean": airport_ci["bootstrap_mean"],
        "bootstrap_std": airport_ci["bootstrap_std"],
        "ci_lower_95": airport_ci["ci_lower_95"],
        "ci_upper_95": airport_ci["ci_upper_95"],
        "n_boot": N_BOOT_DIFF,
    },
    {
        "test": "top_fare_hours_vs_other_hours",
        "group_a_label": "top_fare_hours",
        "group_b_label": "other_hours",
        "observed_difference": observed_diff_top_hours,
        "bootstrap_mean": top_hour_ci["bootstrap_mean"],
        "bootstrap_std": top_hour_ci["bootstrap_std"],
        "ci_lower_95": top_hour_ci["ci_lower_95"],
        "ci_upper_95": top_hour_ci["ci_upper_95"],
        "n_boot": N_BOOT_DIFF,
    },
])

bootstrap_hypothesis_df.to_csv(OUTPUT_DIR / "bootstrap_hypothesis_ci.csv", index=False)
print("\nBootstrap CI for hypothesis differences:")
print(bootstrap_hypothesis_df)

# 模型指标的 bootstrap 区间改由独立脚本输出，这里只读取
# Bootstrap intervals for model metrics are produced by the separate script; only load them here.
bootstrap_metric_path = OUTPUT_DIR / "bootstrap_model_metric_ci.csv"
if bootstrap_metric_path.exists():
    bootstrap_metric_df = pd.read_csv(bootstrap_metric_path)
    print("\nLoaded model metric bootstrap CI:")
    print(bootstrap_metric_df.head(12))
else:
    print("\nModel metric bootstrap CI not found.")
    print("Please run `python proposal_model_training.py` first.")

# 画图：假设差值点估计 + bootstrap 95% CI
# Plot: hypothesis difference point estimates + bootstrap 95% CIs.
plt.figure(figsize=(7, 4))
x_labels = bootstrap_hypothesis_df["test"].tolist()
x = np.arange(len(x_labels))
point = bootstrap_hypothesis_df["observed_difference"].to_numpy()
lower_err = point - bootstrap_hypothesis_df["ci_lower_95"].to_numpy()
upper_err = bootstrap_hypothesis_df["ci_upper_95"].to_numpy() - point

plt.errorbar(
    x,
    point,
    yerr=[lower_err, upper_err],
    fmt="o",
    capsize=6,
    linewidth=1.6
)
plt.axhline(0, color="gray", linestyle="--", linewidth=1)
plt.xticks(x, x_labels, rotation=15, ha="right")
plt.ylabel("difference in mean total_amount")
plt.title("Bootstrap 95% CI for hypothesis differences")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_13_bootstrap_hypothesis_ci.png", dpi=180)
plt.close()


# =========================
# 15. 最终总结输出
# 15. Final summary output
# =========================
print("\n" + "=" * 60)
print("15. Finished")
print("=" * 60)

print("All outputs have been saved to:")
print(OUTPUT_DIR.resolve())

print("\nSaved files include:")
for file in sorted(OUTPUT_DIR.iterdir()):
    print("-", file.name)
