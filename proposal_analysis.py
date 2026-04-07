import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib

# 使用无界面后端，避免 PyCharm 交互后端兼容性问题。
MPL_CONFIG_DIR = Path(".mplconfig")
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR.resolve()))
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# ============================================================
# Proposal Analysis Script for NYC Taxi 2024 Full-Year Sample
# ------------------------------------------------------------
# 这个脚本做的事情：
# 1. 读取你已经做好的 2024 全年样本集 CSV
# 2. 查看数据结构、数据类型、缺失值、描述统计
# 3. 生成 proposal 阶段最关键的 EDA 图
# 4. 做两个简单的 permutation test（假设检验）
# 5. 跑一个基础建模流程（Linear / RF / GB）
# 6. 把图和部分结果保存到本地文件夹
#
# 你只需要改一个地方：
# DATA_PATH = "你的 CSV 文件路径"
# ============================================================


# =========================
# 0. 路径设置
# =========================
# 把这里改成你自己的 CSV 路径
DATA_PATH = "nyc_taxi_2024_cleaned_sample.csv"

# 输出文件夹：图和结果会保存在这里
OUTPUT_DIR = Path("proposal_outputs_2024")
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================
# 1. 读取数据
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
missing_df = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_pct": (df.isna().mean() * 100).round(2)
}).reset_index().rename(columns={"index": "column"})
missing_df.to_csv(OUTPUT_DIR / "missing_summary.csv", index=False)

# 查看月份分布
print("\nPickup month distribution:")
print(df["pickup_month"].value_counts().sort_index())

# 查看周末/工作日比例
print("\nis_weekend distribution:")
print(df["is_weekend"].value_counts(dropna=False))

# 查看时间段分布
print("\nhour_block distribution:")
print(df["hour_block"].value_counts(dropna=False))


# =========================
# 3. 数值变量描述统计
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
# 4. 图 1：total_amount 分布
# =========================
print("\n" + "=" * 60)
print("4. Plotting: Distribution of total_amount")
print("=" * 60)

plt.figure(figsize=(8, 5))
plt.hist(df["total_amount"], bins=60)
plt.xlabel("total_amount")
plt.ylabel("count")
plt.title("Distribution of total_amount")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_1_total_amount_distribution.png", dpi=180)
plt.close()


# =========================
# 5. 图 2：trip_distance vs total_amount
# =========================
print("\n" + "=" * 60)
print("5. Plotting: trip_distance vs total_amount")
print("=" * 60)

plot_df = df.sample(min(5000, len(df)), random_state=42)

plt.figure(figsize=(8, 5))
plt.scatter(plot_df["trip_distance"], plot_df["total_amount"], s=8, alpha=0.35)
plt.xlabel("trip_distance")
plt.ylabel("total_amount")
plt.title("trip_distance vs total_amount")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_2_distance_vs_total_amount.png", dpi=180)
plt.close()


# =========================
# 6. 图 3：trip_duration_minutes vs total_amount
# =========================
print("\n" + "=" * 60)
print("6. Plotting: trip_duration_minutes vs total_amount")
print("=" * 60)

plot_df = df.sample(min(5000, len(df)), random_state=42)

plt.figure(figsize=(8, 5))
plt.scatter(plot_df["trip_duration_minutes"], plot_df["total_amount"], s=8, alpha=0.35)
plt.xlabel("trip_duration_minutes")
plt.ylabel("total_amount")
plt.title("trip_duration_minutes vs total_amount")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_3_duration_vs_total_amount.png", dpi=180)
plt.close()


# =========================
# 7. 图 4：按小时平均 total_amount
# =========================
print("\n" + "=" * 60)
print("7. Plotting: average total_amount by pickup_hour")
print("=" * 60)

hour_avg = (
    df.groupby("pickup_hour", as_index=False)["total_amount"]
      .mean()
      .sort_values("pickup_hour")
)

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


# =========================
# 8. 图 5：按 borough 平均 total_amount
# =========================
print("\n" + "=" * 60)
print("8. Plotting: average total_amount by pickup_borough")
print("=" * 60)

borough_avg = (
    df.groupby("pickup_borough", as_index=False)
      .agg(
          mean_total_amount=("total_amount", "mean"),
          trip_count=("total_amount", "size")
      )
      .sort_values("mean_total_amount", ascending=False)
)

print(borough_avg)

borough_avg.to_csv(OUTPUT_DIR / "borough_avg_total_amount.csv", index=False)

plt.figure(figsize=(8, 5))
plt.bar(borough_avg["pickup_borough"].astype(str), borough_avg["mean_total_amount"])
plt.xlabel("pickup_borough")
plt.ylabel("mean total_amount")
plt.title("Average total_amount by pickup_borough")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_5_borough_avg_total_amount.png", dpi=180)
plt.close()


# =========================
# 9. 图 6：机场 vs 非机场
# =========================
print("\n" + "=" * 60)
print("9. Plotting: airport vs non-airport")
print("=" * 60)

airport_avg = (
    df.groupby("is_airport_trip", as_index=False)
      .agg(
          mean_total_amount=("total_amount", "mean"),
          trip_count=("total_amount", "size")
      )
)

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
# 10. 图 7：工作日 vs 周末
# =========================
print("\n" + "=" * 60)
print("10. Plotting: weekday vs weekend")
print("=" * 60)

weekend_avg = (
    df.groupby("is_weekend", as_index=False)
      .agg(
          mean_total_amount=("total_amount", "mean"),
          trip_count=("total_amount", "size")
      )
)

print(weekend_avg)

weekend_avg.to_csv(OUTPUT_DIR / "weekend_avg_total_amount.csv", index=False)

plt.figure(figsize=(6, 5))
plt.bar(weekend_avg["is_weekend"].astype(str), weekend_avg["mean_total_amount"])
plt.xlabel("is_weekend")
plt.ylabel("mean total_amount")
plt.title("Average total_amount: weekday vs weekend")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_7_weekday_vs_weekend.png", dpi=180)
plt.close()


# =========================
# 11. 相关系数分析
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
# 13. 假设检验 2：周末 vs 工作日
# =========================
print("\n" + "=" * 60)
print("13. Permutation test: weekend vs weekday")
print("=" * 60)

observed_diff_weekend = (
    df[df["is_weekend"] == True]["total_amount"].mean()
    - df[df["is_weekend"] == False]["total_amount"].mean()
)

count = 0
labels = df["is_weekend"].to_numpy()

for _ in range(n_perm):
    shuffled = rng.permutation(labels)
    diff = values[shuffled == True].mean() - values[shuffled == False].mean()
    if abs(diff) >= abs(observed_diff_weekend):
        count += 1

p_value_weekend = (count + 1) / (n_perm + 1)

print("Observed difference:", observed_diff_weekend)
print("Permutation p-value:", p_value_weekend)

# 保存 hypothesis test 结果
hypothesis_df = pd.DataFrame({
    "test": ["airport_vs_nonairport", "weekend_vs_weekday"],
    "observed_difference": [observed_diff_airport, observed_diff_weekend],
    "p_value": [p_value_airport, p_value_weekend]
})
hypothesis_df.to_csv(OUTPUT_DIR / "hypothesis_test_results.csv", index=False)


# =========================
# 14. 基础建模：预测 total_amount
# =========================
print("\n" + "=" * 60)
print("14. Baseline modeling: predict total_amount")
print("=" * 60)

target = "total_amount"

features = [
    "trip_distance",
    "trip_duration_minutes",
    "passenger_count",
    "payment_type",
    "pickup_month",
    "pickup_hour",
    "pickup_weekday",
    "is_weekend",
    "is_airport_trip",
    "pickup_borough",
    "dropoff_borough"
]

model_df = df[features + [target, "tpep_pickup_datetime"]].copy()

numeric_features = [
    "trip_distance",
    "trip_duration_minutes",
    "passenger_count",
    "pickup_month",
    "pickup_hour",
    "pickup_weekday"
]

categorical_features = [
    "payment_type",
    "is_weekend",
    "is_airport_trip",
    "pickup_borough",
    "dropoff_borough"
]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

model_df = model_df.sort_values("tpep_pickup_datetime").reset_index(drop=True)
X = model_df[features]
y = model_df[target]

split_idx = int(len(model_df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

results = []

for name, model in models.items():
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\n{name}")
    print("RMSE =", round(rmse, 4))
    print("MAE  =", round(mae, 4))
    print("R2   =", round(r2, 4))

    results.append({
        "model": name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "model_results.csv", index=False)


# =========================
# 15. 成本效率分析：单位里程成本与速度
# =========================
print("\n" + "=" * 60)
print("15. Cost efficiency analysis")
print("=" * 60)

# 计算基础效率指标，避免除以 0 的问题
eps = 1e-6
df["amount_per_mile"] = df["total_amount"] / (df["trip_distance"] + eps)
df["amount_per_minute"] = df["total_amount"] / (df["trip_duration_minutes"] + eps)
df["speed_mph"] = df["trip_distance"] / (df["trip_duration_minutes"] / 60 + eps)
df["minutes_per_mile"] = df["trip_duration_minutes"] / (df["trip_distance"] + eps)

# 对比例类指标做轻微截尾，减少极端值影响
for col in ["amount_per_mile", "amount_per_minute", "speed_mph", "minutes_per_mile"]:
    q01, q99 = df[col].quantile([0.01, 0.99])
    df[f"{col}_clip"] = df[col].clip(lower=q01, upper=q99)

# 按小时汇总均值和样本量
hour_efficiency = (
    df.groupby("pickup_hour", as_index=False)
      .agg(
          mean_total_amount=("total_amount", "mean"),
          mean_amount_per_mile=("amount_per_mile_clip", "mean"),
          mean_speed_mph=("speed_mph_clip", "mean"),
          mean_minutes_per_mile=("minutes_per_mile_clip", "mean"),
          trip_count=("total_amount", "size")
      )
      .sort_values("pickup_hour")
)

print(hour_efficiency)
hour_efficiency.to_csv(OUTPUT_DIR / "hour_efficiency_summary.csv", index=False)

# 画图：小时维度下单位里程成本与速度
plt.figure(figsize=(9, 5))
plt.plot(hour_efficiency["pickup_hour"], hour_efficiency["mean_amount_per_mile"], marker="o", label="mean amount per mile")
plt.plot(hour_efficiency["pickup_hour"], hour_efficiency["mean_speed_mph"], marker="s", label="mean speed (mph)")
plt.xlabel("pickup_hour")
plt.ylabel("value")
plt.title("Hourly cost efficiency: amount_per_mile vs speed_mph")
plt.xticks(range(0, 24))
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_8_hourly_cost_efficiency.png", dpi=180)
plt.close()


# =========================
# 16. 机场溢价拆解：控制变量 + 距离分段
# =========================
print("\n" + "=" * 60)
print("16. Airport premium decomposition")
print("=" * 60)

# 距离分段后比较机场与非机场均价
df["distance_band"] = pd.cut(
    df["trip_distance"],
    bins=[0, 2, 5, 10, 20, 80],
    labels=["0-2", "2-5", "5-10", "10-20", "20+"],
    include_lowest=True
)

airport_by_dist = (
    df.groupby(["distance_band", "is_airport_trip"], as_index=False, observed=False)
      .agg(
          mean_total_amount=("total_amount", "mean"),
          trip_count=("total_amount", "size")
      )
)

print(airport_by_dist)
airport_by_dist.to_csv(OUTPUT_DIR / "airport_by_distance_band.csv", index=False)

# 构造控制变量线性回归，估计机场额外溢价
control_features = [
    "trip_distance",
    "trip_duration_minutes",
    "pickup_month",
    "pickup_hour",
    "pickup_weekday",
    "passenger_count",
    "is_airport_trip",
    "payment_type",
    "pickup_borough",
    "dropoff_borough"
]

control_numeric_features = [
    "trip_distance",
    "trip_duration_minutes",
    "pickup_month",
    "pickup_hour",
    "pickup_weekday",
    "passenger_count"
]

control_categorical_features = [
    "is_airport_trip",
    "payment_type",
    "pickup_borough",
    "dropoff_borough"
]

control_numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

control_categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

control_preprocessor = ColumnTransformer(
    transformers=[
        ("num", control_numeric_transformer, control_numeric_features),
        ("cat", control_categorical_transformer, control_categorical_features)
    ]
)

control_model = Pipeline(steps=[
    ("preprocessor", control_preprocessor),
    ("model", LinearRegression())
])

control_model.fit(df[control_features], df["total_amount"])

# 提取机场变量的系数
ohe = (
    control_model.named_steps["preprocessor"]
                .named_transformers_["cat"]
                .named_steps["onehot"]
)
cat_feature_names = ohe.get_feature_names_out(control_categorical_features)
all_feature_names = control_numeric_features + list(cat_feature_names)
coef_series = pd.Series(control_model.named_steps["model"].coef_, index=all_feature_names)
airport_controlled_coef = float(coef_series.get("is_airport_trip_True", np.nan))

airport_controlled_df = pd.DataFrame({
    "metric": ["airport_premium_controlled"],
    "value": [airport_controlled_coef]
})
airport_controlled_df.to_csv(OUTPUT_DIR / "airport_controlled_effect.csv", index=False)

print("Controlled airport premium (coef):", round(airport_controlled_coef, 4))

# 画图：距离分段后机场与非机场均价对比
pivot_airport = airport_by_dist.pivot(
    index="distance_band",
    columns="is_airport_trip",
    values="mean_total_amount"
).reset_index()
pivot_airport.columns = ["distance_band", "non_airport_mean", "airport_mean"]

plt.figure(figsize=(8, 5))
x = np.arange(len(pivot_airport))
width = 0.35
plt.bar(x - width / 2, pivot_airport["non_airport_mean"], width=width, label="non-airport")
plt.bar(x + width / 2, pivot_airport["airport_mean"], width=width, label="airport")
plt.xticks(x, pivot_airport["distance_band"].astype(str))
plt.xlabel("distance_band (mile)")
plt.ylabel("mean total_amount")
plt.title("Airport vs non-airport by distance band")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_9_airport_gap_by_distance_band.png", dpi=180)
plt.close()


# =========================
# 17. 风险定价分析：按小时看分位数
# =========================
print("\n" + "=" * 60)
print("17. Risk-based pricing analysis")
print("=" * 60)

# 使用 p50/p90/p95 刻画不同小时的价格风险带
hourly_risk = (
    df.groupby("pickup_hour", as_index=False)
      .agg(
          p50_total_amount=("total_amount", lambda x: x.quantile(0.50)),
          p90_total_amount=("total_amount", lambda x: x.quantile(0.90)),
          p95_total_amount=("total_amount", lambda x: x.quantile(0.95)),
          trip_count=("total_amount", "size")
      )
      .sort_values("pickup_hour")
)

print(hourly_risk)
hourly_risk.to_csv(OUTPUT_DIR / "hourly_risk_quantiles.csv", index=False)

# 画图：不同小时的中位数和高分位价格
plt.figure(figsize=(8, 5))
plt.plot(hourly_risk["pickup_hour"], hourly_risk["p50_total_amount"], marker="o", label="p50")
plt.plot(hourly_risk["pickup_hour"], hourly_risk["p90_total_amount"], marker="s", label="p90")
plt.plot(hourly_risk["pickup_hour"], hourly_risk["p95_total_amount"], marker="^", label="p95")
plt.xlabel("pickup_hour")
plt.ylabel("total_amount")
plt.title("Hourly price risk profile (p50/p90/p95)")
plt.xticks(range(0, 24))
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_10_hourly_price_risk_quantiles.png", dpi=180)
plt.close()


# =========================
# 18. 路线结构分析：上车区-下车区组合
# =========================
print("\n" + "=" * 60)
print("18. Route structure analysis")
print("=" * 60)

# 统计路线层面的价格、距离、时长与效率
route_summary = (
    df.groupby(["pickup_borough", "dropoff_borough"], as_index=False)
      .agg(
          trip_count=("total_amount", "size"),
          mean_total_amount=("total_amount", "mean"),
          mean_trip_distance=("trip_distance", "mean"),
          mean_trip_duration=("trip_duration_minutes", "mean"),
          mean_amount_per_mile=("amount_per_mile_clip", "mean"),
          mean_minutes_per_mile=("minutes_per_mile_clip", "mean")
      )
)

# 过滤小样本路线，避免均值被极端值主导
route_summary_filtered = route_summary[route_summary["trip_count"] >= 300].copy()
route_summary_filtered = route_summary_filtered.sort_values("mean_total_amount", ascending=False)

print(route_summary_filtered.head(20))
route_summary_filtered.to_csv(OUTPUT_DIR / "route_structure_summary.csv", index=False)

# 画图：样本量足够的路线里，均价最高的前 10 条
top_routes = route_summary_filtered.head(10).copy()
top_routes["route"] = top_routes["pickup_borough"].astype(str) + " -> " + top_routes["dropoff_borough"].astype(str)
top_routes = top_routes.sort_values("mean_total_amount", ascending=True)

plt.figure(figsize=(9, 6))
plt.barh(top_routes["route"], top_routes["mean_total_amount"])
plt.xlabel("mean total_amount")
plt.ylabel("route")
plt.title("Top routes by mean total_amount (n >= 300)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_11_top_routes_mean_total_amount.png", dpi=180)
plt.close()


# =========================
# 19. 季节性拆解：月份、周末差异与结构变化
# =========================
print("\n" + "=" * 60)
print("19. Seasonality decomposition")
print("=" * 60)

# 按月份看均价、距离、时长和单位里程价格
monthly_decomp = (
    df.groupby("pickup_month", as_index=False)
      .agg(
          mean_total_amount=("total_amount", "mean"),
          mean_trip_distance=("trip_distance", "mean"),
          mean_trip_duration=("trip_duration_minutes", "mean"),
          mean_amount_per_mile=("amount_per_mile_clip", "mean"),
          trip_count=("total_amount", "size")
      )
      .sort_values("pickup_month")
)

print(monthly_decomp)
monthly_decomp.to_csv(OUTPUT_DIR / "monthly_decomposition.csv", index=False)

# 计算每个月“周末 - 工作日”均价差
monthly_weekend = (
    df.groupby(["pickup_month", "is_weekend"], as_index=False)
      .agg(mean_total_amount=("total_amount", "mean"))
)

monthly_weekend_pivot = monthly_weekend.pivot(
    index="pickup_month",
    columns="is_weekend",
    values="mean_total_amount"
).reset_index()
monthly_weekend_pivot.columns = ["pickup_month", "weekday_mean", "weekend_mean"]
monthly_weekend_pivot["weekend_minus_weekday"] = (
    monthly_weekend_pivot["weekend_mean"] - monthly_weekend_pivot["weekday_mean"]
)

print(monthly_weekend_pivot)
monthly_weekend_pivot.to_csv(OUTPUT_DIR / "monthly_weekend_gap.csv", index=False)

# 画图：月份维度下总价与单位里程成本
plt.figure(figsize=(8, 5))
plt.plot(monthly_decomp["pickup_month"], monthly_decomp["mean_total_amount"], marker="o", label="mean total_amount")
plt.plot(monthly_decomp["pickup_month"], monthly_decomp["mean_amount_per_mile"], marker="s", label="mean amount_per_mile")
plt.xlabel("pickup_month")
plt.ylabel("value")
plt.title("Monthly decomposition: total_amount vs amount_per_mile")
plt.xticks(range(1, 13))
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_12_monthly_decomposition.png", dpi=180)
plt.close()


# =========================
# 20. 最终总结输出
# =========================
print("\n" + "=" * 60)
print("20. Finished")
print("=" * 60)

print("All outputs have been saved to:")
print(OUTPUT_DIR.resolve())

print("\nSaved files include:")
for file in sorted(OUTPUT_DIR.iterdir()):
    print("-", file.name)
