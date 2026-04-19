import pandas as pd
import pyarrow.parquet as pq
import numpy as np

# =========================
# 1. 读取 zone lookup
# 1. Read the zone lookup table
# =========================
lookup = pd.read_csv("taxi_zone_lookup.csv")

# 为了后面做 pickup join，先复制并改列名
# To prepare for the later pickup join, copy it first and rename columns.
pickup_lookup = lookup.rename(columns={
    "LocationID": "PULocationID",
    "Borough": "pickup_borough",
    "Zone": "pickup_zone",
    "service_zone": "pickup_service_zone"
})

# 为了后面做 dropoff join，再复制一份并改列名
# To prepare for the later dropoff join, make another copy and rename columns.
dropoff_lookup = lookup.rename(columns={
    "LocationID": "DOLocationID",
    "Borough": "dropoff_borough",
    "Zone": "dropoff_zone",
    "service_zone": "dropoff_service_zone"
})

# =========================
# 2. 每个月数据的处理函数
# 2. Monthly data processing function
# =========================
def clean_and_sample_month(file_path, sample_size=7500, random_state=42):
    """
    从单个月的 Yellow Taxi parquet 中：
    From one month of Yellow Taxi parquet data:
    1) 读取核心列
    1) Read core columns
    2) 做基础清洗
    2) Perform basic cleaning
    3) 构造时间特征
    3) Build time features
    4) 按 is_weekend + hour_block 分层抽样
    4) Perform stratified sampling by is_weekend + hour_block
    5) 返回抽样后的 DataFrame
    5) Return the sampled DataFrame
    """

    cols = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "passenger_count",
        "trip_distance",
        "PULocationID",
        "DOLocationID",
        "payment_type",
        "fare_amount",
        "tip_amount",
        "total_amount",
        "Airport_fee"
    ]

    # 只读需要的列，避免把整个大表全读进来
    # Read only required columns to avoid loading the entire large table into memory.
    df = pq.read_table(file_path, columns=cols).to_pandas()

    # 转成时间格式
    # Convert to datetime format.
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

    # 构造行程时长（分钟）
    # Build trip duration (in minutes).
    df["trip_duration_minutes"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    # ================
    # 基础清洗
    # Basic cleaning
    # ================
    df = df[
        (df["trip_duration_minutes"] > 1) &
        (df["trip_duration_minutes"] <= 180) &
        (df["trip_distance"] > 0) &
        (df["trip_distance"] <= 100) &
        (df["total_amount"] > 0) &
        (df["total_amount"] <= 300) &
        (df["passenger_count"].fillna(1) >= 0) &
        (df["passenger_count"].fillna(1) <= 8)
    ].copy()

    # ================
    # 构造时间特征
    # Build time features
    # ================
    df["pickup_month"] = df["tpep_pickup_datetime"].dt.month
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_weekday"] = df["tpep_pickup_datetime"].dt.dayofweek
    df["is_weekend"] = df["pickup_weekday"] >= 5

    # 把一天分成几个时间段
    # Split the day into several time blocks.
    df["hour_block"] = pd.cut(
        df["pickup_hour"],
        bins=[-1, 5, 11, 16, 20, 23],
        labels=["late_night", "morning", "midday", "evening", "night"]
    )

    # 是否机场相关行程
    # Whether the trip is airport-related.
    df["is_airport_trip"] = df["Airport_fee"].fillna(0) > 0

    # ================
    # 分层抽样
    # Stratified sampling
    # ================
    # 想从这个月抽 sample_size 条
    # Target sample_size records from this month.
    # 先算整体抽样比例
    # First compute the overall sampling fraction.
    frac = sample_size / len(df)

    # 在每个层里按比例抽
    # Sample proportionally within each stratum.
    sampled = (
        df.groupby(["is_weekend", "hour_block"], group_keys=False)
          .apply(lambda x: x.sample(
              n=max(1, int(round(len(x) * frac))),
              random_state=random_state
          ))
    )

    # 因为四舍五入，抽出来的总数可能不是刚好 sample_size
    # Due to rounding, the sampled total may not be exactly sample_size.
    # 所以再做一次修正
    # So apply one more correction step.
    if len(sampled) > sample_size:
        sampled = sampled.sample(sample_size, random_state=random_state)
    elif len(sampled) < sample_size:
        remaining = df.drop(sampled.index)
        extra = remaining.sample(sample_size - len(sampled), random_state=random_state)
        sampled = pd.concat([sampled, extra], ignore_index=False)

    return sampled.reset_index(drop=True)


# =========================
# 2.5 整合后的二次清洗函数
# 2.5 Secondary cleaning function for the integrated sample
# =========================
def clean_integrated_sample(df):
    """
    对整合后的全年样本做二次清洗：
    Perform secondary cleaning on the integrated full-year sample:
    1) 去重
    1) Remove duplicates
    2) 过滤极端异常值
    2) Filter extreme outliers
    3) 处理缺失值和局部异常
    3) Handle missing values and local anomalies
    4) 重算时间衍生特征，确保字段一致
    4) Recompute derived time features to ensure field consistency
    """
    cleaned = df.copy()
    rows_before = len(cleaned)

    # 去重
    # Remove duplicates.
    cleaned = cleaned.drop_duplicates().copy()

    # 过滤极端异常值（按确认阈值放宽上限）
    # Filter extreme outliers (with relaxed upper bounds based on confirmed thresholds).
    cleaned = cleaned[
        cleaned["trip_distance"].between(0.05, 100) &
        cleaned["trip_duration_minutes"].between(1, 240) &
        cleaned["total_amount"].between(1, 400)
    ].copy()

    # passenger_count：异常值先置空，再填补为 1
    # passenger_count: set abnormal values to null first, then fill with 1.
    cleaned.loc[~cleaned["passenger_count"].between(1, 6), "passenger_count"] = np.nan
    cleaned["passenger_count"] = cleaned["passenger_count"].fillna(1)

    # Airport_fee：缺失或负值统一处理为 0
    # Airport_fee: uniformly treat missing or negative values as 0.
    cleaned["Airport_fee"] = cleaned["Airport_fee"].fillna(0)
    cleaned.loc[cleaned["Airport_fee"] < 0, "Airport_fee"] = 0

    # fare_amount：<=0 先置空，再用距离分箱中位数填补
    # fare_amount: set values <= 0 to null first, then impute with distance-bin medians.
    cleaned.loc[cleaned["fare_amount"] <= 0, "fare_amount"] = np.nan
    distance_bins = pd.cut(
        cleaned["trip_distance"],
        bins=[0, 1, 2, 3, 5, 8, 12, 20, 100],
        include_lowest=True
    )
    fare_median_by_bin = cleaned.groupby(distance_bins)["fare_amount"].transform("median")
    cleaned["fare_amount"] = cleaned["fare_amount"].fillna(fare_median_by_bin)
    cleaned["fare_amount"] = cleaned["fare_amount"].fillna(cleaned["fare_amount"].median())

    # tip_amount：负值修正为 0
    # tip_amount: correct negative values to 0.
    cleaned.loc[cleaned["tip_amount"] < 0, "tip_amount"] = 0

    # zone 相关字段缺失统一填 Unknown（不改列名）
    # For zone-related fields, fill missing values with Unknown (without renaming columns).
    zone_cols = [
        "pickup_borough", "pickup_zone", "pickup_service_zone",
        "dropoff_borough", "dropoff_zone", "dropoff_service_zone"
    ]
    for col in zone_cols:
        cleaned[col] = cleaned[col].fillna("Unknown")

    # 重算时间衍生字段，确保与时间字段一致
    # Recompute derived time fields to ensure consistency with datetime fields.
    cleaned["pickup_month"] = cleaned["tpep_pickup_datetime"].dt.month
    cleaned["pickup_hour"] = cleaned["tpep_pickup_datetime"].dt.hour
    cleaned["pickup_weekday"] = cleaned["tpep_pickup_datetime"].dt.dayofweek
    cleaned["is_weekend"] = cleaned["pickup_weekday"] >= 5
    cleaned["hour_block"] = pd.cut(
        cleaned["pickup_hour"],
        bins=[-1, 5, 11, 16, 20, 23],
        labels=["late_night", "morning", "midday", "evening", "night"]
    )
    cleaned["is_airport_trip"] = cleaned["Airport_fee"] > 0

    rows_after = len(cleaned)
    print("Second cleaning done.")
    print("Rows before second cleaning:", rows_before)
    print("Rows after second cleaning :", rows_after)
    print("Rows removed              :", rows_before - rows_after)

    return cleaned


# =========================
# 3. 分别处理 1-12 月
# 3. Process months 1-12 separately
# =========================
jan = clean_and_sample_month("yellow_tripdata_2024-01.parquet", sample_size=7500, random_state=42)
feb = clean_and_sample_month("yellow_tripdata_2024-02.parquet", sample_size=7500, random_state=42)
mar = clean_and_sample_month("yellow_tripdata_2024-03.parquet", sample_size=7500, random_state=42)
apr = clean_and_sample_month("yellow_tripdata_2024-04.parquet", sample_size=7500, random_state=42)
may = clean_and_sample_month("yellow_tripdata_2024-05.parquet", sample_size=7500, random_state=42)
jun = clean_and_sample_month("yellow_tripdata_2024-06.parquet", sample_size=7500, random_state=42)
jul = clean_and_sample_month("yellow_tripdata_2024-07.parquet", sample_size=7500, random_state=42)
aug = clean_and_sample_month("yellow_tripdata_2024-08.parquet", sample_size=7500, random_state=42)
sep = clean_and_sample_month("yellow_tripdata_2024-09.parquet", sample_size=7500, random_state=42)
oct = clean_and_sample_month("yellow_tripdata_2024-10.parquet", sample_size=7500, random_state=42)
nov = clean_and_sample_month("yellow_tripdata_2024-11.parquet", sample_size=7500, random_state=42)
dec = clean_and_sample_month("yellow_tripdata_2024-12.parquet", sample_size=7500, random_state=42)

# 合并成全年样本集
# Merge into a full-year sample dataset.
q1 = pd.concat([jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec], ignore_index=True)

# =========================
# 4. 连接 zone lookup
# 4. Join with zone lookup
# =========================
q1 = q1.merge(pickup_lookup, on="PULocationID", how="left")
q1 = q1.merge(dropoff_lookup, on="DOLocationID", how="left")

# =========================
# 5. 整合样本二次清洗
# 5. Secondary cleaning on the integrated sample
# =========================
q1 = clean_integrated_sample(q1)

# =========================
# 6. 保存结果
# 6. Save results
# =========================
q1.to_csv("nyc_taxi_2024_cleaned_sample.csv", index=False)

print("Done.")
print("Final shape:", q1.shape)
print(q1.head())
