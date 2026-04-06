import pandas as pd
import pyarrow.parquet as pq

# =========================
# 1. 读取 zone lookup
# =========================
lookup = pd.read_csv("taxi_zone_lookup.csv")

# 为了后面做 pickup join，先复制并改列名
pickup_lookup = lookup.rename(columns={
    "LocationID": "PULocationID",
    "Borough": "pickup_borough",
    "Zone": "pickup_zone",
    "service_zone": "pickup_service_zone"
})

# 为了后面做 dropoff join，再复制一份并改列名
dropoff_lookup = lookup.rename(columns={
    "LocationID": "DOLocationID",
    "Borough": "dropoff_borough",
    "Zone": "dropoff_zone",
    "service_zone": "dropoff_service_zone"
})

# =========================
# 2. 每个月数据的处理函数
# =========================
def clean_and_sample_month(file_path, sample_size=7500, random_state=42):
    """
    从单个月的 Yellow Taxi parquet 中：
    1) 读取核心列
    2) 做基础清洗
    3) 构造时间特征
    4) 按 is_weekend + hour_block 分层抽样
    5) 返回抽样后的 DataFrame
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
    df = pq.read_table(file_path, columns=cols).to_pandas()

    # 转成时间格式
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

    # 构造行程时长（分钟）
    df["trip_duration_minutes"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    # ================
    # 基础清洗
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
    # ================
    df["pickup_month"] = df["tpep_pickup_datetime"].dt.month
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_weekday"] = df["tpep_pickup_datetime"].dt.dayofweek
    df["is_weekend"] = df["pickup_weekday"] >= 5

    # 把一天分成几个时间段
    df["hour_block"] = pd.cut(
        df["pickup_hour"],
        bins=[-1, 5, 11, 16, 20, 23],
        labels=["late_night", "morning", "midday", "evening", "night"]
    )

    # 是否机场相关行程
    df["is_airport_trip"] = df["Airport_fee"].fillna(0) > 0

    # ================
    # 分层抽样
    # ================
    # 想从这个月抽 sample_size 条
    # 先算整体抽样比例
    frac = sample_size / len(df)

    # 在每个层里按比例抽
    sampled = (
        df.groupby(["is_weekend", "hour_block"], group_keys=False)
          .apply(lambda x: x.sample(
              n=max(1, int(round(len(x) * frac))),
              random_state=random_state
          ))
    )

    # 因为四舍五入，抽出来的总数可能不是刚好 sample_size
    # 所以再做一次修正
    if len(sampled) > sample_size:
        sampled = sampled.sample(sample_size, random_state=random_state)
    elif len(sampled) < sample_size:
        remaining = df.drop(sampled.index)
        extra = remaining.sample(sample_size - len(sampled), random_state=random_state)
        sampled = pd.concat([sampled, extra], ignore_index=False)

    return sampled.reset_index(drop=True)


# =========================
# 3. 分别处理 1-12 月
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
q1 = pd.concat([jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec], ignore_index=True)

# =========================
# 4. 连接 zone lookup
# =========================
q1 = q1.merge(pickup_lookup, on="PULocationID", how="left")
q1 = q1.merge(dropoff_lookup, on="DOLocationID", how="left")

# =========================
# 5. 保存结果
# =========================
q1.to_csv("nyc_taxi_2024_sample_90000.csv", index=False)

print("Done.")
print("Final shape:", q1.shape)
print(q1.head())
