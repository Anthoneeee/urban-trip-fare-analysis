from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================
# 1. 基础配置
# =========================
DATA_PATH = Path("nyc_taxi_2024_cleaned_sample.csv")
OUTPUT_DIR = Path("proposal_outputs_2024")
# 保留旧文件名（兼容你当前 LaTeX 引用），但内容改为单图输出
OUTPUT_IMG = OUTPUT_DIR / "fig_13_5am_price_diagnosis.png"
OUTPUT_IMG_DIST = OUTPUT_DIR / "fig_13_5am_distance_mix.png"
OUTPUT_IMG_DECOMP = OUTPUT_DIR / "fig_13_5am_decomposition.png"
OUTPUT_IMG_RATIO = OUTPUT_DIR / "fig_13_5am_key_ratio.png"
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / "hour5_diagnosis_summary.csv"
OUTPUT_DECOMP_CSV = OUTPUT_DIR / "hour5_decomposition.csv"


# =========================
# 2. 读取与预处理
# =========================
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # 兼容 bool/字符串两种存储方式
    if df["is_airport_trip"].dtype != bool:
        df["is_airport_trip"] = (
            df["is_airport_trip"]
            .astype(str)
            .str.lower()
            .map({"true": True, "false": False})
            .fillna(False)
        )

    return df


# =========================
# 3. 计算关键统计
# =========================
def compute_metrics(df: pd.DataFrame) -> dict:
    h5 = df[df["pickup_hour"] == 5].copy()

    summary_rows = [
        ("mean_total_amount", h5["total_amount"].mean(), df["total_amount"].mean()),
        ("median_total_amount", h5["total_amount"].median(), df["total_amount"].median()),
        ("mean_trip_distance", h5["trip_distance"].mean(), df["trip_distance"].mean()),
        (
            "mean_trip_duration_minutes",
            h5["trip_duration_minutes"].mean(),
            df["trip_duration_minutes"].mean(),
        ),
        ("airport_share", h5["is_airport_trip"].mean(), df["is_airport_trip"].mean()),
        ("share_trip_distance_ge_10", (h5["trip_distance"] >= 10).mean(), (df["trip_distance"] >= 10).mean()),
        ("share_trip_distance_ge_20", (h5["trip_distance"] >= 20).mean(), (df["trip_distance"] >= 20).mean()),
    ]

    summary_df = pd.DataFrame(summary_rows, columns=["metric", "hour5", "all_hours"])
    summary_df["ratio_hour5_vs_all"] = summary_df["hour5"] / summary_df["all_hours"]

    # 每英里成本
    h5_amount_per_mile = (h5["total_amount"] / h5["trip_distance"].replace(0, np.nan)).mean()
    all_amount_per_mile = (df["total_amount"] / df["trip_distance"].replace(0, np.nan)).mean()
    extra_row = pd.DataFrame(
        [{
            "metric": "mean_amount_per_mile",
            "hour5": h5_amount_per_mile,
            "all_hours": all_amount_per_mile,
            "ratio_hour5_vs_all": h5_amount_per_mile / all_amount_per_mile,
        }]
    )
    summary_df = pd.concat([summary_df, extra_row], ignore_index=True)

    # 5点均价提升的分解
    s0 = df["is_airport_trip"].mean()
    mn0 = df[df["is_airport_trip"] == False]["total_amount"].mean()
    ma0 = df[df["is_airport_trip"] == True]["total_amount"].mean()
    m0 = df["total_amount"].mean()

    s5 = h5["is_airport_trip"].mean()
    mn5 = h5[h5["is_airport_trip"] == False]["total_amount"].mean()
    ma5 = h5[h5["is_airport_trip"] == True]["total_amount"].mean()
    m5 = h5["total_amount"].mean()

    share_component = (s5 - s0) * (ma0 - mn0)
    non_component = (1 - s5) * (mn5 - mn0)
    airport_component = s5 * (ma5 - ma0)
    delta_total = m5 - m0

    decomp_df = pd.DataFrame(
        [
            {"component": "airport_share_change", "value": share_component},
            {"component": "non_airport_price_change", "value": non_component},
            {"component": "airport_price_change", "value": airport_component},
            {"component": "total_delta_5am_minus_all", "value": delta_total},
        ]
    )

    # 距离分段结构对比
    bins = [0, 2, 5, 10, 20, 100]
    labels = ["0-2", "2-5", "5-10", "10-20", "20+"]
    h5["distance_bin"] = pd.cut(h5["trip_distance"], bins=bins, labels=labels, include_lowest=True)
    df_dist = df.copy()
    df_dist["distance_bin"] = pd.cut(df_dist["trip_distance"], bins=bins, labels=labels, include_lowest=True)
    bin_h5 = h5["distance_bin"].value_counts(normalize=True).sort_index()
    bin_all = df_dist["distance_bin"].value_counts(normalize=True).sort_index()
    dist_compare_df = pd.DataFrame(
        {
            "distance_bin": labels,
            "hour5_share": [bin_h5.get(lb, 0.0) for lb in labels],
            "all_hours_share": [bin_all.get(lb, 0.0) for lb in labels],
        }
    )

    # 小时维度轮廓：用于画“5点为何突出”的常规折线图
    hour_profile_df = (
        df.groupby("pickup_hour", as_index=False)
        .agg(
            mean_total_amount=("total_amount", "mean"),
            median_total_amount=("total_amount", "median"),
            mean_trip_distance=("trip_distance", "mean"),
            airport_share=("is_airport_trip", "mean"),
        )
        .sort_values("pickup_hour")
    )

    return {
        "summary_df": summary_df,
        "decomp_df": decomp_df,
        "dist_compare_df": dist_compare_df,
        "hour_profile_df": hour_profile_df,
        "delta_total": delta_total,
        "m5": m5,
        "m0": m0,
    }


# =========================
# 4. 可视化：拆分为常规单图输出
# =========================
def draw_report(metrics: dict) -> None:
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    summary_df = metrics["summary_df"]
    decomp_df = metrics["decomp_df"]
    dist_df = metrics["dist_compare_df"]
    hour_profile_df = metrics["hour_profile_df"]
    delta_total = metrics["delta_total"]
    m5 = metrics["m5"]
    m0 = metrics["m0"]

    # 图1：小时均价/中位数走势 + 5点高亮（保留旧文件名）
    plt.figure(figsize=(9, 5))
    plt.plot(hour_profile_df["pickup_hour"], hour_profile_df["mean_total_amount"], marker="o", label="hourly mean")
    plt.plot(hour_profile_df["pickup_hour"], hour_profile_df["median_total_amount"], marker="s", label="hourly median")
    plt.scatter([5], [m5], color="red", s=90, zorder=5, label="5AM mean")
    plt.annotate(
        f"5AM={m5:.2f}\nALL={m0:.2f}\nDelta={delta_total:.2f}",
        xy=(5, m5),
        xytext=(7.2, m5 + 2.5),
        arrowprops=dict(arrowstyle="->", lw=1),
        fontsize=9
    )
    plt.xlabel("pickup_hour")
    plt.ylabel("total_amount")
    plt.title("Hourly fare profile with 5AM highlight")
    plt.xticks(range(0, 24))
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=180)
    plt.close()

    # 图2：距离分段结构对比
    x = np.arange(len(dist_df))
    width = 0.36
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, dist_df["hour5_share"], width=width, label="5AM")
    plt.bar(x + width / 2, dist_df["all_hours_share"], width=width, label="ALL")
    plt.xticks(x, dist_df["distance_bin"])
    plt.ylabel("share")
    plt.xlabel("distance bin (mile)")
    plt.title("Distance-band mix: 5AM vs ALL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_DIST, dpi=180)
    plt.close()

    # 图3：5点均价提升分解
    comp = decomp_df[decomp_df["component"] != "total_delta_5am_minus_all"].copy()
    plt.figure(figsize=(8, 5))
    plt.bar(
        comp["component"],
        comp["value"],
        color=["#4C78A8", "#F58518", "#54A24B"],
    )
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("USD contribution")
    plt.title("Decomposition of 5AM mean-fare uplift")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_DECOMP, dpi=180)
    plt.close()

    # 图4：关键指标比值（5AM / ALL）
    keep_metrics = [
        "mean_total_amount",
        "median_total_amount",
        "mean_trip_distance",
        "airport_share",
        "share_trip_distance_ge_10",
        "share_trip_distance_ge_20",
        "mean_amount_per_mile",
    ]
    ratio_df = summary_df[summary_df["metric"].isin(keep_metrics)].copy()
    ratio_df = ratio_df.sort_values("ratio_hour5_vs_all", ascending=False)
    plt.figure(figsize=(9, 5))
    plt.bar(ratio_df["metric"], ratio_df["ratio_hour5_vs_all"], color="#4C78A8")
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    plt.ylabel("ratio (5AM / ALL)")
    plt.title("Key metric ratio: 5AM vs ALL")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_RATIO, dpi=180)
    plt.close()


# =========================
# 5. 主流程
# =========================
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(DATA_PATH)
    metrics = compute_metrics(df)

    metrics["summary_df"].to_csv(OUTPUT_SUMMARY_CSV, index=False)
    metrics["decomp_df"].to_csv(OUTPUT_DECOMP_CSV, index=False)
    draw_report(metrics)

    print("Saved:")
    print("-", OUTPUT_IMG)
    print("-", OUTPUT_IMG_DIST)
    print("-", OUTPUT_IMG_DECOMP)
    print("-", OUTPUT_IMG_RATIO)
    print("-", OUTPUT_SUMMARY_CSV)
    print("-", OUTPUT_DECOMP_CSV)


if __name__ == "__main__":
    main()
