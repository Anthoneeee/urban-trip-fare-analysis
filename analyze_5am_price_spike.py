from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================
# 1. 基础配置
# =========================
DATA_PATH = Path("nyc_taxi_2024_cleaned_sample.csv")
OUTPUT_DIR = Path("proposal_outputs_2024")
OUTPUT_IMG = OUTPUT_DIR / "fig_13_5am_price_diagnosis.png"
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

    return {
        "summary_df": summary_df,
        "decomp_df": decomp_df,
        "dist_compare_df": dist_compare_df,
        "delta_total": delta_total,
        "m5": m5,
        "m0": m0,
    }


# =========================
# 4. 可视化：分析过程+代码+结果合并为一图
# =========================
def draw_report(metrics: dict) -> None:
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    summary_df = metrics["summary_df"]
    decomp_df = metrics["decomp_df"]
    dist_df = metrics["dist_compare_df"]
    delta_total = metrics["delta_total"]
    m5 = metrics["m5"]
    m0 = metrics["m0"]

    fig = plt.figure(figsize=(18, 10), dpi=180, constrained_layout=True)
    grid = fig.add_gridspec(2, 2, wspace=0.25, hspace=0.25)

    # 左上：分析过程 + 核心代码
    ax_text = fig.add_subplot(grid[0, 0])
    ax_text.axis("off")
    text = (
        "5AM fare diagnosis workflow\n"
        "1) Load nyc_taxi_2024_cleaned_sample.csv\n"
        "2) Compare 5AM vs ALL hours (price/distance/duration/airport share)\n"
        "3) Compute long-trip share (>=10mi, >=20mi)\n"
        "4) Decompose 5AM mean-fare uplift into:\n"
        "   - airport share change\n"
        "   - non-airport price structure change\n"
        "   - airport price change\n"
        "5) Export one summary figure + CSV tables\n\n"
        "Core code snippet:\n"
        "h5 = df[df['pickup_hour'] == 5]\n"
        "airport_share = h5['is_airport_trip'].mean()\n"
        "long_trip_share = (h5['trip_distance'] >= 10).mean()\n"
        "delta_total = h5['total_amount'].mean() - df['total_amount'].mean()\n"
    )
    ax_text.text(
        0.0,
        1.0,
        text,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        linespacing=1.45,
    )

    # 右上：关键指标表
    ax_table = fig.add_subplot(grid[0, 1])
    ax_table.axis("off")
    table_df = summary_df.copy()
    table_df["hour5"] = table_df["hour5"].round(4)
    table_df["all_hours"] = table_df["all_hours"].round(4)
    table_df["ratio_hour5_vs_all"] = table_df["ratio_hour5_vs_all"].round(3)
    table = ax_table.table(
        cellText=table_df.values,
        colLabels=["metric", "hour5", "all_hours", "ratio"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.4)
    ax_table.set_title("5AM vs ALL: key metric comparison", fontsize=12, pad=8)

    # 左下：距离结构对比
    ax_dist = fig.add_subplot(grid[1, 0])
    x = np.arange(len(dist_df))
    width = 0.36
    ax_dist.bar(x - width / 2, dist_df["hour5_share"], width=width, label="5AM")
    ax_dist.bar(x + width / 2, dist_df["all_hours_share"], width=width, label="ALL")
    ax_dist.set_xticks(x)
    ax_dist.set_xticklabels(dist_df["distance_bin"])
    ax_dist.set_ylabel("share")
    ax_dist.set_title("Distance-band mix: 5AM vs ALL")
    ax_dist.legend()

    # 右下：均价提升分解
    ax_dec = fig.add_subplot(grid[1, 1])
    comp = decomp_df[decomp_df["component"] != "total_delta_5am_minus_all"].copy()
    ax_dec.bar(
        comp["component"],
        comp["value"],
        color=["#4C78A8", "#F58518", "#54A24B"],
    )
    ax_dec.axhline(0, color="black", linewidth=0.8)
    ax_dec.set_ylabel("USD contribution")
    ax_dec.set_title("Decomposition of 5AM mean-fare uplift")
    ax_dec.tick_params(axis="x", rotation=20)

    # 总标题 + 结论
    fig.suptitle(
        (
            "Why is 5AM average fare high?\n"
            f"5AM mean={m5:.2f}, ALL mean={m0:.2f}, delta={delta_total:.2f} | "
            "Main driver: longer-trip mix, not pure time surcharge."
        ),
        fontsize=13,
        y=0.98,
    )

    fig.savefig(OUTPUT_IMG, dpi=180)
    plt.close(fig)


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
    print("-", OUTPUT_SUMMARY_CSV)
    print("-", OUTPUT_DECOMP_CSV)


if __name__ == "__main__":
    main()
