from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor

try:
    from catboost import CatBoostRegressor
except ImportError as exc:
    raise SystemExit(
        "CatBoost is not installed in the current environment. "
        "Install it first, for example: pip install catboost"
    ) from exc


# =========================
# 0. 路径设置
# 0. Path setup
# =========================
DATA_PATH = "nyc_taxi_2024_cleaned_sample.csv"
OUTPUT_DIR = Path("proposal_outputs_2024")
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================
# 0.5 Bootstrap 工具函数
# 0.5 Bootstrap utility function
# =========================
def bootstrap_prediction_metrics_ci(y_true, y_pred, n_boot=2000, random_state=42):
    """
    对预测指标（RMSE/MAE/R2）做 bootstrap，返回 95% CI。
    Run bootstrap on prediction metrics (RMSE/MAE/R2) and return 95% confidence intervals.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(random_state)

    rmse_boot = np.empty(n_boot)
    mae_boot = np.empty(n_boot)
    r2_boot = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        yp = y_pred[idx]
        rmse_boot[i] = np.sqrt(mean_squared_error(yt, yp))
        mae_boot[i] = mean_absolute_error(yt, yp)
        r2_boot[i] = r2_score(yt, yp)

    def _pack(arr):
        return {
            "bootstrap_mean": float(arr.mean()),
            "bootstrap_std": float(arr.std(ddof=1)),
            "ci_lower_95": float(np.quantile(arr, 0.025)),
            "ci_upper_95": float(np.quantile(arr, 0.975)),
        }

    return {
        "RMSE": _pack(rmse_boot),
        "MAE": _pack(mae_boot),
        "R2": _pack(r2_boot),
    }


# =========================
# 1. 读取数据
# 1. Load data
# =========================
print("=" * 60)
print("1. Reading data for model training...")
print("=" * 60)

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"]
)

print("Data loaded. Shape:", df.shape)


# =========================
# 2. 建模特征与切分
# 2. Modeling features and split
# =========================
print("\n" + "=" * 60)
print("2. Prepare features and time split")
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

# 先按时间排序，再做时间切分，避免信息泄漏
# Sort by time first, then do a time-based split to avoid data leakage.
model_df = df[features + [target, "tpep_pickup_datetime"]].copy()
model_df = model_df.sort_values("tpep_pickup_datetime").reset_index(drop=True)

X = model_df[features]
y = model_df[target]

split_idx = int(len(model_df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print("Train size:", len(X_train), "| Test size:", len(X_test))


# =========================
# 3. 预处理与模型配置
# 3. Preprocessing and model configuration
# =========================
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

# 训练集内部时间序列交叉验证（稳定版：3 折）
# Time-series cross-validation inside the training set (stable version: 3 folds).
tscv = TimeSeriesSplit(n_splits=3)

model_configs = [
    {
        "name": "Linear Regression",
        "estimator": LinearRegression(),
        "param_grid": {
            "model__fit_intercept": [True, False]
        }
    },
    {
        "name": "Ridge",
        "estimator": Ridge(),
        "param_grid": {
            "model__alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]
        }
    },
    {
        "name": "Lasso",
        "estimator": Lasso(random_state=42, max_iter=10000),
        "param_grid": {
            "model__alpha": [0.0005, 0.001, 0.005, 0.01, 0.05]
        }
    },
    {
        "name": "ElasticNet",
        "estimator": ElasticNet(random_state=42, max_iter=10000),
        "param_grid": {
            "model__alpha": [0.0005, 0.001, 0.005, 0.01, 0.05],
            "model__l1_ratio": [0.2, 0.5, 0.8]
        }
    },
    {
        "name": "Random Forest",
        "estimator": RandomForestRegressor(random_state=42, n_jobs=1),
        "param_grid": {
            "model__n_estimators": [200, 350],
            "model__max_depth": [None, 12, 20],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": ["sqrt"]
        }
    }
]


# =========================
# 4. 训练模型并保存结果
# 4. Train models and save results
# =========================
print("\n" + "=" * 60)
print("4. Training models with GridSearchCV")
print("=" * 60)

results = []
test_pred_cache = {}

for cfg in model_configs:
    print("\n" + "-" * 60)
    print(f"Training: {cfg['name']}")
    print("-" * 60)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", cfg["estimator"])
    ])

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=cfg["param_grid"],
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        n_jobs=1,
        verbose=1
    )

    # 只在训练集上调参
    # Tune hyperparameters on the training set only.
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    preds = best_model.predict(X_test)
    test_pred_cache[cfg["name"]] = np.asarray(preds)

    cv_rmse = -search.best_score_
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    medae = median_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("Best params:", search.best_params_)
    print("CV RMSE    =", round(cv_rmse, 4))
    print("Test RMSE  =", round(rmse, 4))
    print("Test MAE   =", round(mae, 4))
    print("Test MedAE =", round(medae, 4))
    print("Test R2    =", round(r2, 4))

    results.append({
        "model": cfg["name"],
        "best_params": str(search.best_params_),
        "CV_RMSE": cv_rmse,
        "RMSE": rmse,
        "MAE": mae,
        "MedAE": medae,
        "R2": r2
    })

# 新版 CatBoost：使用更丰富的 zone 特征单独调参
# Tuned CatBoost: use richer zone features and a dedicated tuning loop.
print("\n" + "-" * 60)
print("Training: CatBoost (Tuned, with zone features)")
print("-" * 60)

cb_numeric_features = [
    "trip_distance",
    "trip_duration_minutes",
    "passenger_count",
    "pickup_month",
    "pickup_hour",
    "pickup_weekday",
]
cb_categorical_features = [
    "payment_type",
    "is_weekend",
    "is_airport_trip",
    "pickup_borough",
    "dropoff_borough",
    "PULocationID",
    "DOLocationID",
    "pickup_zone",
    "dropoff_zone",
    "pickup_service_zone",
    "dropoff_service_zone",
]
cb_features = cb_numeric_features + cb_categorical_features

cb_required = set(cb_features + [target, "tpep_pickup_datetime"])
missing_cb_cols = sorted(cb_required - set(df.columns))
if missing_cb_cols:
    raise ValueError(f"Missing required columns for CatBoost: {missing_cb_cols}")

df_sorted = df.sort_values("tpep_pickup_datetime").reset_index(drop=True)
cb_df = df_sorted[cb_features + [target]].copy()

cb_split_idx = int(len(cb_df) * 0.8)
cb_train_df = cb_df.iloc[:cb_split_idx].copy()
cb_test_df = cb_df.iloc[cb_split_idx:].copy()

cb_val_split_idx = int(len(cb_train_df) * 0.85)
if cb_val_split_idx <= 0 or cb_val_split_idx >= len(cb_train_df):
    cb_val_split_idx = int(len(cb_train_df) * 0.8)

cb_fit_df = cb_train_df.iloc[:cb_val_split_idx].copy()
cb_val_df = cb_train_df.iloc[cb_val_split_idx:].copy()

X_fit_cb = cb_fit_df[cb_features].copy()
y_fit_cb = cb_fit_df[target].copy()
X_val_cb = cb_val_df[cb_features].copy()
y_val_cb = cb_val_df[target].copy()
X_train_cb = cb_train_df[cb_features].copy()
y_train_cb = cb_train_df[target].copy()
X_test_cb = cb_test_df[cb_features].copy()

for c in cb_categorical_features:
    X_fit_cb[c] = X_fit_cb[c].astype(str)
    X_val_cb[c] = X_val_cb[c].astype(str)
    X_train_cb[c] = X_train_cb[c].astype(str)
    X_test_cb[c] = X_test_cb[c].astype(str)

cb_cat_indices = [X_fit_cb.columns.get_loc(c) for c in cb_categorical_features]
cb_candidates = [
    {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3, "bagging_temperature": 0.0},
    {"depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 3, "bagging_temperature": 0.2},
    {"depth": 8, "learning_rate": 0.03, "l2_leaf_reg": 6, "bagging_temperature": 0.5},
    {"depth": 10, "learning_rate": 0.03, "l2_leaf_reg": 6, "bagging_temperature": 0.2},
]

best_cb_val_rmse = np.inf
best_cb_params = None
best_cb_iteration = -1

for params in cb_candidates:
    cb_model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=2500,
        random_seed=42,
        verbose=False,
        **params,
    )
    cb_model.fit(
        X_fit_cb,
        y_fit_cb,
        cat_features=cb_cat_indices,
        eval_set=(X_val_cb, y_val_cb),
        use_best_model=True,
        early_stopping_rounds=120,
    )
    cb_val_pred = cb_model.predict(X_val_cb)
    cb_val_rmse = np.sqrt(mean_squared_error(y_val_cb, cb_val_pred))

    if cb_val_rmse < best_cb_val_rmse:
        best_cb_val_rmse = cb_val_rmse
        best_cb_params = params
        best_cb_iteration = int(cb_model.get_best_iteration())

cb_final_iterations = max(300, int(best_cb_iteration)) if best_cb_iteration > 0 else 1200
cb_final = CatBoostRegressor(
    loss_function="RMSE",
    eval_metric="RMSE",
    iterations=cb_final_iterations,
    random_seed=42,
    verbose=False,
    **best_cb_params,
)
cb_final.fit(X_train_cb, y_train_cb, cat_features=cb_cat_indices)

cb_preds = cb_final.predict(X_test_cb)
if len(cb_preds) != len(y_test):
    raise ValueError(
        "CatBoost test prediction length does not match the main test split. "
        f"CatBoost={len(cb_preds)}, main={len(y_test)}"
    )
test_pred_cache["CatBoost (Tuned)"] = np.asarray(cb_preds)

cb_rmse = np.sqrt(mean_squared_error(y_test, cb_preds))
cb_mae = mean_absolute_error(y_test, cb_preds)
cb_medae = median_absolute_error(y_test, cb_preds)
cb_r2 = r2_score(y_test, cb_preds)

print("Best params:", {**best_cb_params, "best_iteration": int(best_cb_iteration)})
print("Validation RMSE =", round(best_cb_val_rmse, 4))
print("Test RMSE       =", round(cb_rmse, 4))
print("Test MAE        =", round(cb_mae, 4))
print("Test MedAE      =", round(cb_medae, 4))
print("Test R2         =", round(cb_r2, 4))

results.append({
    "model": "CatBoost (Tuned)",
    "best_params": str({**best_cb_params, "best_iteration": int(best_cb_iteration)}),
    "CV_RMSE": float(best_cb_val_rmse),
    "RMSE": float(cb_rmse),
    "MAE": float(cb_mae),
    "MedAE": float(cb_medae),
    "R2": float(cb_r2),
})

results_df = pd.DataFrame(results).sort_values("R2", ascending=False)

# 生成“相对 baseline / 相对 RF”的提升汇总，便于汇报
# Build improvement summary vs baseline and vs random forest for reporting.
if "Linear Regression" in set(results_df["model"]):
    baseline_row = results_df.loc[results_df["model"] == "Linear Regression"].iloc[0]
    baseline_rmse = float(baseline_row["RMSE"])
    baseline_r2 = float(baseline_row["R2"])
else:
    baseline_rmse = np.nan
    baseline_r2 = np.nan

if "Random Forest" in set(results_df["model"]):
    rf_row = results_df.loc[results_df["model"] == "Random Forest"].iloc[0]
    rf_rmse = float(rf_row["RMSE"])
    rf_r2 = float(rf_row["R2"])
else:
    rf_rmse = np.nan
    rf_r2 = np.nan

results_df["RMSE_improvement_vs_baseline_pct"] = (
    (baseline_rmse - results_df["RMSE"]) / baseline_rmse * 100.0
)
results_df["R2_gain_vs_baseline"] = results_df["R2"] - baseline_r2
results_df["RMSE_improvement_vs_rf_pct"] = (
    (rf_rmse - results_df["RMSE"]) / rf_rmse * 100.0
)
results_df["R2_gain_vs_rf"] = results_df["R2"] - rf_r2

results_df.to_csv(OUTPUT_DIR / "model_results_detailed.csv", index=False)
results_df[["model", "RMSE", "MAE", "R2"]].to_csv(OUTPUT_DIR / "model_results.csv", index=False)
results_df[
    [
        "model",
        "RMSE",
        "MAE",
        "R2",
        "RMSE_improvement_vs_baseline_pct",
        "R2_gain_vs_baseline",
        "RMSE_improvement_vs_rf_pct",
        "R2_gain_vs_rf",
    ]
].to_csv(OUTPUT_DIR / "model_improvement_summary.csv", index=False)

print("\nSaved:")
print("-", OUTPUT_DIR / "model_results_detailed.csv")
print("-", OUTPUT_DIR / "model_results.csv")
print("-", OUTPUT_DIR / "model_improvement_summary.csv")


# =========================
# 5. 模型指标 bootstrap 区间
# 5. Bootstrap intervals for model metrics
# =========================
print("\n" + "=" * 60)
print("5. Bootstrap CI for model metrics")
print("=" * 60)

N_BOOT_METRIC = 2000
y_test_np = y_test.to_numpy()
point_metrics = results_df.set_index("model")[["RMSE", "MAE", "R2"]]

metric_rows = []
for model_name, y_pred in test_pred_cache.items():
    metric_ci = bootstrap_prediction_metrics_ci(
        y_test_np,
        y_pred,
        n_boot=N_BOOT_METRIC,
        random_state=42
    )

    for metric_name in ["RMSE", "MAE", "R2"]:
        metric_rows.append({
            "model": model_name,
            "metric": metric_name,
            "point_estimate": float(point_metrics.loc[model_name, metric_name]),
            "bootstrap_mean": metric_ci[metric_name]["bootstrap_mean"],
            "bootstrap_std": metric_ci[metric_name]["bootstrap_std"],
            "ci_lower_95": metric_ci[metric_name]["ci_lower_95"],
            "ci_upper_95": metric_ci[metric_name]["ci_upper_95"],
            "n_boot": N_BOOT_METRIC,
        })

bootstrap_metric_df = pd.DataFrame(metric_rows)
bootstrap_metric_df.to_csv(OUTPUT_DIR / "bootstrap_model_metric_ci.csv", index=False)

print("Saved:")
print("-", OUTPUT_DIR / "bootstrap_model_metric_ci.csv")
print("\nBootstrap CI head:")
print(bootstrap_metric_df.head(12))

print("\nDone.")
