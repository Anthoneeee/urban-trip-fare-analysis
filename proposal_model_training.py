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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# =========================
# 0. 路径设置
# =========================
DATA_PATH = "nyc_taxi_2024_cleaned_sample.csv"
OUTPUT_DIR = Path("proposal_outputs_2024")
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================
# 0.5 Bootstrap 工具函数
# =========================
def bootstrap_prediction_metrics_ci(y_true, y_pred, n_boot=2000, random_state=42):
    """
    对预测指标（RMSE/MAE/R2）做 bootstrap，返回 95% CI。
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
    },
    {
        "name": "Gradient Boosting",
        "estimator": GradientBoostingRegressor(random_state=42),
        "param_grid": {
            "model__n_estimators": [200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__min_samples_split": [2],
            "model__min_samples_leaf": [1, 3],
            "model__subsample": [0.8, 1.0]
        }
    }
]


# =========================
# 4. 训练模型并保存结果
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

results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
results_df.to_csv(OUTPUT_DIR / "model_results_detailed.csv", index=False)
results_df[["model", "RMSE", "MAE", "R2"]].to_csv(OUTPUT_DIR / "model_results.csv", index=False)

print("\nSaved:")
print("-", OUTPUT_DIR / "model_results_detailed.csv")
print("-", OUTPUT_DIR / "model_results.csv")


# =========================
# 5. 模型指标 bootstrap 区间
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
