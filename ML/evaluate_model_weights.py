import os
import json
import warnings
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

DROP_COLS = [
    "excess_mortality_cumulative_absolute",
    "excess_mortality_cumulative",
    "excess_mortality",
    "excess_mortality_cumulative_per_million",
]

META_COLS = ["iso_code", "continent", "location", "date"]

LEAKY_COLS = [
    "total_cases",
    "total_cases_per_million",
    "new_cases_smoothed",
    "new_cases_per_million",
    "new_cases_smoothed_per_million",
]

TARGET_COL = "new_cases"


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop(DROP_COLS, axis=1, inplace=True, errors="ignore")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    df.replace("tests performed", 0, inplace=True)

    df = df.sort_values("date").reset_index(drop=True)
    return df


def build_Xy_dates(df: pd.DataFrame):
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target col: {TARGET_COL}")

    drop_feats = set(META_COLS + [TARGET_COL] + LEAKY_COLS)
    X = df.drop(columns=[c for c in drop_feats if c in df.columns], errors="ignore")

    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    dates = df["date"].copy()

    valid = y.notna().to_numpy()
    X = X.iloc[valid].copy()
    y = y.iloc[valid].copy()
    dates = dates.iloc[valid].reset_index(drop=True)

    X = X.apply(pd.to_numeric, errors="coerce")
    X.fillna(0, inplace=True)

    return X, y, dates


def align_to_features(X: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = X.copy()

    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0

    extra = [c for c in X.columns if c not in feature_cols]
    if extra:
        X.drop(columns=extra, inplace=True)

    X = X[feature_cols]
    return X


def load_model(weights_dir: str, model_name: str):
    joblib_path = os.path.join(weights_dir, f"{model_name}.joblib")
    if os.path.exists(joblib_path):
        return joblib.load(joblib_path), joblib_path

    json_path = os.path.join(weights_dir, f"{model_name}.json")
    if os.path.exists(json_path):
        m = XGBRegressor()
        m.load_model(json_path)
        return m, json_path

    raise FileNotFoundError(f"No weights found for model '{model_name}' in {weights_dir}")


def safe_filename(name: str) -> str:
    bad = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    out = name
    for ch in bad:
        out = out.replace(ch, "_")
    return out.strip()


def merge_country_metrics(country_metrics_path: str, metrics_df_new: pd.DataFrame) -> pd.DataFrame:
    """
    Merge new metrics rows into an existing country metrics_summary.csv.
    - If file exists: replace rows for the same model(s) (update behavior)
    - Else: just write new rows
    """
    metrics_df_new = metrics_df_new.copy()

    if os.path.exists(country_metrics_path):
        try:
            old = pd.read_csv(country_metrics_path)
        except Exception:
            old = pd.DataFrame()

        if not old.empty and "model" in old.columns:
            models_in_run = set(metrics_df_new["model"].astype(str).tolist())
            old = old[~old["model"].astype(str).isin(models_in_run)].copy()

        merged = pd.concat([old, metrics_df_new], ignore_index=True)
    else:
        merged = metrics_df_new

    if "r2" in merged.columns and "mse" in merged.columns:
        merged = merged.sort_values(["r2", "mse"], ascending=[False, True], na_position="last")

    return merged


def main():
    data_folder = r"/Users/tarushshankar/COVID-19-1/data/OWID DataSet/countrywise_data_owid_master_new4"

    output_root = r"/Users/tarushshankar/COVID-19-1/data/NEW TRAINED MODEL PARAMETERS"
    weights_dir = os.path.join(output_root, "weights")
    meta_dir = os.path.join(output_root, "meta")
    eval_dir = os.path.join(output_root, "holdout_eval")
    os.makedirs(eval_dir, exist_ok=True)

    feature_path = os.path.join(meta_dir, "feature_columns.json")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Missing feature_columns.json at: {feature_path}")

    with open(feature_path, "r") as fp:
        feature_cols = json.load(fp)

    IGNORE = {
        "World",
        "Africa",
        "Asia",
        "Europe",
        "North America",
        "South America",
        "Oceania",
        "High-income countries",
        "Upper-middle-income countries",
        "Lower-middle-income countries",
        "Low-income countries",
        "European Union (27)",
    }

    MODEL_NAMES = [
        #"Lasso",
         #"Ridge",
        #"XGBoostRegressor_10000",
         #"GradientBoostingRegressor_10000",
         "RandomForestRegressor_20000",
        # "SVR",
         #"MLPRegressor",
    ]

    if not os.path.isdir(data_folder):
        raise FileNotFoundError(f"Data folder not found: {data_folder}")

    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(f"Weights folder not found: {weights_dir}")

    models = {}
    for name in MODEL_NAMES:
        m, p = load_model(weights_dir, name)
        models[name] = (m, p)

    if not models:
        raise RuntimeError("No models could be loaded. Check MODEL_NAMES and weights_dir contents.")

    run_tag = "__".join([safe_filename(m) for m in MODEL_NAMES]) if MODEL_NAMES else "NO_MODELS"
    all_rows = []

    for country in sorted(IGNORE):
        csv_path = os.path.join(data_folder, f"{country}.csv")
        if not os.path.exists(csv_path):
            print(f"Missing holdout file: {country}.csv")
            continue

        country_dir = os.path.join(eval_dir, safe_filename(country))
        os.makedirs(country_dir, exist_ok=True)

        try:
            df_raw = pd.read_csv(csv_path)
            df = preprocess_df(df_raw)
            X_raw, y, dates = build_Xy_dates(df)
            X = align_to_features(X_raw, feature_cols)
            X.fillna(0, inplace=True)
        except Exception as e:
            print(f"Failed preprocessing for {country}: {e}")
            continue

        metrics_rows = []

        for model_name, (model, wpath) in models.items():
            try:
                preds = model.predict(X)

                mse = mean_squared_error(y, preds)
                r2 = r2_score(y, preds)

                out_df = X.copy()
                out_df.insert(0, "date", dates.values)
                out_df.insert(1, "y_true_new_cases", y.values)
                out_df.insert(2, "y_pred_new_cases", preds)

                pred_csv_name = f"{model_name}_predictions.csv"
                pred_csv_path = os.path.join(country_dir, pred_csv_name)
                out_df.to_csv(pred_csv_path, index=False)

                row = {
                    "country": country,
                    "model": model_name,
                    "mse": mse,
                    "r2": r2,
                    "n": int(len(y)),
                    "weights_file": os.path.basename(wpath),
                    "pred_csv": pred_csv_name,
                }
                metrics_rows.append(row)
                all_rows.append(row)

            except Exception as e:
                row = {
                    "country": country,
                    "model": model_name,
                    "mse": None,
                    "r2": None,
                    "n": int(len(y)),
                    "weights_file": os.path.basename(wpath),
                    "error": str(e),
                }
                metrics_rows.append(row)
                all_rows.append(row)

        metrics_df_new = pd.DataFrame(metrics_rows)

        metrics_path = os.path.join(country_dir, "metrics_summary.csv")
        merged = merge_country_metrics(metrics_path, metrics_df_new)
        merged.to_csv(metrics_path, index=False)

        print(f"Wrote: {country_dir}")



if __name__ == "__main__":
    main()
