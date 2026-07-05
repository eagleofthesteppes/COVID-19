import os
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

DROP_FEATURE_COLS = ["new_cases", "iso_code", "continent", "location", "date"]


def preprocess_country_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.drop(DROP_COLS, axis=1, inplace=True, errors="ignore")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    df.replace("tests performed", 0, inplace=True)

    # time order
    df = df.sort_values("date").reset_index(drop=True)
    return df


def split_country_time(df: pd.DataFrame, train_frac: float = 0.8):
    """
    Return X_test, y_test, date_test, test_start, test_end
    """
    if "new_cases" not in df.columns:
        raise ValueError("Target column 'new_cases' not found.")

    X = df.drop(DROP_FEATURE_COLS, axis=1, errors="ignore")
    y = pd.to_numeric(df["new_cases"], errors="coerce")
    dates = df["date"].copy()

    valid = y.notna().to_numpy()
    X = X.iloc[valid].copy()
    y = y.iloc[valid].copy()
    dates = dates.iloc[valid].reset_index(drop=True)

    X.fillna(0, inplace=True)

    n = len(X)
    if n < 10:
        raise ValueError(f"Too few usable rows after cleaning: {n}")

    split_idx = int(n * train_frac)
    if split_idx <= 0 or split_idx >= n:
        raise ValueError(f"Bad split index computed: {split_idx} for n={n}")

    X_test = X.iloc[split_idx:].copy()
    y_test = y.iloc[split_idx:].copy()
    date_test = dates.iloc[split_idx:].copy()

    test_start = str(date_test.iloc[0].date()) if len(date_test) else None
    test_end = str(date_test.iloc[-1].date()) if len(date_test) else None

    return X_test, y_test, date_test, test_start, test_end


def align_columns(ref_cols, X_other: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure X_other has same columns/order as ref_cols.
    Missing -> 0, extra -> dropped.
    """
    X_other = X_other.copy()

    for c in ref_cols:
        if c not in X_other.columns:
            X_other[c] = 0

    extras = [c for c in X_other.columns if c not in ref_cols]
    if extras:
        X_other.drop(columns=extras, inplace=True)

    X_other = X_other[list(ref_cols)]
    X_other.fillna(0, inplace=True)
    return X_other


def load_model(model_name: str, weights_dir: str, model_type: str):
    """
    model_type:
      - "sklearn" -> loads {model_name}_weights.joblib
      - "xgb"     -> loads {model_name}_weights.json into an XGBRegressor()
    """
    if model_type == "xgb":
        path = os.path.join(weights_dir, f"{model_name}_weights.json")
        m = XGBRegressor()
        m.load_model(path)
        return m, path

    path = os.path.join(weights_dir, f"{model_name}_weights.joblib")
    m = joblib.load(path)
    return m, path


def get_reference_columns(model):
    """
    Best case: sklearn model saved from DataFrame training has feature_names_in_.
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return None


def main():
    # ---- PATHS (use your existing directory) ----
    ROOT = r"/Users/tarushshankar/COVID-19-1/data/ModelWeights_Owid_Master_New3_ALLCOUNTRIES"

    weights_dir = os.path.join(ROOT, "weights")
    per_country_eval_dir = os.path.join(ROOT, "per_country_eval")

    data_folder_path = r"/Users/tarushshankar/COVID-19-1/data/OWID DataSet/countrywise_data_owid_master_new3"

    # ---- CHOOSE ONE MODEL ----
    MODEL_NAME = "XGBoostRegressor_10000"   # change if needed
    MODEL_TYPE = "xgb"                       # "sklearn" or "xgb"
    TRAIN_FRAC = 0.8

    # Optional subset
    TARGETS = None
    # TARGETS = {"Canada", "United States", "India"}

    # ---- checks ----
    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(f"weights_dir not found: {weights_dir}")
    if not os.path.isdir(per_country_eval_dir):
        raise FileNotFoundError(f"per_country_eval_dir not found: {per_country_eval_dir}")
    if not os.path.isdir(data_folder_path):
        raise FileNotFoundError(f"data_folder_path not found: {data_folder_path}")

    # ---- load model ----
    model, model_path = load_model(MODEL_NAME, weights_dir, MODEL_TYPE)
    print(f"Loaded model: {MODEL_NAME}")
    print(f"From: {model_path}")

    ref_cols = get_reference_columns(model)
    if ref_cols is None:
        raise RuntimeError(
            "This saved model does not contain feature_names_in_.\n"
            "That means sklearn doesn't know the training column order.\n\n"
            "Fix options:\n"
            "1) Retrain using pandas DataFrame (not numpy) so feature_names_in_ is stored.\n"
            "2) Save global training columns during training (recommended) and load them here."
        )

    # ---- iterate countries ----
    file_names = [
        f for f in os.listdir(data_folder_path)
        if f.endswith(".csv") and (TARGETS is None or os.path.splitext(f)[0] in TARGETS)
    ]
    file_names.sort()
    if not file_names:
        raise RuntimeError("No country CSVs found.")

    for i, fname in enumerate(file_names, 1):
        country = os.path.splitext(fname)[0]
        csv_path = os.path.join(data_folder_path, fname)

        # write into EXISTING country folder (create if missing)
        country_dir = os.path.join(per_country_eval_dir, country)
        os.makedirs(country_dir, exist_ok=True)

        print(f"\n[{i}/{len(file_names)}] Evaluating: {country}")

        try:
            df = pd.read_csv(csv_path)
            df = preprocess_country_df(df)

            X_test, y_test, date_test, test_start, test_end = split_country_time(df, train_frac=TRAIN_FRAC)

            # align to training columns
            X_test = align_columns(ref_cols, X_test)

            preds = model.predict(X_test)

            mse = mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)

            # predictions CSV
            pred_df = pd.DataFrame({
                "date": date_test.values,
                "y_test_new_cases": y_test.values,
                "y_pred_new_cases": preds
            })
            pred_csv = os.path.join(country_dir, f"{MODEL_NAME}_test_predictions.csv")
            pred_df.to_csv(pred_csv, index=False)

            # per-country metrics CSV (one file per country for THIS model)
            metrics_df = pd.DataFrame([{
                "country": country,
                "model": MODEL_NAME,
                "mse": mse,
                "r2": r2,
                "n_test": int(len(y_test)),
                "test_start": test_start,
                "test_end": test_end,
                "weights_file": os.path.basename(model_path),
                "pred_csv": os.path.basename(pred_csv),
            }])
            metrics_csv = os.path.join(country_dir, f"metrics_summary_{MODEL_NAME}.csv")
            metrics_df.to_csv(metrics_csv, index=False)

            print(f"✅ mse={mse:.4f} r2={r2:.4f}")
            print(f"   wrote: {metrics_csv}")
            print(f"   wrote: {pred_csv}")

        except Exception as e:
            err_df = pd.DataFrame([{
                "country": country,
                "model": MODEL_NAME,
                "error": str(e),
            }])
            metrics_csv = os.path.join(country_dir, f"metrics_summary_{MODEL_NAME}.csv")
            err_df.to_csv(metrics_csv, index=False)
            print(f"❌ Failed: {e}")
            print(f"   wrote error: {metrics_csv}")

    print(f"\nDONE. Results were written into:\n{per_country_eval_dir}")


if __name__ == "__main__":
    main()
