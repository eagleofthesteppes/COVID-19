import os
import warnings
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

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

    # replace known string token
    df.replace("tests performed", 0, inplace=True)

    df = df.sort_values("date").reset_index(drop=True)
    return df


def split_country_time(df: pd.DataFrame, train_frac: float = 0.8):
    """
    Returns:
      X_train, y_train, X_test, y_test, date_test, test_start, test_end
    """
    if "new_cases" not in df.columns:
        raise ValueError("Target column 'new_cases' not found.")

    X = df.drop(DROP_FEATURE_COLS, axis=1, errors="ignore")
    y = pd.to_numeric(df["new_cases"], errors="coerce")
    dates = df["date"].copy()

    # Filter invalid target rows first (keeps alignment)
    valid = y.notna().to_numpy()
    X = X.iloc[valid].copy()
    y = y.iloc[valid].copy()
    dates = dates.iloc[valid].reset_index(drop=True)

    
    # Force every feature column to numeric. Anything non-numeric becomes NaN -> then filled to 0.
    X = X.apply(pd.to_numeric, errors="coerce")
    X.fillna(0, inplace=True)

    n = len(X)
    if n < 10:
        raise ValueError(f"Too few usable rows after cleaning: {n}")

    split_idx = int(n * train_frac)
    if split_idx <= 0 or split_idx >= n:
        raise ValueError(f"Bad split index computed: {split_idx} for n={n}")

    X_train = X.iloc[:split_idx].copy()
    y_train = y.iloc[:split_idx].copy()

    X_test = X.iloc[split_idx:].copy()
    y_test = y.iloc[split_idx:].copy()
    date_test = dates.iloc[split_idx:].copy()

    test_start = str(date_test.iloc[0].date()) if len(date_test) else None
    test_end = str(date_test.iloc[-1].date()) if len(date_test) else None

    return X_train, y_train, X_test, y_test, date_test, test_start, test_end


def align_columns(X_ref: pd.DataFrame, X_other: pd.DataFrame) -> pd.DataFrame:
    X_other = X_other.copy()

    for c in X_ref.columns:
        if c not in X_other.columns:
            X_other[c] = 0

    extra = [c for c in X_other.columns if c not in X_ref.columns]
    if extra:
        X_other.drop(columns=extra, inplace=True)

    X_other = X_other[X_ref.columns]

    # keep numeric
    X_other = X_other.apply(pd.to_numeric, errors="coerce")
    X_other.fillna(0, inplace=True)
    return X_other


def save_model_weights(model, model_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    if hasattr(model, "save_model"):
        out_path = os.path.join(output_dir, f"{model_name}_weights.json")
        model.save_model(out_path)
        return out_path

    out_path = os.path.join(output_dir, f"{model_name}_weights.joblib")
    joblib.dump(model, out_path)
    return out_path


def main():
    models = {
        "XGBoostRegressor_10000": XGBRegressor(
            n_estimators=10000,
            random_state=42,
            verbosity=0,
            use_label_encoder=False,
            tree_method="hist",
        ),
    }

    data_folder_path = r"/Users/tarushshankar/COVID-19-1/data/OWID DataSet/countrywise_data_owid_master_new3"

    output_root = r"/Users/tarushshankar/COVID-19-1/data/ModelWeights_Owid_Master_New3_ALLCOUNTRIES"
    weights_dir = os.path.join(output_root, "weights")
    eval_dir = os.path.join(output_root, "per_country_eval")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    TARGETS = None

    if not os.path.isdir(data_folder_path):
        raise FileNotFoundError(f"The folder path {data_folder_path} does not exist: {data_folder_path}")

    file_names = [
        f for f in os.listdir(data_folder_path)
        if f.endswith(".csv") and (TARGETS is None or os.path.splitext(f)[0] in TARGETS)
    ]
    file_names.sort()
    if not file_names:
        raise RuntimeError("No CSV files found.")

    # Build global train set from per-country train chunks
    X_train_list, y_train_list = [], []
    per_country = {}

    print("\nBuilding per-country 80/20 time split and global training set...")
    total = len(file_names)

    for i, name in enumerate(file_names, 1):
        country = os.path.splitext(name)[0]
        path = os.path.join(data_folder_path, name)
        print(f"[{i}/{total}] {country}")

        try:
            df = pd.read_csv(path)
            df = preprocess_country_df(df)

            X_tr, y_tr, X_te, y_te, date_te, test_start, test_end = split_country_time(df, train_frac=0.8)

            country_dir = os.path.join(eval_dir, country)
            os.makedirs(country_dir, exist_ok=True)
            per_country[country] = {
                "X_test": X_te,
                "y_test": y_te,
                "date_test": date_te,
                "country_dir": country_dir,
                "test_start": test_start,
                "test_end": test_end,
            }

            X_train_list.append(X_tr)
            y_train_list.append(y_tr)

        except Exception as e:
            print(f"   Skipping {country}: {e}")

    if not X_train_list:
        raise RuntimeError("No usable training data was created from the countries.")

    X_train_all = pd.concat(X_train_list, ignore_index=True)
    y_train_all = pd.concat(y_train_list, ignore_index=True)

    # Ensure numeric, just in case
    X_train_all = X_train_all.apply(pd.to_numeric, errors="coerce")
    X_train_all.fillna(0, inplace=True)

    print(f"\nGlobal train shape: X={X_train_all.shape}, y={y_train_all.shape}")
    print(f"Countries kept for per-country eval: {len(per_country)}")

    print("\nTraining global models and saving weights...")
    for model_name, model in models.items():
        print(f"Training: {model_name}")
        model.fit(X_train_all, y_train_all)
        saved = save_model_weights(model, model_name, weights_dir)
        print(f" Saved: {saved}")

    print("\nDONE (training only).")


if __name__ == "__main__":
    main()
