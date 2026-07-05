import os
import json
import warnings
import joblib
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

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


def build_Xy(df: pd.DataFrame):
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target col: {TARGET_COL}")

    drop_feats = set(META_COLS + [TARGET_COL] + LEAKY_COLS)
    X = df.drop(columns=[c for c in drop_feats if c in df.columns], errors="ignore")

    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    valid = y.notna().to_numpy()
    X = X.iloc[valid].copy()
    y = y.iloc[valid].copy()

    # Critical: make every feature numeric (prevents XGBoost object dtype crash)
    X = X.apply(pd.to_numeric, errors="coerce")
    X.fillna(0, inplace=True)

    return X, y


def save_model(model, model_name: str, weights_dir: str):
    os.makedirs(weights_dir, exist_ok=True)

    # XGBoost -> JSON
    if hasattr(model, "save_model"):
        path = os.path.join(weights_dir, f"{model_name}.json")
        model.save_model(path)
        return path

    # sklearn -> joblib
    path = os.path.join(weights_dir, f"{model_name}.joblib")
    joblib.dump(model, path)
    return path


def main():
    data_folder = r"/Users/tarushshankar/COVID-19-1/data/OWID DataSet/countrywise_data_owid_master_new4"

    output_root = r"/Users/tarushshankar/COVID-19-1/data/NEW TRAINED MODEL PARAMETERS"
    weights_dir = os.path.join(output_root, "weights")
    meta_dir = os.path.join(output_root, "meta")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

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

    # Original model set (same names + same big params)
    models = {
        # "GradientBoostingRegressor_10000": GradientBoostingRegressor(
        #     n_estimators=10000, random_state=42
        # ),
 
        "RandomForestRegressor_20000": RandomForestRegressor(
            n_estimators=15000, random_state=42, n_jobs=-1
        ),

        #"SVR": SVR(kernel="rbf", C=10, gamma=0.1),

        # "Lasso": Lasso(alpha=0.01, max_iter=10000),

        #"Ridge": Ridge(alpha=0.01, max_iter=10000),

        # "MLPRegressor": MLPRegressor(
        #     hidden_layer_sizes=(200, 200),
        #     random_state=42,
        #     max_iter=10000,
        #     learning_rate_init=0.001
        # ),

        # "XGBoostRegressor_10000": XGBRegressor(
        #     n_estimators=10000,
        #     random_state=42,
        #     verbosity=0,
        #     use_label_encoder=False,
        #     tree_method="hist",
        #     n_jobs=-1,
        # ),
    }

    if not os.path.isdir(data_folder):
        raise FileNotFoundError(f"Data folder not found: {data_folder}")

    file_names = [f for f in os.listdir(data_folder) if f.endswith(".csv")]
    file_names.sort()
    if not file_names:
        raise RuntimeError("No CSV files found.")

    X_list = []
    y_list = []

    kept = 0
    skipped = 0

    used_countries = []
    skipped_countries = []  # list of dicts: {"country": ..., "reason": ...}

    print("Building global training dataset...")
    for name in file_names:
        country = os.path.splitext(name)[0]

        if country in IGNORE:
            skipped += 1
            skipped_countries.append({"country": country, "reason": "in IGNORE list"})
            continue

        path = os.path.join(data_folder, name)
        try:
            df_raw = pd.read_csv(path)
            df = preprocess_df(df_raw)
            X, y = build_Xy(df)

            if len(X) < 10:
                skipped += 1
                skipped_countries.append({"country": country, "reason": "too few usable rows (<10)"})
                continue

            X_list.append(X)
            y_list.append(y)
            kept += 1
            used_countries.append(country)

        except Exception as e:
            skipped += 1
            skipped_countries.append({"country": country, "reason": str(e)})
            print(f"Skipping {country}: {e}")

    if not X_list:
        raise RuntimeError("No usable training data was created. Check filters and data quality.")

    X_train_all = pd.concat(X_list, ignore_index=True)
    y_train_all = pd.concat(y_list, ignore_index=True)

    # Final safety: numeric enforcement + fill
    X_train_all = X_train_all.apply(pd.to_numeric, errors="coerce")
    X_train_all.fillna(0, inplace=True)

    feature_cols = list(X_train_all.columns)

    feature_path = os.path.join(meta_dir, "feature_columns.json")
    with open(feature_path, "w") as fp:
        json.dump(feature_cols, fp, indent=2)

    used_path = os.path.join(meta_dir, "training_countries.json")
    with open(used_path, "w") as fp:
        json.dump(used_countries, fp, indent=2)

    skipped_path = os.path.join(meta_dir, "skipped_countries.json")
    with open(skipped_path, "w") as fp:
        json.dump(skipped_countries, fp, indent=2)

    print(f"\nCountries used for training: {kept}")
    print(f"Countries skipped/ignored: {skipped}")
    print(f"Global train shape: X={X_train_all.shape}, y={y_train_all.shape}")
    print(f"Saved feature columns to: {feature_path}")
    print(f"Saved training countries list to: {used_path}")
    print(f"Saved skipped countries list to: {skipped_path}")

    print("\nCountries used for training:")
    for c in used_countries:
        print(f"  {c}")

    print("\nCountries skipped/ignored:")
    for item in skipped_countries:
        print(f"  {item['country']} | reason: {item['reason']}")

    print("\nTraining models on global data...")
    for model_name, model in models.items():
        print(f"Training: {model_name}")
        model.fit(X_train_all, y_train_all)

        saved_path = save_model(model, model_name, weights_dir)
        print(f"Saved weights: {saved_path}")

    print("Done.")


if __name__ == "__main__":
    main()
