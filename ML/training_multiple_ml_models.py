import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')


def train_models(path, models, dataset_dir, country_name):
    """
    Per-country training with TIME-ORDERED 80/20 split.
    Saves:
      - metrics (returned to caller)
      - per-model CSV with [date, y_test, y_pred] for the test window
    """
    try:
        data = pd.read_csv(path)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return {model_name: f"Error: {e}" for model_name in models.keys()}

    try:
        # Drop excess mortality columns (safe if missing)
        # data.drop(
        #     [
        #         'excess_mortality_cumulative_absolute',
        #         'excess_mortality_cumulative',
        #         'excess_mortality',
        #         'excess_mortality_cumulative_per_million'
        #     ],
        #     axis=1,
        #     inplace=True,
        #     errors="ignore"
        # )

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
        # Parse date + drop invalid
        data['date'] = pd.to_datetime(data['date'], errors="coerce")
        data = data.dropna(subset=['date'])

        # Date features (same logic)
        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month
        data['day'] = data['date'].dt.day

        # Sort by date (critical)
        data = data.sort_values('date').reset_index(drop=True)

        # Build X/y
        X = data.drop(['new_cases', 'iso_code', 'continent', 'location', 'date'], axis=1, errors="ignore")

        drop_feats = set(META_COLS + [TARGET_COL] + LEAKY_COLS)
        X = data.drop(columns=[c for c in drop_feats if c in data.columns], errors="ignore")
        y = data['new_cases']

        # Replace weird string value in X
        X.replace('tests performed', 0, inplace=True)

        # Coerce y numeric + filter invalid rows
        y = pd.to_numeric(y, errors="coerce")
        valid = y.notna().to_numpy()

        X = X.iloc[valid].copy()
        y = y.iloc[valid].copy()
        dates = data.loc[valid, 'date'].reset_index(drop=True)

        # 80/20 time split
        n = len(X)
        if n < 10:
            raise ValueError(f"Too few usable rows after cleaning: {n}")

        split_idx = int(n * 0.8)
        if split_idx <= 0 or split_idx >= n:
            raise ValueError(f"Bad split index computed: {split_idx} for n={n}")

        X_train = X.iloc[:split_idx].copy()
        y_train = y.iloc[:split_idx].copy()

        X_test = X.iloc[split_idx:].copy()
        y_test = y.iloc[split_idx:].copy()
        date_test = dates.iloc[split_idx:].copy()

        X_train.fillna(0, inplace=True)
        X_test.fillna(0, inplace=True)

        print("Train last date:", dates.iloc[split_idx - 1], " | Test first date:", dates.iloc[split_idx])

    except Exception as e:
        print(f"Error processing data from {path}: {e}")
        return {model_name: f"Error: {e}" for model_name in models.keys()}

    results = {}

    # Ensure directory exists
    os.makedirs(dataset_dir, exist_ok=True)

    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results[model_name] = (mse, r2)

            # ✅ Save test-window CSV with date, actual, prediction
            out_df = pd.DataFrame({
                "date": date_test.values,
                "y_test_new_cases": y_test.values,
                "y_pred_new_cases": y_pred
            })

            out_csv = os.path.join(dataset_dir, f"{model_name}_test_predictions.csv")
            out_df.to_csv(out_csv, index=False)

        except Exception as e:
            results[model_name] = f"Error: {e}"

    return results


def main():
    models = {
        'GradientBoostingRegressor_10000': GradientBoostingRegressor(n_estimators=10000, random_state=42),

        'XGBoostRegressor_10000': XGBRegressor(
            n_estimators=10000,
            random_state=42,
            verbosity=0,
            use_label_encoder=False,
            tree_method='hist'
        ),

        'RandomForestRegressor_20000': RandomForestRegressor(n_estimators=20000, random_state=42, n_jobs=-1),

        'SVR': SVR(kernel='rbf', C=10, gamma=0.1),

        'Lasso': Lasso(alpha=0.01, max_iter=10000),

        'Ridge': Ridge(alpha=0.01, max_iter=10000),

        'MLPRegressor': MLPRegressor(hidden_layer_sizes=(200, 200), random_state=42, max_iter=10000, learning_rate_init=0.001)
    }

    base_result_folder = r"/Users/tarushshankar/COVID-19-1/data/Itr5_Owid_Master_ModelResults"
    os.makedirs(base_result_folder, exist_ok=True)

    data_folder_path = r'/Users/tarushshankar/COVID-19-1/data/OWID DataSet/countrywise_data_owid_master_new4'

    if not os.path.isdir(data_folder_path):
        print(f"The folder path {data_folder_path} does not exist.")
        return

    file_names = [f for f in os.listdir(data_folder_path)
            if f.endswith(".csv")]
    # TARGETS = {
    #     "United States","India","Brazil","United Kingdom","Italy","Spain","France","Germany","China","Russia",
    #     "Canada","Australia","South Korea","Japan","Mexico","Indonesia","South Africa",
    #     "Argentina","Turkey","Sweden"
    # }

    # file_names = [f for f in os.listdir(data_folder_path)
    #               if f.endswith(".csv") and os.path.splitext(f)[0] in TARGETS]

    file_names.sort()
    file_paths = [os.path.join(data_folder_path, f) for f in file_names]

    for path, name in zip(file_paths, file_names):
        country_name = os.path.splitext(name)[0]
        print(f"\nProcessing {name}...")

        dataset_dir = os.path.join(base_result_folder, country_name)
        os.makedirs(dataset_dir, exist_ok=True)

        # metrics files (same as before)
        output_files = {model_name: os.path.join(dataset_dir, f"{model_name}_Results.txt")
                        for model_name in models.keys()}

        file_handlers = {}
        try:
            for model_name, output_file in output_files.items():
                file_handlers[model_name] = open(output_file, 'w')
                file_handlers[model_name].write(f"Results for {model_name}\n{'='*50}\n\n")
        except Exception as e:
            print(f"Error opening output files for {name}: {e}")
            for handler in file_handlers.values():
                handler.close()
            continue

        # Train + also save prediction CSVs
        model_results = train_models(path, models, dataset_dir=dataset_dir, country_name=country_name)

        for model_name, result in model_results.items():
            try:
                file_handlers[model_name].write(f"File: {name}\n")
                if isinstance(result, tuple):
                    mse, r2 = result
                    file_handlers[model_name].write(f"Mean Squared Error (MSE): {mse}\n")
                    file_handlers[model_name].write(f"R² Score: {r2}\n")
                    file_handlers[model_name].write(f"Pred CSV: {model_name}_test_predictions.csv\n\n")
                else:
                    file_handlers[model_name].write(f"{result}\n\n")
            except Exception as e:
                print(f"Error writing results for {model_name} on {name}: {e}")

        for handler in file_handlers.values():
            handler.close()

        print(f" Saved metrics + prediction CSVs to {dataset_dir}")


if __name__ == "__main__":
    main()
