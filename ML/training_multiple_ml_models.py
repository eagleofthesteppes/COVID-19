import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def train_models(path, models):
    """
    Trains multiple regression models on the dataset located at the given path.

    Parameters:
        path (str): The file path to the CSV data.
        models (dict): A dictionary of model names and their corresponding instances.

    Returns:
        dict: A dictionary containing the MSE and R² for each model.
    """
    try:
        # Load the dataset
        data = pd.read_csv(path)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return {model_name: f"Error: {e}" for model_name in models.keys()}
    
    try:
        data = pd.read_csv(path) #path must be a raw string
        missing_data = data.isnull().sum()
        
        data.drop(['excess_mortality_cumulative_absolute', 
            'excess_mortality_cumulative', 
            'excess_mortality', 
            'excess_mortality_cumulative_per_million'], 
            axis=1, 
            inplace=True)
        
        data['date'] = pd.to_datetime(data['date'])

        # Example: Extract year, month, and day as separate features
        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month
        data['day'] = data['date'].dt.day

        # Drop the original 'date' column
        X = data.drop(['new_cases', 'iso_code', 'continent', 'location', 'date'], axis=1)
        y = data['new_cases']

        X.replace('tests performed', 0, inplace=True)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train.fillna(0, inplace=True)
        X_test.fillna(0, inplace=True)

    except Exception as e:
        print(f"Error processing data from {path}: {e}")
        return {model_name: f"Error: {e}" for model_name in models.keys()}
    
    results = {}
    
    for model_name, model in models.items():
        try:
            # Train the model
            model.fit(X_train, y_train)
            # Make predictions
            predictions = model.predict(X_test)
            # Evaluate the model
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            results[model_name] = (mse, r2)
        except Exception as e:
            results[model_name] = f"Error: {e}"
    
    return results

def main():
    # Define the models with their parameters
    models = {
        'GradientBoostingRegressor_10000': GradientBoostingRegressor(n_estimators=10000, random_state=42),
        
        'XGBoostRegressor_10000': XGBRegressor(n_estimators=10000, random_state=42, verbosity=0, use_label_encoder=False, tree_method='gpu_hist'),
        
        'RandomForestRegressor_20000': RandomForestRegressor(n_estimators=20000, random_state=42, n_jobs=-1),
        
        'SVR': SVR(kernel='rbf', C=10, gamma=0.1),  # No native GPU support in scikit-learn for SVR
        
        'Lasso': Lasso(alpha=0.01, max_iter=10000),  # No native GPU support in scikit-learn for Lasso
        
        'Ridge': Ridge(alpha=0.01, max_iter=10000),  # No native GPU support in scikit-learn for Ridge
        
        'MLPRegressor': MLPRegressor(hidden_layer_sizes=(200, 200), random_state=42, max_iter=10000, learning_rate_init=0.001)  # No native GPU support in scikit-learn for MLPRegressor
    }

    # Define the base result folder path
    base_result_folder = r"data\ModelResults_Owid_Old"

    # Verify that the base result folder exists or create it
    os.makedirs(base_result_folder, exist_ok=True)

    # Define the folder path containing the datasets (use raw string to handle backslashes)
    data_folder_path = r'data\OWID DataSet\countrywise_data_owid_old'

    # Verify that the data folder exists
    if not os.path.isdir(data_folder_path):
        print(f"The folder path {data_folder_path} does not exist.")
        return

    # Get list of all CSV files in the data folder
    file_names = [f for f in os.listdir(data_folder_path) if f.endswith('.csv')]
    file_paths = [os.path.join(data_folder_path, f) for f in file_names]

    # Iterate through each file and each model, write results
    for path, name in zip(file_paths, file_names):
        print(f"Processing {name}...")

        # Create a directory named after the dataset (without the .csv extension) in the Model Results folder
        dataset_dir = os.path.join(base_result_folder, os.path.splitext(name)[0])
        os.makedirs(dataset_dir, exist_ok=True)

        # Create output file paths for each model inside the dataset's directory
        output_files = {model_name: os.path.join(dataset_dir, f"{model_name}_Results.txt") for model_name in models.keys()}

        # Open output files for writing
        file_handlers = {}
        try:
            for model_name, output_file in output_files.items():
                file_handlers[model_name] = open(output_file, 'w')
                file_handlers[model_name].write(f"Results for {model_name}\n{'='*50}\n\n")
        except Exception as e:
            print(f"Error opening output files for {name}: {e}")
            # Close any files that were successfully opened before the error
            for handler in file_handlers.values():
                handler.close()
            continue

        # Train models and write results to the respective files
        model_results = train_models(path, models)
        for model_name, result in model_results.items():
            try:
                file_handlers[model_name].write(f"File: {name}\n")
                if isinstance(result, tuple):
                    mse, r2 = result
                    file_handlers[model_name].write(f"Mean Squared Error (MSE): {mse}\n")
                    file_handlers[model_name].write(f"R² Score: {r2}\n\n")
                else:
                    file_handlers[model_name].write(f"{result}\n\n")
            except Exception as e:
                print(f"Error writing results for {model_name} on {name}: {e}")

        # Close all files for the current dataset
        for handler in file_handlers.values():
            handler.close()

        print(f"Results for {name} have been saved to {dataset_dir}.")

if __name__ == "__main__":
    main()