import scipy.stats as stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df1 = pd.read_csv(r'data\OWID DataSet\owid-covid-data-master-file.csv')
df2 = pd.read_csv(r'data\OWID DataSet\owid-covid-data-old.csv')
# # Get numerical columns common in both datasets
# common_numeric_cols = df1.select_dtypes(include=['number']).columns.intersection(df2.select_dtypes(include=['number']).columns)

# # Perform Levene's Test for each common numeric feature
# levene_results = {}
# for col in common_numeric_cols:
#     stat, p_value = stats.levene(df1[col].fillna(0), df2[col].fillna(0))  
#     levene_results[col] = {'Levene Statistic': stat, 'p-value': p_value}

# # Convert results to DataFrame for better readability
# levene_df = pd.DataFrame.from_dict(levene_results, orient='index')

# pd.set_option('display.max_rows', None)  # Show all rows
# pd.set_option('display.max_columns', None)  # Show all columns (if needed)
# pd.set_option('display.expand_frame_repr', False)  # Prevent truncation


# # Display results
# print("\nLevene's Test Results:\n", levene_df)

# # Filter columns where variances are significantly different (p-value < 0.05)
# significant_variance_diff = levene_df[levene_df['p-value'] < 0.05]
# print("\nFeatures with significantly different variances:\n", significant_variance_diff)


# Convert 'date' column to datetime if needed
df1['date'] = pd.to_datetime(df1['date'])
df2['date'] = pd.to_datetime(df2['date'])

# Find common locations between both datasets
common_locations = set(df1['location'].unique()).intersection(df2['location'].unique())

# List of features to test (excluding identifiers and dates)
features = df1.select_dtypes(include=['number']).columns.intersection(df2.select_dtypes(include=['number']).columns)

results = {}

for feature in features:
    for loc in common_locations:
        # Get the time series for this feature and location in both datasets
        series1 = df1[df1['location'] == loc][feature].fillna(0)
        series2 = df2[df2['location'] == loc][feature].fillna(0)
        
        # Only perform the test if both series have more than one observation
        if len(series1) > 1 and len(series2) > 1:
            stat, p_value = stats.levene(series1, series2)
            results[(feature, loc)] = {'Levene Statistic': stat, 'p-value': p_value}

results_df = pd.DataFrame.from_dict(results, orient='index')
results_df.index = pd.MultiIndex.from_tuples(results_df.index, names=['Feature', 'Location'])


pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns 
pd.set_option('display.expand_frame_repr', False)  # Prevent truncation

print("Levene's Test Results (by Feature and Location):")
print(results_df)



# Convert the DataFrame to a string
results_str = results_df.to_string()

# Write the results string to a text file
with open("levene_test_results.txt", "w") as f:
    f.write("Levene's Test Results (by Feature and Location):\n")
    f.write(results_str)

print("Levene's test results have been written to levene_test_results.txt")