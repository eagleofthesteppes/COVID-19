import scipy.stats as stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df1 = pd.read_csv(r'data\OWID DataSet\owid-covid-data-master-file.csv')
df2 = pd.read_csv(r'data\OWID DataSet\owid-covid-data-old.csv')

# Convert 'date' column to datetime for both datasets to align time points
df1['date'] = pd.to_datetime(df1['date'])
df2['date'] = pd.to_datetime(df2['date'])


all_locations = set(df1['location'].unique()).union(df2['location'].unique())

# Get list of features to test (excluding non-numeric columns)
numeric_features = df1.select_dtypes(include=['number']).columns.intersection(df2.select_dtypes(include=['number']).columns)

results = {}

for loc in all_locations:
    # Filter rows by location
    subset1 = df1[df1['location'] == loc]
    subset2 = df2[df2['location'] == loc]
    
    # Merge on 'date' to align time points
    merged = pd.merge(subset1, subset2, on="date", suffixes=("_df1", "_df2"), how="outer")

    for feature in numeric_features:
        # Extract aligned feature values from both datasets
        series1 = merged[f"{feature}_df1"].fillna(0)
        series2 = merged[f"{feature}_df2"].fillna(0)
        
        if len(series1)>1 and len(series2)>1:
            # Perform Levene's test
            stat, p_value = stats.levene(series1, series2)
            results[(feature, loc)] = {'Levene Statistic': stat, 'p-value': p_value}

# Convert results to DataFrame
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