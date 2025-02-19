import scipy.stats as stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df1 = pd.read_csv(r'data\OWID DataSet\owid-covid-data-master-file.csv')
df2 = pd.read_csv(r'data\OWID DataSet\owid-covid-data-old.csv')
# Get numerical columns common in both datasets
common_numeric_cols = df1.select_dtypes(include=['number']).columns.intersection(df2.select_dtypes(include=['number']).columns)

# Perform Levene's Test for each common numeric feature
levene_results = {}
for col in common_numeric_cols:
    stat, p_value = stats.levene(df1[col].fillna(0), df2[col].fillna(0))  
    levene_results[col] = {'Levene Statistic': stat, 'p-value': p_value}

# Convert results to DataFrame for better readability
levene_df = pd.DataFrame.from_dict(levene_results, orient='index')

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns (if needed)
pd.set_option('display.expand_frame_repr', False)  # Prevent truncation


# Display results
print("\nLevene's Test Results:\n", levene_df)

# Filter columns where variances are significantly different (p-value < 0.05)
significant_variance_diff = levene_df[levene_df['p-value'] < 0.05]
print("\nFeatures with significantly different variances:\n", significant_variance_diff)

