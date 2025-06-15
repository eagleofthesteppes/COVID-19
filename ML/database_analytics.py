
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



df1 = pd.read_csv(r'data\OWID DataSet\owid-covid-data-master-file.csv')
df2 = pd.read_csv(r'data\OWID DataSet\owid-covid-data-old.csv')

# Extract column names
columns_1 = set(df1.columns)
columns_2 = set(df2.columns)

# Find common, unique, and mismatched columns
common_columns = columns_1.intersection(columns_2)
unique_to_df1 = columns_1 - columns_2
unique_to_df2 = columns_2 - columns_1

# Check for columns with the same name but different data types
dtype_mismatch = {
    col: (df1[col].dtype, df2[col].dtype)
    for col in common_columns
    if df1[col].dtype != df2[col].dtype
}

# Display results
print("Common columns:", common_columns)
print("\nColumns unique to OWID Data Masterfile:", unique_to_df1)
print("\nColumns unique to OWID Data Old:", unique_to_df2)
print("\nColumns with mismatched data types:", dtype_mismatch)



def missing_values_summary(df, name=""):
    print(f"\n Assessing missing values for {name} ")
    missing = df.isnull().sum() / len(df) * 100  # Percentage of missing values weighed against total number of rows for an input feature
    missing = missing[missing > 0].sort_values(ascending=False)
    return missing

missing_1 = missing_values_summary(df1, "OWID Data Masterfile") 
missing_2 = missing_values_summary(df2, "OWID Data Old")

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns (if needed)
pd.set_option('display.expand_frame_repr', False)  # Prevent truncation

# Combine missing value reports for easy comparison
missing_comparison = pd.DataFrame({'OWID Data Masterfile': missing_1, 'OWID Data Old': missing_2}).fillna(0)
print("\nComparison of Missing Values:\n", missing_comparison)


