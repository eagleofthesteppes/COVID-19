#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.dates as mdates

# Load data with the correct column name for the date
df = pd.read_csv('../data/OWID DataSet/owid-covid-data-master-file.csv', parse_dates=['date'])

# Replace NaN values in the 'total_cases' column with 0
df['total_cases'] = df['total_cases'].fillna(0)

# Filter the DataFrame for the date range from January 2020 to July 2024
start_date = '2020-01-01'
end_date = '2024-07-31'
df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

# Calculate the number of days since the first date
df['Days'] = (df['date'] - df['date'].min()).dt.days

# Define features and target
X = df[['Days']]
y = df['total_cases']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use Polynomial Features for generating artificial trends
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X_train)

# Train the model
model = LinearRegression()
model.fit(X_poly, y_train)

# Generate predictions
X_test_poly = poly.transform(X_test)
y_pred = model.predict(X_test_poly)

# Generate future dates for predictions (up to December 2024)
future_days = np.arange(df['Days'].max() + 1, df['Days'].max() + 31).reshape(-1, 1)
future_days_poly = poly.transform(future_days)
future_predictions = model.predict(future_days_poly)

# Plot the actual and predicted data
plt.figure(figsize=(12, 8))
plt.plot(df['date'], df['total_cases'], label='Actual Cases')
plt.plot(df['date'].iloc[X_test.index], y_pred, label='Predicted Cases', linestyle='--')
plt.plot(df['date'].max() + pd.to_timedelta(future_days.flatten(), unit='D'), future_predictions, label='Future Predictions', linestyle='--')

# Set x-axis and y-axis labels, title
plt.xlabel('Date')
plt.ylabel('Number of Cases')
plt.title('Actual vs Predicted COVID-19 Cases (Jan 2020 - Dec 2024)')

# Set x-axis limits and format
plt.xlim(pd.Timestamp('2020-01-01'), pd.Timestamp('2024-12-31'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Set x-axis ticks yearly
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format x-axis ticks as years

# Set y-axis limits and format
plt.ylim(0, 400000000)  # Set the y-axis limit
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(100000000))  # Set y-axis ticks at intervals of 100,000,000

# Add legend
plt.legend()

# Show plot
plt.show()



###
# add all totals of all continents | keep the timframe constant
###

