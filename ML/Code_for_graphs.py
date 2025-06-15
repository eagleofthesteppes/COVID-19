# import matplotlib.pyplot as plt
# import pandas as pd

# # Sample data for total deaths
# dates = pd.date_range(start='2020-01-22', end='2024-04-06', freq='W')
# total_deaths = [i * 1000 for i in range(len(dates))]  # Example data

# # Sample data for daily new cases
# daily_dates = pd.date_range(start='2020-01-22', end='2024-04-06', freq='D')
# daily_new_cases = [abs(int(10000 * (i % 100))) for i in range(len(daily_dates))]  # Example data

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# # Total deaths line plot
# ax1.plot(dates, total_deaths, color='orange', linewidth=3, label='Deaths')
# ax1.set_title('Total Deaths (Linear Scale)')
# ax1.set_xlabel('Date')
# ax1.set_ylabel('Total Coronavirus Deaths')
# ax1.legend()
# ax1.grid(True)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# Sample data for daily new cases
daily_dates = pd.date_range(start='2020-01-22', end='2024-04-06', freq='D')
daily_new_cases = [abs(int(10000 * (i % 100))) for i in range(len(daily_dates))]

# Create a DataFrame
data = pd.DataFrame({'date': daily_dates, 'new_cases': daily_new_cases})

# Aggregate data by month
data['month'] = data['date'].dt.to_period('M')
monthly_data = data.groupby('month')['new_cases'].sum().reset_index()

# Convert month period to datetime for plotting
monthly_data['month'] = monthly_data['month'].dt.to_timestamp()

fig, ax = plt.subplots(figsize=(12, 6))

# Bar plot
bars = ax.bar(monthly_data['month'], monthly_data['new_cases'], color='skyblue', edgecolor='black', label='Cases per Month')

# Adding data labels
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', fontsize=10, ha='center')

# Setting the title and labels
ax.set_title('Monthly New Cases', fontsize=16, fontweight='bold')
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Cases per Month', fontsize=14)

# Formatting the date on x-axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.autofmt_xdate()

# Adding grid
ax.grid(True, linestyle='--', alpha=0.7)

# Adding legend
ax.legend()

# Tight layout for better spacing
plt.tight_layout()

# Show plot
plt.show()

