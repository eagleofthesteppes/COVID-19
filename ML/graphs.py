import matplotlib.pyplot as plt
import pandas as pd

# Sample data for total deaths
dates = pd.date_range(start='2020-01-22', end='2024-04-06', freq='W')
total_deaths = [i * 1000 for i in range(len(dates))]  # Example data

# Sample data for daily new cases
daily_dates = pd.date_range(start='2020-01-22', end='2024-04-06', freq='D')
daily_new_cases = [abs(int(10000 * (i % 100))) for i in range(len(daily_dates))]  # Example data

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Total deaths line plot
ax1.plot(dates, total_deaths, color='orange', linewidth=3, label='Deaths')
ax1.set_title('Total Deaths (Linear Scale)')
ax1.set_xlabel('Date')
ax1.set_ylabel('Total Coronavirus Deaths')
ax1.legend()
ax1.grid(True)

# # Daily new cases bar plot
ax2.bar(daily_dates, daily_new_cases, color='gray', label='Cases per Day')
ax2.set_title('Daily New Cases')
ax2.set_xlabel('Date')
ax2.set_ylabel('Cases per Day')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show() 
