import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import numpy as np
from prophet import Prophet
from statsmodels.tsa.seasonal import seasonal_decompose
import ruptures as rpt
from statsmodels.tsa.stattools import adfuller

data = pd.read_csv("/Users/anhthientbd/ComputerScience/Transportation Data Science Project/Motor_Vehicle_Collisions_-_Crashes_20250213.csv", low_memory=False)

#TODO: Leverage the describe() function to assess the summary statistics for injuries and fatalities
injury_stats = data['NUMBER OF PERSONS INJURED'].describe()
fatality_stats = data['NUMBER OF PERSONS KILLED'].describe()
injury_stats, fatality_stats
# TODO: Calculate top vehicles for injuries
top_vehicles_injuries = data.groupby('VEHICLE TYPE CODE 1')['NUMBER OF PERSONS INJURED'].sum().sort_values(ascending=False).head(10)
# TODO: Calculate top vehicles for deaths
top_vehicles_deaths = data.groupby('VEHICLE TYPE CODE 1')['NUMBER OF PERSONS KILLED'].sum().sort_values(ascending=False).head(10)

# TODO: Combine the data into a DataFrame
combined_data = pd.DataFrame({'Injuries': top_vehicles_injuries, 'Deaths': top_vehicles_deaths})

# Set the width of the bars
bar_width = 0.35

# Plotting the combined bar chart with bars next to each other
fig, ax1 = plt.subplots(figsize=(12, 7))

# Generate a list of indices for the x-axis
indices = np.arange(len(combined_data))

# Plotting the bars for injuries
ax1.bar(indices - bar_width/2, combined_data['Injuries'], bar_width, color='blue', label='Injuries')

# TODO: Create a secondary y-axis for deaths
ax2 = ax1.twinx()

# Plotting the bars for deaths next to injuries
ax2.bar(indices + bar_width/2, combined_data['Deaths'], bar_width, color='orange', label='Deaths')

# Adding labels and title
ax1.set_title('Top 10 Vehicles in Injuries and Deaths', fontsize=16)
ax1.set_xlabel('Vehicle Type', fontsize=14)
ax1.set_ylabel('Number of Injuries', fontsize=14)
# TODO: Set the label for the secondary y-axis
ax2.set_ylabel('Number of Deaths', fontsize=14)
ax1.set_xticks(indices)
ax1.set_xticklabels(combined_data.index, rotation=45, ha='right')

# Adding legend
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Adjusting layout and displaying the combined chart
plt.tight_layout()
plt.show()

# TODO: Convert all variables in a specific column to upper case
data['VEHICLE TYPE CODE 1'] = data['VEHICLE TYPE CODE 1'].str.upper()

# Combine any repetitive variables
vehicle_types_to_combine = ['STATION WAGON/SPORT UTILITY VEHICLE', 'SPORT UTILITY / STATION WAGON']
# TODO: Replace specified values with a common label in a column
data['VEHICLE TYPE CODE 1'] = data['VEHICLE TYPE CODE 1'].replace(vehicle_types_to_combine, 'SUV/STATION WAGON')
top_vehicle_types = data['VEHICLE TYPE CODE 1'].value_counts().head(10)

# Re-run code from above
top_vehicles_injuries = data.groupby('VEHICLE TYPE CODE 1')['NUMBER OF PERSONS INJURED'].sum().sort_values(ascending=False).head(10)
top_vehicles_deaths = data.groupby('VEHICLE TYPE CODE 1')['NUMBER OF PERSONS KILLED'].sum().sort_values(ascending=False).head(10)
combined_data = pd.DataFrame({'Injuries': top_vehicles_injuries, 'Deaths': top_vehicles_deaths})
bar_width = 0.35
fig, ax1 = plt.subplots(figsize=(12, 7))
indices = np.arange(len(combined_data))
ax1.bar(indices - bar_width/2, combined_data['Injuries'], bar_width, color='blue', label='Injuries')
ax2 = ax1.twinx()
ax2.bar(indices + bar_width/2, combined_data['Deaths'], bar_width, color='orange', label='Deaths')
ax1.set_title('Top 10 Vehicles in Injuries and Deaths', fontsize=16)
ax1.set_xlabel('Vehicle Type', fontsize=14)
ax1.set_ylabel('Number of Injuries', fontsize=14)
ax2.set_ylabel('Number of Deaths', fontsize=14)
ax1.set_xticks(indices)
ax1.set_xticklabels(combined_data.index, rotation=45, ha='right')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.show()

# Load the dataset
file_path = "Motor_Vehicle_Collisions_-_Crashes_20250213.csv"
data = pd.read_csv(file_path, low_memory=False)

# TODO: Convert 'CRASH DATE' to datetime
data['CRASH DATE'] = pd.to_datetime(data['CRASH DATE'])

# Day of the Week Analysis
# TODO: Add a new column for the day of the week from 'CRASH DATE'
data['Day of Week'] = data['CRASH DATE'].dt.day_name()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Group by 'Day of Week' and calculate the average number of crashes per day
average_crashes_per_weekday = data.groupby('Day of Week').size() / data['Day of Week'].nunique()

# Plot the average number of crashes
plt.figure(figsize=(10, 6))
# TODO: Plot a bar chart with seaborn
sns.barplot(x=average_crashes_per_weekday.index, y=average_crashes_per_weekday.values, order=day_order)
plt.title('Average Number of Crashes per Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Average Number of Crashes')
plt.xticks(rotation=45)
plt.show()

# TODO: Count the number of crashes per day
daily_crashes = data.groupby('CRASH DATE').size()

# Perform Augmented Dickey-Fuller test
# TODO: Use adfuller to test if daily crashes are stationary
adf_test = adfuller(daily_crashes)

# Display the ADF test results
# TODO: Create a pandas Series to nicely display test results
adf_output = pd.Series(adf_test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key, value in adf_test[4].items():
    adf_output[f'Critical Value ({key})'] = value

# TODO: Print the ADF test output
print(adf_output)

# TODO: Apply first differencing
daily_crashes_diff = daily_crashes.diff().dropna()

# Perform Augmented Dickey-Fuller test on the differenced series
# TODO: Use adfuller to test if the differenced series is stationary
adf_test_diff = adfuller(daily_crashes_diff)

# Display the ADF test results for the differenced series
# TODO: Create a pandas Series to nicely display test results
adf_output_diff = pd.Series(adf_test_diff[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key, value in adf_test_diff[4].items():
    adf_output_diff[f'Critical Value ({key})'] = value

# TODO: Print the ADF test output for the differenced series
print(adf_output_diff)

# TODO: Convert the 'CRASH DATE' column to datetime format. Replace 'CRASH DATE' if your date column has a different name.
data['CRASH DATE'] = pd.to_datetime(data['CRASH DATE'])

# TODO: Group data by 'CRASH DATE' and count the crashes. Assign this to a new DataFrame called 'daily_crashes'.
daily_crashes = data.groupby('CRASH DATE').size().reset_index(name='crash_count')

# TODO: Rename the columns of 'crash_count' to 'y' for Prophet compatibility.
daily_crashes.rename(columns={'CRASH DATE': 'ds', 'crash_count': 'y'}, inplace=True)

model = Prophet()
model.fit(daily_crashes)

# TODO: Create a future dataframe for forecasting the next 30 days and assign it to a variable called 'future'.
future = model.make_future_dataframe(periods=30)

# TODO: Use the model to make predictions on the 'future' DataFrame and store the result in a variable called 'forecast'.
forecast = model.predict(future)

# TODO: Plot the forecasted data using the Prophet's built-in plot function.
fig = model.plot(forecast)
plt.title('30-Day Forecast of Daily Motor Vehicle Collisions')
plt.xlabel('Date')
plt.ylabel('Number of Crashes')
plt.show()

# Convert 'CRASH DATE' to datetime and extract year and month for easier analysis
data['CRASH DATE'] = pd.to_datetime(data['CRASH DATE'])
data['YEAR'] = data['CRASH DATE'].dt.year
data['MONTH'] = data['CRASH DATE'].dt.month

# Analyzing collision frequencies over time in different boroughs
# Grouping data by Year and Borough
borough_yearly = data.groupby(['YEAR', 'BOROUGH']).size().unstack(fill_value=0)

# Plotting the data
plt.figure(figsize=(15, 7))
# TODO: Use seaborn to create a line plot
sns.lineplot(data=borough_yearly)
plt.title('Yearly Collision Frequencies by Borough')
plt.ylabel('Number of Crashes')
plt.xlabel('Year')
plt.xticks(rotation=45)
plt.legend(title='Borough', loc='upper left')
plt.grid(True)
plt.tight_layout()

plt.show()

# Analyzing specific times or locations with unusually high numbers of crashes
# For this, we will use a heatmap to visualize the distribution of crashes by month and borough

# Grouping data by Month and Borough
borough_monthly = data.groupby(['MONTH', 'BOROUGH']).size().unstack(fill_value=0)

plt.figure(figsize=(12, 8))
# TODO: Create a heatmap to visualize the frequencies
sns.heatmap(borough_monthly, cmap='viridis', annot=True, fmt='d')
plt.title('Monthly Collision Frequencies by Borough')
plt.ylabel('Borough')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

plt.show()

# Creating a correlation matrix for the number of collisions in different boroughs across different years
correlation_matrix = borough_yearly.corr()

plt.figure(figsize=(10, 8))
# TODO: Plot the correlation matrix using a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt='.2f')
plt.title('Correlation Matrix of Collision Frequencies Across Boroughs')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

plt.show()

# aggrate the total crash by dates
Counts=data["CRASH DATE"].value_counts()
Counts.index = pd.to_datetime(Counts.index)
Counts.sort_index(inplace=True)
Counts_2022 = Counts.loc['2022-01-01':'2022-12-31'].rolling(7).mean().dropna()


# implement the algorithm
model = "l2"
algo = rpt.Binseg(model=model).fit(Counts_2022.to_numpy())
my_bkps = algo.predict(n_bkps=10)
# show results
fig, ax = plt.subplots(figsize=(18,8))
ax = plt.gca()
ax.plot_date(Counts_2022.index, Counts_2022.to_numpy(), linestyle='solid')
#rpt.show.display(Counts_2022.to_numpy(), my_bkps, figsize=(10, 6))
plt.ylim([0.9*Counts_2022.min(), 1.1*Counts_2022.max()])

# changing point
my_bkps.pop()
for ix in my_bkps:
    plt.plot([Counts_2022.index[ix], Counts_2022.index[ix]], [0,1000], '--k')

plt.title('Change Point Detection: Binary Segmentation Search Method')
plt.ylabel("Total Crash Count in NYC")
plt.xlabel("Date")
print(my_bkps)

df18 = data.loc[data["CRASH DATE"].dt.year == 2018].reset_index(drop = True)
df18 = df18.dropna(subset="ZIP CODE")
df18["ZIP CODE"] = df18["ZIP CODE"].astype(int)
df18_sort = df18.groupby(['ZIP CODE']).count().sort_values(by = 'CRASH DATE', ascending = False).iloc[0: 10]
df18_sort

#TO DO: Build a visualization, model, or use other statistical methods to gain insights into your data and to support your research question.

# Create a bar chart of the top 10 zip codes with the most crashes in 2018
plt.figure(figsize=(12, 8))
sns.barplot(x=df18_sort.index, y=df18_sort['CRASH DATE'], palette='viridis')

plt.title('Top 10 ZIP Codes with Most Crashes in 2018')
plt.xlabel('ZIP Code')
plt.ylabel('Number of Crashes')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Assuming 'data' is your DataFrame and 'CRASH TIME' is a string in 'HH:MM' format
# Convert 'CRASH TIME' to datetime and extract the hour
data['CRASH TIME'] = pd.to_datetime(data['CRASH TIME'], format='%H:%M')
data['HOUR'] = data['CRASH TIME'].dt.hour

# Group data by 'HOUR' and count the number of crashes
hourly_crashes = data.groupby('HOUR').size().reset_index(name='crash_count')

# Plot the data
plt.figure(figsize=(12, 6))
sns.barplot(x='HOUR', y='crash_count', data=hourly_crashes, palette='viridis')

plt.title('Number of Crashes by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Crashes')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()