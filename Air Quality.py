#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#loading the dataset
data = pd.read_csv("Documents/Air Quality/AirQuality.csv")

#data description
print(data.head())
print(data.info())
print(data.describe())

#identifying missing values
missing_values = data.isna()
num_missing_values = missing_values.sum()
print("Number of missing values in each column:")
print(num_missing_values)

#handling missing values
data = data.fillna(data.mean())
print(data.info())

column_to_delete = 'PM 2.5'
data = data.drop(columns=column_to_delete)
print("\nDataFrame after deleting the column:")
print(data)

#creating heatmap for correlation analysis
correlation_matrix = data.corr()
plt.figure(figsize=(12, 10))
sns.set(style="whitegrid")
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

#creating boxplot to identify outliers
numeric_columns = data.select_dtypes(include='number')
plt.figure(figsize=(15, 10))
sns.set(style="whitegrid")
for column in numeric_columns.columns:
    plt.subplot(1, 4, list(numeric_columns.columns).index(column) + 1)
    sns.boxplot(x=numeric_columns[column], color='skyblue')
    plt.title(column)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.suptitle("OUTLIERS IN Stn Code, SO2, NO2, RSPM/PM10", fontsize=16)
plt.show()

#handling outliers using IQR method
def handle_outliers_with_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (data[column] < lower_bound) | (data[column] > upper_bound)
    data.loc[outliers, column] = upper_bound 
columns_with_outliers = ['Stn Code', 'SO2', 'NO2', 'RSPM/PM10'] 
for column in columns_with_outliers:
    handle_outliers_with_iqr(data, column)

#creating boxplot after handling outliers
plt.figure(figsize=(15, 10))
for i, column in enumerate(numeric_columns.columns):
    plt.subplot(1, 4, i + 1)
    sns.boxplot(x=data[column], color='skyblue')  
    plt.title(column)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.suptitle("AFTER HANDLING OUTLIERS", fontsize=16)
plt.show()

#creating time series graph
data['Sampling Date'] = pd.to_datetime(data['Sampling Date'])
data.set_index('Sampling Date', inplace=True)
data = data.sort_index()

start_date = data.index.min()
end_date = data.index.max()
print("Start Date:", start_date)
print("End Date:", end_date)

#Time series graph for SO2
plt.figure(figsize=(300, 100))
plt.plot(data.index, data['SO2'], label='SO2', color='b')
plt.xlabel("Date")
plt.ylabel("SO2 Value")
plt.title("Time Series Analysis for SO2")
plt.legend()
plt.grid(True)
date_ticks = pd.date_range(start_date, end_date, freq='M')  
plt.xticks(date_ticks, date_ticks.strftime('%Y-%m'))
plt.show()

#Time series graph for NO2
plt.figure(figsize=(300, 100))
plt.plot(data.index, data['NO2'], label='NO2', color='g')
plt.xlabel("Date")
plt.ylabel("NO2 Value")
plt.title("Time Series Analysis for NO2")
plt.legend()
plt.grid(True)
date_ticks = pd.date_range(start_date, end_date, freq='M')  
plt.xticks(date_ticks, date_ticks.strftime('%Y-%m'))
plt.show()

#Time series graph for RSPM
plt.figure(figsize=(300, 100))
plt.plot(data.index, data['RSPM/PM10'], label='RSPM/PM10', color='r')
plt.xlabel("Date")
plt.ylabel("RSPM/PM10 Value")
plt.title("Time Series Analysis for RSPM/PM10")
plt.legend()
plt.grid(True)
date_ticks = pd.date_range(start_date, end_date, freq='M')  
plt.xticks(date_ticks, date_ticks.strftime('%Y-%m'))
plt.show()

#creating sunburst chart
import plotly.express as px

sunburst_data = data[['State', 'Location of Monitoring Station', 'Type of Location', 'NO2', 'SO2', 'RSPM/PM10']]

#sunburst chart for NO2
NO2 = px.sunburst(
    sunburst_data,
    path=['State', 'Location of Monitoring Station', 'Type of Location'],
    values='NO2',  
    width=1500,
    height=1500,
    color_continuous_scale="RdYlGn",
    title="Sunburst Chart for NO2")
NO2.show()

#sunburst chart for SO2
SO2 = px.sunburst(
    sunburst_data,
    path=['State', 'Location of Monitoring Station', 'Type of Location'],
    values='SO2',  
    width=1500,
    height=1500,
    color_continuous_scale="RdYlGn",
    title="Sunburst Chart for SO2")
SO2.show()

#sunburst chart for RSPM
RSPM = px.sunburst(
    sunburst_data,
    path=['State', 'Location of Monitoring Station', 'Type of Location'],
    values='RSPM/PM10',  
    width=1500,
    height=1500,
    color_continuous_scale="RdYlGn",
    title="Sunburst Chart for RSPM/PM10")
RSPM.show()


#importing model & related requirements
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Separate features and target variable
X = data[['NO2', 'SO2']]
y = data['RSPM/PM10']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)


# RandomForestRegressor
rf = RandomForestRegressor(random_state=1)
rf.fit(X_train, y_train)
rf_y_pred = rf.predict(X_test)

#r2 score
print("r2 of rf is:", r2_score(y_test, rf_y_pred))
