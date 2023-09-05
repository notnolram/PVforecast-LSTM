import pandas as pd
import numpy as np
#import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
import random
from datetime import datetime,timedelta

# Load TMY3 data file
df = pd.read_csv('../../data/ES_Linhares.tmy3', skiprows=1)

# Set date and time as index
df['Timestamp'] = pd.to_datetime(df['Date (MM/DD/YYYY)'] + ' ' + df['Time (HH:MM)'])
df.set_index('Timestamp', inplace=True)

# Select relevant columns for PV generation prediction
#df = df[['DNI (W/m^2)', 'DHI (W/m^2)', 'GHI (W/m^2)', 'Dry-bulb (C)', 'Dew-point (C)', 'Wspd (m/s)', 'Wdir (degrees)', 'RHum (%)', 'Alb (unitless)']]
df = df[['DNI (W/m^2)','GHI (W/m^2)', 'Dry-bulb (C)', 'Dew-point (C)', 'Wspd (m/s)', 'Wdir (degrees)', 'RHum (%)', 'Alb (unitless)']]
dftrain = df.loc['2018-01-01 00:00:00':'2020-12-31 23:00:00']

print(dftrain.iloc[::1])
# Define target variable as PV power generation in kW
pv_ghi = dftrain['GHI (W/m^2)']

# Define input features
X = dftrain.drop('GHI (W/m^2)', axis=1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, pv_ghi, test_size=0.1, random_state=42)

# Train random forest regression model
rf = RandomForestRegressor(n_estimators=64, random_state=42)
rf.fit(X_train, y_train)

# Make PV power generation predictions on testing data
y_pred = rf.predict(X_test)

# Calculate mean absolute error (MAE)
mae = np.mean(abs(y_test - y_pred))
rmse = np.sqrt(np.mean((y_test - y_pred)**2))
print('Mean Absolute Error:', round(mae, 2), 'W/m^2')
print('Root Mean Squared Error:', round(rmse, 2), 'W/m^2')

# Select relevant columns for PV generation prediction
df_pred = df.loc['2021-01-01 00:00:00':'2021-12-31 23:00:00']
X = df_pred.drop('GHI (W/m^2)', axis=1)

# Make PV power generation predictions for the next day
pv_ghi_pred = rf.predict(X)

mu = [1, 1, 1, 1, 1]   # média da distribuição normal
sigma = [0.1, 0.081, 0.09, 0.12, 0.13]  # desvio padrão da distribuição normal
a = 0.98                                # limite inferior do intervalo
b = 1.005                               # limite superior do intervalo

start_date = datetime(2021, 1, 1, 0, 0, 0, 0)
current_time = start_date
time_interval = timedelta(hours=1,minutes=0)
current_time.strftime("%Y-%m-%d %H:%M:%S")
x1 = random.normalvariate(mu[0], sigma[0])
x2 = random.normalvariate(mu[1], sigma[1])
x3 = random.normalvariate(mu[2], sigma[2])
x4 = random.normalvariate(mu[3], sigma[3])
x5 = random.normalvariate(mu[4], sigma[4])
x = [x1, x2, x3, x4, x5]

while(current_time.day <= 31 and current_time.month <= 12 and current_time.hour <= 23 and current_time.minute <= 00):
    current_time = current_time + time_interval
    if(current_time.year == 2022):
        break
    current_time.strftime("%Y-%m-%d %H:%M:%S")
    for k in range(len(x)):
        while x[k] < a or x[k] > b:
            x[k] = random.normalvariate(mu[k], sigma[k])
    #df.loc[current_time.strftime("%Y-%m-%d %H:%M:%S"),'DNI (W/m^2)'] = df.loc[current_time.strftime("%Y-%m-%d %H:%M:%S"),'DNI (W/m^2)']*x1
    #df.loc[current_time.strftime("%Y-%m-%d %H:%M:%S"),'DHI (W/m^2)'] = df.loc[current_time.strftime("%Y-%m-%d %H:%M:%S"),'DHI (W/m^2)']*x2
    df.loc[current_time.strftime("%Y-%m-%d %H:%M:%S"),'Dry-bulb (C)'] = df.loc[current_time.strftime("%Y-%m-%d %H:%M:%S"),'Dry-bulb (C)']*x3
    df.loc[current_time.strftime("%Y-%m-%d %H:%M:%S"),'Dew-point (C)'] = df.loc[current_time.strftime("%Y-%m-%d %H:%M:%S"),'Dew-point (C)']*x4
    df.loc[current_time.strftime("%Y-%m-%d %H:%M:%S"),'Wspd (m/s)'] = df.loc[current_time.strftime("%Y-%m-%d %H:%M:%S"),'Wspd (m/s)']*x5
    
# df.loc['2021-01-01 00:00:00':'2021-12-31 23:30:00','DNI (W/m^2)'] = df.loc['2021-01-01 00:00:00':'2021-12-31 23:30:00']['DNI (W/m^2)']*x1
# df.loc['2021-01-01 00:00:00':'2021-12-31 23:30:00','DHI (W/m^2)'] = df.loc['2021-01-01 00:00:00':'2021-12-31 23:30:00']['DHI (W/m^2)']*x2
# df.loc['2021-01-01 00:00:00':'2021-12-31 23:30:00','Dry-bulb (C)'] = df.loc['2021-01-01 00:00:00':'2021-12-31 23:30:00']['Dry-bulb (C)']*x3
# df.loc['2021-01-01 00:00:00':'2021-12-31 23:30:00','Dew-point (C)'] = df.loc['2021-01-01 00:00:00':'2021-12-31 23:30:00']['Dew-point (C)']*x4
# df.loc['2021-01-01 00:00:00':'2021-12-31 23:30:00','Wspd (m/s)'] = df.loc['2021-01-01 00:00:00':'2021-12-31 23:30:00']['Wspd (m/s)']*x5

df_pred = df.loc['2021-01-01 00:00:00':'2021-12-31 23:00:00']
X2 = df_pred.drop('GHI (W/m^2)', axis=1)
pv_ghi_preddev = rf.predict(X2)

# Save PV power generation predictions to a file
df_pred['GHI pred (W/m^2)'] = pv_ghi_pred
df_pred['GHI pred dev (W/m^2)'] = pv_ghi_preddev
df_pred.to_csv('ghi_predictions_2021.csv')