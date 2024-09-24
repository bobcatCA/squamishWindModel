import pandas as pd

# Convert the Weather strings to numerics
weatherCodes = pd.read_csv('weatherCodes.csv')
weatherMap = pd.Series(weatherCodes.value.values, index=weatherCodes.descriptor).to_dict()

# Load dataframes and convert time columns to proper format
vancouverWeather = pd.read_csv('Vancouver.csv')
vancouverWeather['Weather'] = vancouverWeather['Weather'].map(weatherMap)
vancouverWeather['Weather'] = pd.to_numeric(vancouverWeather['Weather'])
whistlerWeather = pd.read_csv('Whistler.csv').set_index('Date/Time (LST)')
whistlerWeather['Weather'] = whistlerWeather['Weather'].map(weatherMap)
whistlerWeather['Weather'] = pd.to_numeric(whistlerWeather['Weather'])
pembertonWeather = pd.read_csv('Pemberton.csv').set_index('Date/Time (LST)')
lillooetWeather = pd.read_csv('Lillooet.csv').set_index('Date/Time (LST)')

squamishWind = pd.read_csv('swsWind.csv')
squamishWind['time'] = pd.to_datetime(squamishWind['time'])
squamishWind.set_index('time', inplace=True)

# Make a dataframe to hold the weather data (features)
df_features = pd.DataFrame()
df_features['datetime'] = vancouverWeather['Date/Time (LST)']
df_features['vancouverDegC'] = vancouverWeather['Temp (\u00b0'+'C)']
df_features['vancouverSky'] = vancouverWeather['Weather']
df_features['vancouverHum'] = vancouverWeather['Rel Hum (%)']
df_features['vancouverKPa'] = vancouverWeather['Stn Press (kPa)']

# Compile the Temperature data from Whistler, Pemberton, Lillooet using merge
# First Whistler, which has cloud cover data
df_features = df_features.merge(whistlerWeather[['Temp (\u00b0'+'C)', 'Weather', 'Stn Press (kPa)', 'Rel Hum (%)']], how='inner', left_on='datetime', right_index=True)
df_features.rename(columns={'Temp (\u00b0'+'C)':'whistlerDegC', 'Weather': 'whistlerSky', 'Stn Press (kPa)': 'whistlerKPa', 'Rel Hum (%)': 'whistlerHum'}, inplace=True)
df_features['whistlerSky'] = pd.to_numeric(df_features['whistlerSky'])
df_features['whistlerHum'] = pd.to_numeric(df_features['whistlerHum'])

# Then Pemperton, Liloet which only have temperature, pressure, and humidity
df_features = df_features.merge(pembertonWeather[['Temp (\u00b0'+'C)', 'Stn Press (kPa)', 'Rel Hum (%)']], how='inner', left_on='datetime', right_index=True)
df_features.rename(columns={'Temp (\u00b0'+'C)':'pembertonDegC', 'Stn Press (kPa)': 'pembertonKPa', 'Rel Hum (%)': 'pembertonHum'}, inplace=True)
df_features = df_features.merge(lillooetWeather[['Temp (\u00b0'+'C)', 'Stn Press (kPa)', 'Rel Hum (%)']], how='inner', left_on='datetime', right_index=True)
df_features.rename(columns={'Temp (\u00b0'+'C)':'lillooetDegC', 'Stn Press (kPa)': 'lillooetKPa', 'Rel Hum (%)': 'lillooetHum'}, inplace=True)
df_features['datetime'] = pd.to_datetime(df_features['datetime'])
df_features.set_index('datetime', inplace=True)

# Merge the two datasets, interpolate for missing data
df_merged = pd.merge(df_features, squamishWind, left_index=True, right_index=True, how='outer')
df_merged['vancouverDegC'] = df_merged['vancouverDegC'].interpolate(method='linear')
df_merged['vancouverSky'] = df_merged['vancouverSky'].interpolate(method='linear')
df_merged['vancouverHum'] = df_merged['vancouverHum'].interpolate(method='linear')
df_merged['vancouverKPa'] = df_merged['vancouverKPa'].interpolate(method='linear')
df_merged['whistlerDegC'] = df_merged['whistlerDegC'].interpolate(method='linear')
df_merged['whistlerSky'] = df_merged['whistlerSky'].interpolate(method='linear')
df_merged['whilstlerHum'] = df_merged['whistlerHum'].interpolate(method='linear')
df_merged['whistlerKPa'] = df_merged['whistlerKPa'].interpolate(method='linear')
df_merged['pembertonDegC'] = df_merged['pembertonDegC'].interpolate(method='linear')
df_merged['pembertonHum'] = df_merged['pembertonHum'].interpolate(method='linear')
df_merged['pembertonKPa'] = df_merged['pembertonKPa'].interpolate(method='linear')
df_merged['lillooetDegC'] = df_merged['lillooetDegC'].interpolate(method='linear')
df_merged['lillooetHum'] = df_merged['lillooetHum'].interpolate(method='linear')
df_merged['lillooetKPa'] = df_merged['lillooetKPa'].interpolate(method='linear')

# Drop unwanted columns and rows without any speed data
df_merged = df_merged.drop(['direction', 'gust', 'lull', 'temperature'], axis=1)
df_merged = df_merged.dropna(subset=['speed'])

# Save to csv file
df_merged.to_csv('mergedOnSpeed.csv')


