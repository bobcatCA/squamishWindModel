import pandas as pd

# Convert the Weather strings to numerics
# Leaving this out for now, as we can keep the text for a categorical variable
# weatherCodes = pd.read_csv('weatherCodes.csv')
# weatherMap = pd.Series(weatherCodes.value.values, index=weatherCodes.descriptor).to_dict()

# Load dataframes and convert time columns to proper format
# Convert the Weather column, which contains a verbal description, to numeric
vancouverWeather = pd.read_csv('Vancouver.csv')
comoxWeather = pd.read_csv('Comox.csv').set_index('Date/Time (LST)')
whistlerWeather = pd.read_csv('Whistler.csv').set_index('Date/Time (LST)')
victoriaWeather = pd.read_csv('Victoria.csv').set_index('Date/Time (LST)')

# Uncomment this if it's desirable to convert the weather text to numeric
# vancouverWeather['Weather'] = vancouverWeather['Weather'].map(weatherMap)
# vancouverWeather['Weather'] = pd.to_numeric(vancouverWeather['Weather'])
# comoxWeather['Weather'] = comoxWeather['Weather'].map(weatherMap)
# comoxWeather['Weather'] = pd.to_numeric(comoxWeather['Weather'])
# whistlerWeather['Weather'] = whistlerWeather['Weather'].map(weatherMap)
# whistlerWeather['Weather'] = pd.to_numeric(whistlerWeather['Weather'])
# victoriaWeather['Weather'] = victoriaWeather['Weather'].map(weatherMap)
# victoriaWeather['Weather'] = pd.to_numeric(victoriaWeather['Weather'])

# The remaining stations don't have Weather description, so just load
pembertonWeather = pd.read_csv('Pemberton.csv').set_index('Date/Time (LST)')
lillooetWeather = pd.read_csv('Lillooet.csv').set_index('Date/Time (LST)')
ballenasWeather = pd.read_csv('Ballenas.csv').set_index('Date/Time (LST)')
pamWeather = pd.read_csv('Pam.csv').set_index('Date/Time (LST)')

# Now load the wind data from the Squamish Windsports website
df_squamishWind = pd.read_csv('swsWind.csv')
df_squamishWind['time'] = pd.to_datetime(df_squamishWind['time'])
df_squamishWind.set_index('time', inplace=True)

# Make a dataframe to hold the weather data (features). Start with Vancouver.
df_features = pd.DataFrame()
df_features['datetime'] = vancouverWeather['Date/Time (LST)']
df_features['vancouverDegC'] = vancouverWeather['Temp (\u00b0'+'C)']
df_features['vancouverSky'] = vancouverWeather['Weather']
df_features['vancouverHum'] = vancouverWeather['Rel Hum (%)']
df_features['vancouverKPa'] = vancouverWeather['Stn Press (kPa)']

# Compile the Temperature data from Whistler, Pemberton, Lillooet using merge
# First Whistler, which has cloud cover data, and merge.
df_features = df_features.merge(whistlerWeather[['Temp (\u00b0'+'C)', 'Weather', 'Stn Press (kPa)', 'Rel Hum (%)']], how='inner', left_on='datetime', right_index=True)
df_features.rename(columns={'Temp (\u00b0'+'C)':'whistlerDegC', 'Weather': 'whistlerSky', 'Stn Press (kPa)': 'whistlerKPa', 'Rel Hum (%)': 'whistlerHum'}, inplace=True)
# df_features['whistlerSky'] = pd.to_numeric(df_features['whistlerSky'])
df_features['whistlerHum'] = pd.to_numeric(df_features['whistlerHum'])

# Same for Comox and Victoria
df_features = df_features.merge(comoxWeather[['Temp (\u00b0'+'C)', 'Weather', 'Stn Press (kPa)', 'Rel Hum (%)']], how='inner', left_on='datetime', right_index=True)
df_features.rename(columns={'Temp (\u00b0'+'C)':'comoxDegC', 'Weather': 'comoxSky', 'Stn Press (kPa)': 'comoxKPa', 'Rel Hum (%)': 'comoxHum'}, inplace=True)
# df_features['comoxSky'] = pd.to_numeric(df_features['comoxSky'])
df_features['comoxHum'] = pd.to_numeric(df_features['comoxHum'])
df_features = df_features.merge(victoriaWeather[['Temp (\u00b0'+'C)', 'Weather', 'Stn Press (kPa)', 'Rel Hum (%)']], how='inner', left_on='datetime', right_index=True)
df_features.rename(columns={'Temp (\u00b0'+'C)':'victoriaDegC', 'Weather': 'victoriaSky', 'Stn Press (kPa)': 'victoriaKPa', 'Rel Hum (%)': 'victoriaHum'}, inplace=True)
# df_features['victoriaSky'] = pd.to_numeric(df_features['victoriaSky'])
df_features['victoriaHum'] = pd.to_numeric(df_features['victoriaHum'])

# Then Pemperton, Liloet, Ballenas, Pam Rocks which only have temperature,
# pressure, and humidity
df_features = df_features.merge(pembertonWeather[['Temp (\u00b0'+'C)', 'Stn Press (kPa)', 'Rel Hum (%)']], how='inner', left_on='datetime', right_index=True)
df_features.rename(columns={'Temp (\u00b0'+'C)':'pembertonDegC', 'Stn Press (kPa)': 'pembertonKPa', 'Rel Hum (%)': 'pembertonHum'}, inplace=True)
df_features = df_features.merge(lillooetWeather[['Temp (\u00b0'+'C)', 'Stn Press (kPa)', 'Rel Hum (%)']], how='inner', left_on='datetime', right_index=True)
df_features.rename(columns={'Temp (\u00b0'+'C)':'lillooetDegC', 'Stn Press (kPa)': 'lillooetKPa', 'Rel Hum (%)': 'lillooetHum'}, inplace=True)
df_features = df_features.merge(pamWeather[['Temp (\u00b0'+'C)', 'Stn Press (kPa)', 'Rel Hum (%)']], how='inner', left_on='datetime', right_index=True)
df_features.rename(columns={'Temp (\u00b0'+'C)':'pamDegC', 'Stn Press (kPa)': 'pamKPa', 'Rel Hum (%)': 'pamHum'}, inplace=True)
df_features = df_features.merge(pamWeather[['Temp (\u00b0'+'C)', 'Stn Press (kPa)', 'Rel Hum (%)']], how='inner', left_on='datetime', right_index=True)
df_features.rename(columns={'Temp (\u00b0'+'C)':'ballenasDegC', 'Stn Press (kPa)': 'ballenasKPa', 'Rel Hum (%)': 'ballenasHum'}, inplace=True)
df_features['datetime'] = pd.to_datetime(df_features['datetime'])
df_features.set_index('datetime', inplace=True)

# Merge the two datasets, interpolate for missing data
# df_merged = pd.merge(df_features, df_squamishWind, left_on='datetime', right_on='time', how='outer')
df_merged = pd.merge(df_features, df_squamishWind, left_index=True, right_index=True, how='outer')
df_merged['vancouverDegC'] = df_merged['vancouverDegC'].interpolate(method='linear')
# df_merged['vancouverSky'] = df_merged['vancouverSky'].interpolate(method='linear')
df_merged['vancouverSky'] = df_merged['vancouverSky'].ffill()  # Fill in seems appropriate, gaps are small enough
df_merged['vancouverHum'] = df_merged['vancouverHum'].interpolate(method='linear')
df_merged['vancouverKPa'] = df_merged['vancouverKPa'].interpolate(method='linear')
df_merged['whistlerDegC'] = df_merged['whistlerDegC'].interpolate(method='linear')
# df_merged['whistlerSky'] = df_merged['whistlerSky'].interpolate(method='linear')
df_merged['whistlerSky'] = df_merged['whistlerSky'].ffill()
df_merged['whistlerHum'] = df_merged['whistlerHum'].interpolate(method='linear')
df_merged['whistlerKPa'] = df_merged['whistlerKPa'].interpolate(method='linear')
df_merged['comoxDegC'] = df_merged['comoxDegC'].interpolate(method='linear')
# df_merged['comoxSky'] = df_merged['comoxSky'].interpolate(method='linear')
df_merged['comoxSky'] = df_merged['comoxSky'].ffill()
df_merged['comoxHum'] = df_merged['comoxHum'].interpolate(method='linear')
df_merged['comoxKPa'] = df_merged['comoxKPa'].interpolate(method='linear')
df_merged['pembertonDegC'] = df_merged['pembertonDegC'].interpolate(method='linear')
df_merged['pembertonHum'] = df_merged['pembertonHum'].interpolate(method='linear')
df_merged['pembertonKPa'] = df_merged['pembertonKPa'].interpolate(method='linear')
df_merged['lillooetDegC'] = df_merged['lillooetDegC'].interpolate(method='linear')
df_merged['lillooetHum'] = df_merged['lillooetHum'].interpolate(method='linear')
df_merged['lillooetKPa'] = df_merged['lillooetKPa'].interpolate(method='linear')
df_merged['ballenasDegC'] = df_merged['ballenasDegC'].interpolate(method='linear')
df_merged['ballenasHum'] = df_merged['ballenasHum'].interpolate(method='linear')
df_merged['ballenasKPa'] = df_merged['ballenasKPa'].interpolate(method='linear')
df_merged['pamDegC'] = df_merged['pamDegC'].interpolate(method='linear')
df_merged['pamHum'] = df_merged['pamHum'].interpolate(method='linear')
df_merged['pamKPa'] = df_merged['pamKPa'].interpolate(method='linear')
df_merged['victoriaDegC'] = df_merged['victoriaDegC'].interpolate(method='linear')
df_merged['victoriaHum'] = df_merged['victoriaHum'].interpolate(method='linear')
df_merged['victoriaKPa'] = df_merged['victoriaKPa'].interpolate(method='linear')
df_merged['victoriaSky'] = df_merged['victoriaSky'].ffill()

# Drop unwanted columns and rows without any speed data
# df_merged = df_merged.drop(['direction', 'gust', 'lull', 'temperature'], axis=1)
df_merged = df_merged.dropna(subset=['speed'])
df_merged = df_merged.reset_index(names=['time'])
df_merged = df_merged[sorted(df_merged.columns)]

# Save to csv file
df_merged.to_csv('mergedOnSpeed.csv')


