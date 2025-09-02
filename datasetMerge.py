import numpy as np
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

# The remaining stations don't have Weather description, so just load
pembertonWeather = pd.read_csv('Pemberton.csv').set_index('Date/Time (LST)')
lillooetWeather = pd.read_csv('Lillooet.csv').set_index('Date/Time (LST)')
ballenasWeather = pd.read_csv('Ballenas.csv').set_index('Date/Time (LST)')
pamWeather = pd.read_csv('Pam.csv').set_index('Date/Time (LST)')

# Now load the wind data from the Squamish Windsports website
df_squamishWind = pd.read_csv('sws_wind_database.csv')
df_squamishWind['datetime'] = pd.to_datetime(df_squamishWind['datetime'], utc=True)
df_squamishWind = df_squamishWind.resample('1H', on='datetime').mean()  # Resample to hourly to match weather df
df_squamishWind.reset_index(inplace=True)
df_squamishWind['datetime'] = df_squamishWind['datetime'].dt.tz_convert('America/Vancouver')  # Back to tz-aware

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
df_features['whistlerHum'] = pd.to_numeric(df_features['whistlerHum'])

# Same for Comox and Victoria
df_features = df_features.merge(comoxWeather[['Temp (\u00b0'+'C)', 'Weather', 'Stn Press (kPa)', 'Rel Hum (%)']], how='inner', left_on='datetime', right_index=True)
df_features.rename(columns={'Temp (\u00b0'+'C)':'comoxDegC', 'Weather': 'comoxSky', 'Stn Press (kPa)': 'comoxKPa', 'Rel Hum (%)': 'comoxHum'}, inplace=True)
df_features['comoxHum'] = pd.to_numeric(df_features['comoxHum'])
df_features = df_features.merge(victoriaWeather[['Temp (\u00b0'+'C)', 'Weather', 'Stn Press (kPa)', 'Rel Hum (%)']], how='inner', left_on='datetime', right_index=True)
df_features.rename(columns={'Temp (\u00b0'+'C)':'victoriaDegC', 'Weather': 'victoriaSky', 'Stn Press (kPa)': 'victoriaKPa', 'Rel Hum (%)': 'victoriaHum'}, inplace=True)
df_features['victoriaHum'] = pd.to_numeric(df_features['victoriaHum'])

# Then Pemperton, Liloet, Ballenas, Pam Rocks which only have temperature,
# pressure, and humidity
df_features = df_features.merge(pembertonWeather[['Temp (\u00b0'+'C)', 'Stn Press (kPa)', 'Rel Hum (%)']], how='inner', left_on='datetime', right_index=True)
df_features.rename(columns={'Temp (\u00b0'+'C)':'pembertonDegC', 'Stn Press (kPa)': 'pembertonKPa', 'Rel Hum (%)': 'pembertonHum'}, inplace=True)
df_features = df_features.merge(lillooetWeather[['Temp (\u00b0'+'C)', 'Stn Press (kPa)', 'Rel Hum (%)']], how='inner', left_on='datetime', right_index=True)
df_features.rename(columns={'Temp (\u00b0'+'C)':'lillooetDegC', 'Stn Press (kPa)': 'lillooetKPa', 'Rel Hum (%)': 'lillooetHum'}, inplace=True)
df_features = df_features.merge(pamWeather[['Temp (\u00b0'+'C)', 'Stn Press (kPa)', 'Rel Hum (%)']], how='inner', left_on='datetime', right_index=True)
df_features.rename(columns={'Temp (\u00b0'+'C)':'pamDegC', 'Stn Press (kPa)': 'pamKPa', 'Rel Hum (%)': 'pamHum'}, inplace=True)
df_features = df_features.merge(ballenasWeather[['Temp (\u00b0'+'C)', 'Stn Press (kPa)', 'Rel Hum (%)']], how='inner', left_on='datetime', right_index=True)
df_features.rename(columns={'Temp (\u00b0'+'C)':'ballenasDegC', 'Stn Press (kPa)': 'ballenasKPa', 'Rel Hum (%)': 'ballenasHum'}, inplace=True)
df_features['datetime'] = pd.to_datetime(df_features['datetime'])
df_features['datetime'] = (df_features['datetime'].dt.tz_localize(
    'America/Vancouver', nonexistent='shift_forward', ambiguous='NaT'))  # To tz-aware

# Merge the two datasets, interpolate for missing data
df_merged = pd.merge(df_features, df_squamishWind, on='datetime', how='inner')
# df_merged = df_features

# TODO: Should I interpolate/fill, or just leave blank and allow_missing_timesteps=True?
# Whistler is only missing data at night, so interpolation should be OK
# df_merged['whistlerDegC'] = df_merged['whistlerDegC'].interpolate(method='linear')
# df_merged['whistlerHum'] = df_merged['whistlerHum'].interpolate(method='linear')
# df_merged['whistlerKPa'] = df_merged['whistlerKPa'].interpolate(method='linear')

# Lillooet is approx 5.1 kPa higher than Whistler, as an approximation. Similar for others.
# df_merged['lillooetKPa'] = df_merged['lillooetKPa'].fillna(df_merged['whistlerKPa'] + 5.1)
# df_merged['ballenasKPa'] = df_merged['ballenasKPa'].fillna(df_merged['victoriaKPa'])
# df_merged['comoxKPa'] = df_merged['comoxKPa'].fillna(df_merged['ballenasKPa'])
# df_merged['victoriaKPa'] = df_merged['victoriaKPa'].fillna(df_merged['comoxKPa'])
# df_merged['pamKPa'] = df_merged['pamKPa'].fillna(df_merged['comoxKPa'] + 0.23)
# df_merged['ballenasDegC'] = df_merged['ballenasDegC'].fillna(df_merged['pamDegC'])
# df_merged['pamDegC'] = df_merged['pamDegC'].fillna(df_merged['ballenasDegC'])
# df_merged['lillooetDegC'] = df_merged['lillooetDegC'].fillna(df_merged['pembertonDegC'] + 1.5)
# df_merged['pembertonDegC'] = df_merged['pembertonDegC'].fillna(df_merged['lillooetDegC'] - 1.5)

# Interpolate a few that have a small number of missing points
columns_interpolate = [
    'comoxKPa', 'lillooetDegC', 'lillooetKPa', 'pamKPa', 'pembertonDegC',
    'vancouverDegC', 'vancouverKPa', 'victoriaDegC', 'victoriaKPa', 'whistlerDegC'
]

for col in columns_interpolate:
    df_merged[col] = df_merged[col].interpolate(method='linear')
    pass

columns_fill_fb = ['pamKPa']
for col in columns_fill_fb:
    df_merged[col] = df_merged[col].ffill()
    df_merged[col] = df_merged[col].bfill()
    pass

# Fill speed and add mask so we can ignore those rows where speed isn't recorded
columns_fill_zero = ['direction', 'gust', 'lull', 'speed']
df_merged['speed_missing'] = df_merged['speed'].isna().astype(int)

for col in columns_fill_zero:
    df_merged[col] = df_merged[col].fillna(0)
    pass

# Fill the sky condition columns with the previous known entry
# Assumption is that the gaps are small enough
# df_merged['vancouverSky'] = df_merged['vancouverSky'].ffill()
# df_merged['whistlerSky'] = df_merged['whistlerSky'].ffill()
# df_merged['comoxSky'] = df_merged['comoxSky'].ffill()
# df_merged['victoriaSky'] = df_merged['victoriaSky'].ffill()

# Add computed gust/lull, they may be easier features than absolute values.
df_merged['gust_relative'] = df_merged['gust'] / df_merged['speed']  # Absolute gust wasn't as accurate, trying to predict relative to speed
df_merged['lull_relative'] = df_merged['lull'] / df_merged['speed']  # Same thing with lull
df_merged['gust_relative'] = df_merged['gust_relative'].replace([np.inf, -np.inf], 3)  # Dividing results in some inf values
df_merged['lull_relative'] = df_merged['lull_relative'].replace([np.inf, -np.inf, np.nan], 0)
df_merged['gust_relative'] = df_merged['gust_relative'].replace(np.nan, 0)  # and some NaN values
df_merged['gust_relative'] = df_merged['gust_relative'].clip(lower=1, upper=3, axis=0)  # Assume gust is only ever 3x wind speed
df_merged['lull_relative'] = df_merged['lull_relative'].clip(lower=0, upper=1, axis=0)  # Similarly, 0 < lull < 1

# Add calculated values
# df_merged['day_fraction'] = (df_merged.index - df_merged.index.normalize()).total_seconds() / 86400  # Add as a categorical feature
df_merged['date'] = df_merged['datetime'].dt.date
df_merged['hour'] = df_merged['datetime'].dt.hour
df_merged['month'] = df_merged['datetime'].dt.month
df_merged['day'] = df_merged['datetime'].dt.day
df_merged['year_fraction'] = (pd.to_timedelta(df_merged['month'] * 30.416 + df_merged['day'], unit='D')).dt.days / 365
df_merged['gustLull_index'] = (df_merged['gust_relative'] - 1) + (1 - df_merged['lull_relative'])
df_merged['sin_hour'] = np.sin(2 * np.pi * df_merged['hour'] / 24)

# The weather columns have many values (eg. "Rain", "Rain Showers", "Ice pellets"). Reduce down.
df_merged['vancouverSky'] = df_merged['vancouverSky'].replace(['Clear', 'Mainly Clear'], 'Fair')  # Clear and mainly clear should be similar
df_merged['vancouverSky'] = df_merged['vancouverSky'].replace(r'^(?!Fair$|Mostly Cloudy|Cloudy$).*', 'Other', regex=True)
df_merged['comoxSky'] = df_merged['comoxSky'].replace(['Clear', 'Mainly Clear'], 'Fair')  # Clear and mainly clear should be similar
df_merged['comoxSky'] = df_merged['comoxSky'].replace(r'^(?!Fair$|Mostly Cloudy|Cloudy$).*', 'Other', regex=True)
df_merged['victoriaSky'] = df_merged['victoriaSky'].replace(['Clear', 'Mainly Clear'], 'Fair')  # Clear and mainly clear should be similar
df_merged['victoriaSky'] = df_merged['victoriaSky'].replace(r'^(?!Fair$|Mostly Cloudy|Cloudy$).*', 'Other', regex=True)
df_merged['whistlerSky'] = df_merged['whistlerSky'].replace(['Clear', 'Mainly Clear'], 'Fair')  # Clear and mainly clear should be similar
df_merged['whistlerSky'] = df_merged['whistlerSky'].replace(r'^(?!Fair$|Mostly Cloudy|Cloudy$).*', 'Other', regex=True)

# Reset the index and sort alphabetically
df_merged = df_merged.sort_values(by='datetime')
df_merged = df_merged[sorted(df_merged.columns)]

# Save to csv file
df_merged.to_csv('hourly_database.csv')


