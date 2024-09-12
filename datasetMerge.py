import pandas as pd

# Load dataframes and convert time columns to proper format
vancouverWeather = pd.read_csv('Vancouver.csv')
whistlerWeather = pd.read_csv('Whistler.csv').set_index('Date/Time (LST)')
pembertonWeather = pd.read_csv('Pemberton.csv').set_index('Date/Time (LST)')
lillooetWeather = pd.read_csv('Lillooet.csv').set_index('Date/Time (LST)')
squamishWind = pd.read_csv('swsWind.csv')
squamishWind['time'] = pd.to_datetime(squamishWind['time'])
squamishWind.set_index('time', inplace=True)

# Make a dataframe to hold the weather data (features)
df_features = pd.DataFrame()
df_features['datetime'] = vancouverWeather['Date/Time (LST)']
df_features['vancouverDegC'] = vancouverWeather['Temp (\u00b0'+'C)']

# Compile the data from Whistler, Pemberton, Lillooet using merge
df_features = df_features.merge(whistlerWeather['Temp (\u00b0'+'C)'], how='inner', left_on='datetime', right_index=True)
df_features.rename(columns={'Temp (\u00b0'+'C)':'whistlerDegC'}, inplace=True)
df_features = df_features.merge(pembertonWeather['Temp (\u00b0'+'C)'], how='inner', left_on='datetime', right_index=True)
df_features.rename(columns={'Temp (\u00b0'+'C)':'pembertonDegC'}, inplace=True)
df_features = df_features.merge(lillooetWeather['Temp (\u00b0'+'C)'], how='inner', left_on='datetime', right_index=True)
df_features.rename(columns={'Temp (\u00b0'+'C)':'lillooetDegC'}, inplace=True)
df_features['datetime'] = pd.to_datetime(df_features['datetime'])
df_features.set_index('datetime', inplace=True)

# Merge the two datasets, interpolate for missing data
df_merged = pd.merge(df_features, squamishWind, left_index=True, right_index=True, how='outer')
df_merged['vancouverDegC'] = df_merged['vancouverDegC'].interpolate(method='linear')
df_merged['whistlerDegC'] = df_merged['whistlerDegC'].interpolate(method='linear')
df_merged['pembertonDegC'] = df_merged['pembertonDegC'].interpolate(method='linear')
df_merged['lillooetDegC'] = df_merged['lillooetDegC'].interpolate(method='linear')

# Drop unwanted columns and rows without any speed data
df_merged = df_merged.drop(['Unnamed: 0', 'direction', 'gust', 'lull', 'temperature'], axis=1)
df_merged = df_merged.dropna(subset=['speed'])

# Save to csv file
df_merged.to_csv('mergedOnSpeed.csv')


