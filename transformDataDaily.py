import numpy as np
import pandas as pd

def compute_5score(value, min_cutoff=0, max_cutoff=0):
    """
    :param value: float
    :return: float, interpolated on a 1 to 5 scale
    """
    return np.clip(5 - 4 * (value - min_cutoff) / (max_cutoff - min_cutoff), 1, 5)

data = pd.read_csv('mergedOnSpeed_hourly.csv')  # Assuming you have your data in a CSV
data['time'] = pd.to_datetime(data['time'])  # Ensure it's in DateTime format
data = data.sort_values('time')  # Sort chronologically (if not already)

# Get df's for each of the daily direction, variability scores, and max_speed
df_directionScore = data[data['speed'] > 15].groupby('date')['direction'].std().reset_index(name='dir_stdev')
df_directionScore['dir_score'] = df_directionScore['dir_stdev'].apply(compute_5score, min_cutoff=0.8, max_cutoff=18)
df_varScore = data[data['speed'] > 15].groupby('date')['gustLull_index'].mean().apply(compute_5score, min_cutoff=0.15, max_cutoff=0.75).reset_index(name='speed_variability')
# df_maxSpeed = data.loc[data.groupby('date')['speed'].idxmax(), ['date', 'speed', 'hour']]
# df_maxSpeed = data.groupby('date')['speed'].max().reset_index(name='max_speed')
data = data[data['hour'] == 14].reset_index(drop=True)

# Merge the daily 2pm sensor readings with the maxSpeed, varScore, and directionScore
data = data.merge(df_directionScore, on='date', how='left')
data = data.merge(df_varScore, on='date', how='left')
# data = data.merge(df_maxSpeed, on='date', how='left')
data.fillna({'dir_score': 0, 'speed_variability': 0}, inplace=True)

data.to_csv('mergedOnSpeed_daily.csv')