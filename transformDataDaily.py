import numpy as np
import pandas as pd


def compute_5score(value, min_cutoff=0, max_cutoff=0):
    """
    :param value: float
    :return: float, interpolated on a 1 to 5 scale
    """
    return np.clip(5 - 4 * (value - min_cutoff) / (max_cutoff - min_cutoff), 1, 5)


def add_scores_to_df(df):
    # Get df's for each of the daily direction, variability scores, and max_speed
    df['date'] = df['datetime'].dt.date
    df['gust_relative'] = df['gust'] / df[
        'speed']  # Absolute gust wasn't as accurate, trying to predict relative to speed
    df['lull_relative'] = df['lull'] / df['speed']  # Same thing with lull
    df['gustLull_index'] = (df['gust_relative'] - 1) + (1 - df['lull_relative'])
    df_directionscore = df[df['speed'] > 15].groupby('date')['direction'].std().reset_index(name='dir_stdev')
    df_directionscore['direction_variability'] = df_directionscore['dir_stdev'].apply(compute_5score, min_cutoff=0.8, max_cutoff=18)
    df_varscore = df[df['speed'] > 15].groupby('date')['gustLull_index'].mean().apply(compute_5score,
                                                                                          min_cutoff=0.15,
                                                                                          max_cutoff=0.75).reset_index(name='speed_variability')

    # Merge the daily 2pm sensor readings with the maxSpeed, varScore, and directionScore
    df_ratings = pd.DataFrame()
    df_ratings['date'] = df['date'].unique()
    df_ratings = df_ratings.merge(df_directionscore, on='date', how='left')
    df_ratings = df_ratings.merge(df_varscore, on='date', how='left')
    df_ratings['date'] = pd.to_datetime(df_ratings['date']) + pd.to_timedelta(14, 'hours')
    df_ratings.rename(columns={'date': 'datetime'}, inplace=True)
    # df_ratings = df_ratings.merge(df, left_on='datetime', right_on='datetime', how='left')
    # df_ratings = df_ratings.merge(df[['datetime', 'speed']], on='datetime', how='left')
    # df_ratings.drop(columns=['time'], inplace=True)
    df_ratings.fillna({'direction_variability': 0, 'speed_variability': 0}, inplace=True)
    return df_ratings

if __name__=='__main__':
    # data = pd.read_csv('mergedOnSpeed_hourly.csv')  # Assuming you have your data in a CSV
    # data['time'] = pd.to_datetime(data['time'])  # Ensure it's in DateTime format
    # data = data.sort_values('time')  # Sort chronologically (if not already)
    # data = add_scores_to_df(data)
    # data.to_csv('mergedOnSpeed_daily.csv')
    pass