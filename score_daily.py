"""Compute daily wind quality scores from hourly SWS observations.

Public API:
    add_scores_to_df(df)   → df with direction_score, speed_score, hours_above_20 columns

Run standalone to rebuild daily_database.csv from hourly_database.csv:
    python score_daily.py
"""

import numpy as np
import pandas as pd


def _to_5_score(value: pd.Series, low: float, high: float) -> pd.Series:
    """Linearly map *value* from [low, high] onto [1, 5], clipped."""
    return np.clip(5 - 4 * (value - low) / (high - low), 1, 5)


def add_scores_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """Append daily quality scores to *df* (which must have hourly rows).

    Adds columns: direction_score, speed_score, hours_above_20.
    Rows are deduplicated to one per day at 14:00 local time.
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert('America/Vancouver')
    df['date'] = df['datetime'].dt.date

    df['gust_relative']  = df['gust'] / df['speed']
    df['lull_relative']  = df['lull'] / df['speed']
    df['gustLull_index'] = (df['gust_relative'] - 1) + (1 - df['lull_relative'])

    sailing = df[df['speed'] > 15]

    dir_stdev = (
        sailing.groupby('date')['direction']
        .std()
        .reset_index(name='dir_stdev')
    )
    dir_stdev['direction_score'] = _to_5_score(dir_stdev['dir_stdev'], low=0.8, high=18)

    speed_score = (
        sailing.groupby('date')['gustLull_index']
        .mean()
        .apply(_to_5_score, low=0.15, high=0.75)
        .reset_index(name='speed_score')
    )

    hours_above_20 = (
        df.assign(above_20=df['speed'] > 20)
        .groupby('date', as_index=False)
        .agg(hours_above_20=('above_20', 'sum'))
    )
    hours_above_20['hours_above_20'] -= 1  # calibration offset

    daily = pd.DataFrame({'date': df['date'].unique()})
    daily = (daily
             .merge(dir_stdev[['date', 'direction_score']], on='date', how='left')
             .merge(speed_score,  on='date', how='left')
             .merge(hours_above_20, on='date', how='left'))

    daily['date'] = (
        pd.to_datetime(daily['date'])
        + pd.to_timedelta(14, 'hours')
    ).dt.tz_localize('America/Vancouver')
    daily = daily.rename(columns={'date': 'datetime'})

    result = daily.merge(df, on='datetime', how='left')
    result.drop(columns='dir_stdev', inplace=True, errors='ignore')
    result.fillna({'direction_score': 0, 'speed_score': 0}, inplace=True)
    result.sort_values('datetime', inplace=True)
    return result


if __name__ == '__main__':
    data = pd.read_csv('hourly_database.csv')
    data = add_scores_to_df(data)
    data.to_csv('daily_database.csv', index=False)
    print(f'Saved daily_database.csv  {len(data):,} rows')
