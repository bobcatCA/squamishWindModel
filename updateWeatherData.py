import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from envCanadaForecastPull import pull_forecast
from envCanadaStationPull import pull_past_24hrs

def get_conditions_table(target_columns, known_columns, encoder_length=30, prediction_length=8):
    """
    :return: DataFrame, Concat'd with observed values from the past 24hrs, and forecast hourly
    """
    # Pull the past and forecast data
    df_past24 = pull_past_24hrs()
    for col in target_columns:
        df_past24[col] = np.random.randint(1, 8, df_past24.shape[0])  # Add in the labels from the main file. TODO: pull SWS wind data
    df_forecast = pull_forecast()

    # Put the two dataframes together
    df_data = pd.concat([df_forecast, df_past24], ignore_index=True, sort=False)
    df_data.sort_values(by='datetime', inplace=True)
    df_data.reset_index(drop=True, inplace=True)

    # Make a new DF for the desired dates
    nowTime = pd.Timestamp.now().ceil('h')
    startTime = nowTime - timedelta(hours=encoder_length)
    endTime = nowTime + timedelta(hours=prediction_length)
    timeValues = pd.date_range(start=startTime, end=endTime, freq='h')
    df = pd.DataFrame(columns=['datetime'])
    df['datetime'] = timeValues
    df = df.merge(df_data, on='datetime', how='left')
    df.sort_values(by='datetime', inplace=True)

    # Add calculated columns
    df['sin_hour'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
    df['month'] = df['datetime'].dt.month
    df['year_fraction'] = (pd.to_timedelta(df['month'] * 30.416, unit='D')).dt.days / 365
    df.drop(columns=['month'], inplace=True)

    # There might be one row missing, due to an hour gap between observed and forecast. Interpolate/ffill
    # df[known_columns].iloc[:, 1:] = df[known_columns].iloc[:, 1:].apply(lambda col: col.interpolate() if col.dtype.kind in 'biufc' else col.ffill())
    df.loc[:, known_columns] = df.loc[:, known_columns].apply(lambda col: col.interpolate() if col.dtype.kind in 'biufc' else col.ffill())
    df.dropna(axis=0, inplace=True, subset=known_columns)  # Drop missing rows at the start of the datset
    df.reset_index(drop=True, inplace=True)

    return df

if __name__=='__main__':
    pass