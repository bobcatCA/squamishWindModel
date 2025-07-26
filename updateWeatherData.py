import numpy as np
import os
import pandas as pd
import pytz
import sqlite3
from datetime import timedelta
from dotenv import load_dotenv
from envCanadaForecastPull import pull_forecast_hourly, pull_forecast_daily
from envCanadaStationPull import pull_past_hrs_weather
from pathlib import Path
from swsDataPull import get_sws_df
from transformDataDaily import add_scores_to_df


# Load environment variables
load_dotenv()
sql_database_path = Path(os.getenv('WORKING_DIRECTORY')) / 'weather_data_hourly.db'


def update_sql_db_hourly(df):
    """
    :param df: Pandas DataFrame, containing time-series weather station data
    :return: None
    """
    conn = sqlite3.connect(sql_database_path)

    # Get existing table column names, keep only those matching, and commit.
    sql_columns = pd.read_sql("PRAGMA table_info(weather)", conn)['name'].tolist()
    conn.close()

    # Add missing columns with NaN if the columns exist in SQL, but not in the df
    df = df.copy()
    for col in sql_columns:
        if col not in df.columns:
            print(col)
            df[col] = np.nan

    df = df[sql_columns]  # Remove any columns from df that don't exist in SQL database

    # Convert DataFrame to a list of tuples, and use Unix timestamps
    # df['datetime'] = (df['datetime'] - pd.Timestamp('1970-01-01', tz='America/Vancouver')) // pd.Timedelta('1s')
    df['datetime'] = df['datetime'].dt.tz_convert('UTC')
    df['datetime'] = df['datetime'].astype('int64') // 10**9
    data = list(df.itertuples(index=False, name=None))

    # Create INSERT OR IGNORE query dynamically
    sql = """
    INSERT OR IGNORE INTO weather ({})
    VALUES ({})
    """.format(", ".join(df.columns), ", ".join(["?"] * len(df.columns)))

    # Connect to database and insert data
    with sqlite3.connect(sql_database_path) as conn:
        cursor = conn.cursor()

        # Get initial row count before insertion
        cursor.execute('SELECT COUNT(*) FROM weather')
        before_count = cursor.fetchone()[0]

        # Insert new data
        cursor.executemany(sql, data)
        conn.commit()

        # Get new row count after insertion
        cursor.execute('SELECT COUNT(*) FROM weather')
        after_count = cursor.fetchone()[0]

        # Calculate how many rows were inserted vs. ignored
        inserted_rows = after_count - before_count
        ignored_rows = len(data) - inserted_rows

    # Output results
    print(f'Total rows attempted: {len(data)}')
    print(f'Inserted: {inserted_rows}')
    print(f'Ignored (duplicates): {ignored_rows}')

    return


def get_conditions_table_daily(encoder_length=8, prediction_length=5):
    """
    :param encoder_length: Int, number of time steps to look back/encode
    :param prediction_length: Int, number of time steps to predict/look forward
    :return: DataFrame, Concat'd with observed values from the past/upcoming days
    """
    today_14 = pd.to_datetime(pd.Timestamp.now(tz='America/Vancouver').date()) + pd.to_timedelta(14, 'hours')
    start_time = today_14 - timedelta(days=encoder_length)
    end_time = today_14 + timedelta(days=prediction_length - 1)
    time_values = pd.date_range(start=start_time, end=end_time, freq='d')

    # Get corresponding recent data from SQL server
    conn = sqlite3.connect(sql_database_path)
    df_encoder = pd.read_sql_query('SELECT * FROM weather WHERE datetime > ?', conn, params=(start_time.timestamp(), ))
    df_encoder['datetime'] = pd.to_datetime(df_encoder['datetime'], unit='s', utc=True)
    df_encoder['datetime'] = df_encoder['datetime'].dt.tz_convert('America/Vancouver')
    conn.close()

    # Merge SQL data with desired date range
    df = pd.DataFrame()
    df['datetime'] = time_values
    df['datetime'] = df['datetime'].dt.tz_localize('America/Vancouver')
    df = df.merge(df_encoder, on='datetime', how='left')

    # Add in the Quality scores, these are daily labels to predict
    df_ratings = add_scores_to_df(df_encoder)
    df = df.merge(df_ratings, on='datetime', how='left')

    # Get forecast data from Environment Canada and concatenate
    df_forecast = pull_forecast_daily(time_values)
    df = pd.concat([df, df_forecast])
    df['year_fraction'] = ((df['datetime'].dt.month - 1) * 30.416 + df['datetime'].dt.day - 1) / 365
    print('pause')

    # Categorize weather columns into 'Fair', 'Mostly Cloudy', 'Cloudy', and 'Other'
    df.loc[:, df.columns.str.contains('Sky')] = df.loc[:, df.columns.str.contains('Sky')].replace(['Clear',
                                                                                                   'Mainly Clear',
                                                                                                   'Sunny'
                                                                                                   ], 'Fair')
    df.loc[:, df.columns.str.contains('Sky')] = df.loc[:, df.columns.str.contains('Sky')].replace(['A mix of sun and cloud',
                                                                                                   'Partly cloudy',
                                                                                                   'Mainly cloudy',
                                                                                                   ], 'Mostly Cloudy')
    df.loc[:, df.columns.str.contains('Sky')] = df.loc[:, df.columns.str.contains('Sky')].replace(['A mix of sun and cloud',
                                                                                                   ], 'Mostly Cloudy')
    df.loc[:, df.columns.str.contains('Sky')] = df.loc[:, df.columns.str.contains('Sky')].replace(r'^(?!Fair$|Mostly Cloudy|Cloudy$).*',
                                                                                                  'Other', regex = True)

    # Final cleaning/re-ordering
    df.sort_values(by='datetime', inplace=True)
    df.dropna(inplace=True, thresh=14)  # TODO: Better way than thresh=14?
    df.reset_index(inplace=True, drop=True)
    return df


def get_conditions_table_hourly(encoder_length=12, prediction_length=8):
    """
    :param encoder_length: Int, number of time steps to look back/encode
    :param prediction_length: Int, number of time steps to predict/look forward
    :return: DataFrame, Concat'd with observed values from the past/upcomind days
    """
    # Pull the past and forecast data. Update the SQL database with recent data
    df_weather_recent = pull_past_hrs_weather()
    past24_dates = list(df_weather_recent['datetime'].dt.date.unique().astype(str))
    df_sws = get_sws_df(past24_dates)
    df_recent = pd.merge_asof(df_weather_recent, df_sws, on='datetime', direction='nearest')
    update_sql_db_hourly(df_recent)

    # Make a new DF for the desired dates
    now_time = pd.Timestamp.now(tz='America/Vancouver').ceil('h')
    start_time = now_time - timedelta(hours=encoder_length)
    end_time = now_time + timedelta(hours=prediction_length - 1)  # TODO: does this have to match the max_encoder_length exactly?
    time_values = pd.date_range(start=start_time, end=end_time, freq='h')
    df = pd.DataFrame(columns=['datetime'])

    # Get the recent data from SQL DB, per the encoder_length (df_recent may be smaller than encoder_length)
    conn = sqlite3.connect(sql_database_path)
    df_encoder = pd.read_sql_query('SELECT * FROM weather WHERE datetime > ?', conn, params=(start_time.timestamp(), ))
    df_encoder['datetime'] = pd.to_datetime(df_encoder['datetime'], unit='s', utc=True)
    df_encoder['datetime'] = df_encoder['datetime'].dt.tz_convert('America/Vancouver')

    # Pull forecast data and put the two dataframes together
    df_forecast = pull_forecast_hourly()
    df_data = pd.concat([df_forecast, df_encoder], ignore_index=True, sort=False)
    df_data.sort_values(by='datetime', inplace=True)
    df_data.reset_index(drop=True, inplace=True)

    # Merge past and forecast data with the desired time interval
    df['datetime'] = time_values
    df = df.merge(df_data, on='datetime', how='left')
    df.sort_values(by='datetime', inplace=True)

    # Add calculated columns
    df['sin_hour'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
    df['year_fraction'] = ((df['datetime'].dt.month - 1) * 30.416 + df['datetime'].dt.day - 1) / 365
    df['is_daytime'] = df['datetime'].dt.hour.between(10, 17).astype(str)
    df['is_thermal'] = ((df['lillooetDegC'] - df['vancouverDegC']) > 5).astype(str)

    # Categorize weather columns into 'Fair', 'Mostly Cloudy', 'Cloudy', and 'Other'
    df.loc[:, df.columns.str.contains('Sky')] = df.loc[:, df.columns.str.contains('Sky')].replace(['Clear',
                                                                                                   'Mainly Clear',
                                                                                                   'Sunny'
                                                                                                   ], 'Fair')
    df.loc[:, df.columns.str.contains('Sky')] = df.loc[:, df.columns.str.contains('Sky')].replace(['A mix of sun and cloud',
                                                                                                   'Partly cloudy',
                                                                                                   'Mainly cloudy',
                                                                                                   ], 'Mostly Cloudy')
    df.loc[:, df.columns.str.contains('Sky')] = df.loc[:, df.columns.str.contains('Sky')].replace(['A mix of sun and cloud',
                                                                                                   ], 'Mostly Cloudy')
    df.loc[:, df.columns.str.contains('Sky')] = df.loc[:, df.columns.str.contains('Sky')].replace(r'^(?!Fair$|Mostly Cloudy|Cloudy$).*',
                                                                                                  'Other', regex = True)

    # There might be one row missing, due to an hour gap between observed and forecast. Interpolate/ffill
    df.loc[:, df.select_dtypes(include=['number']).columns] = df.select_dtypes(include=['number']).apply(lambda x: x.interpolate(limit=1))
    df.dropna(thresh=14, inplace=True)  # TODO: better way than Thresh=14?
    df.sort_values(by='datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

if __name__=='__main__':
    # df = pd.read_csv('mergedOnSpeed_forSQL.csv')
    # df['datetime'] = pd.to_datetime(df['datetime'])
    # update_sql_db_hourly(df)

    cnxn = sqlite3.connect(sql_database_path)
    df_test = pd.read_sql('SELECT * FROM weather', cnxn)
    pass
