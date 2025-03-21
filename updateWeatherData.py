import numpy as np
import pandas as pd
import sqlite3
from datetime import timedelta
from envCanadaForecastPull import pull_forecast_hourly, pull_forecast_daily
from envCanadaStationPull import pull_past_24hrs_weather
from swsDataPull import get_sws_df
from transformDataDaily import add_scores_to_df

def update_sql_db_hourly(df):
    """
    :param df: Pandas DataFrame, containing time-series weather station data
    :return: None
    """
    conn = sqlite3.connect('weather_data_hourly.db')

    # Get existing table column names, keep only those matching, and commit.
    sql_columns = pd.read_sql("PRAGMA table_info(weather)", conn)['name'].tolist()
    conn.close()

    # Add missing columns with NaN if the columns exist in SQL, but not in the df
    for col in sql_columns:
        if col not in df.columns:
            print(col)
            df[col] = np.nan

    df = df[sql_columns]  # Remove any columns from df that don't exist in SQL database
    # df.loc[:, 'datetime'] = df['datetime'].astype(int)  # TODO: Decide if the database should be Int timestamps or Y-m-d
    df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Convert DataFrame to a list of tuples
    records = df.to_records(index=False)
    data = list(records)

    # Create INSERT OR IGNORE query dynamically
    sql = """
    INSERT OR IGNORE INTO weather ({})
    VALUES ({})
    """.format(", ".join(df.columns), ", ".join(["?"] * len(df.columns)))

    # Connect to database and insert data
    with sqlite3.connect("weather_data_hourly.db") as conn:
        cursor = conn.cursor()

        # Get initial row count before insertion
        cursor.execute("SELECT COUNT(*) FROM weather")
        before_count = cursor.fetchone()[0]

        # Insert new data
        cursor.executemany(sql, data)
        conn.commit()

        # Get new row count after insertion
        cursor.execute("SELECT COUNT(*) FROM weather")
        after_count = cursor.fetchone()[0]

        # Calculate how many rows were inserted vs. ignored
        inserted_rows = after_count - before_count
        ignored_rows = len(data) - inserted_rows

    # Output results
    print(f"Total rows attempted: {len(data)}")
    print(f"Inserted: {inserted_rows}")
    print(f"Ignored (duplicates): {ignored_rows}")

    return


def get_conditions_table_daily(encoder_length=8, prediction_length=5):
    """
    :param encoder_length: Int, number of time steps to look back/encode
    :param prediction_length: Int, number of time steps to predict/look forward
    :return: DataFrame, Concat'd with observed values from the past/upcoming days
    """
    today_14 = pd.to_datetime(pd.Timestamp.now().date()) + pd.to_timedelta(14, 'hours')
    startTime = today_14 - timedelta(days=encoder_length)
    endTime = today_14 + timedelta(days=prediction_length)
    timeValues = pd.date_range(start=startTime, end=endTime, freq='d')

    # Get corresponding recent data from SQL server
    conn = sqlite3.connect('weather_data_hourly.db')
    df_recent = pd.read_sql('SELECT * FROM weather', conn)
    df_recent['datetime'] = pd.to_datetime(df_recent['datetime'])

    # TODO: Delete, once SWS is up and running
    df_recent['direction'] = np.random.randint(50, 361, df_recent.shape[0])
    df_recent['lull'] = np.random.randint(1, 25, df_recent.shape[0])
    df_recent['gust'] = np.random.randint(1, 25, df_recent.shape[0])
    df_recent['speed'] = np.random.randint(1, 25, df_recent.shape[0])

    conn.close()

    # Merge SQL data with desired date range
    df = pd.DataFrame()
    df['datetime'] = timeValues
    df = df.merge(df_recent, on='datetime', how='left')

    # Add in the Quality scores, these are daily labels to predict
    df_ratings = add_scores_to_df(df_recent)
    df = df.merge(df_ratings, on='datetime', how='left')

    # Get forecast data from Environment Canada and concatenate
    df_forecast = pull_forecast_daily(timeValues)
    df = pd.concat([df, df_forecast])
    df['year_fraction'] = ((df['datetime'].dt.month - 1) * 30.416 + df['datetime'].dt.day - 1) / 365

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
    df.reset_index(inplace=True, drop=True)
    return df


def get_conditions_table_hourly(encoder_length=50, prediction_length=8):
    """
    :param encoder_length: Int, number of time steps to look back/encode
    :param prediction_length: Int, number of time steps to predict/look forward
    :return: DataFrame, Concat'd with observed values from the past/upcomind days
    """
    # Pull the past and forecast data. Update the SQL database with recent data
    df_weather_recent = pull_past_24hrs_weather()
    past24_dates = list(df_weather_recent['datetime'].dt.date.unique().astype(str))
    # df_sws = get_sws_df(past24_dates)  # TODO: uncomment v
    df_sws = get_sws_df(['2024-09-02', '2024-09-03'])  # TODO: Erase this once SWS weather station is up and running
    df_sws['datetime'] + pd.to_timedelta(198, 'days')  # TODO: Ditto
    df_recent = pd.merge_asof(df_weather_recent, df_sws, on='datetime', direction='nearest')
    update_sql_db_hourly(df_recent)

    # Pull forecast data and put the two dataframes together
    df_forecast = pull_forecast_hourly()
    df_data = pd.concat([df_forecast, df_recent], ignore_index=True, sort=False)
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
    df['year_fraction'] = ((df['datetime'].dt.month - 1) * 30.416 + df['datetime'].dt.day - 1) / 365

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
    # df.loc[:, known_columns] = df.loc[:, known_columns].apply(lambda col: col.interpolate() if col.dtype.kind in 'biufc' else col.ffill())
    df = df.apply(lambda col: col.interpolate() if col.dtype.kind in 'biufc' else col.ffill())
    df.sort_values(by='datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

if __name__=='__main__':
    # df = pd.read_csv('mergedOnSpeed_forSQL.csv')
    # df['datetime'] = pd.to_datetime(df['datetime'])
    # update_sql_db_hourly(df)

    conn = sqlite3.connect('weather_data_hourly.db')
    df_test = pd.read_sql('SELECT * FROM weather', conn)
    pass