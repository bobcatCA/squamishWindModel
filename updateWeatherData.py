import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from envCanadaForecastPull import pull_forecast
from envCanadaStationPull import pull_past_24hrs_weather
from swsDataPull import get_sws_df

def update_sql_db_hourly(df):
    # Connect to the database and append new data
    conn = sqlite3.connect('weather_data_hourly.db')
    # cursor = conn.cursor()

    # Get existing table column names, keep only those matching, and commit.
    sql_columns = pd.read_sql("PRAGMA table_info(weather)", conn)['name'].tolist()
    conn.close()

    # Add missing columns with NaN if the columns exist in SQL, but not in the df
    for col in sql_columns:
        if col not in df.columns:
            print(col)
            df[col] = np.nan

    df = df[sql_columns]  # Remove any columns from df that don't exist in SQL database
    # df.loc[:, 'datetime'] = df['datetime'].astype(int)
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

    # # Append recent records to the database
    # df.to_sql('weather', conn, if_exists='append', index=False, method='multi')
    #
    # # Remove duplicates if they exist
    # cursor.execute("""
    # DELETE FROM weather
    # WHERE rowid NOT IN (
    #     SELECT MIN(rowid)
    #     FROM weather
    #     GROUP BY datetime
    # )
    # """)


    return


def get_conditions_table(target_columns, known_columns, encoder_length=30, prediction_length=8):
    """
    :return: DataFrame, Concat'd with observed values from the past 24hrs, and forecast hourly
    """
    # Pull the past and forecast data
    df_weatherRecent = pull_past_24hrs_weather()
    # df_weatherRecent = pd.DataFrame()  # TODO: DELETE
    past24_dates = list(df_weatherRecent['datetime'].dt.date.unique().astype(str))
    # df_sws = get_sws_df(past24_dates)
    df_sws = get_sws_df(['2024-09-02', '2024-09-03'])
    df_sws['datetime'] + pd.to_timedelta(198, 'days')
    df_recent = pd.merge_asof(df_weatherRecent, df_sws, on='datetime', direction='nearest')
    update_sql_db_hourly(df_recent)

    df_forecast = pull_forecast()

    # Put the two dataframes together
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