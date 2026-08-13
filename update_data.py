"""Pull the latest EC and SWS observations into the SQLite database.

    python update_data.py       # update DB only (no forecast generated)

The forecast scripts call update_db() automatically, so this script is only
needed when you want to refresh the DB without generating a new forecast.
"""

import numpy as np
import os
import pandas as pd
import sqlite3
from datetime import timedelta
from dotenv import load_dotenv
from pathlib import Path

from build_dataset import wind_to_hourly_index
from ec_scrape import pull_past_hrs_weather
from sws_pull import get_sws_df

load_dotenv()


def _db_path() -> Path:
    return Path(os.getenv('WORKING_DIRECTORY')) / 'web_data' / 'weather_data_hourly.db'


_DB_COL_RENAME = {
    'speed':       'squamishSpeed',
    'gust':        'squamishGust',
    'lull':        'squamishLull',
    'direction':   'squamishDirection',
    'temperature': 'squamishDegC',
}


def _insert_rows(df: pd.DataFrame) -> None:
    with sqlite3.connect(_db_path()) as conn:
        sql_columns = pd.read_sql('PRAGMA table_info(weather)', conn)['name'].tolist()

    df = df.copy()
    for col in sql_columns:
        if col not in df.columns:
            df[col] = np.nan
    df = df[sql_columns]

    epoch = pd.Timestamp('1970-01-01', tz='UTC')
    df['datetime'] = (df['datetime'].dt.tz_convert('UTC') - epoch).dt.total_seconds().astype('int64')
    rows = list(df.itertuples(index=False, name=None))

    sql = 'INSERT OR IGNORE INTO weather ({}) VALUES ({})'.format(
        ', '.join(df.columns), ', '.join(['?'] * len(df.columns)),
    )

    with sqlite3.connect(_db_path()) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM weather')
        before = cursor.fetchone()[0]
        cursor.executemany(sql, rows)
        conn.commit()
        cursor.execute('SELECT COUNT(*) FROM weather')
        after = cursor.fetchone()[0]

    inserted = after - before
    print(f'Rows attempted: {len(rows)}  inserted: {inserted}  duplicates: {len(rows) - inserted}')


def update_db() -> None:
    """Pull latest EC and SWS observations and insert new rows into the SQLite DB."""
    df_recent_weather = pull_past_hrs_weather()
    past_dates = list(df_recent_weather['datetime'].dt.date.unique().astype(str))
    try:
        df_sws = get_sws_df(past_dates)
    except Exception as e:
        print(f'Warning: SWS pull failed ({e}), continuing without wind data')
        df_sws = pd.DataFrame()

    if not df_sws.empty:
        # Average raw (few-minutes-resolution) SWS readings within ±10min of each
        # EC on-hour timestamp, same aggregation used when building training CSVs.
        df_sws = df_sws.rename(columns=_DB_COL_RENAME).set_index('datetime')
        df_wind = wind_to_hourly_index(df_sws, df_recent_weather['datetime'])
        df_recent = df_recent_weather.merge(df_wind, left_on='datetime', right_index=True, how='left')
    else:
        df_recent = df_recent_weather

    _insert_rows(df_recent)


def get_conditions_table_hourly(encoder_length: int = 12, prediction_length: int = 8,
                                 specified_start: pd.Timestamp = None,
                                 update: bool = True) -> pd.DataFrame:
    if update:
        update_db()

    now   = specified_start or pd.Timestamp.now(tz='America/Vancouver').ceil('h')
    start = now - timedelta(hours=encoder_length)
    end   = now + timedelta(hours=prediction_length - 1)
    time_index = pd.date_range(start=start, end=end, freq='h')

    with sqlite3.connect(_db_path()) as conn:
        df_hist = pd.read_sql_query(
            'SELECT * FROM weather WHERE datetime > ?',
            conn, params=(start.timestamp(),),
        )
    df_hist['datetime'] = (
        pd.to_datetime(df_hist['datetime'], unit='s', utc=True)
        .dt.tz_convert('America/Vancouver')
    )

    df_hist.sort_values('datetime', inplace=True)

    df = pd.DataFrame({'datetime': time_index})
    df = df.merge(df_hist, on='datetime', how='left')

    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df[num_cols].interpolate(limit=1)
    df.sort_values('datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


if __name__ == '__main__':
    import pytz
    from datetime import datetime
    tz = pytz.timezone('America/Vancouver')
    print(f'DB update started at {datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")}')
    update_db()
    print(f'DB update complete at {datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")}')
