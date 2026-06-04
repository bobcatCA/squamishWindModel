import numpy as np
import os
import pandas as pd
import sqlite3
from datetime import timedelta
from dotenv import load_dotenv
from pathlib import Path

from ec_scrape import (
    normalize_sky_series,
    pull_forecast_daily,
    pull_forecast_hourly,
    pull_past_hrs_weather,
)
from score_daily import add_scores_to_df
from sws_pull import get_sws_df

load_dotenv()


def _db_path() -> Path:
    return Path(os.getenv('WORKING_DIRECTORY')) / 'weather_data_hourly.db'


def _normalize_sky_cols(df: pd.DataFrame) -> pd.DataFrame:
    sky_cols = df.columns[df.columns.str.contains('Sky')]
    for col in sky_cols:
        df[col] = normalize_sky_series(df[col])
    return df


def update_sql_db_hourly(df: pd.DataFrame) -> None:
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
        ', '.join(df.columns),
        ', '.join(['?'] * len(df.columns)),
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
    """Pull latest EC and SWS data and insert new rows into the SQLite DB."""
    df_recent_weather = pull_past_hrs_weather()
    past_dates = list(df_recent_weather['datetime'].dt.date.unique().astype(str))
    try:
        df_sws = get_sws_df(past_dates)
    except Exception as e:
        print(f'Warning: SWS pull failed ({e}), continuing without wind data')
        df_sws = pd.DataFrame()
    if not df_sws.empty:
        df_recent_weather['datetime'] = df_recent_weather['datetime'].dt.as_unit('us')
        df_sws['datetime'] = df_sws['datetime'].dt.as_unit('us')
        df_recent = pd.merge_asof(df_recent_weather, df_sws, on='datetime', direction='nearest')
    else:
        df_recent = df_recent_weather
    update_sql_db_hourly(df_recent)


def get_conditions_table_daily(encoder_length: int = 8,
                               prediction_length: int = 5) -> pd.DataFrame:
    update_db()
    today_14 = pd.Timestamp.now(tz='America/Vancouver').normalize() + pd.Timedelta(hours=14)
    start = today_14 - timedelta(days=encoder_length)
    end = today_14 + timedelta(days=prediction_length - 1)
    time_index = pd.date_range(start=start, end=end, freq='d')

    with sqlite3.connect(_db_path()) as conn:
        df_hist = pd.read_sql_query(
            'SELECT * FROM weather WHERE datetime > ?',
            conn,
            params=(start.timestamp(),),
        )
    df_hist['datetime'] = (
        pd.to_datetime(df_hist['datetime'], unit='s', utc=True)
        .dt.tz_convert('America/Vancouver')
    )

    df_scored = add_scores_to_df(df_hist)
    df_forecast = pull_forecast_daily(time_index)
    df_out = pd.concat([df_scored, df_forecast], ignore_index=True)

    df_out['year_fraction'] = (df_out['datetime'].dt.month * 30.416 + df_out['datetime'].dt.day) / 365
    df_out = _normalize_sky_cols(df_out)
    df_out.sort_values('datetime', inplace=True)
    df_out.dropna(thresh=14, inplace=True)
    df_out.reset_index(drop=True, inplace=True)
    return df_out


def get_conditions_table_hourly(encoder_length: int = 12,
                                prediction_length: int = 8,
                                specified_start: pd.Timestamp = None) -> pd.DataFrame:
    update_db()

    now = specified_start or pd.Timestamp.now(tz='America/Vancouver').ceil('h')
    start = now - timedelta(hours=encoder_length)
    end = now + timedelta(hours=prediction_length - 1)
    time_index = pd.date_range(start=start, end=end, freq='h')

    with sqlite3.connect(_db_path()) as conn:
        df_hist = pd.read_sql_query(
            'SELECT * FROM weather WHERE datetime > ?',
            conn,
            params=(start.timestamp(),),
        )
    df_hist['datetime'] = (
        pd.to_datetime(df_hist['datetime'], unit='s', utc=True)
        .dt.tz_convert('America/Vancouver')
    )

    df_forecast = pull_forecast_hourly()
    df_data = pd.concat([df_forecast, df_hist], ignore_index=True, sort=False)
    df_data.sort_values('datetime', inplace=True)
    df_data.reset_index(drop=True, inplace=True)

    df = pd.DataFrame({'datetime': time_index})
    df = df.merge(df_data, on='datetime', how='left')
    df.sort_values('datetime', inplace=True)

    df['sin_hour'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
    df['year_fraction'] = (df['datetime'].dt.month * 30.416 + df['datetime'].dt.day) / 365
    df = _normalize_sky_cols(df)

    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df[num_cols].interpolate(limit=1)
    df.dropna(thresh=14, inplace=True)
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
