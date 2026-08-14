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

from build_dataset import add_direction_steadiness, add_speed_steadiness, wind_to_hourly_index
from collect_data import migrate_schema
from ec_scrape import _FORECAST_DAILY_URLS, pull_forecast_daily, pull_past_hrs_weather
from sws_pull import get_sws_df

load_dotenv()

DAILY_HOUR = 14  # matches build_dataset.py / tft_daily_model.py — the 2pm daily snapshot


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
    # A `weather` table copied in from another deployment (different code
    # version — e.g. the Raspberry Pi) can be on an older schema at any time,
    # not just at first-time setup, so migrate on every call rather than only
    # via collect_data.py --init-db.
    with sqlite3.connect(_db_path()) as conn:
        migrate_schema(conn)
        conn.commit()

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


_DAILY_FORECAST_STATIONS = list(_FORECAST_DAILY_URLS)  # comox, lillooet, pemberton, vancouver, victoria, whistler
_PLAUSIBLE_DEGC_RANGE = (-40, 50)  # generous bound for a BC daily high — wide enough to never flag real
                                    # weather, tight enough to catch a scrape/parse failure that slips
                                    # through as a number instead of NaN (e.g. a mis-matched regex capture)


def _pull_daily_forecast(prediction_length: int = 5) -> pd.DataFrame:
    """Scrape EC's multi-day temp/sky forecast for the daily model's real_known
    stations. Pulled fresh, in-memory, on every call — never persisted, since a
    forecast is only ever useful to the run that requested it and EC's own page
    is re-scraped fresh next time anyway.

    Validates rather than trusting the scrape blindly:
      - a row-count mismatch (usually one station returning misaligned dates,
        which pull_forecast_daily's per-station inner join then applies to
        every station) discards the whole pull
      - a station missing from the response entirely is left absent
      - an individual missing/implausible temperature is blanked to NaN
    forecast.py no longer fills any gap this leaves — a blanked/missing value
    here means that real_known column reaches TimeSeriesDataSet with a NaN in
    it, which pytorch_forecasting refuses outright rather than silently
    predicting from a fabricated number. This is a real, permanent condition
    for pamDegC (no EC daily-forecast page exists for that station at all)
    until it's retrained out of real_known.
    """
    next_2pm = pd.Timestamp.now(tz='America/Vancouver').normalize() + timedelta(hours=DAILY_HOUR)
    if next_2pm <= pd.Timestamp.now(tz='America/Vancouver'):
        next_2pm += timedelta(days=1)
    time_index = pd.date_range(start=next_2pm, periods=prediction_length, freq='D')

    df = pull_forecast_daily(time_index)

    if len(df) != prediction_length:
        print(f'Warning: daily forecast scrape returned {len(df)}/{prediction_length} days '
              '(likely a station returning mismatched dates) — discarding this pull entirely')
        return pd.DataFrame(columns=['datetime'])

    missing = [s for s in _DAILY_FORECAST_STATIONS if f'{s}DegC' not in df.columns]
    if missing:
        print(f'Warning: daily forecast scrape has no data for {missing} — '
              'that real_known column will be NaN for the prediction window')

    for station in _DAILY_FORECAST_STATIONS:
        col = f'{station}DegC'
        if col not in df.columns:
            continue
        bad = df[col].isna() | ~df[col].between(*_PLAUSIBLE_DEGC_RANGE)
        if bad.any():
            print(f'Warning: daily forecast {col} has {bad.sum()}/{len(df)} missing/implausible value(s) '
                  '— blanking them to NaN rather than trusting them')
            df.loc[bad, col] = np.nan

    return df


def get_conditions_table_daily(encoder_length: int = 5, prediction_length: int = 5,
                                specified_start: pd.Timestamp = None,
                                update: bool = True) -> pd.DataFrame:
    if update:
        update_db()

    now = specified_start or pd.Timestamp.now(tz='America/Vancouver').normalize() + timedelta(hours=DAILY_HOUR)
    if now <= pd.Timestamp.now(tz='America/Vancouver'):
        now += timedelta(days=1)
    start = now - timedelta(days=encoder_length)
    end   = now + timedelta(days=prediction_length - 1)
    day_index = pd.date_range(start=start, end=end, freq='D')

    with sqlite3.connect(_db_path()) as conn:
        df_hist = pd.read_sql_query(
            # >= not >: pd.date_range(start=start, ...) below is inclusive of
            # start, so a strict > would silently drop a real row that lands
            # exactly on the window boundary (which happens routinely, since
            # `now`/`start` are anchored to actual observed timestamps).
            'SELECT * FROM weather WHERE datetime >= ?',
            conn, params=(start.timestamp(),),
        )
    df_hist['datetime'] = (
        pd.to_datetime(df_hist['datetime'], unit='s', utc=True)
        .dt.tz_convert('America/Vancouver')
    )
    df_hist = df_hist.rename(columns=_DB_COL_RENAME)
    df_hist_2pm = df_hist[df_hist['datetime'].dt.hour == DAILY_HOUR]

    df_fcst = _pull_daily_forecast(prediction_length)

    df = pd.DataFrame({'datetime': day_index})
    df = df.merge(df_hist_2pm, on='datetime', how='left')
    df = df.merge(df_fcst, on='datetime', how='left', suffixes=('', '_fcst'))
    for col in df_fcst.columns:
        fcst_col = f'{col}_fcst'
        if fcst_col in df.columns:
            df[col] = df[col].combine_first(df[fcst_col])
            df.drop(columns=[fcst_col], inplace=True)

    # speed_steadiness/direction_steadiness aren't stored anywhere — they're
    # derived from the FULL hourly (not just 2pm) squamish sensor history for
    # each day's sailable hours. Reusing build_dataset.py's exact formula
    # (rather than reimplementing it here) means live inference can't drift
    # from the definition the models were actually trained on — including its
    # 0-for-no-sailable-hours convention, which the trained models expect.
    df = add_speed_steadiness(df, hourly=df_hist)
    df = add_direction_steadiness(df, hourly=df_hist)

    # A column that's entirely NULL over the queried window reads back from
    # SQLite as object dtype (pandas has nothing to infer a numeric type
    # from), which TimeSeriesDataSet can't convert to a tensor. Force numeric
    # columns to float64 regardless of how much real data came back — this is
    # a dtype fix only, it doesn't change which values are NaN vs real.
    sky_cols = [c for c in df.columns if c.endswith('Sky')]
    num_cols = [c for c in df.columns if c != 'datetime' and c not in sky_cols]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

    df.sort_values('datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_conditions_table_hourly(encoder_length: int = 12, prediction_length: int = 8,
                                 specified_start: pd.Timestamp = None,
                                 update: bool = True) -> pd.DataFrame:
    if update:
        update_db()

    if specified_start is not None:
        now = specified_start
    else:
        wall_now = pd.Timestamp.now(tz='America/Vancouver').ceil('h')
        with sqlite3.connect(_db_path()) as conn:
            last_ts = conn.execute('SELECT MAX(datetime) FROM weather').fetchone()[0]
        # EC's live feed routinely lags wall-clock by an hour or more. Anchoring
        # to raw wall-clock time means that lag can push the *entire* encoder
        # window past the last real reading (worse the smaller encoder_length
        # is — hourly's is only 2h), leaving zero real data to predict from and
        # crashing downstream instead of degrading gracefully. Anchor to
        # whichever is earlier: the window still ends at "now" when data is
        # current, and simply starts from the latest data we actually have when
        # it isn't, rather than assuming EC is always caught up to the present.
        if last_ts is not None:
            last_observed = pd.to_datetime(last_ts, unit='s', utc=True).tz_convert('America/Vancouver')
            now = min(wall_now, last_observed)
        else:
            now = wall_now

    start = now - timedelta(hours=encoder_length)
    end   = now + timedelta(hours=prediction_length - 1)
    time_index = pd.date_range(start=start, end=end, freq='h')

    with sqlite3.connect(_db_path()) as conn:
        df_hist = pd.read_sql_query(
            # >= not >: pd.date_range(start=start, ...) below is inclusive of
            # start, so a strict > would silently drop a real row that lands
            # exactly on the window boundary (which happens routinely, since
            # `now`/`start` are anchored to actual observed timestamps).
            'SELECT * FROM weather WHERE datetime >= ?',
            conn, params=(start.timestamp(),),
        )
    df_hist['datetime'] = (
        pd.to_datetime(df_hist['datetime'], unit='s', utc=True)
        .dt.tz_convert('America/Vancouver')
    )

    df_hist.sort_values('datetime', inplace=True)

    df = pd.DataFrame({'datetime': time_index})
    df = df.merge(df_hist, on='datetime', how='left')

    # A column that's entirely NULL over the queried window reads back from
    # SQLite as object dtype, which `.interpolate()` can't handle (and
    # select_dtypes(include='number') would just silently skip it). Coerce
    # first so a fully-missing column still ends up as float NaN, not object.
    sky_cols = [c for c in df.columns if c.endswith('Sky')]
    num_cols = [c for c in df.columns if c != 'datetime' and c not in sky_cols]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
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
