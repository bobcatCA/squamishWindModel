"""Build hourly and daily training datasets from raw station CSVs and SWS wind data.

    python build_dataset.py             # build both hourly and daily CSVs
    python build_dataset.py --mode hourly
    python build_dataset.py --mode daily  (reads hourly_database.csv)

Column naming convention
    squamishSpeed / squamishGust / squamishLull / squamishDirection / squamishDegC
        — measurements from the SWS sensor at the Squamish spit
    {station}DegC / {station}KPa / {station}Hum / {station}Sky
        — measurements from the corresponding EC weather station
"""

import argparse
import numpy as np
import pandas as pd

from ec_scrape import normalize_sky_series

TIME_COL  = 'Date/Time (LST)'
TEMP_COL  = 'Temp (°C)'
PRESS_COL = 'Stn Press (kPa)'
HUM_COL   = 'Rel Hum (%)'
SKY_COL   = 'Weather'
TZ        = 'America/Vancouver'

# (csv filename, has sky-condition column)
STATIONS = {
    'vancouver': ('web_data/Vancouver.csv', True),
    'whistler':  ('web_data/Whistler.csv',  True),
    'comox':     ('web_data/Comox.csv',     True),
    'victoria':  ('web_data/Victoria.csv',  True),
    'pemberton': ('web_data/Pemberton.csv', False),
    'lillooet':  ('web_data/Lillooet.csv',  False),
    'pam':       ('web_data/Pam.csv',       False),
    'ballenas':  ('web_data/Ballenas.csv',  False),
}


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_station(filename: str, has_sky: bool, name: str) -> pd.DataFrame:
    """Load one EC station CSV → tz-aware DatetimeIndex, prefixed column names."""
    cols = [TIME_COL, TEMP_COL, PRESS_COL, HUM_COL] + ([SKY_COL] if has_sky else [])
    df = pd.read_csv(filename, usecols=cols)
    df[HUM_COL] = pd.to_numeric(df[HUM_COL], errors='coerce')

    idx = pd.to_datetime(df[TIME_COL]).dt.tz_localize(
        TZ, nonexistent='shift_forward', ambiguous='NaT'
    )
    df = df.drop(columns=[TIME_COL])
    df.index = idx
    df.index.name = 'datetime'
    df = df[~df.index.isna() & ~df.index.duplicated(keep='first')]

    rename = {TEMP_COL: f'{name}DegC', PRESS_COL: f'{name}KPa', HUM_COL: f'{name}Hum'}
    if has_sky:
        rename[SKY_COL] = f'{name}Sky'
    return df.rename(columns=rename)


def _load_wind(filename: str) -> pd.DataFrame:
    """Load SWS CSV → resample to hourly → squamish-prefixed column names."""
    df = pd.read_csv(filename)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert(TZ)
    df = df.sort_values('datetime').drop_duplicates(subset='datetime').set_index('datetime')

    # Sensor sometimes reports temperature in tenths of a degree (e.g. 150 → 15.0°C).
    # Divide by 10 until all values are within a plausible outdoor range (±50°C).
    if 'temperature' in df.columns:
        df['temperature'] = df['temperature'].astype(float)
        while (df['temperature'].abs() > 50).any():
            df.loc[df['temperature'].abs() > 50, 'temperature'] /= 10

    # Average direction via unit-vector decomposition to avoid wrap-around artefacts
    df['dir_sin'] = np.sin(np.radians(df['direction']))
    df['dir_cos'] = np.cos(np.radians(df['direction']))
    scalar_cols = [c for c in ('speed', 'gust', 'lull', 'temperature') if c in df.columns]
    hourly = df[scalar_cols + ['dir_sin', 'dir_cos']].resample('h').mean()
    hourly['direction'] = np.degrees(np.arctan2(hourly['dir_sin'], hourly['dir_cos'])) % 360
    hourly = hourly[scalar_cols + ['direction']].dropna(how='all')

    return hourly.rename(columns={
        'speed':       'squamishSpeed',
        'gust':        'squamishGust',
        'lull':        'squamishLull',
        'direction':   'squamishDirection',
        'temperature': 'squamishDegC',
    })


# ── Hourly dataset ────────────────────────────────────────────────────────────

def build_hourly(out_path: str = 'training_data/hourly_database.csv') -> pd.DataFrame:
    df_wind = _load_wind('web_data/sws_wind_database.csv')
    print(f'SWS: {len(df_wind):,} hourly rows  ({df_wind.index.min()} → {df_wind.index.max()})')

    ec = {name: _load_station(f, sky, name) for name, (f, sky) in STATIONS.items()}
    df_ec = pd.concat(ec.values(), axis=1)

    df = df_wind.join(df_ec, how='left').reset_index()

    # Fill zero for missing SWS readings (calm / sensor down)
    wind_cols = ['squamishSpeed', 'squamishGust', 'squamishLull', 'squamishDirection']
    df[wind_cols] = df[wind_cols].fillna(0)

    # Temporal features
    h = df['datetime'].dt.hour
    df['sin_hour']     = np.sin(2 * np.pi * h / 24)
    df['wind_hour']    = np.where(
        (h >= 10) & (h <= 13), 0.5 * (1 - np.cos(np.pi * (h - 10) / 3)),
        np.where((h > 13) & (h < 18), 0.5 * (1 + np.cos(np.pi * (h - 13) / 5)), 0.0)
    )
    df['year_fraction'] = (df['datetime'].dt.month * 30.416 + df['datetime'].dt.day) / 365

    for col in df.columns:
        if col.endswith('Sky'):
            df[col] = normalize_sky_series(df[col])

    df = df.sort_values('datetime')[sorted(df.columns)]
    df.to_csv(out_path, index=False)
    print(f'Saved {out_path}  {len(df):,} rows  '
          f'{df["datetime"].min()} → {df["datetime"].max()}')
    return df


# ── Daily dataset (scoring) ───────────────────────────────────────────────────

def _to_5_score(value: pd.Series, low: float, high: float) -> pd.Series:
    """Linearly map *value* from [low, high] onto [1, 5], clipped."""
    return np.clip(5 - 4 * (value - low) / (high - low), 1, 5)


def add_scores_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily quality scores from hourly observations.

    Adds direction_score, speed_score, hours_above_20. Returns one row per day
    at 14:00 local time.
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert('America/Vancouver')
    df['date'] = df['datetime'].dt.date

    sailing = df[df['squamishSpeed'] > 15].copy()
    sailing['gust_relative']   = sailing['squamishGust'] / sailing['squamishSpeed']
    sailing['lull_relative']   = sailing['squamishLull'] / sailing['squamishSpeed']
    sailing['gust_lull_index'] = (sailing['gust_relative'] - 1) + (1 - sailing['lull_relative'])

    dir_stdev = (
        sailing.groupby('date')['squamishDirection']
        .std()
        .reset_index(name='dir_stdev')
    )
    dir_stdev['direction_score'] = _to_5_score(dir_stdev['dir_stdev'], low=0.8, high=18)

    speed_score = (
        sailing.groupby('date')['gust_lull_index']
        .mean()
        .apply(_to_5_score, low=0.15, high=0.75)
        .reset_index(name='speed_score')
    )

    hours_above_20 = (
        df.assign(above_20=df['squamishSpeed'] > 20)
        .groupby('date', as_index=False)
        .agg(hours_above_20=('above_20', 'sum'))
    )
    hours_above_20['hours_above_20'] -= 1

    daily = pd.DataFrame({'date': df['date'].unique()})
    daily = (daily
             .merge(dir_stdev[['date', 'direction_score']], on='date', how='left')
             .merge(speed_score, on='date', how='left')
             .merge(hours_above_20, on='date', how='left'))

    daily['date'] = (
        pd.to_datetime(daily['date']) + pd.to_timedelta(14, 'hours')
    ).dt.tz_localize('America/Vancouver')
    daily = daily.rename(columns={'date': 'datetime'})

    result = daily.merge(df, on='datetime', how='left')
    result.drop(columns='dir_stdev', inplace=True, errors='ignore')
    result.fillna({'direction_score': 0, 'speed_score': 0}, inplace=True)
    result.sort_values('datetime', inplace=True)
    return result


def build_daily(hourly_path: str = 'training_data/hourly_database.csv',
                out_path: str = 'training_data/daily_database.csv') -> pd.DataFrame:
    data = pd.read_csv(hourly_path)
    data = add_scores_to_df(data)
    data.to_csv(out_path, index=False)
    print(f'Saved {out_path}  {len(data):,} rows')
    return data


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Build TFT training datasets')
    p.add_argument('--mode', choices=['hourly', 'daily', 'both'], default='both',
                   help='Which dataset to build (default: both)')
    args = p.parse_args()

    if args.mode in ('hourly', 'both'):
        build_hourly()
    if args.mode in ('daily', 'both'):
        build_daily()
