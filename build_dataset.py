"""Build hourly and daily training datasets from raw station CSVs and SWS wind data.

    python build_dataset.py             # build both hourly and daily CSVs
    python build_dataset.py --mode hourly
    python build_dataset.py --mode daily  (reads hourly_database.csv)
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

STATIONS = {
    'vancouver': ('Vancouver.csv', True),
    'whistler':  ('Whistler.csv',  True),
    'comox':     ('Comox.csv',     True),
    'victoria':  ('Victoria.csv',  True),
    'pemberton': ('Pemberton.csv', False),
    'lillooet':  ('Lillooet.csv',  False),
    'pam':       ('Pam.csv',       False),
    'ballenas':  ('Ballenas.csv',  False),
}


# ── Hourly dataset ────────────────────────────────────────────────────────────

def _load_station(filename: str, has_sky: bool) -> pd.DataFrame:
    cols = [TIME_COL, TEMP_COL, PRESS_COL, HUM_COL] + ([SKY_COL] if has_sky else [])
    df = pd.read_csv(filename, usecols=cols)
    df[HUM_COL] = pd.to_numeric(df[HUM_COL], errors='coerce')
    return df.set_index(TIME_COL)


def _load_wind(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename, index_col=0)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert(TZ)
    df = df.sort_values('datetime').drop_duplicates(subset='datetime').set_index('datetime')
    df['dir_sin'] = np.sin(np.radians(df['direction']))
    df['dir_cos'] = np.cos(np.radians(df['direction']))
    scalar_cols = [c for c in ('speed', 'gust', 'lull', 'temperature') if c in df.columns]
    hourly = df[scalar_cols + ['dir_sin', 'dir_cos']].resample('h').mean()
    hourly['direction'] = np.degrees(np.arctan2(hourly['dir_sin'], hourly['dir_cos'])) % 360
    return hourly[scalar_cols + ['direction']].dropna(how='all')


def build_hourly(out_path: str = 'hourly_database.csv') -> pd.DataFrame:
    stations = {name: _load_station(f, has_sky) for name, (f, has_sky) in STATIONS.items()}

    base = stations['vancouver']
    df = base[[TEMP_COL, PRESS_COL, HUM_COL, SKY_COL]].rename(columns={
        TEMP_COL: 'vancouverDegC', PRESS_COL: 'vancouverKPa',
        HUM_COL: 'vancouverHum',  SKY_COL: 'vancouverSky',
    }).reset_index().rename(columns={TIME_COL: 'datetime'})

    for name, (_, has_sky) in STATIONS.items():
        if name == 'vancouver':
            continue
        rename = {TEMP_COL: f'{name}DegC', PRESS_COL: f'{name}KPa', HUM_COL: f'{name}Hum'}
        if has_sky:
            rename[SKY_COL] = f'{name}Sky'
        df = df.merge(
            stations[name][list(rename)].rename(columns=rename),
            how='left', left_on='datetime', right_index=True,
        )

    df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(
        TZ, nonexistent='shift_forward', ambiguous='NaT',
    )
    df = df.set_index('datetime')
    df = df[~df.index.isna() & ~df.index.duplicated(keep='first')]

    df_wind = _load_wind('sws_wind_database.csv')
    print(f'SWS hourly rows: {len(df_wind):,}  ({df_wind.index.min()} → {df_wind.index.max()})')

    sky_cols = [c for c in df.columns if c.endswith('Sky')]
    df = df_wind.join(df, how='left').reset_index()
    df = df.rename(columns={'index': 'datetime'})
    df[['direction', 'gust', 'lull', 'speed']] = (
        df[['direction', 'gust', 'lull', 'speed']].fillna(0)
    )

    df['hour']         = df['datetime'].dt.hour
    df['date']         = df['datetime'].dt.date
    df['month']        = df['datetime'].dt.month
    df['day']          = df['datetime'].dt.day
    df['sin_hour']     = np.sin(2 * np.pi * df['hour'] / 24)
    df['year_fraction'] = (df['month'] * 30.416 + df['day']) / 365
    df['gust_relative']  = (df['gust'] / df['speed']).replace([np.inf, -np.inf], 3).fillna(0).clip(1, 3)
    df['lull_relative']  = (df['lull'] / df['speed']).replace([np.inf, -np.inf], 0).fillna(0).clip(0, 1)
    df['gustLull_index'] = (df['gust_relative'] - 1) + (1 - df['lull_relative'])

    for col in sky_cols:
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


def build_daily(hourly_path: str = 'hourly_database.csv',
                out_path: str = 'daily_database.csv') -> pd.DataFrame:
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
