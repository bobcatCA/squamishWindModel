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
    """Load one EC station CSV → tz-aware DatetimeIndex, prefixed column names.

    'Date/Time (LST)' is Environment Canada's *Local Standard Time* — a fixed
    UTC-8 offset that does NOT observe daylight saving (verified: EC's raw CSVs
    run continuously through both DST transitions, with no repeated/skipped
    hour). Localizing it directly to the DST-aware 'America/Vancouver' zone
    would shift every summer timestamp by an hour, so first anchor it to UTC
    at the fixed +8h offset, then convert to local wall-clock time.
    """
    cols = [TIME_COL, TEMP_COL, PRESS_COL, HUM_COL] + ([SKY_COL] if has_sky else [])
    df = pd.read_csv(filename, usecols=cols)
    df[HUM_COL] = pd.to_numeric(df[HUM_COL], errors='coerce')

    idx = (pd.to_datetime(df[TIME_COL]) + pd.Timedelta(hours=8)).dt.tz_localize('UTC').dt.tz_convert(TZ)
    df = df.drop(columns=[TIME_COL])
    df.index = idx
    df.index.name = 'datetime'
    df = df[~df.index.isna() & ~df.index.duplicated(keep='first')]

    rename = {TEMP_COL: f'{name}DegC', PRESS_COL: f'{name}KPa', HUM_COL: f'{name}Hum'}
    if has_sky:
        rename[SKY_COL] = f'{name}Sky'
    return df.rename(columns=rename)


SWS_MERGE_TOLERANCE = pd.Timedelta('10min')


def _load_wind(filename: str) -> pd.DataFrame:
    """Load SWS CSV → tz-aware DatetimeIndex, squamish-prefixed column names (raw, un-aggregated)."""
    df = pd.read_csv(filename)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert(TZ)
    df = df.sort_values('datetime').drop_duplicates(subset='datetime').set_index('datetime')

    # Sensor sometimes reports temperature in tenths of a degree (e.g. 150 → 15.0°C).
    # Divide by 10 until all values are within a plausible outdoor range (±50°C).
    if 'temperature' in df.columns:
        df['temperature'] = df['temperature'].astype(float)
        while (df['temperature'].abs() > 50).any():
            df.loc[df['temperature'].abs() > 50, 'temperature'] /= 10

    return df.rename(columns={
        'speed':       'squamishSpeed',
        'gust':        'squamishGust',
        'lull':        'squamishLull',
        'direction':   'squamishDirection',
        'temperature': 'squamishDegC',
    })


def wind_to_hourly_index(df_wind: pd.DataFrame, hourly_index: pd.DatetimeIndex,
                          tolerance: pd.Timedelta = SWS_MERGE_TOLERANCE) -> pd.DataFrame:
    """Aggregate raw SWS readings onto *hourly_index*.

    Every reading is matched to the nearest timestamp in *hourly_index*
    (dropped if none falls within *tolerance*), then all readings matched to
    the same hour are averaged. Comparisons run on tz-aware timestamps
    directly, so DST transitions can't shift a reading into the wrong bin.
    """
    hours = pd.DataFrame({'datetime': pd.DatetimeIndex(hourly_index).sort_values().unique()})
    hours['hour'] = hours['datetime']

    wind = df_wind.sort_index().reset_index()
    matched = pd.merge_asof(wind, hours, on='datetime', direction='nearest', tolerance=tolerance)
    matched = matched.dropna(subset=['hour'])

    # Average direction via unit-vector decomposition to avoid wrap-around artefacts
    matched['dir_sin'] = np.sin(np.radians(matched['squamishDirection']))
    matched['dir_cos'] = np.cos(np.radians(matched['squamishDirection']))
    scalar_cols = [c for c in ('squamishSpeed', 'squamishGust', 'squamishLull', 'squamishDegC')
                   if c in matched.columns]

    grouped = matched.groupby('hour')[scalar_cols + ['dir_sin', 'dir_cos']].mean()
    grouped['squamishDirection'] = np.degrees(np.arctan2(grouped['dir_sin'], grouped['dir_cos'])) % 360
    grouped = grouped[scalar_cols + ['squamishDirection']]
    grouped.index.name = 'datetime'
    return grouped


# ── Hourly dataset ────────────────────────────────────────────────────────────

def build_hourly(out_path: str = 'training_data/hourly_database.csv') -> pd.DataFrame:
    ec = {name: _load_station(f, sky, name) for name, (f, sky) in STATIONS.items()}
    df_ec = pd.concat(ec.values(), axis=1).sort_index()
    print(f'EC: {len(df_ec):,} hourly rows  ({df_ec.index.min()} → {df_ec.index.max()})')

    df_wind_raw = _load_wind('web_data/sws_wind_database.csv')
    df_wind = wind_to_hourly_index(df_wind_raw, df_ec.index)
    matched = df_wind['squamishSpeed'].notna().sum()
    tolerance_min = int(SWS_MERGE_TOLERANCE.total_seconds() // 60)
    print(f'SWS: {matched:,} of {len(df_ec):,} EC hours matched '
          f'(±{tolerance_min}min) from {len(df_wind_raw):,} raw readings')

    df = df_ec.join(df_wind, how='left').reset_index()

    # Drop hours with no genuine wind reading. A missing SWS match (sensor off —
    # mainly outside the May-Sept season) and a reported 0 kt reading are both
    # stored as 0 and are indistinguishable from each other, so both are dropped
    # rather than anchoring the model on a large mass of spurious/off-season zeros.
    wind_cols = ['squamishSpeed', 'squamishGust', 'squamishLull', 'squamishDirection']
    before = len(df)
    df = df[df['squamishSpeed'].fillna(0) != 0].reset_index(drop=True)
    print(f'Dropped {before - len(df):,} of {before:,} rows (squamishSpeed missing or 0)')
    df[wind_cols] = df[wind_cols].fillna(0)

    # Temporal features
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
