"""Build the hourly training dataset from raw station CSVs and SWS wind data.

    python build_dataset.py

Column naming convention
    squamishSpeed / squamishGust / squamishLull / squamishDirection / squamishDegC
        — measurements from the SWS sensor at the Squamish spit
    {station}DegC / {station}KPa / {station}Hum / {station}Sky
        — measurements from the corresponding EC weather station
"""

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

    for col in df.columns:
        if col.endswith('Sky'):
            df[col] = normalize_sky_series(df[col])

    df = df.sort_values('datetime')[sorted(df.columns)]
    df.to_csv(out_path, index=False)
    print(f'Saved {out_path}  {len(df):,} rows  '
          f'{df["datetime"].min()} → {df["datetime"].max()}')
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    build_hourly()
