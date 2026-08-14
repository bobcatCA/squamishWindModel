"""Build the hourly and daily training datasets from raw station CSVs and SWS wind data.

    python build_dataset.py                  # build both
    python build_dataset.py --mode hourly
    python build_dataset.py --mode daily

Column naming convention
    squamishSpeed / squamishGust / squamishLull / squamishDirection / squamishDegC
        — measurements from the SWS sensor at the Squamish spit
    {station}DegC / {station}KPa / {station}Hum / {station}Sky
        — measurements from the corresponding EC weather station

build_hourly() is the ONLY place raw web_data/ files are read. build_daily()
derives entirely from training_data/hourly_database.csv — it just keeps each
day's DAILY_HOUR (2pm) row, already merged/cleaned by build_hourly() — so it
requires build_hourly() to have already run, and any data-quality issue only
ever needs tracking down in one place.
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


# ── Shared build logic ────────────────────────────────────────────────────────

def _load_ec_hourly() -> pd.DataFrame:
    """Load and concatenate all EC stations onto a shared hourly DatetimeIndex."""
    ec = {name: _load_station(f, sky, name) for name, (f, sky) in STATIONS.items()}
    return pd.concat(ec.values(), axis=1).sort_index()


def _merge_wind_and_save(df_ec: pd.DataFrame, ec_index: pd.DatetimeIndex, out_path: str,
                          tolerance: pd.Timedelta = SWS_MERGE_TOLERANCE) -> pd.DataFrame:
    """Join EC rows at *ec_index* with SWS readings averaged within *tolerance*,
    drop rows with no genuine wind reading, normalize sky text, and save.
    """
    df_wind_raw = _load_wind('web_data/sws_wind_database.csv')
    df_wind = wind_to_hourly_index(df_wind_raw, ec_index, tolerance=tolerance)
    matched = df_wind['squamishSpeed'].notna().sum()
    tolerance_min = int(tolerance.total_seconds() // 60)
    print(f'SWS: {matched:,} of {len(ec_index):,} rows matched '
          f'(±{tolerance_min}min) from {len(df_wind_raw):,} raw readings')

    df = df_ec.loc[ec_index].join(df_wind, how='left').reset_index()

    # Drop rows with no genuine wind reading. A missing SWS match (sensor off —
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


# ── Hourly dataset ────────────────────────────────────────────────────────────

def build_hourly(out_path: str = 'training_data/hourly_database.csv') -> pd.DataFrame:
    df_ec = _load_ec_hourly()
    print(f'EC: {len(df_ec):,} hourly rows  ({df_ec.index.min()} → {df_ec.index.max()})')
    return _merge_wind_and_save(df_ec, df_ec.index, out_path)


# ── Daily dataset ─────────────────────────────────────────────────────────────

DAILY_HOUR = 14  # 2pm local — anchor timestamp for the daily dataset

# Below this speed there's no meaningful gust/lull structure to score (and the
# ratio blows up as speed -> 0), so speed_steadiness is fixed at 0 ("not
# sailable") rather than evaluated.
STEADINESS_MIN_SPEED = 15

# speed_steadiness calibration: p10/p90 of (gust-lull)/speed averaged over each
# day's sailable hours, from the hourly dataset. Fixed constants (not
# recomputed per build) so the score's meaning stays stable over time rather
# than drifting as more data accumulates.
SPEED_STEADINESS_LOW  = 0.2394  # gustiness at this level or steadier -> 5
SPEED_STEADINESS_HIGH = 0.5340  # gustiness at this level or gustier  -> 0

# direction_steadiness needs >=2 sailable hours to measure variation at all —
# a single reading trivially has zero spread, which would misleadingly score
# as perfectly steady.
DIRECTION_MIN_SAILABLE_HOURS = 2

# direction_steadiness calibration: p10/p90 of circular variance (1 - mean
# resultant length) of squamishDirection over each day's sailable hours
# (with >=2 of them), from the hourly dataset.
DIRECTION_STEADINESS_LOW  = 0.0002  # circular variance at this level or steadier -> 5
DIRECTION_STEADINESS_HIGH = 0.0264  # circular variance at this level or scattered -> 0


def _to_5_score(value: pd.Series, low: float, high: float) -> pd.Series:
    """Linearly map *value* from [low, high] onto [5, 0] (higher value = worse), clipped."""
    return np.clip(5 - 5 * (value - low) / (high - low), 0, 5)


def _sailable_hours(hourly_path: str) -> pd.DataFrame:
    """Hourly rows where squamishSpeed > STEADINESS_MIN_SPEED, with a 'date' column."""
    hourly = pd.read_csv(hourly_path)
    hourly['datetime'] = pd.to_datetime(hourly['datetime'], utc=True).dt.tz_convert('America/Vancouver')
    hourly['date'] = hourly['datetime'].dt.date
    return hourly[hourly['squamishSpeed'] > STEADINESS_MIN_SPEED].copy()


def _daily_score(df: pd.DataFrame, daily_value: pd.Series, low: float, high: float, col: str) -> pd.DataFrame:
    """Map a per-date raw value onto [0, 5] via _to_5_score and join onto *df* by date."""
    df = df.copy()
    date = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert('America/Vancouver').dt.date
    df[col] = _to_5_score(date.map(daily_value), low, high).fillna(0)
    return df


def add_speed_steadiness(df: pd.DataFrame, hourly_path: str = 'training_data/hourly_database.csv') -> pd.DataFrame:
    """Score 0-5 (5=steadiest) from the gust/lull spread relative to speed:
    (squamishGust - squamishLull) / squamishSpeed, averaged over each day's
    sailable hours (squamishSpeed > STEADINESS_MIN_SPEED) from the hourly
    dataset — NOT just the daily row's own 2pm reading. Days with no sailable
    hours score 0.
    """
    sailable = _sailable_hours(hourly_path)
    sailable['gustiness'] = (sailable['squamishGust'] - sailable['squamishLull']) / sailable['squamishSpeed']
    daily_gustiness = sailable.groupby('date')['gustiness'].mean()
    return _daily_score(df, daily_gustiness, SPEED_STEADINESS_LOW, SPEED_STEADINESS_HIGH, 'speed_steadiness')


def add_direction_steadiness(df: pd.DataFrame, hourly_path: str = 'training_data/hourly_database.csv') -> pd.DataFrame:
    """Score 0-5 (5=steadiest) from how much squamishDirection varies over each
    day's sailable hours (squamishSpeed > STEADINESS_MIN_SPEED), using CIRCULAR
    variance (1 - mean resultant length) rather than a plain std — direction
    wraps at 360°, so e.g. 350° and 10° are 20° apart, not ~340°. Days with
    fewer than DIRECTION_MIN_SAILABLE_HOURS sailable hours score 0 (a single
    reading has no measurable spread).
    """
    sailable = _sailable_hours(hourly_path)

    def circular_variance(deg: pd.Series) -> float:
        rad = np.radians(deg)
        r = np.sqrt(np.mean(np.cos(rad)) ** 2 + np.mean(np.sin(rad)) ** 2)
        return 1 - r

    counts = sailable.groupby('date').size()
    enough = counts[counts >= DIRECTION_MIN_SAILABLE_HOURS].index
    daily_variance = (sailable[sailable['date'].isin(enough)]
                       .groupby('date')['squamishDirection']
                       .apply(circular_variance))
    return _daily_score(df, daily_variance, DIRECTION_STEADINESS_LOW, DIRECTION_STEADINESS_HIGH, 'direction_steadiness')


def build_daily(out_path: str = 'training_data/daily_database.csv',
                 hourly_path: str = 'training_data/hourly_database.csv') -> pd.DataFrame:
    """Build the daily dataset entirely FROM the hourly one — no raw web_data/
    reads here. build_hourly() is the single point where raw EC/SWS data enters
    the pipeline; every other dataset derives from its output, so a bad
    reading only needs tracking down in one place.
    """
    hourly = pd.read_csv(hourly_path)
    hourly['datetime'] = pd.to_datetime(hourly['datetime'], utc=True).dt.tz_convert(TZ)

    df = hourly[hourly['datetime'].dt.hour == DAILY_HOUR].copy()
    print(f'Hourly: {len(df):,} of {len(hourly):,} rows at {DAILY_HOUR}:00 '
          f'({df["datetime"].min()} → {df["datetime"].max()})')

    df = add_speed_steadiness(df, hourly_path)
    df = add_direction_steadiness(df, hourly_path)
    df = df.sort_values('datetime')[sorted(df.columns)]
    df.to_csv(out_path, index=False)
    print(f'Saved {out_path}  {len(df):,} rows  '
          f'{df["datetime"].min()} → {df["datetime"].max()}')
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Build training datasets')
    p.add_argument('--mode', choices=['hourly', 'daily', 'both'], default='both',
                   help='Which dataset to build (default: both)')
    args = p.parse_args()

    if args.mode in ('hourly', 'both'):
        build_hourly()
    if args.mode in ('daily', 'both'):
        build_daily()
