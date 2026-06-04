"""Build hourly_database.csv from EC weather station CSVs and SWS wind data.

Weather stations are recorded hourly; SWS wind data is resampled from its
native ~3-minute resolution to hourly means. A left join on the SWS index
keeps only hours that have wind observations.

Run from the project directory:
    python build_dataset.py
"""

import numpy as np
import pandas as pd

from ec_scrape import normalize_sky_series

TIME_COL = 'Date/Time (LST)'
TEMP_COL = 'Temp (°C)'
PRESS_COL = 'Stn Press (kPa)'
HUM_COL = 'Rel Hum (%)'
SKY_COL = 'Weather'
TZ = 'America/Vancouver'

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


def load_station(filename: str, has_sky: bool) -> pd.DataFrame:
    cols = [TIME_COL, TEMP_COL, PRESS_COL, HUM_COL] + ([SKY_COL] if has_sky else [])
    df = pd.read_csv(filename, usecols=cols)
    df[HUM_COL] = pd.to_numeric(df[HUM_COL], errors='coerce')
    return df.set_index(TIME_COL)


def load_wind(filename: str) -> pd.DataFrame:
    """Load SWS wind data and resample to hourly means."""
    df = pd.read_csv(filename, index_col=0)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert(TZ)
    df = df.sort_values('datetime').drop_duplicates(subset='datetime').set_index('datetime')

    df['dir_sin'] = np.sin(np.radians(df['direction']))
    df['dir_cos'] = np.cos(np.radians(df['direction']))
    scalar_cols = [c for c in ('speed', 'gust', 'lull', 'temperature') if c in df.columns]
    hourly = df[scalar_cols + ['dir_sin', 'dir_cos']].resample('h').mean()
    hourly['direction'] = np.degrees(np.arctan2(hourly['dir_sin'], hourly['dir_cos'])) % 360
    return hourly[scalar_cols + ['direction']].dropna(how='all')


if __name__ == '__main__':
    # ── Weather stations ─────────────────────────────────────────────────────
    stations = {name: load_station(f, has_sky) for name, (f, has_sky) in STATIONS.items()}

    base = stations['vancouver']
    df = base[[TEMP_COL, PRESS_COL, HUM_COL, SKY_COL]].rename(columns={
        TEMP_COL: 'vancouverDegC', PRESS_COL: 'vancouverKPa',
        HUM_COL: 'vancouverHum', SKY_COL: 'vancouverSky',
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

    # ── Merge with hourly-resampled SWS wind ─────────────────────────────────
    df_wind = load_wind('sws_wind_database.csv')
    print(f'SWS hourly rows: {len(df_wind):,}  ({df_wind.index.min()} → {df_wind.index.max()})')

    sky_cols = [c for c in df.columns if c.endswith('Sky')]
    df = df_wind.join(df, how='left').reset_index()
    df = df.rename(columns={'index': 'datetime'})

    df[['direction', 'gust', 'lull', 'speed']] = (
        df[['direction', 'gust', 'lull', 'speed']].fillna(0)
    )

    # ── Derived features ──────────────────────────────────────────────────────
    df['hour'] = df['datetime'].dt.hour
    df['date'] = df['datetime'].dt.date
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['year_fraction'] = (df['month'] * 30.416 + df['day']) / 365
    df['gust_relative'] = (df['gust'] / df['speed']).replace([np.inf, -np.inf], 3).fillna(0).clip(1, 3)
    df['lull_relative'] = (df['lull'] / df['speed']).replace([np.inf, -np.inf], 0).fillna(0).clip(0, 1)
    df['gustLull_index'] = (df['gust_relative'] - 1) + (1 - df['lull_relative'])

    for col in sky_cols:
        df[col] = normalize_sky_series(df[col])

    # ── Save ──────────────────────────────────────────────────────────────────
    df = df.sort_values('datetime')[sorted(df.columns)]
    df.to_csv('hourly_database.csv', index=False)
    print(f'Saved hourly_database.csv  {len(df):,} rows  '
          f'{df["datetime"].min()} → {df["datetime"].max()}')
