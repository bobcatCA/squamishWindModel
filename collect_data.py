"""Download historical weather data and initialise the SQLite database.

Run for first-time setup or to refresh raw data sources:

    python collect_data.py              # EC station CSVs + SWS wind data + init DB
    python collect_data.py --ec-only    # only EC station CSVs
    python collect_data.py --sws-only   # only SWS wind data  (requires Selenium + Chrome)
    python collect_data.py --init-db    # only initialise the SQLite DB
"""

import argparse
import os
import sqlite3
import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from dateutil import rrule
from dotenv import load_dotenv

from sws_pull import get_sws_df, update_sws_csv

load_dotenv()

# ── EC climate bulk-download ──────────────────────────────────────────────────

_SESSION = requests.Session()
_SESSION.headers['User-Agent'] = 'Mozilla/5.0'

EC_STATIONS = {
    'Vancouver': 51442,
    'Whistler':  52178,
    'Pemberton': 536,
    'Lillooet':  27388,
    'Victoria':  51337,
    'Ballenas':  138,
    'Pam':       6817,
    'Comox':     155,
}

EC_START = 'Jan2016'


def _fetch_station_month(station_id: int, year: int, month: int,
                          retries: int = 3) -> pd.DataFrame:
    url = (
        f'https://climate.weather.gc.ca/climate_data/bulk_data_e.html?'
        f'format=csv&stationID={station_id}&Year={year}&Month={month}&Day=1&timeframe=1'
    )
    for attempt in range(retries):
        try:
            resp = _SESSION.get(url, timeout=30)
            resp.raise_for_status()
            if b'Date/Time' not in resp.content[:500]:
                raise ValueError(f'Unexpected response: {resp.content[:100]}')
            return pd.read_csv(StringIO(resp.content.decode('utf-8-sig')))
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f'    Warning: failed after {retries} attempts — {exc}')
    return pd.DataFrame()


def download_ec_history(end: str = None) -> None:
    """Download monthly EC hourly CSVs for all stations from EC_START to *end*."""
    start_dt = datetime.strptime(EC_START, '%b%Y')
    end_dt   = datetime.strptime(end, '%b%Y') if end else datetime.now().replace(day=1)
    months   = list(rrule.rrule(rrule.MONTHLY, dtstart=start_dt, until=end_dt))
    delay    = 0.4

    for name, station_id in EC_STATIONS.items():
        print(f'\n{name} (ID {station_id}) — {len(months)} months')
        frames = []
        for i, dt in enumerate(months):
            df = _fetch_station_month(station_id, dt.year, dt.month)
            if not df.empty:
                frames.append(df)
            if (i + 1) % 12 == 0:
                print(f'  {dt.year} done')
            time.sleep(delay)

        if not frames:
            print('  No data — skipping')
            continue

        out = pd.concat(frames, ignore_index=True)
        out['Date/Time (LST)'] = pd.to_datetime(out['Date/Time (LST)'])
        out['Temp (°C)'] = pd.to_numeric(out['Temp (°C)'], errors='coerce')
        out.to_csv(f'web_data/{name}.csv', index=False)
        print(f'  Saved web_data/{name}.csv  ({len(out):,} rows)')


def update_ec_history(end: str = None) -> None:
    """Append new EC data to each station CSV since its last recorded timestamp."""
    end_dt = datetime.strptime(end, '%b%Y') if end else datetime.now().replace(day=1)
    delay  = 0.4

    for name, station_id in EC_STATIONS.items():
        csv_path = Path(f'web_data/{name}.csv')
        if not csv_path.exists():
            print(f'\n{name}: {csv_path} not found — skipping (run --ec-only to build from scratch)')
            continue

        existing = pd.read_csv(csv_path)
        existing['Date/Time (LST)'] = pd.to_datetime(existing['Date/Time (LST)'])
        if existing.empty:
            print(f'\n{name}: CSV is empty — skipping')
            continue

        # Base start month on the last row with actual temperature data.
        valid_temps = existing.loc[existing['Temp (°C)'].notna(), 'Date/Time (LST)']
        last_dt  = valid_temps.max() if not valid_temps.empty else existing['Date/Time (LST)'].max()
        start_dt = last_dt.replace(day=1)

        # Scan the most recent 4 months for sparse/placeholder months (< 50% non-NaN
        # temperature) and strip those rows so they get cleanly re-fetched below.
        recent_start = start_dt - pd.DateOffset(months=3)
        recent_rows  = existing[existing['Date/Time (LST)'] >= recent_start].copy()
        recent_rows['_month'] = recent_rows['Date/Time (LST)'].dt.to_period('M')
        completeness = recent_rows.groupby('_month')['Temp (°C)'].apply(
            lambda x: x.notna().mean()
        )
        current_period = pd.Timestamp.now().to_period('M')
        sparse = completeness[
            (completeness < 0.5) & (completeness.index < current_period)
        ].index
        if not sparse.empty:
            first_sparse = sparse.min().to_timestamp()
            drop_mask = existing['Date/Time (LST)'].dt.to_period('M').isin(sparse)
            existing = existing[~drop_mask].reset_index(drop=True)
            start_dt = min(start_dt, first_sparse)
            print(f'  Stripped {int(drop_mask.sum()):,} placeholder rows from '
                  f'{[str(s) for s in sparse]} — will re-fetch')
        months    = list(rrule.rrule(rrule.MONTHLY, dtstart=start_dt, until=end_dt))

        print(f'\n{name} (ID {station_id}) — {len(months)} months ({start_dt.strftime("%b%Y")} → {end_dt.strftime("%b%Y")})')
        frames = []
        for i, dt in enumerate(months):
            df = _fetch_station_month(station_id, dt.year, dt.month)
            if not df.empty:
                frames.append(df)
            time.sleep(delay)

        if not frames:
            print('  No new data.')
            continue

        new_data = pd.concat(frames, ignore_index=True)
        new_data['Date/Time (LST)'] = pd.to_datetime(new_data['Date/Time (LST)'])
        new_data['Temp (°C)'] = pd.to_numeric(new_data['Temp (°C)'], errors='coerce')

        combined = (
            pd.concat([existing, new_data], ignore_index=True)
            .drop_duplicates(subset=['Date/Time (LST)'])
            .sort_values('Date/Time (LST)')
            .reset_index(drop=True)
        )
        combined.to_csv(csv_path, index=False)
        added = len(combined) - len(existing)
        print(f'  Added {added:,} rows → {csv_path} now has {len(combined):,} rows total')


# ── SWS historical download ───────────────────────────────────────────────────

SWS_START = '2016-05-01'


def download_sws_history(start: str = SWS_START, end: str = None) -> None:
    """Download all SWS wind readings from *start* to *end* → sws_wind_database.csv."""
    start_dt = datetime.strptime(start, '%Y-%m-%d')
    end_dt   = datetime.strptime(end, '%Y-%m-%d') if end else datetime.now()
    date_list = [
        (start_dt + timedelta(days=i)).strftime('%Y-%m-%d')
        for i in range((end_dt - start_dt).days + 1)
    ]
    print(f'Downloading SWS data for {len(date_list)} days ({start} → {end_dt.date()})')
    print('  (requires Selenium + Chrome — this may take a while)')
    df = get_sws_df(date_list)
    if df.empty:
        print('  No SWS data returned.')
        return
    df.to_csv('web_data/sws_wind_database.csv', index=False)
    print(f'  Saved web_data/sws_wind_database.csv  ({len(df):,} rows)')


# ── DB initialisation ─────────────────────────────────────────────────────────

_CREATE_WEATHER = """
CREATE TABLE IF NOT EXISTS weather (
    datetime      INTEGER PRIMARY KEY,
    ballenasDegC  REAL, ballenasKPa   REAL,
    comoxDegC     REAL, comoxKPa      REAL, comoxSky      TEXT,
    lillooetDegC  REAL, lillooetKPa   REAL,
    pamDegC       REAL, pamKPa        REAL,
    pembertonDegC REAL, pembertonKPa  REAL,
    vancouverDegC REAL, vancouverKPa  REAL, vancouverSky  TEXT,
    victoriaDegC  REAL, victoriaKPa   REAL, victoriaSky   TEXT,
    whistlerDegC  REAL, whistlerKPa   REAL, whistlerSky   TEXT,
    speed         REAL, gust          REAL, lull          REAL,
    direction     REAL, temperature   REAL
)
"""


def init_db(db_path: Path = None) -> None:
    if db_path is None:
        db_path = Path(os.getenv('WORKING_DIRECTORY', '.')) / 'web_data' / 'weather_data_hourly.db'
    with sqlite3.connect(db_path) as conn:
        conn.execute(_CREATE_WEATHER)
        conn.commit()
    print(f'Initialised {db_path}')


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Download historical weather data')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--ec-only',    action='store_true', help='Only download EC station CSVs (full rebuild)')
    mode.add_argument('--ec-update',  action='store_true', help='Append EC data since last CSV entry for each station')
    mode.add_argument('--sws-only',   action='store_true', help='Only download SWS wind data (full rebuild)')
    mode.add_argument('--sws-update', action='store_true', help='Append SWS data since the last CSV entry')
    mode.add_argument('--init-db',    action='store_true', help='Only initialise the SQLite DB')
    p.add_argument('--ec-end',  metavar='MonYYYY', help='Last month to download for EC (e.g. Jun2026)')
    p.add_argument('--sws-start', default=SWS_START, help=f'SWS start date (default: {SWS_START})')
    p.add_argument('--sws-end',   help='SWS end date YYYY-MM-DD (default: today)')
    args = p.parse_args()

    if args.init_db:
        init_db()
    elif args.ec_update:
        update_ec_history(end=args.ec_end)
    elif args.ec_only:
        download_ec_history(end=args.ec_end)
    elif args.sws_update:
        update_sws_csv()
    elif args.sws_only:
        download_sws_history(start=args.sws_start, end=args.sws_end)
    else:
        init_db()
        download_ec_history(end=args.ec_end)
        download_sws_history(start=args.sws_start, end=args.sws_end)
