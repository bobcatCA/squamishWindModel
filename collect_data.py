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

# Same rationale as update_data.py's _PLAUSIBLE_DEGC_RANGE: wide enough to
# never flag real weather, tight enough to catch a value that's numeric but
# wrong (e.g. an EC placeholder/sentinel) — pd.to_numeric(errors='coerce')
# only catches values that aren't numbers at all, not ones that are numbers
# but nonsense.
_PLAUSIBLE_DEGC_RANGE = (-40, 50)

_REQUIRED_EC_COLUMNS = ['Date/Time (LST)', 'Temp (°C)']


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
            df = pd.read_csv(StringIO(resp.content.decode('utf-8-sig')))
            missing = [c for c in _REQUIRED_EC_COLUMNS if c not in df.columns]
            if missing:
                # The raw-bytes sniff above only checks the response looks
                # roughly EC-shaped before parsing; this catches a CSV that
                # parsed fine but doesn't actually have the columns we need
                # (e.g. EC changed the header format).
                raise ValueError(f'Parsed CSV missing columns {missing}')
            return df
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f'    Warning: failed after {retries} attempts — {exc}')
    return pd.DataFrame()


def _blank_implausible_temps(df: pd.DataFrame, label: str) -> pd.DataFrame:
    bad = df['Temp (°C)'].notna() & ~df['Temp (°C)'].between(*_PLAUSIBLE_DEGC_RANGE)
    if bad.any():
        print(f'    Warning: {label} has {int(bad.sum())} implausible Temp (°C) value(s) '
              f'(outside {_PLAUSIBLE_DEGC_RANGE}) — blanking to NaN')
        df.loc[bad, 'Temp (°C)'] = pd.NA
    return df


def _sparse_months(df: pd.DataFrame, threshold: float = 0.5) -> pd.PeriodIndex:
    """Return the calendar months where Temp (°C) completeness is below
    *threshold* — EC's bulk endpoint sometimes serves a shaped-correctly but
    mostly/entirely-empty CSV for a month instead of erroring, which neither
    the response-format check nor the plausibility check catches (an absent
    reading isn't an implausible one)."""
    tmp = df[['Date/Time (LST)', 'Temp (°C)']].copy()
    tmp['_month'] = tmp['Date/Time (LST)'].dt.to_period('M')
    completeness = tmp.groupby('_month')['Temp (°C)'].apply(lambda x: x.notna().mean())
    return completeness[completeness < threshold].index


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
        out = _blank_implausible_temps(out, f'{name}.csv')

        # Full rebuild covers years of already-published EC data, so unlike
        # update_ec_history's recent-months check (where sparseness likely
        # just means EC hasn't finished publishing that month yet, and
        # re-fetching the same run fixes it), a sparse month found here is
        # more likely a genuine permanent gap in EC's records — report it
        # rather than assume a retry would help.
        sparse = _sparse_months(out)
        if not sparse.empty:
            print(f'  Warning: sparse Temp (°C) coverage (<50%) in {len(sparse)} month(s): '
                  f'{[str(m) for m in sparse]}')

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
        # (Unlike download_ec_history's full-history check, sparseness here is
        # treated as fixable-by-retry: a recent month is plausibly just not
        # fully published by EC yet, so exclude the still-in-progress current
        # month and re-fetch the rest within this same run.)
        recent_start   = start_dt - pd.DateOffset(months=3)
        current_period = pd.Timestamp.now().to_period('M')
        sparse = _sparse_months(existing[existing['Date/Time (LST)'] >= recent_start])
        sparse = sparse[sparse < current_period]
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
        new_data = _blank_implausible_temps(new_data, f'{name}.csv (incremental)')

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
    ballenasDegC  REAL, ballenasKPa   REAL, ballenasHum   REAL,
    comoxDegC     REAL, comoxKPa      REAL, comoxHum      REAL, comoxSky      TEXT,
    lillooetDegC  REAL, lillooetKPa   REAL, lillooetHum   REAL,
    pamDegC       REAL, pamKPa        REAL, pamHum        REAL,
    pembertonDegC REAL, pembertonKPa  REAL, pembertonHum  REAL,
    vancouverDegC REAL, vancouverKPa  REAL, vancouverHum  REAL, vancouverSky  TEXT,
    victoriaDegC  REAL, victoriaKPa   REAL, victoriaHum   REAL, victoriaSky   TEXT,
    whistlerDegC  REAL, whistlerKPa   REAL, whistlerHum   REAL, whistlerSky   TEXT,
    squamishSpeed REAL, squamishGust   REAL, squamishLull  REAL,
    squamishDirection REAL, squamishDegC REAL
)
"""

# All 8 stations scraped by ec_scrape.pull_past_hrs_weather().
_HUM_STATIONS = ['ballenas', 'comox', 'lillooet', 'pam', 'pemberton', 'vancouver', 'victoria', 'whistler']


def _migrate_add_hum_columns(conn: sqlite3.Connection) -> None:
    """Add {station}Hum columns to a pre-existing `weather` table that predates
    humidity capture. No-op (and no data loss) if they're already there —
    ALTER TABLE ADD COLUMN only appends NULLs for existing rows, it never
    touches prior data."""
    existing = {row[1] for row in conn.execute('PRAGMA table_info(weather)')}
    for station in _HUM_STATIONS:
        col = f'{station}Hum'
        if col not in existing:
            conn.execute(f'ALTER TABLE weather ADD COLUMN {col} REAL')


# The SWS sensor columns' pre-rename names (see git history: "Rename SWS
# columns in SQLite DB to squamish* convention"). A `weather` table copied in
# from a deployment running older code (e.g. the Raspberry Pi) may still be
# on this naming.
_LEGACY_SWS_RENAME = {
    'speed':       'squamishSpeed',
    'gust':        'squamishGust',
    'lull':        'squamishLull',
    'direction':   'squamishDirection',
    'temperature': 'squamishDegC',
}


def _migrate_rename_legacy_sws_columns(conn: sqlite3.Connection) -> None:
    """Rename a pre-migration `weather` table's raw SWS column names to the
    current squamish* convention. Safe/idempotent — RENAME COLUMN preserves
    every existing value, it only relabels the column, and this only fires
    for a column still under its old name."""
    existing = {row[1] for row in conn.execute('PRAGMA table_info(weather)')}
    for old, new in _LEGACY_SWS_RENAME.items():
        if old in existing and new not in existing:
            conn.execute(f'ALTER TABLE weather RENAME COLUMN {old} TO {new}')


def migrate_schema(conn: sqlite3.Connection) -> None:
    """Bring an existing `weather` table up to the current schema in place.
    Safe to call on every DB touch, not just --init-db — a DB file copied in
    from another deployment (different code version, e.g. the Pi) can be on
    an older schema at any time, not only at first-time setup."""
    _migrate_add_hum_columns(conn)
    _migrate_rename_legacy_sws_columns(conn)


def init_db(db_path: Path = None) -> None:
    if db_path is None:
        db_path = Path(os.getenv('WORKING_DIRECTORY', '.')) / 'weather_data_hourly.db'
    with sqlite3.connect(db_path) as conn:
        conn.execute(_CREATE_WEATHER)
        migrate_schema(conn)
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
