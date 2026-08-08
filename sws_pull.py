"""Fetch wind data from the Squamish Windsports sensor via headless Chrome.

Public API:
    get_sws_df(dates)        → DataFrame of raw wind readings for the given date strings
    update_sws_csv(csv_path) → append data since the last date in csv_path

Run standalone:
    python sws_pull.py           # full rebuild of sws_wind_database.csv
    python sws_pull.py --update  # append only new data since last CSV entry
"""

import argparse
import json
import platform
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

_SWS_BASE_URL = 'https://squamishwindsports.com/wind-data/getmet.php?wind_src=spit&reqdate={date}&reqtime=0'

_COL_RENAME = {
    'dt': 'datetime',
    'ws': 'speed',
    'wd': 'direction',
    'wg': 'gust',
    'wl': 'lull',
    't':  'temperature',
}


def _make_driver() -> webdriver.Chrome:
    opts = Options()
    opts.add_argument('--no-sandbox')
    opts.add_argument('--disable-dev-shm-usage')
    opts.add_argument('--headless')

    if 'aarch64' in platform.machine():
        opts.binary_location = '/usr/bin/chromium-browser'
        return webdriver.Chrome(service=Service('/usr/bin/chromedriver'), options=opts)

    opts.add_argument(f'--user-data-dir={tempfile.mkdtemp()}')
    return webdriver.Chrome(options=opts)


def get_sws_df(dates: list[str]) -> pd.DataFrame:
    """Fetch SWS wind data for each date string in *dates* (format 'YYYY-MM-DD').

    Returns a DataFrame sorted by datetime with columns:
        datetime, speed, direction, gust, lull, temperature
    Returns an empty DataFrame if no data is retrieved.
    """
    driver = _make_driver()
    frames = []

    try:
        for date in dates:
            driver.get(_SWS_BASE_URL.format(date=date))
            time.sleep(2)
            text = driver.find_element(By.CSS_SELECTOR, 'body').text.strip()
            if not text:
                print(f'Warning: empty response for {date}')
                continue
            try:
                frames.append(pd.DataFrame(json.loads(text)))
            except json.JSONDecodeError:
                print(f'Warning: invalid JSON for {date}')
    finally:
        driver.quit()

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns=_COL_RENAME)
    df = df.apply(pd.to_numeric, errors='coerce')
    df['datetime'] = (
        pd.to_datetime(df['datetime'], unit='s')
        .dt.tz_localize('America/Vancouver')
    )
    return df.sort_values('datetime').reset_index(drop=True)


_SWS_CSV = 'web_data/sws_wind_database.csv'


def update_sws_csv(csv_path: str = _SWS_CSV) -> None:
    """Append SWS data since the last recorded date in *csv_path*."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f'{csv_path} not found — run without --update to build from scratch.')
        return

    existing = pd.read_csv(csv_path)
    existing = existing.loc[:, ~existing.columns.str.match(r'^Unnamed')]
    # utc=True handles mixed PST/PDT offsets across seasons
    existing['datetime'] = pd.to_datetime(existing['datetime'], utc=True).dt.tz_convert('America/Vancouver')

    if existing.empty:
        print('CSV is empty — run without --update to build from scratch.')
        return

    last_date = existing['datetime'].max().date()
    end_date  = datetime.now().date()
    date_list = [
        (datetime(last_date.year, last_date.month, last_date.day) + timedelta(days=i)).strftime('%Y-%m-%d')
        for i in range((end_date - last_date).days + 1)
    ]

    print(f'Fetching {len(date_list)} days ({last_date} → {end_date})…')
    df_new = get_sws_df(date_list)
    if df_new.empty:
        print('No new data returned.')
        return

    # Both are now America/Vancouver-aware; drop_duplicates works correctly
    df_new['datetime'] = df_new['datetime'].dt.tz_convert('America/Vancouver')

    df_out = (
        pd.concat([existing, df_new], ignore_index=True)
        .drop_duplicates(subset=['datetime'])
        .sort_values('datetime')
        .reset_index(drop=True)
    )
    df_out.to_csv(csv_path, index=False)
    added = len(df_out) - len(existing)
    print(f'Added {added:,} rows → {csv_path} now has {len(df_out):,} rows total')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Fetch SWS wind data')
    p.add_argument('--update', action='store_true',
                   help='Append data since the last date in the CSV (default: full rebuild)')
    p.add_argument('--csv', default=_SWS_CSV, metavar='PATH',
                   help=f'CSV path (default: {_SWS_CSV})')
    args = p.parse_args()

    if args.update:
        update_sws_csv(args.csv)
    else:
        start_dt = datetime.strptime('2016-05-01', '%Y-%m-%d')
        end_dt   = datetime.strptime('2026-06-01', '%Y-%m-%d')
        date_list = [
            (start_dt + timedelta(days=i)).strftime('%Y-%m-%d')
            for i in range((end_dt - start_dt).days + 1)
        ]
        df_sws = get_sws_df(date_list)
        df_sws.to_csv(args.csv)
        print(f'Saved {args.csv}  {len(df_sws):,} rows')
