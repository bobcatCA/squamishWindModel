"""Fetch wind data from the Squamish Windsports sensor via headless Chrome.

Public API:
    get_sws_df(dates)   → DataFrame of raw wind readings for the given date strings

Run standalone to rebuild sws_wind_database.csv:
    python sws_pull.py
"""

import json
import platform
import tempfile
import time
from datetime import datetime, timedelta

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


if __name__ == '__main__':
    start_dt = datetime.strptime('2016-05-01', '%Y-%m-%d')
    end_dt   = datetime.strptime('2026-06-01', '%Y-%m-%d')
    date_list = [
        (start_dt + timedelta(days=i)).strftime('%Y-%m-%d')
        for i in range((end_dt - start_dt).days + 1)
    ]

    df_sws = get_sws_df(date_list)
    df_sws.to_csv('sws_wind_database.csv')
    print(f'Saved sws_wind_database.csv  {len(df_sws):,} rows')
