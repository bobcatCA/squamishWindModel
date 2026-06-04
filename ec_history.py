"""Download historical hourly data from the EC climate bulk-data API.

Run from the project directory to (re)build the per-station CSVs:
    python ec_history.py
"""

import time
from datetime import datetime
from io import StringIO

import pandas as pd
import requests
from dateutil import rrule

_SESSION = requests.Session()
_SESSION.headers['User-Agent'] = 'Mozilla/5.0'

# Station name → EC climate station ID
STATIONS = {
    'Vancouver': 51442,
    'Whistler':  52178,
    'Pemberton': 536,
    'Lillooet':  27388,
    'Victoria':  51337,
    'Ballenas':  138,
    'Pam':       6817,
    'Comox':     155,
}


def fetch_station_month(station_id: int, year: int, month: int,
                        retries: int = 3) -> pd.DataFrame:
    """Fetch one month of hourly data from the EC climate API."""
    url = (
        f'https://climate.weather.gc.ca/climate_data/bulk_data_e.html?'
        f'format=csv&stationID={station_id}&Year={year}&Month={month}&Day=1&timeframe=1'
    )
    for attempt in range(retries):
        try:
            resp = _SESSION.get(url, timeout=30)
            resp.raise_for_status()
            if b'Date/Time' not in resp.content[:500]:
                raise ValueError(f'Unexpected response (HTML?): {resp.content[:100]}')
            return pd.read_csv(StringIO(resp.content.decode('utf-8-sig')))
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f'    Warning: failed after {retries} attempts — {exc}')
                return pd.DataFrame()
    return pd.DataFrame()


if __name__ == '__main__':
    start = datetime.strptime('Jan2016', '%b%Y')
    end   = datetime.strptime('Jun2026', '%b%Y')
    months = list(rrule.rrule(rrule.MONTHLY, dtstart=start, until=end))
    delay  = 0.4  # seconds between API calls

    for name, station_id in STATIONS.items():
        print(f'\n{name} (ID {station_id}) — {len(months)} months')
        frames = []
        for i, dt in enumerate(months):
            df = fetch_station_month(station_id, dt.year, dt.month)
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
        last_valid = out['Date/Time (LST)'].loc[out['Rel Hum (%)'].last_valid_index()]
        print(f'  Last valid humidity: {last_valid}')
        out.to_csv(f'{name}.csv', index=False)
        print(f'  Saved {name}.csv ({len(out):,} rows)')
