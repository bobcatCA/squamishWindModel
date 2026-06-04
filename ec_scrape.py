"""Scrape real-time and forecast weather data from Environment Canada.

Public API:
    pull_past_hrs_weather()     → DataFrame of the last 24 h from 8 stations
    pull_forecast_hourly()      → DataFrame of the hourly EC forecast (6 stations)
    pull_forecast_daily(range)  → DataFrame of the daily EC forecast (6 stations)
"""

import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup
from thefuzz import process

# ── Shared helpers ────────────────────────────────────────────────────────────

def _clean_col_name(name: str) -> str:
    return ' '.join(name.split())


def _extract_bracket_value(text: str) -> float | None:
    """Return the first numeric value found inside parentheses, e.g. '(13.2)' → 13.2."""
    import re
    m = re.search(r'\((\d+\.?\d*)\)', text)
    return float(m.group(1)) if m else None


def _find_col(target: str, cols, prefer: str = None, threshold: int = 80) -> str | None:
    """Fuzzy-match *target* against *cols*, optionally preferring matches that contain *prefer*."""
    if prefer:
        for match, score in process.extract(target, cols, limit=2):
            if prefer in match and score >= threshold:
                return match
        return None
    match, score = process.extractOne(target, cols)
    return match if score >= threshold else None


def _parse_ec_table(html: str) -> tuple[pd.DataFrame, dict]:
    """Parse a standard EC station HTML table.

    Returns (row_df, day_row_index) where day_row_index maps a calendar date
    to the first data-row index that belongs to that date.
    """
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table', attrs={'class': 'table'})
    headers = [th.text.strip() for th in table.find('thead').find_all('th')]

    rows = []
    day_row_index = {}
    for tr in table.find('tbody').find_all('tr'):
        cells = [td.text.strip() for td in tr.find_all('td')]
        if cells:
            rows.append(cells)
        else:
            try:
                day_row_index[pd.to_datetime(tr.text)] = len(rows)
            except (ValueError, TypeError):
                pass

    return pd.DataFrame(rows, columns=headers), day_row_index


def _attach_dates(df: pd.DataFrame, day_row_index: dict) -> pd.DataFrame:
    """Resolve hour-offset timestamps to full tz-aware datetimes."""
    day_df = pd.DataFrame(day_row_index.items(), columns=['day', 'start_idx'])
    df = df.merge(day_df, left_on=df.index, right_on='start_idx', how='left')
    df['day'] = df['day'].ffill()
    df['datetime'] = df['day'] + pd.to_timedelta(df['datetime'] + ':00')
    df['datetime'] = df['datetime'].dt.tz_localize('America/Vancouver')
    return df


# ── Sky-condition normalisation ───────────────────────────────────────────────

_SKY_REWRITES = [
    (['Clear', 'Mainly Clear', 'Sunny'], 'Fair'),
    (['A mix of sun and cloud', 'Partly cloudy', 'Mainly cloudy'], 'Mostly Cloudy'),
]


def normalize_sky_series(series: pd.Series) -> pd.Series:
    """Normalise raw EC weather strings to Fair / Mostly Cloudy / Cloudy / Other."""
    for originals, replacement in _SKY_REWRITES:
        series = series.replace(originals, replacement)
    return series.replace(r'^(?!Fair$|Mostly Cloudy|Cloudy$).*', 'Other', regex=True)


# ── Station URLs ──────────────────────────────────────────────────────────────

_PAST_URLS = {
    'ballenas':  'https://weather.gc.ca/past_conditions/index_e.html?station=voq',
    'comox':     'https://weather.gc.ca/past_conditions/index_e.html?station=yqq',
    'lillooet':  'https://weather.gc.ca/past_conditions/index_e.html?station=wkf',
    'pam':       'https://weather.gc.ca/past_conditions/index_e.html?station=was',
    'pemberton': 'https://weather.gc.ca/past_conditions/index_e.html?station=wgp',
    'vancouver': 'https://weather.gc.ca/past_conditions/index_e.html?station=yvr',
    'victoria':  'https://weather.gc.ca/past_conditions/index_e.html?station=yyj',
    'whistler':  'https://weather.gc.ca/past_conditions/index_e.html?station=wae',
}

_FORECAST_HOURLY_URLS = {
    'comox':     'https://weather.gc.ca/en/forecast/hourly/index.html?coords=49.674,-124.928',
    'lillooet':  'https://weather.gc.ca/en/forecast/hourly/index.html?coords=50.694,-121.939',
    'pemberton': 'https://weather.gc.ca/en/forecast/hourly/index.html?coords=50.317,-122.8',
    'vancouver': 'https://weather.gc.ca/en/forecast/hourly/index.html?coords=49.245,-123.115',
    'victoria':  'https://weather.gc.ca/en/forecast/hourly/index.html?coords=48.433,-123.362',
    'whistler':  'https://weather.gc.ca/en/forecast/hourly/index.html?coords=50.117,-122.955',
}

_FORECAST_DAILY_URLS = {
    'comox':     'https://weather.gc.ca/en/location/index.html?coords=49.674,-124.928',
    'lillooet':  'https://weather.gc.ca/en/location/index.html?coords=50.694,-121.939',
    'pemberton': 'https://weather.gc.ca/en/location/index.html?coords=50.317,-122.800',
    'vancouver': 'https://weather.gc.ca/en/location/index.html?coords=49.245,-123.115',
    'victoria':  'https://weather.gc.ca/en/location/index.html?coords=48.433,-123.362',
    'whistler':  'https://weather.gc.ca/en/location/index.html?coords=50.117,-122.955',
}


# ── Public pull functions ─────────────────────────────────────────────────────

def pull_past_hrs_weather() -> pd.DataFrame:
    """Return a DataFrame of the past 24 h of observed conditions for all stations."""
    df_all = pd.DataFrame()

    for station, url in _PAST_URLS.items():
        resp = requests.get(url)
        if resp.status_code != 200:
            print(f'Warning: no response for {station} ({url})')
            continue

        df_station, day_row_index = _parse_ec_table(resp.text)
        cols = [_clean_col_name(c) for c in df_station.columns]
        df_station.columns = cols

        date_col  = _find_col('Date', cols)
        press_col = _find_col('Press', cols, prefer='kPa')
        temp_col  = _find_col('Temp', cols, prefer='C')
        cond_col  = _find_col('Condition', cols)

        df_station = df_station.rename(columns={
            date_col:  'datetime',
            cond_col:  f'{station}Sky',
            press_col: f'{station}KPa',
            temp_col:  f'{station}DegC',
        })
        df_station = _attach_dates(df_station, day_row_index)
        df_station = df_station[['datetime', f'{station}Sky', f'{station}KPa', f'{station}DegC']]

        df_station[f'{station}DegC'] = df_station[f'{station}DegC'].apply(_extract_bracket_value)
        df_station[f'{station}DegC'] = pd.to_numeric(df_station[f'{station}DegC'], errors='coerce')
        df_station[f'{station}KPa']  = pd.to_numeric(df_station[f'{station}KPa'],  errors='coerce')

        df_all = df_station if df_all.empty else df_all.merge(df_station, on='datetime', how='inner')

    df_all.sort_values('datetime', inplace=True)
    return df_all


def pull_forecast_hourly() -> pd.DataFrame:
    """Return a DataFrame of the hourly EC forecast for the next ~24 h."""
    df_all = pd.DataFrame()

    for station, url in _FORECAST_HOURLY_URLS.items():
        resp = requests.get(url)
        if resp.status_code != 200:
            print(f'Warning: no response for {station} ({url})')
            continue

        df_station, day_row_index = _parse_ec_table(resp.text)
        cols = list(df_station.columns)

        date_col = _find_col('Date', cols)
        temp_col = _find_col('Temp', cols, prefer='C')
        cond_col = _find_col('Condition', cols)

        df_station = df_station.rename(columns={
            date_col: 'datetime',
            cond_col: f'{station}Sky',
            temp_col: f'{station}DegC',
        })
        df_station = _attach_dates(df_station, day_row_index)
        df_station = df_station[['datetime', f'{station}Sky', f'{station}DegC']]
        df_station[f'{station}DegC'] = pd.to_numeric(df_station[f'{station}DegC'], errors='coerce')

        df_all = df_station if df_all.empty else df_all.merge(df_station, on='datetime', how='inner')

    df_all.sort_values('datetime', inplace=True)
    return df_all


def pull_forecast_daily(time_range) -> pd.DataFrame:
    """Return a DataFrame of the daily EC high-temp forecast aligned to *time_range*."""
    df_all = pd.DataFrame({'datetime': time_range})
    current_year = datetime.date.today().year

    for station, url in _FORECAST_DAILY_URLS.items():
        resp = requests.get(url)
        if resp.status_code != 200:
            print(f'Warning: no response for {station} ({url})')
            continue

        soup = BeautifulSoup(resp.text, 'html.parser')
        table = soup.find(attrs={'class': 'div-table'})
        dates = pd.Series([
            col.text.strip()
            for col in table.find_all(attrs={'class': 'div-row div-row1 div-row-head'})
        ])
        forecast_data = pd.Series([
            col.text.strip()
            for col in table.find_all(attrs={'class': [
                'div-row div-row2 div-row-data',
                'div-row div-row2 div-row-data linkdate',
            ]})
        ])

        high_temps  = forecast_data.str.extract(r'(\d+)(?=°)')
        conditions  = forecast_data.str.extract(r'(?<=°C)\s*(.*)')
        days        = dates.str.extract(r'(\d+)')
        months      = dates.str.extract(r'([A-Za-z]+)$')

        df_station = pd.DataFrame()
        df_station['datetime'] = (
            pd.to_datetime(days[0] + ' ' + months[0] + ' ' + str(current_year), format='mixed')
            + pd.to_timedelta(14, 'hours')
        )
        df_station['datetime'] = df_station['datetime'].dt.tz_localize('America/Vancouver')
        df_station[f'{station}DegC'] = pd.to_numeric(high_temps[0], errors='coerce')
        df_station[f'{station}Sky']  = conditions[0]

        df_all = df_all.merge(df_station, on='datetime', how='inner')

    return df_all
