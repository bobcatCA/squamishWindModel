import requests
import pandas as pd
import re
from bs4 import BeautifulSoup
from thefuzz import process

def clean_column_names(name):
    """
    :param name: Name with potentially a lot of spaces, new-lines
    :return: name with spaces and new lines removed
    """
    return ' '.join(name.split())

def extract_in_brackets(text):
    """
    :param text: String, of text to look in
    :return: Float, of first value found in brackets
    """
    match = re.search(r'\((\d+\.\d+|\d+)\)', text)  # Look for text in brackets
    return float(match.group(1)) if match else None  # Convert to an ineger if found


def find_best_name_match(target, list_of_possible, prefer=None, threshold=80):
    """
    :param target: String, that you wish to find in a list of items
    :param list_of_possible: List in which you're looking to find the target
    :param prefer: String, for potential multiple matches if another part in the string is preferred
    :param threshold: Measure of match quality
    :return: The matching value from the list, if it meets threshold
    """
    if prefer:
        matches = process.extract(target, list_of_possible, limit=2)  # Get top 2 matches

        for match, score, in matches:
            if prefer in match and score >= threshold:
                return match
            else:
                return None
    else:
        match, score = process.extractOne(target, list_of_possible)
    return match if score >= threshold else None


def pull_past_hrs_weather():
    """
    :return: DataFrame containing time-series of past 24hrs of all weather stations
    """
    # Retrieve hourly information from Env Canada website
    urls = {
        'ballenas': 'https://weather.gc.ca/past_conditions/index_e.html?station=voq',
        'comox': 'https://weather.gc.ca/past_conditions/index_e.html?station=yqq',
        'lillooet': 'https://weather.gc.ca/past_conditions/index_e.html?station=wkf',
        'pam': 'https://weather.gc.ca/past_conditions/index_e.html?station=was',
        'pemberton': 'https://weather.gc.ca/past_conditions/index_e.html?station=wgp',
        'vancouver': 'https://weather.gc.ca/past_conditions/index_e.html?station=yvr',
        'victoria': 'https://weather.gc.ca/past_conditions/index_e.html?station=yyj',
        'whistler': 'https://weather.gc.ca/past_conditions/index_e.html?station=wae',
    }
    # Fetch the page content
    df = pd.DataFrame()
    first_station = True

    for key, value in urls.items():
        print(f'Fetching values for {key}')
        response = requests.get(value)
        if response.status_code == 200:

            # Find the forecast table and headers
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', attrs={'class': 'table'})
            headers = [th.text.strip() for th in table.find('thead').find_all('th')]

            # Extract table rows for each 'tr' class
            rows = []
            day_first_idx = {}
            for tr in table.find('tbody').find_all('tr'):
                cells = [td.text.strip() for td in tr.find_all('td')]
                if cells:  # Build list of hourly entries
                    rows.append(cells)
                else:  # There is a banner above the cells on this table, that has the day
                    try:
                        tableDate = pd.to_datetime(tr.text)
                        day_first_idx[tableDate] = len(rows)
                    except:  # TODO: Update. This should occur for all cases without cells that don't have a date/time value
                        print('Unknown table cell!')

            # Format weather data into a dataframe. Remove white space and re-name to standard heading names
            df_station = pd.DataFrame(rows, columns=headers)
            html_names = df_station.columns
            html_names = [clean_column_names(col) for col in html_names]
            df_station.columns = html_names
            html_date = find_best_name_match('Date', html_names)
            html_condition = find_best_name_match('Condition', html_names)
            html_pressure = find_best_name_match('Press', html_names, prefer='kPa')
            html_temperature = find_best_name_match('Temp', html_names, prefer='C')
            html_names = [html_date, html_condition, html_pressure, html_temperature]
            new_names = ['datetime', f'{key}Sky', f'{key}KPa', f'{key}DegC']
            dict_names = dict(zip(html_names, new_names))
            df_station = df_station.rename(columns=dict_names)
            df_station['datetime'] = pd.to_timedelta(df_station['datetime'] + ':00')

            # Add day to the hour column, so it can be converted to a proper time stamp
            df_date_idx = pd.DataFrame(day_first_idx.items(), columns=['day', 'startIdx'])
            df_station = df_station.merge(df_date_idx, left_on=df_station.index, right_on='startIdx', how='left')
            df_station['day'] = df_station['day'].ffill()
            df_station['datetime'] = df_station['day'] + df_station['datetime']

            # Narrow down to required columns, change values to numeric or standard strings
            df_station = df_station[new_names]
            df_station[f'{key}DegC'] = df_station[f'{key}DegC'].apply(extract_in_brackets)
            df_station[f'{key}DegC'] = pd.to_numeric(df_station[f'{key}DegC'], errors='coerce')
            df_station[f'{key}KPa'] = pd.to_numeric(df_station[f'{key}KPa'], errors='coerce')
            df_station[f'{key}Sky'] = df_station[f'{key}Sky'].replace(['Clear', 'Mainly Clear'],
                                                                          'Fair')  # Clear and mainly clear should be similar
            df_station[f'{key}Sky'] = df_station[f'{key}Sky'].replace(r'^(?!Fair$|Mostly Cloudy|Cloudy|n/a$).*',
                                                                          'Other', regex=True)

            # Merge multiple cities into the big dataframe
            if first_station:
                df = df_station
                first_station = False
            else:
                df = df.merge(df_station, on='datetime', how='inner')

        else:  # If no response from URL get
            print('Error: invalid URL or no response')

        pass

    # Sort by date ascending and return
    df.sort_values(by='datetime', inplace=True)
    return df

if __name__=='__main__':
    pass
