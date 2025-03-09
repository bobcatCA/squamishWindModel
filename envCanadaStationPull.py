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


def find_best_name_match(target, list, prefer=None, threshold=80):
    """
    :param target: String, that you wish to find in a list of items
    :param list: List in which you're looking to find the target
    :param prefer: String, for potential multiple matches if another part in the string is preferred
    :param threshold: Measure of match quality
    :return: The matching value from the list, if it meets threshold
    """
    if prefer:
        matches = process.extract(target, list, limit=2)  # Get top 2 matches

        for match, score, in matches:
            if prefer in match and score >= threshold:
                return match
            else:
                return None
    else:
        match, score = process.extractOne(target, list)
    return match if score >= threshold else None


def pull_past_24hrs():
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
    startTime = pd.Timestamp.now().ceil('h')
    timeValues = pd.date_range(end=startTime, periods=30, freq='h')
    df = pd.DataFrame({'datetime': timeValues})

    for key, value in urls.items():
        print(f'Fetching values for {key}')
        response = requests.get(value)
        if response.status_code == 200:

            # Find the forecast table and headers
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', attrs={'class': 'table'})
            headers = [th.text.strip() for th in table.find('thead').find_all('th')]

            # Extract table rows
            rows = []
            dayFirstIdx = {}
            for tr in table.find('tbody').find_all('tr'):
                cells = [td.text.strip() for td in tr.find_all('td')]
                if cells:  # Build list of hourly entries
                    rows.append(cells)
                else:  # There is a banner above the cells on this table, that has the day
                    try:
                        tableDate = pd.to_datetime(tr.text)
                        dayFirstIdx[tableDate] = len(rows)
                    except:  # TODO: Update. This should occur for all cases without cells that don't have a date/time value
                        print('Unknown table cell!')

            # Format weather data into a dataframe. Remove white space and re-name to standard heading names
            df_station = pd.DataFrame(rows, columns=headers)
            htmlNames = df_station.columns
            htmlNames = [clean_column_names(col) for col in htmlNames]
            df_station.columns = htmlNames
            htmlDate = find_best_name_match('Date', htmlNames)
            htmlCondition = find_best_name_match('Condition', htmlNames)
            htmlPress = find_best_name_match('Press', htmlNames, prefer='kPa')
            htmlTemp = find_best_name_match('Temp', htmlNames, prefer='C')
            htmlNames = [htmlDate, htmlCondition, htmlPress, htmlTemp]
            newNames = ['datetime', f'{key}Sky', f'{key}KPa', f'{key}DegC']
            dictNames = dict(zip(htmlNames, newNames))
            df_station = df_station.rename(columns=dictNames)
            df_station['datetime'] = pd.to_timedelta(df_station['datetime'] + ':00')

            # Add day to the hour column, so it's a proper time stamp
            df_dateIdx = pd.DataFrame(dayFirstIdx.items(), columns=['day', 'startIdx'])
            df_station = df_station.merge(df_dateIdx, left_on=df_station.index, right_on='startIdx', how='left')
            df_station['day'] = df_station['day'].ffill()
            df_station['datetime'] = df_station['day'] + df_station['datetime']

            # Narrow down to required columns, change values to numeric or standard strings
            df_station = df_station[newNames]
            df_station[f'{key}DegC'] = df_station[f'{key}DegC'].apply(extract_in_brackets)
            df_station[f'{key}DegC'] = pd.to_numeric(df_station[f'{key}DegC'], errors='coerce')
            df_station[f'{key}KPa'] = pd.to_numeric(df_station[f'{key}KPa'], errors='coerce')
            df_station[f'{key}Sky'] = df_station[f'{key}Sky'].replace(['Clear', 'Mainly Clear'],
                                                                          'Fair')  # Clear and mainly clear should be similar
            df_station[f'{key}Sky'] = df_station[f'{key}Sky'].replace(r'^(?!Fair$|Mostly Cloudy|Cloudy|n/a$).*',
                                                                          'Other', regex=True)

            # Merge multiple cities into the big dataframe
            df = df.merge(df_station, on='datetime', how='inner')

        else:  # If no response from URL get
            print('Error: invalid URL or no response')

        pass

    # Sort by date ascending and return
    df.sort_values(by='datetime', inplace=True)
    return df

if __name__=='__main__':
    pass
