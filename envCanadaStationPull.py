import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

def extract_in_brackets(text):
    """
    :param text: String, of text to look in
    :return: Float, of first value found in brackets
    """
    match = re.search(r'\((\d+\.\d+|\d+)\)', text)  # Look for text in brackets
    return float(match.group(1)) if match else None  # Convert to an ineger if found

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

            # Format weather data into a dataframe
            df_station = pd.DataFrame(rows, columns=headers)
            df_station['Date / Time(PST)'] = pd.to_timedelta(df_station['Date / Time(PST)'] + ':00')

            # Add day to the hour column
            df_dateIdx = pd.DataFrame(dayFirstIdx.items(), columns=['day', 'startIdx'])
            df_station = df_station.merge(df_dateIdx, left_on=df_station.index, right_on='startIdx', how='left')
            df_station['day'] = df_station['day'].ffill()
            df_station['datetime'] = df_station['day'] + df_station['Date / Time(PST)']
            df_station = df_station[['datetime', 'Conditions',
                                     'Temperature\n                                            (\u00b0'+'C)',
                                     'Pressure(kPa)']]
            # TODO: make this more general... the table format might not always be the same?
            df_station = df_station.rename(columns={'Temperature\n                                            (\u00b0'+'C)': f'{key}degC',
                                                    'Conditions': f'{key}Sky', 'Pressure(kPa)': f'{key}KPa'})
            df_station[f'{key}degC'] = df_station[f'{key}degC'].apply(extract_in_brackets)
            df_station[f'{key}KPa'] = pd.to_numeric(df_station[f'{key}KPa'], errors='coerce')

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
