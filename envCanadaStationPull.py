import requests
from bs4 import BeautifulSoup
from fuzzywuzzy import process
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import dump_svmlight_file

def extract_in_brackets(text):
    match = re.search(r'\((\d+\.\d+|\d+)\)', text)  # Look for text in brackets
    return float(match.group(1)) if match else None  # Convert to an ineger if found

# Retrieve hourly information from Env Canada website
urls = {
    'vancouver': 'https://weather.gc.ca/past_conditions/index_e.html?station=yvr'
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
        df_station[f'{key}KPa'] = pd.to_numeric(df_station[f'{key}KPa'])

        # Merge multiple cities into the big dataframe
        df = df.merge(df_station, on='datetime', how='inner')

        # df.drop(columns='Date / Time (PST)', inplace=True)

    else:  # If no response from URL get
        print('Error: invalid URL or no response')

    pass

print('done')
