import requests
from bs4 import BeautifulSoup
from fuzzywuzzy import process
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Retrieve hourly information from Env Canada website
urls = {
    'comox': 'https://weather.gc.ca/en/forecast/hourly/index.html?coords=49.674,-124.928',
    'lillooet': 'https://weather.gc.ca/en/forecast/hourly/index.html?coords=50.694,-121.939',
    'pemberton': 'https://weather.gc.ca/en/forecast/hourly/index.html?coords=50.317,-122.8',
    'vancouver': 'https://weather.gc.ca/en/forecast/hourly/index.html?coords=49.245,-123.115',
    'victoria': 'https://weather.gc.ca/en/forecast/hourly/index.html?coords=48.433,-123.362',
    'whistler': 'https://weather.gc.ca/en/forecast/hourly/index.html?coords=50.117,-122.955',
}
# Fetch the page content
startTime = pd.Timestamp.now().ceil('h')
timeValues = pd.date_range(start=startTime, periods=10, freq='h')
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
        for tr in table.find('tbody').find_all('tr'):
            cells = [td.text.strip() for td in tr.find_all('td')]
            if cells:
                rows.append(cells)
            else:  # There is a banner above the cells on this table, that has the day
                try:
                    tableDate = pd.to_datetime(tr.text)
                    if pd.to_datetime('today').normalize() == tableDate:
                        forecastDate = tableDate
                    else:  # Only pull today's forecast data, so break loop if not
                        break
                except:  # TODO: Update. This should occur for all cases without cells that don't have a date/time value
                    print('Unknown table cell!')

        # Format weather data into a dataframe
        df_station = pd.DataFrame(rows, columns=headers)
        df_station['Date/Time (PST)'] = pd.to_timedelta(df_station['Date/Time (PST)'] + ':00') + forecastDate
        df_station = df_station[['Date/Time (PST)', 'Temp. (\u00b0'+'C)', 'Weather Conditions']]

        # Merge multiple cities into the big dataframe
        df = df.merge(df_station, left_on='datetime', right_on='Date/Time (PST)', how='inner')
        df = df.rename(columns={'Temp. (\u00b0'+'C)': f'{key}degC', 'Weather Conditions': f'{key}Sky'})
        df.drop(columns='Date/Time (PST)', inplace=True)

    else:  # If no response from URL get
        print('Error: invalid URL or no response')

    pass

print('done')