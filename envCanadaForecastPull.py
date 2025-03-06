import requests
from bs4 import BeautifulSoup
import pandas as pd

def pull_forecast():
    """
    :return: DataFrame, with the time-series forecast data for all stations
    """
    # Retrieve hourly information from Env Canada website
    urls = {
        'comox': 'https://weather.gc.ca/en/forecast/hourly/index.html?coords=49.674,-124.928',
        'lillooet': 'https://weather.gc.ca/en/forecast/hourly/index.html?coords=50.694,-121.939',
        'pemberton': 'https://weather.gc.ca/en/forecast/hourly/index.html?coords=50.317,-122.8',
        'vancouver': 'https://weather.gc.ca/en/forecast/hourly/index.html?coords=49.245,-123.115',
        'victoria': 'https://weather.gc.ca/en/forecast/hourly/index.html?coords=48.433,-123.362',
        'whistler': 'https://weather.gc.ca/en/forecast/hourly/index.html?coords=50.117,-122.955',
    }

    # Create an empty DF with hourly timestamps for the next n hours
    startTime = pd.Timestamp.now().ceil('h')
    timeValues = pd.date_range(start=startTime, periods=20, freq='h')
    df = pd.DataFrame({'datetime': timeValues})

    # Loop through all the URLs and compile a dataframe of all the data. Merge it with df
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
                if cells:
                    rows.append(cells)
                else:  # There is a banner above the cells on this table, that has the day
                    try:
                        tableDate = pd.to_datetime(tr.text)
                        dayFirstIdx[tableDate] = len(rows)
                        # if pd.to_datetime('today').normalize() == tableDate:
                        #     forecastDate = tableDate
                        # else:  # Only pull today's forecast data, so break loop if not
                        #     break
                    except:  # TODO: Update. This should occur for all cases without cells that don't have a date/time value
                        print('Unknown table cell!')

            # Format weather data into a dataframe
            df_station = pd.DataFrame(rows, columns=headers)
            df_station['Date/Time (PST)'] = pd.to_timedelta(df_station['Date/Time (PST)'] + ':00')

            # Add day to the hour column
            df_dateIdx = pd.DataFrame(dayFirstIdx.items(), columns=['day', 'startIdx'])
            df_station = df_station.merge(df_dateIdx, left_on=df_station.index, right_on='startIdx', how='left')
            df_station['day'] = df_station['day'].ffill()
            df_station['datetime'] = df_station['day'] + df_station['Date/Time (PST)']
            df_station = df_station[['datetime', 'Temp. (\u00b0' + 'C)', 'Weather Conditions']]

            # Merge multiple cities into the big dataframe
            df = df.merge(df_station, on='datetime', how='inner')
            df = df.rename(columns={'Temp. (\u00b0'+'C)': f'{key}degC', 'Weather Conditions': f'{key}Sky'})
            # df.drop(columns='Date/Time (PST)', inplace=True)

        else:  # If no response from URL get
            print('Error: invalid URL or no response')

        pass  # Loop to the next weather station

    # Sort by date ascending and return
    df.sort_values(by='datetime', inplace=True)
    return df

if __name__=='__main__':
    pass
