import pandas as pd
import requests
from bs4 import BeautifulSoup
from thefuzz import process

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

            # Format weather data into a dataframe, rename columns to standard headings
            df_station = pd.DataFrame(rows, columns=headers)
            htmlNames = df_station.columns
            htmlDate = find_best_name_match('Date', htmlNames)
            htmlCondition = find_best_name_match('Condition', htmlNames)
            htmlTemp = find_best_name_match('Temp', htmlNames, prefer='C')
            htmlNames = [htmlDate, htmlCondition, htmlTemp]
            newNames = ['datetime', f'{key}Sky', f'{key}DegC']
            dictNames = dict(zip(htmlNames, newNames))
            df_station = df_station.rename(columns=dictNames)
            df_station['datetime'] = pd.to_timedelta(df_station['datetime'] + ':00')

            # Add day to the hour column
            df_dateIdx = pd.DataFrame(dayFirstIdx.items(), columns=['day', 'startIdx'])
            df_station = df_station.merge(df_dateIdx, left_on=df_station.index, right_on='startIdx', how='left')
            df_station['day'] = df_station['day'].ffill()
            df_station['datetime'] = df_station['day'] + df_station['datetime']
            df_station = df_station[newNames]

            # Convert to the proper type (numeric, string of standard categories)
            df_station[f'{key}DegC'] = pd.to_numeric(df_station[f'{key}DegC'], errors='coerce')
            # df_station[f'{key}KPa'] = pd.to_numeric(df_station[f'{key}KPa'], errors='coerce')
            df_station[f'{key}Sky'] = df_station[f'{key}Sky'].replace(['Clear', 'Mainly Clear'],
                                                                      'Fair')  # Clear and mainly clear should be similar
            df_station[f'{key}Sky'] = df_station[f'{key}Sky'].replace(r'^(?!Fair$|Mostly Cloudy|Cloudy|n/a$).*',
                                                                      'Other', regex=True)

            # Merge multiple cities into the big dataframe
            df = df.merge(df_station, on='datetime', how='inner')

        else:  # If no response from URL get
            print('Error: invalid URL or no response')

        pass  # Loop to the next weather station

    # Sort by date ascending and return
    df.sort_values(by='datetime', inplace=True)
    return df

if __name__=='__main__':
    pass
