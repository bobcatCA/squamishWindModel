import datetime
import pandas as pd
import requests
from bs4 import BeautifulSoup
from thefuzz import process

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


def pull_forecast_daily(time_range):
    df = pd.DataFrame()
    df['datetime'] = time_range
    df['datetime'] = df['datetime'].dt.tz_localize('America/Vancouver')

    urls = {
        'comox': 'https://weather.gc.ca/en/location/index.html?coords=49.674,-124.928',
        'lillooet': 'https://weather.gc.ca/en/location/index.html?coords=50.694,-121.939',
        'pemberton': 'https://weather.gc.ca/en/location/index.html?coords=50.317,-122.800',
        'vancouver': 'https://weather.gc.ca/en/location/index.html?coords=49.245,-123.115',
        'victoria': 'https://weather.gc.ca/en/location/index.html?coords=48.433,-123.362',
        'whistler': 'https://weather.gc.ca/en/location/index.html?coords=50.117,-122.955',
    }

    current_year = datetime.date.today().year

    for key, value in urls.items():
        # print(f'fetching dailys for {key}')
        response = requests.get(value)
        
        if response.status_code == 200:
            # Find the forecast table, headers (dates), and contents
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find(attrs={'class': 'div-table'})
            dates = pd.Series([col.text.strip() for col in table.find_all(attrs={'class': 'div-row div-row1 div-row-head'})])
            forecast_data = pd.Series([col.text.strip() for col in table.find_all(attrs={
                'class': ["div-row div-row2 div-row-data", "div-row div-row2 div-row-data linkdate"]})])
            high_temps = forecast_data.str.extract(r'(\d+)(?=°)')
            weather_conditions = forecast_data.str.extract(r'(?<=°C)\s*(.*)')

            # Assemble into a dataframe
            days = dates.str.extract(r'(\d+)')  # Extract numeric day
            months = dates.str.extract(r'([A-Za-z]+)$')  # Extract month
            df_station = pd.DataFrame()
            df_station['datetime'] = days + ' ' + months + ' ' + str(current_year)
            df_station['datetime'] = (pd.to_datetime(df_station['datetime'], format='mixed')
                                      + pd.to_timedelta(14, 'hours'))
            df_station['datetime'] = df_station['datetime'].dt.tz_localize('America/Vancouver')
            df_station[f'{key}DegC'] = high_temps
            df_station[f'{key}Sky'] = weather_conditions
            df = df.merge(df_station, on='datetime', how='inner')

    return df


def pull_forecast_hourly():
    """
    :return: DataFrame, with the time-series forecast data for all stations
    """
    # Retrieve hourly information from Env Canada websites
    urls = {
        'comox': 'https://weather.gc.ca/en/forecast/hourly/index.html?coords=49.674,-124.928',
        'lillooet': 'https://weather.gc.ca/en/forecast/hourly/index.html?coords=50.694,-121.939',
        'pemberton': 'https://weather.gc.ca/en/forecast/hourly/index.html?coords=50.317,-122.8',
        'vancouver': 'https://weather.gc.ca/en/forecast/hourly/index.html?coords=49.245,-123.115',
        'victoria': 'https://weather.gc.ca/en/forecast/hourly/index.html?coords=48.433,-123.362',
        'whistler': 'https://weather.gc.ca/en/forecast/hourly/index.html?coords=50.117,-122.955',
    }

    df = pd.DataFrame()
    first_station = True

    # Loop through all the URLs and compile a dataframe of all the data. Merge it with df
    for key, value in urls.items():
        # print(f'Fetching values for {key}')
        response = requests.get(value)
        if response.status_code == 200:

            # Find the forecast table and headers
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', attrs={'class': 'table'})
            headers = [th.text.strip() for th in table.find('thead').find_all('th')]

            # Extract table rows
            rows = []
            day_first_idx = {}
            for tr in table.find('tbody').find_all('tr'):
                cells = [td.text.strip() for td in tr.find_all('td')]
                if cells:
                    rows.append(cells)
                else:  # There is a banner above the cells on this table, that has the day
                    try:
                        table_date = pd.to_datetime(tr.text)
                        day_first_idx[table_date] = len(rows)
                        # if pd.to_datetime('today').normalize() == table_date:
                        #     forecastDate = table_date
                        # else:  # Only pull today's forecast data, so break loop if not
                        #     break
                    except:  # TODO: Update. This should occur for all cases without cells that don't have a date/time value
                        print('Unknown table cell!')

            # Format weather data into a dataframe, rename columns to standard headings
            df_station = pd.DataFrame(rows, columns=headers)
            html_names = df_station.columns
            html_date = find_best_name_match('Date', html_names)
            html_condition = find_best_name_match('Condition', html_names)
            html_temperature = find_best_name_match('Temp', html_names, prefer='C')
            html_names = [html_date, html_condition, html_temperature]
            new_names = ['datetime', f'{key}Sky', f'{key}DegC']
            dict_names = dict(zip(html_names, new_names))
            df_station = df_station.rename(columns=dict_names)
            df_station['datetime'] = pd.to_timedelta(df_station['datetime'] + ':00')

            # Add day to the hour column
            df_date_idx = pd.DataFrame(day_first_idx.items(), columns=['day', 'startIdx'])
            df_station = df_station.merge(df_date_idx, left_on=df_station.index, right_on='startIdx', how='left')
            df_station['day'] = df_station['day'].ffill()
            df_station['datetime'] = df_station['day'] + df_station['datetime']
            df_station['datetime'] = df_station['datetime'].dt.tz_localize('America/Vancouver')
            df_station = df_station[new_names]

            # Convert to the proper type (numeric, string of standard categories)
            df_station[f'{key}DegC'] = pd.to_numeric(df_station[f'{key}DegC'], errors='coerce')
            # df_station[f'{key}KPa'] = pd.to_numeric(df_station[f'{key}KPa'], errors='coerce')
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

        pass  # Loop to the next weather station

    # Sort by date ascending and return
    df.sort_values(by='datetime', inplace=True)
    return df

if __name__=='__main__':
    pass
