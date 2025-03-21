import pandas as pd
from dateutil import rrule
from datetime import datetime, timedelta

# Van 51442
# Whistler 43443 (missing cover), 52178 (day only) (x348, 47867
# Lillooet 27388  (missing cover) (x955, 48308
# Pemberton 536 (missing cover)
# Entrance 132 no data
# Bowser 141 no data
# Cambbell 145 no data
# Nanaimo 192 no data

# Call Environment Canada API
# Returns a dataframe of data
def get_hourly_data_api(stationID, year, month):
    base_url = "http://climate.weather.gc.ca/climate_data/bulk_data_e.html?"
    query_url = "format=csv&stationID={}&Year={}&Month={}&timeframe=1".format(stationID, year, month)
    # query_url = "format=csv&stationID={}&Year={}&timeframe=1".format(stationID, year)
    api_endpoint = base_url + query_url
    return pd.read_csv(api_endpoint, skiprows=0)

if __name__=='__main__':
    # stationID = 51442
    stations = {
        # "Vancouver" : 51442,
        "Whistler": 52178,
        # "Pemberton": 536,
        # "Lillooet": 27388,
        # "Victoria": 51337,
        # "Ballenas": 138,
        # "Pam": 6817,
        # "Comox": 155
        # 'Squamish': 336
    }
    start_date = datetime.strptime('Jan2024', '%b%Y')
    end_date = datetime.strptime('Apr2025', '%b%Y')

    # Loop through the stations and gather data for the last 8 years
    for station, id in stations.items():
        print(station, id)
        filename = station + '.csv'
        frames = []
        for dt in rrule.rrule(rrule.MONTHLY, dtstart=start_date, until=end_date):
            df = get_hourly_data_api(id, dt.year, dt.month)
            frames.append(df)

        weather_data = pd.concat(frames)
        weather_data['Date/Time (LST)'] = pd.to_datetime(weather_data['Date/Time (LST)'])
        weather_data['Temp (°C)'] = pd.to_numeric(weather_data['Temp (°C)'])
        print(weather_data['Date/Time (LST)'].loc[weather_data['Rel Hum (%)'].last_valid_index()])
        weather_data.to_csv(filename)

