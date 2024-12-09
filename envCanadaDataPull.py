import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import rrule
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
import requests
import re

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
def getHourlyData(stationID, year, month):
    base_url = "http://climate.weather.gc.ca/climate_data/bulk_data_e.html?"
    query_url = "format=csv&stationID={}&Year={}&Month={}&timeframe=1".format(stationID, year, month)
    # query_url = "format=csv&stationID={}&Year={}&timeframe=1".format(stationID, year)
    api_endpoint = base_url + query_url
    return pd.read_csv(api_endpoint, skiprows=0)

# stationID = 51442
stations = {
    # "Vancouver" : 51442,
    # "Whistler": 52178,
    # "Pemberton": 536,
    # "Lillooet": 27388,
    # "Victoria": 51337,
    # "Ballenas": 138,
    "Pam": 6817,
    "Comox": 155
}
start_date = datetime.strptime('May2016', '%b%Y')
end_date = datetime.strptime('Sep2024', '%b%Y')

# Loop through the stations and gather data for the last 8 years
for station, id in stations.items():
    print(station, id)
    filename = station + '.csv'
    frames = []
    for dt in rrule.rrule(rrule.MONTHLY, dtstart=start_date, until=end_date):
        df = getHourlyData(id, dt.year, dt.month)
        frames.append(df)

    weather_data = pd.concat(frames)
    weather_data['Date/Time (LST)'] = pd.to_datetime(weather_data['Date/Time (LST)'])
    weather_data['Temp (°C)'] = pd.to_numeric(weather_data['Temp (°C)'])
    weather_data.to_csv(filename)

# Plot to show
# sns.set_style('whitegrid')
# fig = plt.figure(figsize=(15, 5))
# plt.plot(weather_data['Date/Time (LST)'], weather_data['Temp (°C)'], '-o', alpha=0.8, markersize=2)
# plt.plot(weather_data['Date/Time (LST)'], weather_data['Temp (°C)'].rolling(window=250,center=False).mean(), '-k', alpha=1.0)
# plt.ylabel('Temp (°C)')
# plt.xlabel('Time')
# plt.show()
