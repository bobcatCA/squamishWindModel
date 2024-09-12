# Library imports
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Initialize a start date and empty lists to hold the data we pull from Json files
startDate = pd.to_datetime('2016-05-05').date()
date = startDate
swsDate = []
swsSpeed = []
swsDir = []
swsLull = []
swsGust = []
swsTemp = []

while date < pd.Timestamp('today').date():
    # Look in the folder for the Json file corresponding to the date
    filename = str(date)+'.json'
    date += pd.Timedelta(days=1)  # Increment to next day

    # If the file doesn't exist, move to the next day
    if not os.path.isfile(filename):
        continue

    with open(filename) as jsonData:
        # Get the contents of the Json-formatted data from the file
        jsonContents = json.load(jsonData)
        jsonData.close()

        # Append/extend each list with the data from the new file
        swsDate.extend(jsonContents['dt'])
        swsSpeed.extend(jsonContents['ws'])
        swsDir.extend(jsonContents['wd'])
        swsGust.extend(jsonContents['wg'])
        swsLull.extend(jsonContents['wl'])
        swsTemp.extend(jsonContents['t'])
        pass

# Save the lists to a dataframe, convert from string to floats/dates
df_swsData = pd.DataFrame()
df_swsData['time'] = pd.to_datetime(swsDate, unit='s')
df_swsData['speed'] = pd.to_numeric(swsSpeed, errors='coerce').astype(np.float32)
df_swsData['direction'] = pd.to_numeric(swsDir, errors='coerce').astype(np.float32)
df_swsData['gust'] = pd.to_numeric(swsGust, errors='coerce').astype(np.float32)
df_swsData['lull'] = pd.to_numeric(swsLull, errors='coerce').astype(np.float32)
df_swsData['temperature'] = pd.to_numeric(swsTemp, errors='coerce').astype(np.float32)

# Save to csv
df_swsData.to_csv('swsWind.csv')
