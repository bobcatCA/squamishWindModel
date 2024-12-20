import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('mergedOnSpeed.csv')
df = df.iloc[294162:-1]
df['time'] = pd.to_datetime(df['time'])
# df = df[df['speed'] > 8]
# df = df[(220 < df['direction']) & (245 > df['direction'])]
# df = df[df['lillooetDegC'] - df['vancouverDegC'] > 7]


x_range = df['time']
y_rangeWind = {
    "Speed": df['speed'],
    "Gust": df['gust'],
    "Lull": df['lull'],
    "Direction": df['direction']
}

y_rangeTemp = {
    "Comox_Temperature": df['comoxDegC'],
    "Pam_Temperature": df['pamDegC']
}

y_rangeWeather = {
    # "Comox_Barometric": df['comoxKPa'],
    # "Pam_Barometric": df['pamKPa'],
    'Delta: Comox-Pam': df['comoxKPa'] - df['pamKPa']
}

fig, axs = plt.subplots(3, 1, sharex=True)

# Plot Speed, lull, gust on the top plot, then Direction on the bottom
for series in y_rangeWind:
    if series == 'Direction':
        plotNumber = 1
    else: plotNumber = 0

    axs[plotNumber].plot(x_range, y_rangeWind[series], label=series)
    pass

for series in y_rangeTemp:
    axs[0].plot(x_range, y_rangeTemp[series], label=series)
    pass

for series in y_rangeWeather:
    axs[2].plot(x_range, y_rangeWeather[series], label=series)
    pass

# Add labels and legend
axs[0].set_ylabel('Wind Speed (knots)')
axs[0].legend()
axs[1].set_ylabel('Direction (degrees)')
axs[1].legend()
axs[2].legend()
axs[2].set_ylabel('BP (kPa)')
plt.legend()
plt.show()
