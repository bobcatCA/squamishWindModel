import sqlite3

# Connect to (or create) the database
conn = sqlite3.connect('weather_data_hourly.db')
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS weather (
    datetime TEXT PRIMARY KEY,
    comoxSky REAL,
    vancouverSky REAL,
    victoriaSky REAL,
    whistlerSky REAL,
    comoxDegC REAL,
    lillooetDegC REAL,
    pembertonDegC REAL,
    vancouverDegC REAL,
    victoriaDegC REAL,
    whistlerDegC REAL,
    comoxKPa REAL,
    vancouverKPa REAL,
    lillooetKPa REAL,
    pamKPa REAL,
    ballenasKPa REAL,
    speed REAL,
    gust REAL,
    lull REAL,
    direction REAL
)
""")

conn.commit()
conn.close()
#
# # Connect to (or create) the database
# conn = sqlite3.connect('weather_data_daily.db')
# cursor = conn.cursor()
#
# # Create table if it doesn't exist
# cursor.execute("""
# CREATE TABLE IF NOT EXISTS weather (
#     datetime REAL PRIMARY KEY,
#     comoxSky REAL,
#     vancouverSky REAL,
#     victoriaSky REAL,
#     whistlerSky REAL,
#     comoxDegC REAL,
#     lillooetDegC REAL,
#     pembertonDegC REAL,
#     vancouverDegC REAL,
#     victoriaDegC REAL,
#     whistlerDegC REAL,
#     comoxKPa REAL,
#     vancouverKPa REAL,
#     lillooetKPa REAL,
#     pamKPa REAL,
#     ballenasKPa REAL,
#     speed REAL,
#     speed_variability REAL,
#     dir_score REAL
# )
# """)
#
# conn.commit()
# conn.close()