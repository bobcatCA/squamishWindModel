"""Initialise the SQLite weather database.

Run once before first use:
    python db_init.py [path/to/weather_data_hourly.db]
"""

import os
import sqlite3
import sys
from dotenv import load_dotenv
from pathlib import Path

_CREATE_WEATHER = """
CREATE TABLE IF NOT EXISTS weather (
    datetime     INTEGER PRIMARY KEY,
    ballenasDegC REAL,
    ballenasKPa  REAL,
    comoxDegC    REAL,
    comoxKPa     REAL,
    comoxSky     TEXT,
    lillooetDegC REAL,
    lillooetKPa  REAL,
    pamDegC      REAL,
    pamKPa       REAL,
    pembertonDegC REAL,
    pembertonKPa  REAL,
    vancouverDegC REAL,
    vancouverKPa  REAL,
    vancouverSky  TEXT,
    victoriaDegC  REAL,
    victoriaKPa   REAL,
    victoriaSky   TEXT,
    whistlerDegC  REAL,
    whistlerKPa   REAL,
    whistlerSky   TEXT,
    speed         REAL,
    gust          REAL,
    lull          REAL,
    direction     REAL,
    temperature   REAL
)
"""


def init_db(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(_CREATE_WEATHER)
        conn.commit()
    print(f'Initialised {db_path}')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        load_dotenv()
        path = Path(os.getenv('WORKING_DIRECTORY', '.')) / 'weather_data_hourly.db'
    init_db(path)
