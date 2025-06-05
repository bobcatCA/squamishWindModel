import json
import pandas as pd
import platform
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import tempfile
import time

def get_sws_df(dates):
    # Set up the WebDriver (e.g., for Chrome)
    # Make sure to provide the path to your Chrome WebDriver if necessary
    # Create a temporary directory for the user data dir
    temp_user_data_dir = tempfile.mkdtemp()
    chrome_options = Options() 
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--headless')  # Optional: if running without GUI

    if 'aarch64' in platform.machine():
        chrome_options.binary_location = '/usr/bin/chromium-browser'
        service = Service(executable_path='/usr/bin/chromedriver')
        driver = webdriver.Chrome(service=service, options=chrome_options)

    elif 'x86_64' in platform.machine():
        chrome_options.add_argument(f'--user-data-dir={temp_user_data_dir}')
        driver = webdriver.Chrome(options=chrome_options)

    else:
        driver = None
        pass

    # Initialize empty start points and loop throuhgh dates, adding data for each
    first_date = True
    df = None
    for date in dates:

        string1 = 'https://squamishwindsports.com/wind-data/getmet.php?wind_src=spit&reqdate='
        string2 = '&reqtime=0'
        # date = date + pd.Timedelta(days=1)
        url = string1 + str(date) + string2
        # Load the page
        # url = 'https://squamishwindsports.com/wind-data/getmet.php?wind_src=spit&reqdate=2024-09-10&reqtime=0'
        driver.get(url)

        # Allow time for the page to fully load
        time.sleep(2)

        # Grab the page source
        page_source = driver.page_source

        # Parse the page with BeautifulSoup
        soup = BeautifulSoup(page_source, 'html.parser')

        # Find the data related to the wind chart
        # This will depend on how the data is structured on the page.
        # For demonstration, I'm using a placeholder for the data extraction.
        # Replace 'your_selector' with the actual selector used to target the chart data.
        chart_data = driver.find_element(By.CSS_SELECTOR, 'body').text
        if not chart_data.strip():
            print(f'warning: empty chart data for {date}')
        else:
            try:
                data_json = json.loads(chart_data)
                if first_date:
                    df = pd.DataFrame(data_json)
                    first_date = False
                else:
                    df_date = pd.DataFrame(data_json)
                    df = pd.concat([df, df_date])

                # with open(str(date) + '.json', 'w', encoding='utf-8') as f:
                #     json.dump(data_json, f, ensure_ascii=False, indent=4)
                #
            except json.JSONDecodeError:
                print(f'Error in {date} is not valid JSON format')
        pass  # for loop

    if df.empty:
        pass
    else:
        dict_names = {
            'dt': 'datetime',
            'ws': 'speed',
            'wd': 'direction',
            'wg': 'gust',
            'wl': 'lull',
            't': 'temperature'
        }
        df.rename(columns=dict_names, inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce')
        df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
        df[['direction', 'gust', 'lull']] = df[['direction', 'gust', 'lull']].astype(float)
        df.sort_values('datetime', inplace=True)

        # Close the WebDriver
        driver.quit()
    return df

if __name__=='__main__':
    # date_of_query = '2024-09-10'
    # df_sws = get_sws_df(date_of_query)
    pass
