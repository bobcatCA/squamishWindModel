import json
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time

# Set up the WebDriver (e.g., for Chrome)
# Make sure to provide the path to your Chrome WebDriver if necessary
driver = webdriver.Chrome()

startDate = pd.to_datetime('2016-09-21').date()
endDate = pd.to_datetime('2024-09-10').date()
date = startDate

string1 = 'https://squamishwindsports.com/wind-data/getmet.php?wind_src=spit&reqdate='
string2 = '&reqtime=0'

while date <= endDate:

    date = date + pd.Timedelta(days=1)
    # Don't pull any data before May, or after September.
    if date.month < 5 or date.month > 9:
        continue
    url = string1 + str(date) + string2
    print(url)
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
            with open(str(date) + '.json', 'w', encoding='utf-8') as f:
                json.dump(data_json, f, ensure_ascii=False, indent=4)

        except json.JSONDecodeError:
            print(f'Error in {date} is not valid JSON format')
    pass

# Close the WebDriver
driver.quit()
