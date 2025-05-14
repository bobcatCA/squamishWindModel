import os
import logging
from pathlib import Path
from io import StringIO
import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def setup_logging(log_file):
    """
    This function configures the logging module to log messages with INFO level or higher
    to the given file, with a standard timestamped format.

    :param log_file: (Path or str) The path to the log file where logs should be written.
    :return: None
    """

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def get_session(api_key, retries=3, backoff_factor=2):
    """
    Create and return a configured requests session with authentication headers.
    This helps reuse TCP connections for multiple uploads, improving performance.

    :param api_key: (str) The API key to include in the session headers for authentication.
    :param retries: (int) Number of times to attempt the session request
    :param backoff_factor: (int) Relative measure of sleep() between retries
    :return: session: A session object with the 'X-API-KEY' header set.
    """

    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        status_forcelist=[429, 500, 502, 503, 504],
        backoff_factor=backoff_factor
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount('https://', adapter)
    session.headers.update({'X-API-KEY': api_key})
    return session


def upload_csv(filepath, session, url):
    """
    Upload a CSV file to the specified URL using a POST request. Exceptions are caught
    and logged if the upload fails.

    :param filepath: (Path or str), The path to the CSV file to upload.
    :param session: (requests.Session), An existing requests session with headers pre-set.
    :param url: (str), the destination URL to send the file
    :return:
    """

    # Read .csv file into dataframe, create csv buffer
    df = pd.read_csv(filepath)
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    files = {'file': ('data.csv', csv_buffer.getvalue())}
    headers = {'X-Filename': filepath.name}

    # Send csv with headers to the destination
    try:
        response = session.post(url, files=files, headers=headers, verify=False)
        if response.ok:
            logging.info(f"Sent {filepath.name} - Status: {response.status_code} - Response: {response.text}")
        else:
            logging.warning(f"Failed to send {filepath.name} - Status: {response.status_code} - Response: {response.text}")
    except requests.RequestException as e:
        logging.error(f"Exception when sending {filepath.name} - Error: {e}")


def main():
    """
    Main entry point of the script:
    -Loads environment variables
    -Sets up logging
    -Creates session with API authentication
    -Iterates through csv files and uploads them
    :return:
    """

    # Load environment
    load_dotenv()
    working_directory = Path(os.getenv('WORKING_DIRECTORY'))
    log_directory = os.getenv('LOG_DIRECTORY')
    url = os.getenv('UPLOAD_URL')
    api_key = os.getenv('API_KEY')

    # Set up logging and get a session object
    setup_logging(log_directory)
    session = get_session(api_key)

    csv_filenames = os.getenv('CSV_FILES').split(',')

    for filename in csv_filenames:
        upload_csv(working_directory / filename, session, url)


if __name__ == '__main__':
    main()
