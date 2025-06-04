from dotenv import load_dotenv
from pathlib import Path
import os
import subprocess


# Load environment variables
load_dotenv()
JSON_PATH = Path(os.getenv('WORKING_DIRECTORY'))
REMOTE_USER = Path(os.getenv('REMOTE_USER'))
REMOTE_HOST = Path(os.getenv('REMOTE_HOST'))
REMOTE_PATH = Path(os.getenv('REMOTE_PATH'))


def send_file_via_scp(local_path, remote_user, remote_host, remote_path):
    """
    :param local_path: Path object, pointing to Json file
    :param remote_user: Str, username for remote machine
    :param remote_host: Str, name of remote machine
    :param remote_path: Str, path to save file on remote machine
    :return: None
    """
    scp_command = ["scp", local_path, f"{remote_user}@{remote_host}:{remote_path}"]
    print(f"[>] Sending {local_path} to {remote_user}@{remote_host}:{remote_path}")
    result = subprocess.run(scp_command, capture_output=True, text=True)

    if result.returncode == 0:
        print("[✓] File sent successfully!")
    else:
        print("[✗] Failed to send file.")
        print("stderr:", result.stderr)


if __name__ == "__main__":
    json_filenames = ['hourly_speed_predictions.json', 'daily_speed_predictions.json']
    for file in json_filenames:
        json_file = JSON_PATH / file
        remote_dir = REMOTE_PATH / file
        send_file_via_scp(json_file, REMOTE_USER, REMOTE_HOST, remote_dir)
        # os.remove(json_file)  # Clean up the local JSON file
        # print("[✓] Done.")
        pass
