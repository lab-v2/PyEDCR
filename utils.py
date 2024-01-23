import os
import sys
import pathlib
import datetime
from typing import Union
import dropbox

CHUNK_SIZE = 4 * 1024 * 1024


def format_seconds(seconds: int):
    # Create a timedelta object with the given seconds
    time_delta = datetime.timedelta(seconds=seconds)

    # Use the total_seconds() method to get the total number of seconds
    total_seconds = time_delta.total_seconds()

    # Use divmod to get the hours and minutes
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Create the formatted string
    if hours > 0:
        return f"{int(hours)} hour{'s' if hours > 1 else ''}"
    elif minutes > 0:
        return f"{int(minutes)} minute{'s' if minutes > 1 else ''}"
    else:
        return f"{int(seconds)} second{'s' if seconds > 1 else ''}"


# Function to create a directory if it doesn't exist
def create_directory(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f'Created {directory}')


def is_running_in_colab() -> bool:
    """
    Check if the code is running in Google Colab.
    Returns:
        True if running in Google Colab, False otherwise.
    """

    return 'google.colab' in sys.modules


def is_local() -> bool:
    return pathlib.Path(__file__).parent.parent.name == 'PycharmProjects'


def is_debug_mode():
    # Check if the script was launched with the -d or --debug flag
    return is_local() and (any(arg in sys.argv for arg in ['-d', '--debug']) or sys.gettrace() is not None)


def colored_text(color: str):
    index = {'red': 1,
             'green': 2,
             'blue': 4}[color]

    return lambda s: f"\033[9{index}m{s}\033[0m"


def green_text(s: Union[str, float]) -> str:
    return colored_text('green')(s)


def red_text(s: Union[str, float]) -> str:
    return colored_text('red')(s)


def blue_text(s: Union[str, float]) -> str:
    return colored_text('blue')(s)

class TransferData:
    def __init__(self, access_token):
        self.access_token = access_token

    def upload_file(self, file_from, file_to):
        """upload a file to Dropbox using API v2
        """
        dbx = dropbox.Dropbox(self.access_token)

        f = open(file_from, "rb")

        file_size = os.path.getsize(file_from)

        if file_size <= CHUNK_SIZE:
        
            dbx.files_upload(f.read(), file_to)
        
        else:
        
            upload_session_start_result = dbx.files_upload_session_start(f.read(CHUNK_SIZE))
            cursor = dropbox.files.UploadSessionCursor(session_id=upload_session_start_result.session_id,
                                                       offset=f.tell())
            commit = dropbox.files.CommitInfo(path=file_to)
        
            while f.tell() < file_size:
                if ((file_size - f.tell()) <= CHUNK_SIZE):
                    dbx.files_upload_session_finish(f.read(CHUNK_SIZE),
                                                    cursor,
                                                    commit)
                else:
                    dbx.files_upload_session_append(f.read(CHUNK_SIZE),
                                                    cursor.session_id,
                                                    cursor.offset)
                    cursor.offset = f.tell()

    def download_file(self, file_from, file_to):
        """Download a file from Dropbox to a local file."""
        dbx = dropbox.Dropbox(self.access_token)

        try:
            # Download the file from Dropbox
            metadata, response = dbx.files_download(file_from)

            # Save the file locally
            with open(file_to, 'wb') as f:
                f.write(response.content)

            print(f"File downloaded successfully: {file_to}")

        except dropbox.exceptions.ApiError as err:
            print(f"Error downloading file: {err}")