import os
import sys
from datetime import timedelta


def format_seconds(seconds: int):
    # Create a timedelta object with the given seconds
    time_delta = timedelta(seconds=seconds)

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
    return Path(__file__).parent.parent.name == 'PycharmProjects'

