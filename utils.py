import os
import sys
import pathlib
import datetime
import math
import typing


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
        return f"{math.floor(hours)} hour{'s' if hours > 1 else ''}"
    elif minutes > 0:
        return f"{math.floor(minutes)} minute{'s' if minutes > 1 else ''}"
    else:
        return f"{math.floor(seconds)} second{'s' if seconds > 1 else ''}"


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


def green_text(s: typing.Union[str, float]) -> str:
    return colored_text('green')(s)


def red_text(s: typing.Union[str, float]) -> str:
    return colored_text('red')(s)


def blue_text(s: typing.Union[str, float]) -> str:
    return colored_text('blue')(s)


def format_integer(n):
    if n == 0:
        return "0"

    # Extracting the sign
    sign = "-" if n < 0 else ""
    n = abs(n)

    # Finding 'a' and 'b'
    b = 0
    while n >= 10:
        n /= 10
        b += 1
    a = int(n)

    # Constructing the string representation
    if b == 0:
        return f"{sign}{a}"
    else:
        return f"{sign}{a} * 10^{b}"


