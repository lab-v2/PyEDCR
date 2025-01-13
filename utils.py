import os
import sys
import pathlib
import datetime
import math
import typing


def format_seconds(seconds: int):
    """
    Formats a given number of seconds into a human-readable string indicating the largest
    unit of time (hours, minutes, or seconds) and the corresponding value. This function
    utilizes a `datetime.timedelta` object for precise time arithmetic and makes use of
    Python's built-in methods for conversion to readable format.

    This function helps in converting raw time durations into user-friendly textual
    representations, such as "2 hours", "45 minutes", or "30 seconds", depending on the
    input duration.

    :param seconds: The time duration in seconds to format
    :return: A string representation of the time duration in a human-readable form, such as
             "x hours", "x minutes", or "x seconds"
    """
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


def create_directory(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f'Created {directory}')


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
    """
    Formats an integer into a string representation including its sign and scientific
    notation (if necessary). The function handles positive, negative, and zero values
    appropriately before determining the base (`a`) and exponent (`b`) for numbers
    greater than or equal to 10.

    :param n: Integer to be formatted
    :return: The formatted string representation of the integer
    """
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


def expand_ranges(tuples: list[tuple[int, int]]) -> list[int]:
    """
    Expands a list of tuples of integers into a list containing all numbers within the ranges.
    :param tuples: A list of tuples of integers representing ranges (start, end).
    :returns: A list containing all numbers within the specified ranges.
    """

    result = []
    for start, end in tuples:
        # Ensure start is less than or equal to end
        if start > end:
            start, end = end, start
        # Add all numbers from start (inclusive) to end (exclusive)
        result.extend(range(start, end + 1))
    return result
