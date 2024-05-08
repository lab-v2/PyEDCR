import os
import typing
import time
import google_auth_oauthlib.flow
import google.auth.transport.requests
import google.oauth2.credentials
import googleapiclient.discovery
import googleapiclient.errors
import numpy as np

spreadsheet_id = '1JVLylVDMcYZgabsO2VbNCJLlrj7DSlMxYhY6YwQ38ck'


def initiate_api() -> googleapiclient.discovery.Resource:
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time.
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    if os.path.exists("token.json"):
        creds = (google.oauth2.credentials.Credentials.from_authorized_user_file(filename="token.json",
                                                                                 scopes=scopes))

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(google.auth.transport.requests.Request())
        else:
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                client_secrets_file="credentials.json",
                scopes=scopes)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    service = googleapiclient.discovery.build(serviceName="sheets",
                                              version="v4",
                                              credentials=creds)
    sheet = service.spreadsheets()

    return sheet


__sheet: googleapiclient.discovery.Resource = initiate_api()


def get_sheet_tab_name(main_model_name: str,
                       data_str: str,
                       secondary_model_name: str = None,
                       binary: bool = False) -> str:

    main_model_name_str = {'vit_b_16': 'VIT_b_16',
                           'dinov2_vits14': 'DINO V2 VIT14_s',
                           'tresnet_m': 'Tresnet M'}[main_model_name]
    data_set_str = {'military_vehicles': 'Military Vehicles',
                    'imagenet': 'ImageNet',
                    'openimage': 'OpenImage'}[data_str]
    secondary_model_str = ((' with ' + "DINO V2 VIT14_l" if data_str == 'imagenet' else 'VIT_l_16')
                           if secondary_model_name is not None else '')
    binary_str = ' with Binary' if binary else ''

    return f"{main_model_name_str} on {data_set_str}{binary_str}{secondary_model_str}"


def exponential_backoff(func: typing.Callable) -> typing.Callable:
    """Decorator to retry with exponential backoff when rate limited."""

    def wrapper(*args, **kwargs):
        wait = 30  # Start with 30 seconds
        while True:
            try:
                return func(*args, **kwargs)
            except googleapiclient.errors.HttpError as e:
                error_code = e.resp.status
                if error_code == 429:
                    print(f"Rate limit exceeded, waiting {wait} seconds...")
                    time.sleep(wait)
                    wait *= 1.1  # Exponential backoff
                else:
                    print(e)
                    time.sleep(60)

    return wrapper


@exponential_backoff
def update_sheet(range_: str,
                 body: typing.Dict[str, typing.List[typing.List[typing.Union[float, str]]]]):
    """Function to update Google Sheet and handle retries on rate limits."""

    result = __sheet.values().update(
        spreadsheetId=spreadsheet_id,
        range=range_,
        valueInputOption='USER_ENTERED',
        body=body).execute()

    print(f"{result.get('updatedCells')} cell updated to {range_}")


@exponential_backoff
def find_empty_rows_in_column(sheet_tab_name: str,
                              column_letter: str):
    # Fetch the column data
    values = __sheet.values().get(spreadsheetId=spreadsheet_id,
                                  range=f'{sheet_tab_name}!{column_letter}:{column_letter}').execute().get('values', [])

    total_value_num = len(values)

    # Identify empty rows
    empty_row_indices = []
    for index, value in enumerate(values, start=1):  # Starts counting from 1 (Google Sheets row numbers)
        if not value:  # If the list is empty, the row is empty
            empty_row_indices.append(index)

    return empty_row_indices, total_value_num


@exponential_backoff
def get_values_from_columns(sheet_tab_name: str,
                            column_letters: typing.List[str]):
    ranges = [f'{sheet_tab_name}!{letter}2:{letter}' for letter in column_letters]
    response = __sheet.values().batchGet(
        spreadsheetId=spreadsheet_id,
        ranges=ranges
    ).execute()

    return [np.array([e[0].strip('%') if e[0] != 'None' else 0
                      for e in response_i.get('values', []) if e[0] != '#N/A'],
                     dtype=float) for response_i in response['valueRanges']]


@exponential_backoff
def get_maximal_epsilon(sheet_tab_name: str):
    # Specify the separate ranges to fetch
    data_range_b_to_e = f'{sheet_tab_name}!B2:E'
    data_range_g = f'{sheet_tab_name}!G2:G'
    column_a_range = f'{sheet_tab_name}!A2:A'

    # Fetch the data using batchGet
    response = __sheet.values().batchGet(
        spreadsheetId=spreadsheet_id,
        ranges=[data_range_b_to_e, data_range_g, column_a_range]
    ).execute()

    # Extract the values for each range
    data_values_b_to_e = response['valueRanges'][0].get('values', [])
    data_values_g = response['valueRanges'][1].get('values', [])
    column_a_values = response['valueRanges'][2].get('values', [])

    # Standardize the length of each row
    max_length_b_to_e = max((len(row) for row in data_values_b_to_e), default=0)
    data_values_b_to_e = [row + [None] * (max_length_b_to_e - len(row)) for row in data_values_b_to_e]

    max_length_g = max((len(row) for row in data_values_g), default=0)
    data_values_g = [row + [None] * (max_length_g - len(row)) for row in data_values_g]

    # Convert data to NumPy arrays, handling percentages and missing values
    data_array_b_to_e = np.array(
        [[float(item.strip('%')) if isinstance(item, str) and item else 0 for item in row] for row in
         data_values_b_to_e])
    data_array_g = np.array([[float(row[0]) if row and row[0] else 0] for row in data_values_g])

    # Concatenate columns B-E with column G
    data_array = np.hstack((data_array_b_to_e, data_array_g))

    # Calculate the sum of each row using NumPy's sum function along axis 1 (rows)
    row_sums = np.sum(data_array, axis=1)

    # Find the index of the row with the maximum sum
    max_index = np.argmax(row_sums)

    # Retrieve the value from column A for the row with the maximum sum
    if max_index < len(column_a_values):
        return column_a_values[max_index][0]
    else:
        return None
