import os
import typing
import time
import google_auth_oauthlib.flow
import google.auth.transport.requests
import google.oauth2.credentials
import googleapiclient.discovery
import googleapiclient.errors


def initiate_api():
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


def exponential_backoff(func: typing.Callable):
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
def update_sheet(spreadsheet_id: str,
                 range_: str,
                 body: typing.Dict[str, typing.List[typing.List[typing.Union[float, str]]]],
                 sheet: googleapiclient.discovery.Resource = None):
    """Function to update Google Sheet and handle retries on rate limits."""

    if sheet is None:
        sheet = initiate_api()

    result = sheet.values().update(
        spreadsheetId=spreadsheet_id,
        range=range_,
        valueInputOption='USER_ENTERED',
        body=body).execute()

    print(f"{result.get('updatedCells')} cell updated.")


@exponential_backoff
def find_empty_rows_in_column(sheet_id: str,
                              tab_name: str,
                              column: str,
                              sheet: googleapiclient.discovery.Resource = None):
    if sheet is None:
        sheet = initiate_api()

    # Fetch the column data
    result = sheet.values().get(spreadsheetId=sheet_id,
                                range=f'{tab_name}!{column}:{column}').execute()
    values = result.get('values', [])

    total_value_num = len(values)

    # Identify empty rows
    empty_row_indices = []
    for index, value in enumerate(values, start=1):  # Starts counting from 1 (Google Sheets row numbers)
        if not value:  # If the list is empty, the row is empty
            empty_row_indices.append(index)

    return empty_row_indices, total_value_num
