import pathlib

from google.auth.environment_vars import CREDENTIALS

ROOT_PATH = pathlib.Path(__file__).parent.parent.parent.parent
DATA_FOLDER = rf'{ROOT_PATH}/data'
RESULTS_FOLDER = rf'{ROOT_PATH}/results'
BINARY_RESULTS_FOLDER = rf'{ROOT_PATH}/binary_results'
CREDENTIALS_FOLDER = rf'{ROOT_PATH}/credentials'
