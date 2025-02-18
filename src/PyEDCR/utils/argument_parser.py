import os
import sys
import argparse


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run f-EDR pipeline")
    parser.add_argument('--data_str',
                        type=str,
                        help='Dataset to use')
    parser.add_argument('--main_model_name',
                        type=str,
                        help='Main model to detect errors on')
    parser.add_argument('--secondary_model_name',
                        type=str,
                        help='Secondary model to define conditions with')
    parser.add_argument('--main_lr',
                        type=float,
                        help='Learning rate for the main model')
    parser.add_argument('--secondary_lr',
                        type=float,
                        help='Learning rate for the secondary model')
    parser.add_argument('--binary_lr',
                        type=float,
                        help='Learning rate for the binary model')
    parser.add_argument('--original_num_epochs',
                        type=int,
                        help='Number of epochs for the original training')
    parser.add_argument('--secondary_num_epochs',
                        type=int,
                        help='Number of epochs for the secondary model training')
    parser.add_argument('--binary_num_epochs',
                        type=int,
                        help='Number of epochs for the binary model training')

    # Check if running in a Jupyter notebook
    if 'ipykernel_launcher' in sys.argv[0] or 'jupyter' in os.path.basename(sys.executable):
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()

    return args
