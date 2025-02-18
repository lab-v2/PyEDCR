import torch.utils.data

from src.PyEDCR.neural_fine_tuning import models
from src.PyEDCR.utils import utils


def print_fine_tuning_initialization(fine_tuner: models.FineTuner,
                                     num_epochs: int,
                                     lr: float,
                                     device: torch.device,
                                     early_stopping: bool = False):
    early_stopping_str = ' maximal' if early_stopping else ''
    print(f'\nFine-tuning {fine_tuner} with {utils.format_integer(len(fine_tuner))} '
          f'parameters for {num_epochs}{early_stopping_str} epochs using lr={lr} on {device}...')
