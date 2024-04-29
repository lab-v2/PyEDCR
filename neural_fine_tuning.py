import torch.utils.data

import models
import utils
import data_preprocessing


def print_fine_tuning_initialization(fine_tuner: models.FineTuner,
                                     num_epochs: int,
                                     lr: float,
                                     device: torch.device,
                                     experiment_name: str = None):
    print(f'\nFine-tuning {fine_tuner} with {utils.format_integer(len(fine_tuner))} '
          f'parameters for {num_epochs} epochs for {experiment_name if experiment_name is not None else ""} '
          f'using lr={lr} on {device}...')
