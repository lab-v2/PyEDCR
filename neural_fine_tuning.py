import torch.utils.data


import models
import utils



def print_fine_tuning_initialization(fine_tuner: models.FineTuner,
                                     num_epochs: int,
                                     lr: float,
                                     device: torch.device):
    print(f'\nFine-tuning {fine_tuner} with {utils.format_integer(len(fine_tuner))} '
          f'parameters for {num_epochs} epochs using lr={lr} on {device}...')
