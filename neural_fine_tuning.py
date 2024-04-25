import torch.utils.data
from tqdm import tqdm

import models
import utils


def get_fine_tuning_batches(train_loader: torch.utils.data.DataLoader,
                            num_batches: int,
                            debug: bool = False):

    batches = tqdm(enumerate([list(train_loader)[0]] if debug else train_loader, 0),
                   total=num_batches)

    return batches



def print_fine_tuning_initialization(fine_tuner: models.FineTuner,
                                     num_epochs: int,
                                     lr: float,
                                     device: torch.device):
    print(f'\nFine-tuning {fine_tuner} with {utils.format_integer(len(fine_tuner))} '
          f'parameters for {num_epochs} epochs using lr={lr} on {device}...')
