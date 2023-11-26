import os
import torch
import torchvision
import torch.utils.data
import pathlib
import typing

import models

granularities = {0: 'coarse',
                 1: 'fine'}


def get_transforms(train_or_test: str) -> torchvision.transforms.Compose:
    resize_num = 224
    means = stds = [0.5] * 3

    return torchvision.transforms.Compose(
        ([torchvision.transforms.RandomResizedCrop(resize_num),
          torchvision.transforms.RandomHorizontalFlip()] if train_or_test == 'train' else
         [torchvision.transforms.Resize(int(resize_num / 224 * 256)),
          torchvision.transforms.CenterCrop(resize_num)]) +
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(means, stds)
         ])


def get_datasets(cwd: typing.Union[str, pathlib.Path],
                 granularity: str
                 ) -> (dict[str, models.ImageFolderWithName], int):
    data_dir = pathlib.Path.joinpath(cwd, '.')
    datasets = {f'{train_or_test}': models.ImageFolderWithName(root=os.path.join(data_dir,
                                                                                 f'{train_or_test}_{granularity}'),
                                                               transform=get_transforms(train_or_test=train_or_test))
                for train_or_test in ['train', 'test']}

    num_train = len(datasets['train'])
    num_test = len(datasets['test'])
    print(f'Total number of train images: {num_train}\nTotal number of test images: {num_test}')

    classes = datasets['train'].classes
    num_classes = len(classes)

    return datasets, num_classes


def get_loaders(datasets: dict[str, models.ImageFolderWithName],
                batch_size: int) -> dict[str, torch.utils.data.DataLoader]:
    return {train_or_test: torch.utils.data.DataLoader(
        dataset=datasets[train_or_test],
        batch_size=batch_size,
        shuffle=train_or_test == 'train') for train_or_test in ['train', 'test']}
