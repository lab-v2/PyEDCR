import os
import torch
import torchvision
import torch.utils.data
import pathlib
import typing

from models import ImageFolderWithName

granularities = {0: 'coarse',
                 1: 'fine'}


def get_transforms(train_or_val: str,
                   model_name: str,
                   train_folder_name: str) -> torchvision.transforms.Compose:
    if model_name == 'inception_v3':
        resize_num = 299
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]
    else:
        resize_num = 518 if 'h_14' in model_name else 224
        means = stds = [0.5] * 3

    return torchvision.transforms.Compose(
        ([torchvision.transforms.RandomResizedCrop(resize_num),
          torchvision.transforms.RandomHorizontalFlip()] if train_or_val == train_folder_name else
         [torchvision.transforms.Resize(int(resize_num / 224 * 256)),
          torchvision.transforms.CenterCrop(resize_num)]) +
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(means, stds)
         ])


def get_datasets(model_names: list[str],
                 cwd: typing.Union[str, pathlib.Path],
                 train_folder_name: str,
                 test_folder_name: str) -> (dict[str, ImageFolderWithName], int):
    data_dir = pathlib.Path.joinpath(cwd, '.')
    datasets = {f'{model_name}_{train_or_val}': ImageFolderWithName(root=os.path.join(data_dir, train_or_val),
                                                                    transform=get_transforms(
                                                                        train_or_val=train_or_val,
                                                                        model_name=model_name,
                                                                        train_folder_name=train_folder_name))
                for model_name in model_names for train_or_val in [train_folder_name, test_folder_name]}

    assert all(datasets[f'{model_name}_{train_folder_name}'].classes ==
               datasets[f'{model_name}_{test_folder_name}'].classes
               for model_name in model_names)

    num_train_examples = len(datasets[f'{model_names[0]}_{train_folder_name}'])
    num_test_examples = len(datasets[f'{model_names[0]}_{test_folder_name}'])
    train_ratio = round(num_train_examples / (num_train_examples + num_test_examples) * 100, 2)
    test_ratio = round(num_test_examples / (num_train_examples + num_test_examples) * 100, 2)

    print(f"Total num of examples: train: {num_train_examples} ({train_ratio}%), "
          f"test: {num_test_examples} ({test_ratio}%)")

    classes = datasets[f'{model_names[0]}_{train_folder_name}'].classes
    print(f'Classes: {classes}')
    n = len(classes)

    return datasets, n


def get_loaders(datasets: dict[str, ImageFolderWithName],
                batch_size: int,
                model_names: list[str],
                train_folder_name: str,
                test_folder_name: str) -> dict[str, torch.utils.data.DataLoader]:
    return {f"{model_name}_{train_or_val}": torch.utils.data.DataLoader(
        dataset=datasets[f'{model_name}_{train_or_val}'],
        batch_size=batch_size,
        shuffle=train_or_val == train_folder_name)
        for model_name in model_names for train_or_val in [train_folder_name, test_folder_name]}
