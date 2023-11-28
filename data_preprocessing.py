import os
import torch
import torchvision
import pandas as pd
import torch.utils.data
import pathlib
import typing

data_file_path = rf'data/WEO_Data_Sheet.xlsx'
dataframes_by_sheet = pd.read_excel(data_file_path, sheet_name=None)
fine_grain_results_df = dataframes_by_sheet['Fine-Grain Results']
fine_grain_classes = sorted(fine_grain_results_df['Class Name'].to_list())
coarse_grain_results_df = dataframes_by_sheet['Coarse-Grain Results']
coarse_grain_classes = sorted(coarse_grain_results_df['Class Name'].to_list())


def get_fine_to_coarse() -> (dict[str, str], dict[int, int]):
    fine_to_coarse = {}
    fine_to_course_idx = {}
    training_df = dataframes_by_sheet['Training']

    assert (set(training_df['Fine-Grain Ground Truth'].unique().tolist()).intersection(fine_grain_classes)
            == set(fine_grain_classes))

    for fine_grain_class_idx, fine_grain_class in enumerate(fine_grain_classes):
        curr_fine_grain_training_data = training_df[training_df['Fine-Grain Ground Truth'] == fine_grain_class]
        assert curr_fine_grain_training_data['Course-Grain Ground Truth'].nunique() == 1

        coarse_grain_class = curr_fine_grain_training_data['Course-Grain Ground Truth'].iloc[0]
        coarse_grain_class_idx = coarse_grain_classes.index(coarse_grain_class)

        fine_to_coarse[fine_grain_class] = coarse_grain_class
        fine_to_course_idx[fine_grain_class_idx] = coarse_grain_class_idx

    return fine_to_coarse, fine_to_course_idx


fine_to_coarse, fine_to_course_idx = get_fine_to_coarse()


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


class ImageFolderWithName(torchvision.datasets.ImageFolder):
    def __getitem__(self,
                    index: int) -> (torch.tensor, int, str):
        path, y = self.samples[index]

        y_coarse_grain = fine_to_course_idx[y]

        x = self.loader(path)

        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        name = os.path.basename(path)
        folder_path = os.path.basename(os.path.dirname(path))

        x_identifier = f'{folder_path}/{name}'

        return x, y, x_identifier, y_coarse_grain


def get_datasets(cwd: typing.Union[str, pathlib.Path],
                 ) -> (dict[str, ImageFolderWithName], int):
    data_dir = pathlib.Path.joinpath(cwd, '.')
    datasets = {f'{train_or_test}': ImageFolderWithName(root=os.path.join(data_dir,
                                                                          f'{train_or_test}_fine'),
                                                        transform=get_transforms(train_or_test=train_or_test))
                for train_or_test in ['train', 'test']}

    print(f"Total number of train images: {len(datasets['train'])}\n"
          f"Total number of test images: {len(datasets['test'])}")

    classes = datasets['train'].classes
    assert classes == sorted(classes) == fine_grain_classes

    num_fine_grain_classes = len(classes)
    num_coarse_grain_classes = len(coarse_grain_classes)

    return datasets, num_fine_grain_classes, num_coarse_grain_classes


def get_loaders(datasets: dict[str, ImageFolderWithName],
                batch_size: int) -> dict[str, torch.utils.data.DataLoader]:
    return {train_or_test: torch.utils.data.DataLoader(
        dataset=datasets[train_or_test],
        batch_size=batch_size,
        shuffle=train_or_test == 'train') for train_or_test in ['train', 'test']}
