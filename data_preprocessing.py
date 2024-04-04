import os
import numpy as np
import torch
import torchvision
import pandas as pd
import torch.utils.data
import pathlib
import typing
import abc
import random

from typing import List

random.seed(42)

current_file_location = pathlib.Path(__file__).parent.resolve()
os.chdir(current_file_location)

data_file_path = rf'data/WEO_Data_Sheet.xlsx'
dataframes_by_sheet = pd.read_excel(data_file_path, sheet_name=None)
fine_grain_results_df = dataframes_by_sheet['Fine-Grain Results']
fine_grain_classes_str = sorted(fine_grain_results_df['Class Name'].to_list())
coarse_grain_results_df = dataframes_by_sheet['Coarse-Grain Results']
coarse_grain_classes_str = sorted(coarse_grain_results_df['Class Name'].to_list())
granularities_str = ['fine', 'coarse']

# Data for our use case

test_true_fine_data = np.load(r'data/test_fine/test_true_fine.npy')
test_true_coarse_data = np.load(r'data/test_coarse/test_true_coarse.npy')

train_true_fine_data = np.load(r'data/train_fine/train_true_fine.npy')
train_true_coarse_data = np.load(r'data/train_coarse/train_true_coarse.npy')

num_fine_grain_classes = len(fine_grain_classes_str)
num_coarse_grain_classes = len(coarse_grain_classes_str)


def is_monotonic(input_arr: np.array) -> bool:
    return np.all(input_arr[:-1] <= input_arr[1:])


def expand_ranges(tuples):
    """
    Expands a list of tuples of integers into a list containing all numbers within the ranges.
    :param tuples: A list of tuples of integers representing ranges (start, end).
    :returns: A list containing all numbers within the specified ranges.
    """

    result = []
    for start, end in tuples:
        # Ensure start is less than or equal to end
        if start > end:
            start, end = end, start
        # Add all numbers from start (inclusive) to end (exclusive)
        result.extend(range(start, end + 1))
    return result


def get_fine_to_coarse() -> (dict[str, str], dict[int, int]):
    """
    Creates and returns a dictionary with fine-grain labels as keys and their corresponding coarse grain-labels
    as values, and a dictionary with fine-grain label indices as keys and their corresponding coarse-grain label
    indices as values
    """

    output_fine_to_coarse = {}
    output_fine_to_course_idx = {}
    training_df = dataframes_by_sheet['Training']

    assert (set(training_df['Fine-Grain Ground Truth'].unique().tolist()).intersection(fine_grain_classes_str)
            == set(fine_grain_classes_str))

    for fine_grain_class_idx, fine_grain_class in enumerate(fine_grain_classes_str):
        curr_fine_grain_training_data = training_df[training_df['Fine-Grain Ground Truth'] == fine_grain_class]
        assert curr_fine_grain_training_data['Course-Grain Ground Truth'].nunique() == 1

        coarse_grain_class = curr_fine_grain_training_data['Course-Grain Ground Truth'].iloc[0]
        coarse_grain_class_idx = coarse_grain_classes_str.index(coarse_grain_class)

        output_fine_to_coarse[fine_grain_class] = coarse_grain_class
        output_fine_to_course_idx[fine_grain_class_idx] = coarse_grain_class_idx

    return output_fine_to_coarse, output_fine_to_course_idx


fine_to_coarse, fine_to_course_idx = get_fine_to_coarse()


class Granularity(typing.Hashable):
    def __init__(self,
                 g_str: str):
        self.g_str = g_str

    def __str__(self):
        return self.g_str

    def __hash__(self):
        return hash(self.g_str)

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


def get_ground_truths(test: bool,
                      K: List[int] = None,
                      g: Granularity = None) -> np.array:
    if K is None:
        K = [idx for idx in range(0, len(test_true_coarse_data))]
    if test:
        true_fine_data = test_true_fine_data
        true_coarse_data = test_true_coarse_data
    else:
        true_fine_data = train_true_fine_data
        true_coarse_data = train_true_coarse_data

    if g is None:
        return true_fine_data[K], true_coarse_data[K]
    else:
        return true_fine_data[K] if str(g) == 'fine' else true_coarse_data[K]


granularities = {g_str: Granularity(g_str=g_str) for g_str in granularities_str}


class Label(typing.Hashable, abc.ABC):
    def __init__(self,
                 l_str: str,
                 index: int):
        self._l_str = l_str
        self._index = index
        self._g = None

    def __str__(self):
        return self._l_str

    @property
    def index(self):
        return self._index

    @property
    def g(self):
        return self._g

    def __hash__(self):
        return hash(f'{self.g}_{self._l_str}')

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class FineGrainLabel(Label):
    def __init__(self,
                 l_str: str):
        super().__init__(l_str=l_str,
                         index=fine_grain_classes_str.index(l_str))
        assert l_str in fine_grain_classes_str
        self.__correct_coarse = fine_to_coarse[l_str]
        self._g = granularities['fine']

    @classmethod
    def with_index(cls,
                   l_index: int):
        l = fine_grain_classes_str[l_index]
        instance = cls(l_str=l)

        return instance


class CoarseGrainLabel(Label):
    def __init__(self,
                 l_str: str):
        super().__init__(l_str=l_str,
                         index=coarse_grain_classes_str.index(l_str))
        assert l_str in coarse_grain_classes_str
        self.correct_fine = coarse_to_fine[l_str]
        self._g = granularities['coarse']

    @classmethod
    def with_index(cls,
                   i_l: int):
        l = coarse_grain_classes_str[i_l]
        instance = cls(l_str=l)

        return instance


def get_num_inconsistencies(fine_labels: typing.Union[np.array, torch.Tensor],
                            coarse_labels: typing.Union[np.array, torch.Tensor]) -> int:
    inconsistencies = 0

    if isinstance(fine_labels, torch.Tensor):
        fine_labels = np.array(fine_labels.cpu())
        coarse_labels = np.array(coarse_labels.cpu())

    for fine_prediction, coarse_prediction in zip(fine_labels, coarse_labels):
        if fine_to_course_idx[fine_prediction] != coarse_prediction:
            inconsistencies += 1

    return inconsistencies


def get_dataset_transforms(train_or_test: str,
                           vit_model_name='vit_b_16',
                           error_fixing: bool = False,
                           weight: str = 'DEFAULT') -> torchvision.transforms.Compose:
    """
    Returns the transforms required for the VIT for training or test datasets
    """

    resize_num = 518 if vit_model_name == 'vit_h_14' else (224 if weight == 'DEFAULT' else 512)
    means = stds = [0.5] * 3

    standard_transforms = [torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize(means, stds)]
    train_transforms = [torchvision.transforms.RandomResizedCrop(resize_num),
                        torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.RandomRotation(15),  # Random rotation
                        torchvision.transforms.RandomVerticalFlip(),  # Random vertical flip
                        torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                        ]
    # Additional error-fixing-specific augmentations
    if error_fixing:
        train_transforms += [
            torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
            torchvision.transforms.RandomPerspective(distortion_scale=0.05, p=0.5),  # Random perspective
            torchvision.transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # Gaussian blur
        ]

    test_transforms = [torchvision.transforms.Resize(256),
                       torchvision.transforms.CenterCrop(resize_num)]
    return torchvision.transforms.Compose(
        (train_transforms if train_or_test == 'train' else test_transforms) + standard_transforms)


class EDCRImageFolder(torchvision.datasets.ImageFolder):
    def find_classes(self, directory: str) -> typing.Tuple[List[str], typing.Dict[str, int]]:
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir() and
                         not entry.name.startswith('.'))
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: index for index, cls_name in enumerate(classes)}
        return classes, class_to_idx


class CombinedImageFolderWithName(EDCRImageFolder):
    """
    Subclass of torchvision.datasets for a combined coarse and fine grain models that returns an image with its filename
    """

    def __getitem__(self,
                    index: int) -> (torch.tensor, int, str):
        """
        Returns one image from the dataset

        Parameters
        ----------

        index: Index of the image in the dataset
        """

        path, y_fine_grain = self.samples[index]
        y_coarse_grain = fine_to_course_idx[y_fine_grain]
        x = self.loader(path)

        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y_fine_grain = self.target_transform(y_fine_grain)
        name = os.path.basename(path)
        folder_path = os.path.basename(os.path.dirname(path))

        x_identifier = f'{folder_path}/{name}'

        return x, y_fine_grain, x_identifier, y_coarse_grain


class BinaryImageFolder(EDCRImageFolder):
    def __init__(self,
                 root: str,
                 l: Label,
                 transform: typing.Optional[typing.Callable] = None,
                 evaluation: bool = False):
        self.evaluation = evaluation

        if not evaluation:
            super().__init__(root=root, transform=transform)
            self.l = l.index
            self.balanced_samples = []

            # Count the number of images per class
            class_counts = {target: 0 for _, target in self.samples}
            for _, target in self.samples:
                class_counts[target] += 1

            # Number of positive examples
            positive_count = class_counts[self.l]

            # Calculate the number of negatives to sample from each of the other classes
            other_classes = [cls for cls in class_counts.keys() if cls != self.l]
            negative_samples_per_class = positive_count // len(other_classes)

            # Sample negatives

            for cls in other_classes:
                cls_samples = [(path, target) for path, target in self.samples if target == cls]
                self.balanced_samples.extend(
                    random.sample(cls_samples, min(negative_samples_per_class, len(cls_samples))))

            # Add positive examples
            self.balanced_samples.extend([(path, target) for path, target in self.samples if target == self.l])

            # Shuffle the dataset
            random.shuffle(self.balanced_samples)
        else:
            super().__init__(root=root,
                             transform=transform,
                             target_transform=lambda y: int(y == l.index))

    def __getitem__(self, index: int):
        if self.evaluation:
            return super().__getitem__(index)

        path, target = self.balanced_samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        # Convert the target to binary (1 for the chosen class, 0 for others)
        target = int(target == self.l)

        return sample, target

    def __len__(self):
        return len(self.samples if self.evaluation else self.balanced_samples)


class IndividualImageFolderWithName(EDCRImageFolder):
    """
    Subclass of torchvision.datasets for individual coarse or fine grain models that returns an image with its filename
    """

    def __getitem__(self,
                    index: int) -> (torch.tensor, int, str):
        """
        Returns one image from the dataset

        Parameters
        ----------

        index: Index of the image in the dataset
        """

        path, y = self.samples[index]
        x = self.loader(path)

        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)

        name = os.path.basename(path)
        folder_path = os.path.basename(os.path.dirname(path))

        x_identifier = f'{folder_path}/{name}'

        return x, y, x_identifier


def get_datasets(vit_model_names: list[str] = ['vit_b_16'],
                 weights: list[str] = ['DEFAULT'],
                 cwd: typing.Union[str, pathlib.Path] = os.getcwd(),
                 combined: bool = True,
                 binary_label: Label = None,
                 evaluation: bool = False,
                 error_fixing: bool = False) -> \
        (dict[str, torchvision.datasets.ImageFolder], int, int):
    """
    Instantiates and returns train and test datasets

    Parameters
    ----------
        :param weights:
        :param vit_model_names:
        :param error_fixing:
        :param evaluation:
        :param binary_label:
        :param cwd:
        :param combined:
    """

    data_dir = pathlib.Path.joinpath(pathlib.Path(cwd), '.')

    datasets = {}

    for train_or_test in ['train', 'test']:
        if binary_label is not None:
            datasets[train_or_test] = BinaryImageFolder(root=os.path.join(data_dir, f'data/{train_or_test}_fine'),
                                                        transform=get_dataset_transforms(train_or_test=train_or_test),
                                                        l=binary_label,
                                                        evaluation=evaluation)
        elif combined:
            datasets[train_or_test] = CombinedImageFolderWithName(root=os.path.join(data_dir,
                                                                                    f'data/{train_or_test}_fine'),
                                                                  transform=get_dataset_transforms(
                                                                      train_or_test=train_or_test,
                                                                      error_fixing=error_fixing,
                                                                      vit_model_name=vit_model_names[0],
                                                                      weight=weights[0]
                                                                  ))
        else:
            datasets[train_or_test] = IndividualImageFolderWithName(
                root=os.path.join(data_dir, f'{train_or_test}_fine'),
                transform=
                get_dataset_transforms(train_or_test=train_or_test))

    return datasets


class DatasetWithIndices(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = np.arange(len(dataset))

    def __getitem__(self, index):
        x, y_fine_grain, x_identifier, y_coarse_grain = self.dataset[index]
        return x, y_fine_grain, x_identifier, y_coarse_grain, self.indices[index]

    def __len__(self):
        return len(self.dataset)


def get_loaders(datasets: dict[str, torchvision.datasets.ImageFolder],
                batch_size: int,
                subset_indices: typing.Sequence = None,
                evaluation: bool = None,
                train_eval_split: float = None,
                get_indices: bool = None,
                get_fraction_of_example_with_label: dict[Label, float] = None,
                binary: bool = False) -> dict[
    str, torch.utils.data.DataLoader]:
    """
    Instantiates and returns train and test torch data loaders

    Parameters
    ----------
        :param binary:
        :param get_indices:
        :param get_fraction_of_example_with_label:
        :param train_eval_split:
        :param evaluation:
        :param datasets:
        :param batch_size:
        :param subset_indices:
    """
    loaders = {}

    all_indices = None
    train_size = None

    label_counts = {}
    total_count = 0

    if not binary:
        for element in datasets['train']:  # Iterate through the original dataset
            if element[-1] not in label_counts:
                label_counts[element[-1]] = 0
            label_counts[element[-1]] += 1
            total_count += 1
        print(f"Label counts in original train dataset: {total_count} examples in total, in which")

        # Sort label counts by label value (assuming labels are integers)
        sorted_label_counts = sorted(label_counts.items(), key=lambda item: item[0])

        for label, count in sorted_label_counts:
            print(f"  - Label {label}: {count} examples")

    for split in ['train', 'test'] + (['train_eval'] if train_eval_split is not None else []):
        relevant_dataset = datasets[split if split != 'train_eval' else 'train']

        if get_indices:
            relevant_dataset = DatasetWithIndices(relevant_dataset)

        if subset_indices is None or relevant_dataset != 'train':
            loader_dataset = relevant_dataset
        else:
            loader_dataset = torch.utils.data.Subset(dataset=relevant_dataset,
                                                     indices=subset_indices)

        if train_eval_split is not None:
            # Shuffle the indices in-place
            if all_indices is None:
                dataset_size = len(loader_dataset)
                all_indices = list(range(dataset_size))
                random.shuffle(all_indices)
                train_size = int(train_eval_split * dataset_size)

            if split == 'train':
                loader_dataset = torch.utils.data.Subset(dataset=relevant_dataset,
                                                         indices=all_indices[:train_size])
            elif split == 'train_eval':
                loader_dataset = torch.utils.data.Subset(dataset=relevant_dataset,
                                                         indices=all_indices[train_size:])

        if get_fraction_of_example_with_label is not None and split == 'train' and get_indices:
            # Remove the fraction of label based on the value specify in the dict remove_label_with_fraction
            label_counts = {}
            for _, _, _, y_coarse_grain, _ in loader_dataset:  # Iterate through data and labels
                if y_coarse_grain not in label_counts:
                    label_counts[y_coarse_grain] = 0
                label_counts[y_coarse_grain] += 1  # Count occurrences of each label

            for label, remove_fraction in get_fraction_of_example_with_label.items():
                num_to_add = int(remove_fraction * label_counts[label.index])  # Calculate number to remove

                # Get datapoints with the specified label
                label_indices = [idx for _, _, _, y_coarse_grain, idx in loader_dataset
                                 if y_coarse_grain == label.index]

                # Sample indices to add the desired number
                add_indices = random.sample(label_indices, num_to_add)
                filtered_indices = [idx for _, _, _, _, idx in loader_dataset
                                    if (idx in add_indices) or (idx not in label_indices)]

                print(f'add {remove_fraction * 100}% of examples out of {label_counts[label.index]} '
                      f'from label {label.index} in train')

                loader_dataset = torch.utils.data.Subset(relevant_dataset, filtered_indices)

        shuffle = split == 'train' and (evaluation is None or not evaluation)

        # if not binary:
            # if split in ['train', 'train_eval']:  # Only for train and train_eval loaders
            #     label_counts = {}
            #     total_count = 0
            #     for _, _, _, y_coarse_grain, _ in loader_dataset:  # Iterate through the loader dataset
            #         if y_coarse_grain not in label_counts:
            #             label_counts[y_coarse_grain] = 0
            #         label_counts[y_coarse_grain] += 1
            #         total_count += 1
            #
            #     print(f"Label counts in {split} loader: {total_count} examples in total, in which")
            #     # Sort label counts by label value (assuming labels are integers)
            #     sorted_label_counts = sorted(label_counts.items(), key=lambda item: item[0])
            #
            #     for label, count in sorted_label_counts:
            #         print(f"  - Label {label}: {count} examples")

        loaders[split] = torch.utils.data.DataLoader(
            dataset=loader_dataset,
            batch_size=batch_size,
            shuffle=shuffle)

    return loaders


def get_one_hot_encoding(input_arr: np.array) -> np.array:
    return np.eye(np.max(input_arr) + 1)[input_arr].T


for i, arr in enumerate([train_true_fine_data, test_true_fine_data]):
    assert is_monotonic(arr)

coarse_to_fine = {
    'Air Defense': ['30N6E', 'Iskander', 'Pantsir-S1', 'Rs-24'],
    'BMP': ['BMP-1', 'BMP-2', 'BMP-T15'],
    'BTR': ['BRDM', 'BTR-60', 'BTR-70', 'BTR-80'],
    'Tank': ['T-14', 'T-62', 'T-64', 'T-72', 'T-80', 'T-90'],
    'Self Propelled Artillery': ['2S19_MSTA', 'BM-30', 'D-30', 'Tornado', 'TOS-1'],
    'BMD': ['BMD'],
    'MT_LB': ['MT_LB']
}

fine_grain_labels = {l: FineGrainLabel(l) for l in fine_grain_classes_str}
coarse_grain_labels = {l: CoarseGrainLabel(l) for l in coarse_grain_classes_str}


def get_labels(g: Granularity) -> dict[str, Label]:
    return fine_grain_labels if str(g) == 'fine' else coarse_grain_labels
