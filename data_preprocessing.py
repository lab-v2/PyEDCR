import os
import numpy as np
import torch
import torchvision
import pandas as pd
import torch.utils.data
import pathlib
import typing
import abc

from typing import List

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
        K = [(0, len(test_true_coarse_data) - 1)]
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


def get_dataset_transforms(train_or_test: str) -> torchvision.transforms.Compose:
    """
    Returns the transforms required for the VIT for training or test datasets
    """

    resize_num = 224
    means = stds = [0.5] * 3

    standard_transforms = [torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize(means, stds)]
    train_transforms = [torchvision.transforms.RandomResizedCrop(resize_num),
                        torchvision.transforms.RandomHorizontalFlip()]
    test_transforms = [torchvision.transforms.Resize(int(resize_num / 224 * 256)),
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
    """
    Subclass of torchvision.datasets for a binary classifier that returns an image with its filename
    """

    def __init__(self,
                 root: str,
                 l: Label,
                 transform: typing.Optional[typing.Callable] = None):
        super().__init__(root=root,
                         transform=transform,
                         target_transform=lambda y: int(y == l.index))


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


def get_datasets(cwd: typing.Union[str, pathlib.Path] = os.getcwd(),
                 combined: bool = True,
                 binary_label: Label = None) -> \
        (dict[str, torchvision.datasets.ImageFolder], int, int):
    """
    Instantiates and returns train and test datasets

    Parameters
    ----------
        :param binary_label:
        :param cwd:
        :param combined:
    """

    data_dir = pathlib.Path.joinpath(pathlib.Path(cwd), '.')

    datasets = {
        f'{train_or_test}': BinaryImageFolder(root=os.path.join(data_dir, f'data/{train_or_test}_fine'),
                                              transform=get_dataset_transforms(train_or_test=train_or_test),
                                              l=binary_label)
        if binary_label is not None else
        (CombinedImageFolderWithName(root=os.path.join(data_dir, f'data/{train_or_test}_fine'),
                                     transform=get_dataset_transforms(
                                         train_or_test=train_or_test))
         if combined else IndividualImageFolderWithName(root=os.path.join(data_dir, f'{train_or_test}_fine'),
                                                        transform=
                                                        get_dataset_transforms(train_or_test=train_or_test)))
        for train_or_test in ['train', 'test']}

    return datasets


def get_loaders(datasets: dict[str, torchvision.datasets.ImageFolder],
                batch_size: int,
                indices: typing.Sequence = None,
                evaluation: bool = None) -> dict[str, torch.utils.data.DataLoader]:
    """
    Instantiates and returns train and test torch data loaders

    Parameters
    ----------
        :param evaluation:
        :param datasets:
        :param batch_size:
        :param indices:
    """

    return {train_or_test_dataset: torch.utils.data.DataLoader(
        dataset=datasets[train_or_test_dataset if train_or_test_dataset != 'train_eval' else 'train']
        if indices is None or train_or_test_dataset != 'train'
        else torch.utils.data.Subset(dataset=
                                     datasets[
                                         train_or_test_dataset if train_or_test_dataset != 'train_eval' else 'train'],
                                     indices=indices),
        batch_size=batch_size,
        shuffle=train_or_test_dataset == 'train' and (evaluation is None or not evaluation))
        for train_or_test_dataset in ['train', 'train_eval', 'test']}


def get_one_hot_encoding(injput_arr: np.array) -> np.array:
    return np.eye(np.max(injput_arr) + 1)[injput_arr].T


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
