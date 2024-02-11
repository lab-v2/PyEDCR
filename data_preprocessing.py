import os
import numpy as np
import torch
import torchvision
import pandas as pd
import torch.utils.data
import pathlib
import typing

data_file_path = rf'data/WEO_Data_Sheet.xlsx'
dataframes_by_sheet = pd.read_excel(data_file_path, sheet_name=None)
fine_grain_results_df = dataframes_by_sheet['Fine-Grain Results']
fine_grain_classes_str = sorted(fine_grain_results_df['Class Name'].to_list())
coarse_grain_results_df = dataframes_by_sheet['Coarse-Grain Results']
coarse_grain_classes_str = sorted(coarse_grain_results_df['Class Name'].to_list())
granularities_str = ['fine', 'coarse']

# Data for our use case

test_true_fine_data = np.load(r'test_fine/test_true_fine.npy')
test_true_coarse_data = np.load(r'test_coarse/test_true_coarse.npy')

train_true_fine_data = np.load(r'train_fine/train_true_fine.npy')
train_true_coarse_data = np.load(r'train_coarse/train_true_coarse.npy')

# Data for checking mode

check_test_true_fine_data = np.load(r'test_data/check_test_true_fine_data.npy')
check_test_true_coarse_data = np.load(r'test_data/check_test_true_coarse_data.npy')

check_train_true_fine_data = np.load(r'test_data/check_train_true_fine_data.npy')
check_train_true_coarse_data = np.load(r'test_data/check_train_true_coarse_data.npy')


def is_monotonic(arr: np.array):
    return np.all(arr[:-1] <= arr[1:])


def get_fine_to_coarse() -> (dict[str, str], dict[int, int]):
    """
    Creates and returns a dictionary with fine-grain labels as keys and their corresponding coarse grain-labels
    as values, and a dictionary with fine-grain label indices as keys and their corresponding coarse-grain label
    indices as values
    """

    fine_to_coarse = {}
    fine_to_course_idx = {}
    training_df = dataframes_by_sheet['Training']

    assert (set(training_df['Fine-Grain Ground Truth'].unique().tolist()).intersection(fine_grain_classes_str)
            == set(fine_grain_classes_str))

    for fine_grain_class_idx, fine_grain_class in enumerate(fine_grain_classes_str):
        curr_fine_grain_training_data = training_df[training_df['Fine-Grain Ground Truth'] == fine_grain_class]
        assert curr_fine_grain_training_data['Course-Grain Ground Truth'].nunique() == 1

        coarse_grain_class = curr_fine_grain_training_data['Course-Grain Ground Truth'].iloc[0]
        coarse_grain_class_idx = coarse_grain_classes_str.index(coarse_grain_class)

        fine_to_coarse[fine_grain_class] = coarse_grain_class
        fine_to_course_idx[fine_grain_class_idx] = coarse_grain_class_idx

    return fine_to_coarse, fine_to_course_idx


fine_to_coarse, fine_to_course_idx = get_fine_to_coarse()


class Granularity:
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
                      g: Granularity = None,
                      check_mode: bool = False):
    
    # if check mode, data is taken from test_data:
    if check_mode:
        if test:
            true_fine_data = check_test_true_fine_data
            true_coarse_data = check_test_true_coarse_data
        else:
            true_fine_data = check_train_true_fine_data
            true_coarse_data = check_train_true_coarse_data
    else:
        if test:
            true_fine_data = test_true_fine_data
            true_coarse_data = test_true_coarse_data
        else:
            true_fine_data = train_true_fine_data
            true_coarse_data = train_true_coarse_data

    if g is None:
        return true_fine_data, true_coarse_data
    else:
        return true_fine_data if str(g) == 'fine' else true_coarse_data


granularities = [Granularity(g) for g in granularities_str]


def get_classes(g: Granularity) -> list:
    return fine_grain_classes_str if g == granularities[0] else coarse_grain_classes_str


class Label:
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
        return hash(self._l_str)

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class FineGrainLabel(Label):
    def __init__(self,
                 l_str: str):
        super().__init__(l_str=l_str, index=fine_grain_classes_str.index(l_str))
        assert l_str in fine_grain_classes_str
        self.__correct_coarse = fine_to_coarse[l_str]
        self._g = granularities[0]

    @classmethod
    def with_index(cls,
                   l_index: int):
        l = fine_grain_classes_str[l_index]
        instance = cls(l_str=l)

        return instance


class CoarseGrainLabel(Label):
    def __init__(self,
                 l_str: str):
        super().__init__(l_str=l_str, index=coarse_grain_classes_str.index(l_str))
        assert l_str in coarse_grain_classes_str
        self.correct_fine = coarse_to_fine[l_str]
        self._g = granularities[1]

    @classmethod
    def with_index(cls,
                   i_l: int):
        l = coarse_grain_classes_str[i_l]
        instance = cls(l_str=l)

        return instance


# class Example:
#     def __init__(self,
#                  i: int):
#         self.__index = i
#
#     @property
#     def index(self):
#         return self.__index


def get_labels(g: Granularity):
    return fine_grain_labels if str(g) == 'fine' else coarse_grain_labels


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


for test in [True, False]:
    true_fine_data, true_coarse_data = get_ground_truths(test=test)
    assert get_num_inconsistencies(fine_labels=true_fine_data,
                                   coarse_labels=true_coarse_data) == 0


def get_dataset_transforms(train_or_test: str) -> torchvision.transforms.Compose:
    """
    Returns the transforms required for the VIT for training or test datasets
    """

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


class CombinedImageFolderWithName(torchvision.datasets.ImageFolder):
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


class IndividualImageFolderWithName(torchvision.datasets.ImageFolder):
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


def get_datasets(cwd: typing.Union[str, pathlib.Path],
                 combined: bool = True) -> \
        (dict[str, typing.Union[CombinedImageFolderWithName, IndividualImageFolderWithName]], int, int):
    """
    Instantiates and returns train and test datasets

    Parameters
    ----------
        cwd: Path to the current working folder
        combined: Whether the model is combining fine and coarse grain or not
    """

    data_dir = pathlib.Path.joinpath(cwd, '.')
    datasets = {f'{train_or_test}': CombinedImageFolderWithName(root=os.path.join(data_dir, f'{train_or_test}_fine'),
                                                                transform=get_dataset_transforms(
                                                                    train_or_test=train_or_test))
    if combined else IndividualImageFolderWithName(root=os.path.join(data_dir, f'{train_or_test}_fine'),
                                                   transform=get_dataset_transforms(train_or_test=train_or_test))
                for train_or_test in ['train', 'test']}

    print(f"Total number of train images: {len(datasets['train'])}\n"
          f"Total number of test images: {len(datasets['test'])}")

    classes = datasets['train'].classes
    assert classes == sorted(classes) == fine_grain_classes_str

    num_fine_grain_classes = len(classes)
    num_coarse_grain_classes = len(coarse_grain_classes_str)

    return datasets, num_fine_grain_classes, num_coarse_grain_classes


def get_loaders(datasets: dict[str, typing.Union[CombinedImageFolderWithName, IndividualImageFolderWithName]],
                batch_size: int) -> dict[str, torch.utils.data.DataLoader]:
    """
    Instantiates and returns train and test torch data loaders

    Parameters
    ----------
        datasets: Train and test datasets to use with the loaders
        batch_size
    """

    return {train_or_test_dataset: torch.utils.data.DataLoader(
        dataset=datasets[train_or_test_dataset if train_or_test_dataset != 'train_eval' else 'train'],
        batch_size=batch_size,
        shuffle=train_or_test_dataset == 'train')
        for train_or_test_dataset in ['train', 'train_eval', 'test']}


def get_one_hot_encoding(arr: np.array) -> np.array:
    return np.eye(np.max(arr) + 1)[arr].T


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

fine_grain_labels = [FineGrainLabel(l) for l in fine_grain_classes_str]
coarse_grain_labels = [CoarseGrainLabel(l) for l in coarse_grain_classes_str]

# train_examples = [Example(i) for i in range(train_true_fine_data.shape[0])]
# test_examples = [Example(i) for i in range(test_true_fine_data.shape[0])]

all_labels = [label for granularity in granularities for label in get_labels(granularity)]
