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

random.seed(42)
np.random.seed(42)

current_file_location = pathlib.Path(__file__).parent.resolve()
os.chdir(current_file_location)


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


class Label(typing.Hashable, abc.ABC):
    def __init__(self,
                 l_str: str,
                 index: int):
        self.l_str = l_str
        self.index = index
        self.g = None

    def __str__(self):
        return self.l_str

    def __hash__(self):
        return hash(f'{self.g}_{self.l_str}')

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class FineGrainLabel(Label):
    def __init__(self,
                 l_str: str,
                 fine_grain_classes_str: typing.List[str]):
        super().__init__(l_str=l_str,
                         index=fine_grain_classes_str.index(l_str))
        assert l_str in fine_grain_classes_str

        self.g = DataPreprocessor.granularities['fine']

    @classmethod
    def with_index(cls,
                   fine_grain_classes_str: typing.List[str],
                   l_index: int):
        l = fine_grain_classes_str[l_index]
        instance = cls(l_str=l,
                       fine_grain_classes_str=fine_grain_classes_str)

        return instance


class CoarseGrainLabel(Label):
    def __init__(self,
                 l_str: str,
                 coarse_grain_classes_str: typing.List[str]):
        super().__init__(l_str=l_str,
                         index=coarse_grain_classes_str.index(l_str))
        assert l_str in coarse_grain_classes_str
        self.g = DataPreprocessor.granularities['coarse']

    @classmethod
    def with_index(cls,
                   i_l: int,
                   coarse_grain_classes_str: typing.List[str]):
        l = coarse_grain_classes_str[i_l]
        instance = cls(l_str=l,
                       coarse_grain_classes_str=coarse_grain_classes_str)

        return instance


class DataPreprocessor:
    granularities_str = ['fine', 'coarse']
    granularities = {g_str: Granularity(g_str=g_str) for g_str in granularities_str}

    def __init__(self,
                 data_str: str):
        self.data_str = data_str
        self.fine_to_course_idx = {}

        if data_str == 'imagenet':
            self.coarse_grain_classes_str = [
                'bird', 'snake', 'spider', 'small fish', 'turtle', 'lizard', 'crab', 'shark'
            ]

            self.fine_grain_mapping_dict = {
                'n01818515': 'macaw',
                'n01537544': 'indigo bunting, indigo finch, indigo bird, Passerina cyanea',
                'n02007558': 'flamingo',
                'n02002556': 'white stork, Ciconia ciconia',
                'n01614925': 'bald eagle, American eagle, Haliaeetus leucocephalus',
                'n01582220': 'magpie',
                'n01806143': 'peacock',
                'n01795545': 'black grouse',
                'n01531178': 'goldfinch, Carduelis carduelis',
                'n01622779': 'great grey owl, great gray owl, Strix nebulosa',
                'n01833805': 'hummingbird',
                'n01740131': 'night snake, Hypsiglena torquata',
                'n01735189': 'garter snake, grass snake',
                'n01755581': 'diamondback, diamondback rattlesnake, Crotalus adamanteus',
                'n01751748': 'sea snake',
                'n01729977': 'green snake, grass snake',
                'n01729322': 'hognose snake, puff adder, sand viper',
                'n01734418': 'king snake, kingsnake',
                'n01728572': 'thunder snake, worm snake, Carphophis amoenus',
                'n01739381': 'vine snake',
                'n01756291': 'sidewinder, horned rattlesnake, Crotalus cerastes',
                'n01773797': 'garden spider, Aranea diademata',
                'n01775062': 'wolf spider, hunting spider',
                'n01773549': 'barn spider, Araneus cavaticus',
                'n01774384': 'black widow, Latrodectus mactans',
                'n01774750': 'tarantula',
                'n01440764': 'tench, Tinca tinca',
                'n01443537': 'goldfish, Carassius auratus',
                'n01667778': 'terrapin',
                'n01667114': 'mud turtle',
                'n01664065': 'loggerhead, loggerhead turtle, Caretta caretta',
                'n01665541': 'leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea',
                'n01687978': 'agama',
                'n01677366': 'common iguana, iguana, Iguana iguana',
                'n01695060': 'Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis',
                'n01685808': 'whiptail, whiptail lizard',
                'n01978287': 'Dungeness crab, Cancer magister',
                'n01986214': 'hermit crab',
                'n01978455': 'rock crab, Cancer irroratus',
                'n01491361': 'tiger shark, Galeocerdo cuvieri',
                'n01484850': 'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias',
                'n01494475': 'hammerhead, hammerhead shark'
            }
            self.fine_grain_classes_str = list(self.fine_grain_mapping_dict.values())

            self.fine_to_coarse = {
                'macaw': 'bird',
                'indigo bunting, indigo finch, indigo bird, Passerina cyanea': 'bird',
                'flamingo': 'bird',
                'white stork, Ciconia ciconia': 'bird',
                'bald eagle, American eagle, Haliaeetus leucocephalus': 'bird',
                'magpie': 'bird',
                'peacock': 'bird',
                'black grouse': 'bird',
                'goldfinch, Carduelis carduelis': 'bird',
                'great grey owl, great gray owl, Strix nebulosa': 'bird',
                'hummingbird': 'bird',
                'night snake, Hypsiglena torquata': 'snake',
                'garter snake, grass snake': 'snake',
                'diamondback, diamondback rattlesnake, Crotalus adamanteus': 'snake',
                'sea snake': 'snake',
                'green snake, grass snake': 'snake',
                'hognose snake, puff adder, sand viper': 'snake',
                'king snake, kingsnake': 'snake',
                'thunder snake, worm snake, Carphophis amoenus': 'snake',
                'vine snake': 'snake',
                'sidewinder, horned rattlesnake, Crotalus cerastes': 'snake',
                'garden spider, Aranea diademata': 'spider',
                'wolf spider, hunting spider': 'spider',
                'barn spider, Araneus cavaticus': 'spider',
                'black widow, Latrodectus mactans': 'spider',
                'tarantula': 'spider',
                'tench, Tinca tinca': 'small fish',
                'goldfish, Carassius auratus': 'small fish',
                'terrapin': 'turtle',
                'mud turtle': 'turtle',
                'loggerhead, loggerhead turtle, Caretta caretta': 'turtle',
                'leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea': 'turtle',
                'agama': 'lizard',
                'common iguana, iguana, Iguana iguana': 'lizard',
                'Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis': 'lizard',
                'whiptail, whiptail lizard': 'lizard',
                'Dungeness crab, Cancer magister': 'crab',
                'hermit crab': 'crab',
                'rock crab, Cancer irroratus': 'crab',
                'tiger shark, Galeocerdo cuvieri': 'shark',
                'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias': 'shark',
                'hammerhead, hammerhead shark': 'shark'
            }

            self.coarse_to_fine = {}

            for fine_idx, (fine_class, coarse_class) in enumerate(self.fine_to_coarse.items()):
                coarse_idx = self.coarse_grain_classes_str.index(coarse_class)
                self.fine_to_course_idx[fine_idx] = coarse_idx

                if coarse_class not in coarse_to_fine:
                    coarse_to_fine[coarse_class] = [fine_class]
                else:
                    coarse_to_fine[coarse_class].append(fine_class)

        elif data_str == 'openimage':
            self.coarse_to_fine = {
                "Animal": [
                    "Mammal",
                    "Bird",
                    "Invertebrate",
                    "Fish",
                    "Reptile",
                ],
                "Building": [
                    "House",
                    "Skyscraper",
                    "Tower",
                    "Office building",
                    "Castle",
                    "Lighthouse",
                    "Convenience store"
                ],
                "Clothing": [
                    "Footwear",
                    "Fashion accessory",
                    "Dress",
                    "Suit",
                    "Hat",
                    "Trousers",
                    "Jacket"
                ],
                "Drink": [
                    "Beer",
                    "Wine",
                    "Cocktail",
                    "Coffee",
                    "Juice",
                ],
                "Vehicle": [
                    "Land vehicle",
                    "Watercraft",
                    "Aircraft",
                ],
                "Hat": [  # Handle case where a category appears as both coarse and fine label
                    "Sun hat",
                    "Fedora",
                    "Cowboy hat"
                ]
            }
            self.fine_grain_classes_str = sorted(
                [item for category, items in self.coarse_to_fine.items() for item in items])
            self.coarse_grain_classes_str = sorted([item for item in self.coarse_to_fine.keys()])

            self.fine_to_coarse = {}
            for fine_grain_class_idx, fine_grain_class in enumerate(self.fine_grain_classes_str):
                for coarse_grain_class, fine_grain_class_list in self.coarse_to_fine.items():
                    if fine_grain_class in fine_grain_class_list:
                        self.fine_to_coarse[fine_grain_class] = coarse_grain_class

            for fine_idx, (fine_class, coarse_class) in enumerate(self.fine_to_coarse.items()):
                coarse_idx = self.coarse_grain_classes_str.index(coarse_class)
                self.fine_to_course_idx[fine_idx] = coarse_idx

                if coarse_class not in coarse_to_fine:
                    coarse_to_fine[coarse_class] = [fine_class]
                else:
                    coarse_to_fine[coarse_class].append(fine_class)

        else:
            data_file_path = rf'data/WEO_Data_Sheet.xlsx'
            dataframes_by_sheet = pd.read_excel(data_file_path, sheet_name=None)

            fine_grain_results_df = dataframes_by_sheet['Fine-Grain Results']
            self.fine_grain_classes_str = sorted(fine_grain_results_df['Class Name'].to_list())
            coarse_grain_results_df = dataframes_by_sheet['Coarse-Grain Results']
            self.coarse_grain_classes_str = sorted(coarse_grain_results_df['Class Name'].to_list())

            self.fine_to_coarse = {}

            training_df = dataframes_by_sheet['Training']

            assert (set(training_df['Fine-Grain Ground Truth'].unique().tolist()).intersection(
                self.fine_grain_classes_str) == set(self.fine_grain_classes_str))

            for fine_grain_class_idx, fine_grain_class in enumerate(self.fine_grain_classes_str):
                curr_fine_grain_training_data = training_df[training_df['Fine-Grain Ground Truth'] == fine_grain_class]
                assert curr_fine_grain_training_data['Course-Grain Ground Truth'].nunique() == 1

                coarse_grain_class = curr_fine_grain_training_data['Course-Grain Ground Truth'].iloc[0]
                coarse_grain_class_idx = self.coarse_grain_classes_str.index(coarse_grain_class)

                self.fine_to_coarse[fine_grain_class] = coarse_grain_class
                self.fine_to_course_idx[fine_grain_class_idx] = coarse_grain_class_idx

        if data_str == 'imagenet':
            data_path_str = 'data/ImageNet100/'
        elif data_str == 'openimage':
            data_path_str = 'scratch/ngocbach/OpenImage'
        else:
            data_path_str = 'data/'

        self.test_true_fine_data = np.load(rf'{data_path_str}test_fine/test_true_fine.npy')
        self.test_true_coarse_data = np.load(rf'{data_path_str}test_coarse/test_true_coarse.npy')

        self.train_true_fine_data = np.load(rf'{data_path_str}train_fine/train_true_fine.npy')
        self.train_true_coarse_data = np.load(rf'{data_path_str}train_coarse/train_true_coarse.npy')

        self.num_fine_grain_classes = len(self.fine_grain_classes_str)
        self.num_coarse_grain_classes = len(self.coarse_grain_classes_str)

        self.fine_unique, self.fine_counts = np.unique(self.train_true_fine_data, return_counts=True)
        self.coarse_unique, self.coarse_counts = np.unique(self.train_true_coarse_data, return_counts=True)

        # # Create dictionaries from unique labels and counts
        self.fine_data_counts = dict(zip(self.fine_unique, self.fine_counts))
        self.coarse_data_counts = dict(zip(self.coarse_unique, self.coarse_counts))

        self.fine_grain_labels = {l: FineGrainLabel(l, fine_grain_classes_str=self.fine_grain_classes_str)
                                  for l in self.fine_grain_classes_str}
        self.coarse_grain_labels = {l: CoarseGrainLabel(l, coarse_grain_classes_str=self.coarse_grain_classes_str)
                                    for l in self.coarse_grain_classes_str}

        assert (self.get_num_inconsistencies(fine_labels=self.train_true_fine_data,
                                             coarse_labels=self.train_true_coarse_data)[0] ==
                self.get_num_inconsistencies(fine_labels=self.test_true_fine_data,
                                             coarse_labels=self.test_true_coarse_data)[0] == 0)

    def get_ground_truths(self,
                          test: bool,
                          K: typing.List[int] = None,
                          g: Granularity = None) -> np.array:
        if K is None:
            K = [idx for idx in range(0, len(self.test_true_coarse_data))]
        if test:
            true_fine_data = self.test_true_fine_data
            true_coarse_data = self.test_true_coarse_data
        else:
            true_fine_data = self.train_true_fine_data
            true_coarse_data = self.train_true_coarse_data

        if g is None:
            return true_fine_data[K], true_coarse_data[K]
        else:
            return true_fine_data[K] if str(g) == 'fine' else true_coarse_data[K]

    def get_num_inconsistencies(self,
                                fine_labels: typing.Union[np.array, torch.Tensor],
                                coarse_labels: typing.Union[np.array, torch.Tensor]) -> typing.Tuple[int, int]:
        inconsistencies = 0
        unique_inconsistencies = {}

        if isinstance(fine_labels, torch.Tensor):
            fine_labels = np.array(fine_labels.cpu())
            coarse_labels = np.array(coarse_labels.cpu())

        for fine_prediction, coarse_prediction in zip(fine_labels, coarse_labels):
            if self.fine_to_course_idx[fine_prediction] != coarse_prediction:
                inconsistencies += 1

                if fine_prediction not in unique_inconsistencies:
                    unique_inconsistencies[fine_prediction] = {coarse_prediction}
                else:
                    unique_inconsistencies[fine_prediction] \
                        = (unique_inconsistencies[fine_prediction].union({coarse_prediction}))

        unique_inconsistencies_num = sum(len(coarse_dict) for coarse_dict in unique_inconsistencies.values())

        return inconsistencies, unique_inconsistencies_num

    def get_labels(self,
                   g: Granularity) -> typing.Dict[str, Label]:
        return self.fine_grain_labels if str(g) == 'fine' else self.coarse_grain_labels

    def get_subset_indices_for_train_and_train_eval(self,
                                                    train_eval_split: float,
                                                    get_fraction_of_example_with_label: typing.Dict[
                                                        Label, float] = None, ):
        """
            Splits indices into train and train_eval sets, respecting train_eval_split
            and removing examples from train based on get_fraction_of_example_with_label.

            Args:
                get_fraction_of_example_with_label: Optional dict mapping coarse-grained
                    labels to fractions of examples to keep for training.
                train_eval_split: Float between 0 and 1, indicating the proportion
                    of examples to assign to the train set.

            Returns:
                Tuple of NumPy arrays containing indices for train and train_eval sets.
            """

        num_examples = len(self.train_true_coarse_data)
        num_train = int(num_examples * train_eval_split)

        # Split indices initially based on train_eval_split
        all_indices = np.arange(num_examples)
        np.random.shuffle(all_indices)  # Shuffle for random sampling
        train_indices = all_indices[:num_train]
        train_eval_indices = all_indices[num_train:]

        # Filter train indices based on get_fraction_of_example_with_label if provided
        if get_fraction_of_example_with_label is not None:
            filter_label = {l.index: int((1 - frac) * self.coarse_data_counts[l.index])
                            for l, frac in get_fraction_of_example_with_label.items()}
            count_labels = {l: 0 for l in range(self.num_coarse_grain_classes)}
            filtered_train_indices = []
            for idx in train_indices:
                label = self.train_true_coarse_data[idx]
                if label in list(filter_label.keys()) and filter_label[label] > 0:
                    filter_label[label] -= 1
                    continue
                else:
                    count_labels[label] += 1
                filtered_train_indices.append(idx)

            print(f"\nCoarse data counts: {self.coarse_data_counts}")
            print(f"train eval split is: {train_eval_split}")
            print(f'Coarse data counts for train after remove: {count_labels}\n')
            train_indices = np.array(filtered_train_indices)

        return train_indices, train_eval_indices

    def get_imbalance_weight(self,
                             l: Label,
                             train_images_num: int,
                             evaluation: bool = False) -> typing.List[float]:
        g_ground_truth = self.train_true_fine_data if l.g.g_str == 'fine' else self.train_true_coarse_data
        positive_examples_num = np.sum(np.where(g_ground_truth == l.index, 1, 0))
        # negative_examples_num = train_images_num - positive_examples_num

        positive_class_weight = train_images_num / positive_examples_num

        if not evaluation:
            print(f'\nl={l}:\n'
                  f'weight of positive class: {positive_class_weight}')

        return positive_class_weight


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


def get_dataset_transforms(data: str,
                           train_or_test: str,
                           model_name: str = 'vit_b_16',
                           error_fixing: bool = False,
                           weight: str = 'DEFAULT') -> torchvision.transforms.Compose:
    """
    Returns the transforms required for the VIT for training or test datasets
    """

    if data == 'imagenet':
        model_size = model_name.split('dinov2_vit')[-1][0]
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])

        standard_transforms = [
            torchvision.transforms.ToTensor(),
            normalize
        ]
        train_transforms = [
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip()
        ]

        test_transforms = [torchvision.transforms.Resize(256 if model_size == 's' else 1024),
                           torchvision.transforms.CenterCrop(224)
                           ]

    else:

        resize_num = 518 if model_name == 'vit_h_14' else (224 if weight == 'DEFAULT' else 512)
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
    def __init__(
            self,
            root: str,
            transform=None,
            target_transform=None,
            is_valid_file=None,
            relevant_classes: typing.List[str] = None
    ):
        self.relevant_classes = relevant_classes
        super().__init__(root=root,
                         transform=transform,
                         target_transform=target_transform,
                         is_valid_file=is_valid_file)

    def find_classes(self,
                     directory: str) -> (typing.List[str], typing.Dict[str, int]):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir() and
                         not entry.name.startswith('.') and (self.relevant_classes is None
                                                             or entry.name in self.relevant_classes))
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: index for index, cls_name in enumerate(classes)}
        return classes, class_to_idx


class CombinedImageFolderWithName(EDCRImageFolder):
    """
    Subclass of torchvision.datasets for a combined coarse and fine grain models that returns an image with its filename
    """

    def __init__(self,
                 root,
                 preprocessor: DataPreprocessor,
                 transform=None,
                 target_transform=None):
        self.preprocessor = preprocessor
        super(CombinedImageFolderWithName, self).__init__(root=root,
                                                          transform=transform,
                                                          target_transform=target_transform,
                                                          relevant_classes=
                                                          list(self.preprocessor.fine_grain_mapping_dict.keys())
                                                          if preprocessor.data_str == 'imagenet' else None)

    def __getitem__(self,
                    index: int) -> (torch.tensor, int, str):
        """
        Returns one image from the dataset

        Parameters
        ----------

        index: Index of the image in the dataset
        """

        path, y_fine_grain = self.samples[index]
        y_coarse_grain = self.preprocessor.fine_to_course_idx[y_fine_grain]
        x = self.loader(path)

        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y_fine_grain = self.target_transform(y_fine_grain)
        name = os.path.basename(path)
        folder_path = os.path.basename(os.path.dirname(path))

        x_identifier = f'{folder_path}/{name}'

        return x, y_fine_grain, x_identifier, y_coarse_grain, index


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


def get_datasets(preprocessor: DataPreprocessor,
                 model_names: typing.List[str],
                 weights: typing.List[str] = ['DEFAULT'],
                 cwd: typing.Union[str, pathlib.Path] = os.getcwd(),
                 combined: bool = True,
                 binary_label: Label = None,
                 evaluation: bool = False,
                 error_fixing: bool = False) -> \
        (typing.Dict[str, torchvision.datasets.ImageFolder], int, int):
    """
    Instantiates and returns train and test datasets

    Parameters
    ----------
        :param preprocessor:
        :param weights:
        :param model_names:
        :param error_fixing:
        :param evaluation:
        :param binary_label:
        :param cwd:
        :param combined:
    """

    data_dir = pathlib.Path.joinpath(pathlib.Path(cwd), 'data')

    datasets = {}

    for train_or_test in ['train', 'test']:

        if preprocessor.data_str == 'openimage':
            full_data_dir = f'scratch/ngocbach/OpenImage/{train_or_test}_fine'
        else:
            data_dir_name = f'ImageNet100/{train_or_test}_fine' if preprocessor.data_str == 'imagenet' \
                else f'{train_or_test}_fine'
            full_data_dir = os.path.join(data_dir, data_dir_name)

        if binary_label is not None:
            datasets[train_or_test] = BinaryImageFolder(root=full_data_dir,
                                                        transform=get_dataset_transforms(data=preprocessor.data_str,
                                                                                         train_or_test=train_or_test),
                                                        l=binary_label,
                                                        evaluation=evaluation)
        elif combined:
            datasets[train_or_test] = CombinedImageFolderWithName(preprocessor=preprocessor,
                                                                  root=full_data_dir,
                                                                  transform=get_dataset_transforms(
                                                                      data=preprocessor.data_str,
                                                                      train_or_test=train_or_test,
                                                                      error_fixing=error_fixing,
                                                                      model_name=model_names[0],
                                                                      weight=weights[0],
                                                                  ))
        else:
            datasets[train_or_test] = IndividualImageFolderWithName(
                root=full_data_dir,
                transform=
                get_dataset_transforms(data=preprocessor.data_str,
                                       train_or_test=train_or_test))

    return datasets


def get_loaders(preprocessor: DataPreprocessor,
                datasets: typing.Dict[str, torchvision.datasets.ImageFolder],
                batch_size: int,
                subset_indices: typing.Sequence = None,
                evaluation: bool = None,
                train_eval_split: float = None,
                get_indices: bool = False,
                get_fraction_of_example_with_label: typing.Dict[Label, float] = None,
                binary: bool = False
                ) -> typing.Dict[str, torch.utils.data.DataLoader]:
    """
    Instantiates and returns train and test torch data loaders

    Parameters
    ----------
        :param preprocessor:
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
    train_indices = None
    train_eval_indices = None

    if train_eval_split is not None:
        # Shuffle the indices in-place
        train_indices, train_eval_indices = preprocessor.get_subset_indices_for_train_and_train_eval(
            train_eval_split=train_eval_split,
            get_fraction_of_example_with_label=get_fraction_of_example_with_label
        )

    for split in ['train', 'test'] + (['train_eval'] if train_eval_split is not None else []):
        relevant_dataset = datasets[split if split != 'train_eval' else 'train']

        if subset_indices is None or split != 'train':
            loader_dataset = relevant_dataset
        else:
            loader_dataset = torch.utils.data.Subset(dataset=relevant_dataset,
                                                     indices=subset_indices)

        if split == 'train' and train_eval_split is not None:
            loader_dataset = torch.utils.data.Subset(dataset=relevant_dataset,
                                                     indices=train_indices)
        elif split == 'train_eval' and train_eval_split is not None:
            loader_dataset = torch.utils.data.Subset(dataset=relevant_dataset,
                                                     indices=train_eval_indices)

        shuffle = split == 'train' and (evaluation is None or not evaluation)

        loaders[split] = torch.utils.data.DataLoader(
            dataset=loader_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
        )

    return loaders


def get_one_hot_encoding(input_arr: np.array) -> np.array:
    return np.eye(np.max(input_arr) + 1)[input_arr].T


coarse_to_fine = {
    'Air Defense': ['30N6E', 'Iskander', 'Pantsir-S1', 'Rs-24'],
    'BMP': ['BMP-1', 'BMP-2', 'BMP-T15'],
    'BTR': ['BRDM', 'BTR-60', 'BTR-70', 'BTR-80'],
    'Tank': ['T-14', 'T-62', 'T-64', 'T-72', 'T-80', 'T-90'],
    'Self Propelled Artillery': ['2S19_MSTA', 'BM-30', 'D-30', 'Tornado', 'TOS-1'],
    'BMD': ['BMD'],
    'MT_LB': ['MT_LB']
}
