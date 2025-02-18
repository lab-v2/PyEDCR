import os
import typing
import numpy as np
import pandas as pd
import pathlib
import random
import shutil
import tqdm
import abc

import torch

from src.PyEDCR.classes import granularity, label
from src.PyEDCR.utils import utils
from src.PyEDCR.utils import paths as paths

random.seed(42)
np.random.seed(42)

current_file_location = pathlib.Path(__file__).parent.resolve()
os.chdir(current_file_location)


def get_filepath(data_str: str,
                 model_name,
                 test: bool,
                 pred: bool,
                 loss: str = None,
                 lr: typing.Union[str, float] = None,
                 combined: bool = True,
                 l: label.Label = None,
                 epoch: int = None,
                 granularity: str = None,
                 lower_prediction_index: int = None,
                 additional_info: str = None) -> str:
    """
    Constructs the file path to the model output / ground truth data.

    :param additional_info:
    :param data_str:
    :param l:
    :param lower_prediction_index:
    :param model_name: The name of the model or `FineTuner` object.
    :param combined: Whether the model are individual or combine one.
    :param test: Whether the data is getting from testing or training set.
    :param granularity: The granularity level.
    :param loss: The loss function used during training.
    :param lr: The learning rate used during training.
    :param pred: Whether the data is a prediction from neural or ground truth
    :param epoch: The epoch number (optional, only for training data).
    :return: The generated file path.
    """
    epoch_str = f'_e{epoch - 1}' if epoch is not None else ''
    granularity_str = f'_{granularity}' if granularity is not None else ''
    test_str = 'test' if test else 'train'
    pred_str = 'pred' if pred else 'true'
    folder_str = (('BINARY' if l is not None else ('COMBINED' if combined else 'INDIVIDUAL')) + '_RESULTS')
    lower_prediction_index_str = f'_lower_{lower_prediction_index}' if lower_prediction_index is not None else ''
    lower_prediction_folder_str = 'lower_prediction/' if lower_prediction_index is not None else ''
    l_str = f'_{l}' if l is not None else ''
    additional_str = f'_{additional_info}' if additional_info is not None else ''
    loss_str = f'_{loss}' if loss is not None else ''
    lr_str = f'_lr{lr}' if lr is not None else ''

    return (f"{paths.ROOT_PATH}/{folder_str}/{lower_prediction_folder_str}"
            f"{data_str}_{model_name}_{test_str}{granularity_str}_{pred_str}{loss_str}{lr_str}{epoch_str}"
            f"{lower_prediction_index_str}{l_str}{additional_str}.npy")


class DataPreprocessor(abc.ABC):
    """
    Abstract base class for data preprocessing.

    This class defines the blueprint for preprocessing data, including obtaining
    ground truths and labels. It is intended to be subclassed, where concrete
    implementations must be provided for its abstract methods. The purpose of
    this class is to ensure a standardized interface for handling data
    preprocessing tasks.

    :ivar data_str: String representation of the data to be processed.
    """

    def __init__(self,
                 data_str: str):
        self.data_str = data_str

    @abc.abstractmethod
    def get_ground_truths(self,
                          *args,
                          **kwargs):
        """
        Retrieve the ground truth values for a given dataset or context.

        This method is an abstract definition that must be implemented
        by any subclass. It defines the structure for obtaining the
        ground truths required for evaluation or further processing.
        """
        pass

    @abc.abstractmethod
    def get_labels(self,
                   *args,
                   **kwargs) -> typing.Dict[str, label.Label]:
        """
        Retrieve a dictionary of labels based on the provided arguments and keyword arguments.

        This method is an abstract method that must be implemented by any subclass.
        It should return a mapping of label names to their corresponding `Label` objects.
        The implementation details and parameters usage remain dependent on the inheriting
        class's specific requirements.
        """
        pass


class OneLevelDataPreprocessor(DataPreprocessor):
    """
    Handles preprocessing of data at one level of granularity.

    The `OneLevelDataPreprocessor` class is responsible for reading dataset files,
    transforming and validating ground truth and prediction data, and preparing necessary
    data structures for further processing. It assigns labels to classes and loads
    preprocessed data for training and testing. This class ensures alignment and consistency
    between ground truth and predicted labels.

    :ivar df: DataFrame containing the dataset loaded from the given CSV file.
    :ivar data_with_gt: Subset of the data frame containing rows with non-null ground truth values.
    :ivar Y_gt_transformed: Transformed ground truth data as integer indices.
    :ivar gt_labels: Array of unique ground truth label names.
    :ivar Y_pred_transformed: Transformed predicted data as integer indices.
    :ivar pred_labels: Array of unique predicted label names.
    :ivar labels: Dictionary mapping ground truth label strings to `label.Label` objects.
    :ivar num_classes: Number of unique ground truth classes in the dataset.
    :ivar main_model_name: Name of the primary model used for processing the data.
    :ivar train_true_data: Ground truth data for the training set.
    :ivar test_true_data: Ground truth data for the testing set.
    """

    def __init__(self,
                 data_str: str):
        super().__init__(data_str=data_str)
        self.df = pd.read_csv('COX/dataset_may_2024.csv')
        self.data_with_gt = self.df[self.df['gt'].notna()]
        self.Y_gt_transformed, self.gt_labels = pd.factorize(self.data_with_gt['gt'])
        self.Y_pred_transformed, self.pred_labels = pd.factorize(self.data_with_gt['pred'])

        assert np.all(
            [self.data_with_gt['gt'].iloc[i] == self.gt_labels[self.Y_gt_transformed[i]]
             for i in range(self.data_with_gt.shape[0])])
        assert np.all(
            [self.data_with_gt['pred'].iloc[i] == self.pred_labels[self.Y_pred_transformed[i]]
             for i in range(self.data_with_gt.shape[0])])
        assert set(self.gt_labels).intersection(set(self.pred_labels)) == set(self.pred_labels)

        self.labels = {str(int(l_original)): label.Label(l_str=str(int(l_original)),
                                                         index=l_index)
                       for l_index, l_original in enumerate(self.gt_labels)}
        self.num_classes = self.data_with_gt['gt'].nunique()

        self.main_model_name = 'main_model'
        self.train_true_data = np.load(get_filepath(data_str=data_str,
                                                    model_name=self.main_model_name,
                                                    test=False,
                                                    pred=False))
        self.test_true_data = np.load(get_filepath(data_str=data_str,
                                                   model_name=self.main_model_name,
                                                   test=True,
                                                   pred=False))

    def get_ground_truths(self,
                          test: bool) -> np.array:
        return self.test_true_data if test else self.train_true_data

    def get_labels(self) -> typing.Dict[str, label.Label]:
        return self.labels


class FineCoarseDataPreprocessor(DataPreprocessor):
    """
    FineCoarseDataPreprocessor class provides data preprocessing mechanisms for handling fine-grained
    to coarse-grained mappings of classes using datasets such as 'imagenet' and 'openimage'. It relies
    on configurations that define relationships between specific fine-grained classes and their broader
    coarse-grained categories. This class is designed to streamline the process of creating such mappings
    and supports customization for different datasets.

    :ivar granularities_str: List of granularity identifiers as strings, used for establishing
        different levels of granularity.
    :ivar granularities: Dictionary mapping granularity strings to their respective granularity objects,
        allowing access to granularity-specific information.
    :ivar fine_to_course_idx: Dictionary mapping coarse-grained categories to a list of their corresponding
        fine-grained indices. Populated during preprocessing.
    :ivar coarse_grain_classes_str: List of coarse-grained class strings based on the dataset chosen
        during initialization, representing larger category names.
    :ivar fine_grain_classes_str: List of fine-grained class strings derived from mapping dictionaries,
        representing specific subcategories related to raw data classes.
    :ivar fine_grain_mapping_dict: Dictionary containing mappings of image class IDs to their detailed
        textual names, which is relevant for fine-grained processing.
    :ivar fine_to_coarse: Dictionary mapping fine-grained class strings to the corresponding
        coarse-grained category label, organizing detailed observations into higher-level insights.
    :ivar coarse_to_fine: Dictionary mapping coarse-grained category labels to lists of their
        corresponding fine-grained class strings, enabling granular inspection within broad categories.
    """
    granularities_str = ['fine', 'coarse']
    granularities = {g_str: granularity.Granularity(g_str=g_str) for g_str in granularities_str}

    def __init__(self,
                 data_str: str):
        super().__init__(data_str=data_str)
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

        elif data_str == 'openimage':
            self.coarse_to_fine = {
                "Clothing": [
                    "Footwear",
                    "Fashion accessory",
                    "Dress",
                    "Suit",
                    "Sports uniform",
                    "Trousers",
                    "Shorts",
                    "Swimwear",
                    "Jacket"
                ],
                "Food": [
                    "Snack",
                    "Baked goods",
                    "Dessert",
                    "Fruit",
                    "Vegetable",
                    "Fast food",
                    "Seafood"
                ],
                "Furniture": [
                    "Table",
                    "Chair",
                    "Shelf",
                    "Desk"
                ],
                "Land vehicle": [
                    "Car",
                    "Truck",
                    "Bicycle",
                    "Motorcycle"
                ],
                "Mammal": [
                    "Carnivore",
                    "Horse",
                    "Monkey"
                ],
                "Plant": [
                    "Tree",
                    "Flower",
                    "Houseplant"
                ],
            }
            self.fine_grain_classes_str = sorted(
                [item for category, items in self.coarse_to_fine.items() for item in items])
            self.coarse_grain_classes_str = sorted([item for item in self.coarse_to_fine.keys()])

            self.fine_to_coarse = {}
            for fine_grain_class_idx, fine_grain_class in enumerate(self.fine_grain_classes_str):
                for coarse_grain_class, fine_grain_class_list in self.coarse_to_fine.items():
                    if fine_grain_class in fine_grain_class_list:
                        self.fine_to_coarse[fine_grain_class] = coarse_grain_class

        elif data_str == 'coco':
            self.coarse_to_fine = {
                "vehicle": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
                "animal": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
                "sports": ["ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis"],
                "food": ["banana", "apple", "sandwich", "orange", "broccoli", "carrot"],
                "electronic": ["tv", "laptop", "mouse", "remote", "keyboard"],
                "kitchen": ["microwave", "oven", "toaster", "sink", "refrigerator"]
            }
            self.fine_grain_classes_str = sorted(
                [item for category, items in self.coarse_to_fine.items() for item in items])
            self.coarse_grain_classes_str = sorted([item for item in self.coarse_to_fine.keys()])

            self.fine_to_coarse = {}
            for fine_grain_class_idx, fine_grain_class in enumerate(self.fine_grain_classes_str):
                for coarse_grain_class, fine_grain_class_list in self.coarse_to_fine.items():
                    if fine_grain_class in fine_grain_class_list:
                        self.fine_to_coarse[fine_grain_class] = coarse_grain_class
        else:
            self.coarse_to_fine = {
                'Air Defense': ['30N6E', 'Iskander', 'Pantsir-S1', 'Rs-24'],
                'BMP': ['BMP-1', 'BMP-2', 'BMP-T15'],
                'BTR': ['BRDM', 'BTR-60', 'BTR-70', 'BTR-80'],
                'Tank': ['T-14', 'T-62', 'T-64', 'T-72', 'T-80', 'T-90'],
                'Self Propelled Artillery': ['2S19_MSTA', 'BM-30', 'D-30', 'Tornado', 'TOS-1'],
                'BMD': ['BMD'],
                'MT_LB': ['MT_LB']
            }

            data_file_path = rf'../../../data/Military Vehicles/WEO_Data_Sheet.xlsx'
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

        self.coarse_to_fine = {}

        for fine_idx, (fine_class, coarse_class) in enumerate(self.fine_to_coarse.items()):
            coarse_idx = self.coarse_grain_classes_str.index(coarse_class)
            self.fine_to_course_idx[fine_idx] = coarse_idx

            if coarse_class not in self.coarse_to_fine:
                self.coarse_to_fine[coarse_class] = [fine_class]
            else:
                self.coarse_to_fine[coarse_class].append(fine_class)

        if data_str == 'imagenet':
            self.data_path_str = '../../../data/ImageNet50/'
        elif data_str == 'openimage':
            self.data_path_str = (f'../../ngocbach/' if not utils.is_local() else 'data/') + 'OpenImage/'
        elif data_str == 'coco':
            self.data_path_str = 'scratch/ngocbach/COCO/'
        else:
            self.data_path_str = '../../../data/Military Vehicles/'

        self.num_fine_grain_classes = len(self.fine_grain_classes_str)
        self.num_coarse_grain_classes = len(self.coarse_grain_classes_str)

        self.test_true_fine_data = np.load(rf'{self.data_path_str}test_fine/test_true_fine.npy')
        self.test_true_coarse_data = np.load(rf'{self.data_path_str}test_coarse/test_true_coarse.npy')

        self.train_true_fine_data = np.load(rf'{self.data_path_str}train_fine/train_true_fine.npy')
        self.train_true_coarse_data = np.load(rf'{self.data_path_str}train_coarse/train_true_coarse.npy')

        # self.noisy_train_true_fine_data = self.train_true_fine_data.copy()
        # self.noisy_train_true_coarse_data = self.train_true_coarse_data.copy()

        self.fine_unique, self.fine_counts = np.unique(self.train_true_fine_data, return_counts=True)
        self.coarse_unique, self.coarse_counts = np.unique(self.train_true_coarse_data, return_counts=True)

        # # Create dictionaries from unique labels and counts
        self.fine_data_counts = dict(zip(self.fine_unique, self.fine_counts))
        self.coarse_data_counts = dict(zip(self.coarse_unique, self.coarse_counts))

        self.fine_grain_labels = {l_str: label.FineGrainLabel(g=self.granularities['fine'],
                                                              l_str=l_str,
                                                              fine_grain_classes_str=self.fine_grain_classes_str)
                                  for l_str in self.fine_grain_classes_str}
        self.coarse_grain_labels = {l_str: label.CoarseGrainLabel(g=self.granularities['coarse'],
                                                                  l_str=l_str,
                                                                  coarse_grain_classes_str=
                                                                  self.coarse_grain_classes_str)
                                    for l_str in self.coarse_grain_classes_str}

        assert (self.get_num_inconsistencies(fine_labels=self.train_true_fine_data,
                                             coarse_labels=self.train_true_coarse_data)[0] ==
                self.get_num_inconsistencies(fine_labels=self.test_true_fine_data,
                                             coarse_labels=self.test_true_coarse_data)[0] == 0)

    def get_ground_truths(self,
                          test: bool,
                          K: typing.List[int] = None,
                          g: typing.Union[granularity.Granularity, str] = None,
                          ) -> np.array:
        if test:
            true_fine_data = self.test_true_fine_data
            true_coarse_data = self.test_true_coarse_data
        else:
            true_fine_data = self.train_true_fine_data
            true_coarse_data = self.train_true_coarse_data

        if g is None:
            return (true_fine_data[K], true_coarse_data[K]) if K is not None else (true_fine_data, true_coarse_data)
        else:
            return (true_fine_data[K] if str(g) == 'fine' else true_coarse_data[K]) if K is not None else \
                (true_fine_data if str(g) == 'fine' else true_coarse_data)

    def get_num_of_train_fine_examples(self,
                                       fine_l_index: int) -> int:
        return np.where(self.train_true_fine_data == fine_l_index)[0].shape[0]

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
                   g: granularity.Granularity) -> typing.Dict[str, label.Label]:
        return self.fine_grain_labels if str(g) == 'fine' else self.coarse_grain_labels

    def get_subset_indices_for_train_and_train_eval(self,
                                                    train_eval_split: float,
                                                    get_fraction_of_example_with_label: typing.Dict[
                                                        label.Label, float] = None, ):
        """
        Splits the dataset indices into training and training-evaluation sets based on
        a specified ratio and optionally adjusts the training dataset indices based on the
        fractions of examples to retain for each label. This method is primarily utilized
        to control and balance the proportions of data labels in the respective datasets
        after the split.

        :param train_eval_split: A float between 0 and 1 representing the fraction of
            the dataset to be included in the training set. The complement is used for
            the training-evaluation set. The value should ideally represent the
            proportionality required for training data.
        :param get_fraction_of_example_with_label: A dictionary mapping each label to a
            float that defines the fraction of examples corresponding to that label to
            remove from the training dataset. This parameter allows fine control over
            label representation in the training set. If None, the dataset will only be
            split by the train_eval_split parameter without considering label-specific
            fractions.
        :return: A tuple containing:
            - train_indices: A NumPy array of indices corresponding to the training
              subset of the original dataset. If `get_fraction_of_example_with_label` is
              provided, the indices are filtered according to the specified fractions for
              each label.
            - train_eval_indices: A NumPy array of indices corresponding to the
              training-evaluation subset of the original dataset. This set is computed
              using the complement of train_eval_split.
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

    def copy_subset_files(self, destination_folder: str):
        """
        Copies the subset of image files defined by the class into a designated folder.

        :param destination_folder: The folder where the image files will be copied.
        """
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        for dataset_type in ['train', 'test']:
            for granularity in ['fine', 'coarse']:
                data_path = rf"{self.data_path_str}{dataset_type}_{granularity}/"
                dataset_destination_folder = os.path.join(destination_folder, f"{dataset_type}_{granularity}")

                if not os.path.exists(dataset_destination_folder):
                    os.makedirs(dataset_destination_folder)

                if granularity == 'fine':
                    for identifier, class_name in tqdm.tqdm(self.fine_grain_mapping_dict.items(),
                                                            desc=f'Copying {dataset_type} {granularity} files'):
                        class_folder = os.path.join(data_path, identifier)
                        if os.path.exists(class_folder):
                            destination_class_folder = os.path.join(dataset_destination_folder, identifier)
                            if not os.path.exists(destination_class_folder):
                                os.makedirs(destination_class_folder)
                            for file in os.listdir(class_folder):
                                file_path = os.path.join(class_folder, file)
                                shutil.copy(file_path, destination_class_folder)

                # Copy ground truth .npy file
                ground_truth_file = os.path.join(self.data_path_str, f"{dataset_type}_{granularity}",
                                                 f"{dataset_type}_true_{granularity}.npy")
                if os.path.exists(ground_truth_file):
                    shutil.copy(ground_truth_file, dataset_destination_folder)



if __name__ == '__main__':
    preprocessor = FineCoarseDataPreprocessor(data_str='imagenet')
    # print(preprocessor.train_true_fine_data.shape)
    preprocessor.copy_subset_files('data/ImageNet100/subset')
