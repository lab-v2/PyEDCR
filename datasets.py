import os
import numpy as np
import pathlib
import typing
import abc
import random

import torch
import torch.utils.data
import torchvision

import data_preprocessor
import config
import label


random.seed(42)
np.random.seed(42)

current_file_location = pathlib.Path(__file__).parent.resolve()
os.chdir(current_file_location)



def get_dataset_transforms(data: str,
                           train_or_test: str,
                           model_name: str = 'vit_b_16',
                           error_fixing: bool = False,
                           weight: str = 'DEFAULT') -> torchvision.transforms.Compose:
    """
    Returns the transforms required for the VIT for training or test datasets
    """

    if data == 'imagenet':
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

        test_transforms = [torchvision.transforms.Resize(256),
                           torchvision.transforms.CenterCrop(224)
                           ]

    elif data == 'openimage' or data == 'coco':
        # normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                              std=[0.229, 0.224, 0.225])
        standard_transforms = [
            torchvision.transforms.ToTensor(),
        ]
        train_transforms = [
            torchvision.transforms.Resize((224, 224)),
        ]

        test_transforms = [torchvision.transforms.Resize((224, 224)),
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


class EDCRImageFolder(abc.ABC, torchvision.datasets.ImageFolder):
    """
    Represents a custom image folder dataset for loading specific classes of images.

    This class extends the torchvision.datasets.ImageFolder and is designed to include
    functionality for selectively loading and processing only relevant classes of images
    from a given root directory. It provides mechanisms to identify valid image classes
    and implements additional dataset balancing functionality.

    :ivar relevant_classes: A list of relevant class names to include in the dataset.
        If None, all classes in the dataset directory are considered relevant.
    """

    def __init__(
            self,
            root: str,
            transform: typing.Callable = None,
            target_transform: typing.Callable = None,
            relevant_classes: typing.List[str] = None):
        self.relevant_classes = relevant_classes
        super().__init__(root=root,
                         transform=transform,
                         target_transform=target_transform)

    def find_classes(self,
                     directory: str) -> (typing.List[str], typing.Dict[str, int]):
        """
        Finds and returns class folder names and a mapping of these names to numeric indices.

        This method scans through a given directory looking for subdirectories
        interpreted as "class folders". It excludes directories starting with a
        dot (hidden directories) and can optionally filter directories based on
        a specified set of relevant class names. An exception is raised if no
        such class folders are found in the directory.

        :param directory: The path to the directory to scan for class folders.
        :raises FileNotFoundError: If no valid class folders are found in the directory.
        :return: A tuple containing a list of class folder names sorted
                 alphabetically and a dictionary mapping class folder names
                 to numeric indices.
        """
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir() and
                         not entry.name.startswith('.') and (self.relevant_classes is None
                                                             or entry.name in self.relevant_classes))
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: index for index, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def balance_samples(self):
        """
        Balances the dataset by resampling or adjusting the counts of
        different classes to ensure they are equally represented. This
        method is typically used in cases where class imbalance might
        negatively impact the performance of a machine learning model.
        It adjusts the sample distribution without modifying the original
        data integrity.
        """
        pass


class CombinedImageFolderWithName(EDCRImageFolder):
    """
    CombinedImageFolderWithName class that extends functionality of EDCRImageFolder.

    This class provides functionality for loading and processing images from a
    specific folder structure, while incorporating a preprocessor for handling
    fine-grained and coarse-grained class mappings. It also appends additional
    information, such as file identifiers and coarse-grain indices, to the loaded
    data samples.

    :ivar preprocessor: Instance of FineCoarseDataPreprocessor that handles fine to
        coarse index mapping and relevant class filtering.
    """

    def __init__(self,
                 root: str,
                 preprocessor: data_preprocessor.FineCoarseDataPreprocessor,
                 transform=None,
                 target_transform=None):
        self.preprocessor = preprocessor
        super().__init__(root=root,
                         transform=transform,
                         target_transform=target_transform,
                         relevant_classes=
                         list(self.preprocessor.fine_grain_mapping_dict.keys())
                         if preprocessor.data_str == 'imagenet' else None)

    def __getitem__(self,
                    index: int) -> (torch.tensor, int, str):
        """
        Fetches the sample data indexed by the input index from the dataset and
        applies any defined transformations to the input data and its target. Returns
        the processed sample data, including the input tensor, fine-grained target,
        unique identifier of the input, coarse-grained target, and the index itself.

        :param index: The position of the sample in the dataset.

        :return: A tuple containing the input tensor, the fine-grained target index,
            a unique string identifier for the data sample, a coarse-grained target
            index, and the sample's index in the dataset.
        """
        x_path, y_fine_grain = self.samples[index]
        y_coarse_grain = self.preprocessor.fine_to_course_idx[y_fine_grain]
        x = self.loader(x_path)

        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y_fine_grain = self.target_transform(y_fine_grain)
        name = os.path.basename(x_path)
        folder_path = os.path.basename(os.path.dirname(x_path))

        x_identifier = f'{folder_path}/{name}'

        return x, y_fine_grain, x_identifier, y_coarse_grain, index


class ErrorDetectorImageFolder(EDCRImageFolder):
    """
    Handles image datasets, incorporating fine and coarse-grained prediction results to identify
    errors and differentiate samples into positive and negative ones based on a defined criterion.
    Facilitates error detection and balanced sample preparation for evaluation and further processing.

    This class extends `EDCRImageFolder`, adding functionality for precise error detection by
    utilizing both fine-grained and coarse-grained predictions and ground truth values. It determines
    if an error occurred and categorizes the dataset into positive and negative samples accordingly.
    Additionally, it supports balanced sampling for training or evaluation purposes when required.

    :ivar preprocessor: Preprocessor for handling data transformations, including mappings for fine
        and coarse-grained classifications.
    :ivar fine_predictions: Fine-grained prediction labels for the dataset.
    :ivar coarse_predictions: Coarse-grained prediction labels for the dataset.
    :ivar original_number_of_samples: Total number of samples originally in the dataset.
    :ivar samples: Processed list of samples containing paths, predictions, and an error flag.
    :ivar positive_samples: Subset of the dataset containing only positive (error) samples.
    :ivar negative_samples: Subset of the dataset containing only negative (non-error) samples.
    """

    def __init__(self,
                 root: str,
                 preprocessor: data_preprocessor.FineCoarseDataPreprocessor,
                 fine_predictions: np.array,
                 coarse_predictions: np.array,
                 transform=None,
                 target_transform=None,
                 evaluation=True):
        self.preprocessor = preprocessor
        self.fine_predictions = fine_predictions
        self.coarse_predictions = coarse_predictions
        self.original_number_of_samples = fine_predictions.shape[0]

        super().__init__(root=root,
                         transform=transform,
                         target_transform=target_transform,
                         relevant_classes=
                         list(self.preprocessor.fine_grain_mapping_dict.keys())
                         if preprocessor.data_str == 'imagenet' else None)

        samples = []
        self.positive_samples = []
        self.negative_samples = []

        for index, (x_path, y_true_fine) in enumerate(self.samples):
            y_pred_fine = self.fine_predictions[index]
            y_pred_coarse = self.coarse_predictions[index]
            y_true_coarse = self.preprocessor.fine_to_course_idx[y_true_fine]
            error = self.get_error(y_pred_fine=y_pred_fine,
                                   y_pred_coarse=y_pred_coarse,
                                   y_true_fine=y_true_fine,
                                   y_true_coarse=y_true_coarse)
            element = (x_path, y_pred_fine, y_pred_coarse, error)
            if error:
                self.positive_samples += [element]
            else:
                self.negative_samples += [element]
            samples += [element]

        self.samples = samples
        if not evaluation:
            self.balance_samples()

    @staticmethod
    def get_error(y_pred_fine: int,
                  y_pred_coarse,
                  y_true_fine,
                  y_true_coarse) -> int:
        """
        Computes an error flag indicating whether the predictions match the true
        values for both fine and coarse labels. Returns 1 if there is any mismatch
        between predictions and true values for the fine or coarse labels; otherwise,
        returns 0.

        :param y_pred_fine: Predicted value for the fine-grained classification.
        :param y_pred_coarse: Predicted value for the coarse-grained classification.
        :param y_true_fine: True value for the fine-grained classification.
        :param y_true_coarse: True value for the coarse-grained classification.
        :return: Error flag, 1 if there is a mismatch, 0 otherwise.
        """
        return int(y_pred_fine != y_true_fine or y_pred_coarse != y_true_coarse)

    def balance_samples(self):
        """
        Balances the samples by extending the positive samples to match
        the ratio of negative to positive samples. This method calculates
        the ratio of the number of negative samples to positive samples and
        duplicates the positive samples in such a way that the distribution
        is balanced relative to the negative samples.

        :raises ZeroDivisionError: if the length of `positive_samples` is zero.
        """
        ratio = int(len(self.negative_samples) / len(self.positive_samples))
        self.samples.extend(self.positive_samples * ratio)

    def __getitem__(self,
                    index: int) -> (torch.tensor, int, str):
        """
        Retrieves and processes the item at the given index in the dataset. The data
        sample includes an input tensor obtained by loading and optionally transforming
        the data, as well as the corresponding predictions and error.

        :param index: The index of the sample to retrieve.
        :return: A tuple containing the processed input tensor, fine-grained prediction,
            coarse-grained prediction, and associated error of the sample.
        """
        x_path, y_pred_fine, y_pred_coarse, error = self.samples[index]
        x = self.loader(x_path)

        if self.transform is not None:
            x = self.transform(x)

        return x, y_pred_fine, y_pred_coarse, error


class BinaryImageFolder(EDCRImageFolder):
    def __init__(self,
                 root: str,
                 preprocessor: data_preprocessor.FineCoarseDataPreprocessor,
                 l: label.Label,
                 transform: typing.Optional[typing.Callable] = None,
                 evaluation: bool = False, ):
        self.evaluation = evaluation
        self.preprocessor = preprocessor
        self.l = l

        if self.preprocessor.data_str == 'imagenet':
            relevant_class = list(preprocessor.fine_grain_mapping_dict.keys())
        else:
            relevant_class = None

        super().__init__(root=root,
                         transform=transform,
                         target_transform=transform,
                         relevant_classes=relevant_class)
        # self.balance_samples()

    # def balance_samples(self):
    #     if not self.evaluation:
    #         positive_example = [(x_path, target) for x_path, target in self.samples if target == self.l.index]
    #         self.samples.extend(positive_example * self.preprocessor.num_fine_grain_classes)

    def __getitem__(self, index: int):
        x_path, y = self.samples[index]
        x = self.loader(x_path)

        if self.transform is not None:
            x = self.transform(x)

        if self.l.g.g_str == 'fine':
            y = int(y == self.l.index)
        else:
            y = int(self.preprocessor.fine_to_course_idx[y] == self.l.index)

        return x, y


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

        x_path, y = self.samples[index]
        x = self.loader(x_path)

        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)

        name = os.path.basename(x_path)
        folder_path = os.path.basename(os.path.dirname(x_path))

        x_identifier = f'{folder_path}/{name}'

        return x, y, x_identifier


def get_datasets(preprocessor: data_preprocessor.FineCoarseDataPreprocessor,
                 model_name: str,
                 weights: str = 'DEFAULT',
                 cwd: typing.Union[str, pathlib.Path] = os.getcwd(),
                 combined: bool = True,
                 binary_label: label.Label = None,
                 evaluation: bool = False,
                 error_fixing: bool = False,
                 train_fine_predictions: np.array = None,
                 train_coarse_predictions: np.array = None,
                 test_fine_predictions: np.array = None,
                 test_coarse_predictions: np.array = None,
                 ) -> \
        (typing.Dict[str, torchvision.datasets.ImageFolder], int, int):
    """
    Instantiates and returns train and test datasets

    Parameters
    ----------
        :param test_coarse_predictions:
        :param test_fine_predictions:
        :param train_coarse_predictions:
        :param train_fine_predictions:
        :param preprocessor:
        :param weights:
        :param model_name:
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
            full_data_dir = f'~/../../scratch/ngocbach/OpenImage/{train_or_test}_fine' if not config.running_on_sol \
                else f'~/../../scratch/ngocbach/OpenImage/{train_or_test}_fine'
        elif preprocessor.data_str == 'coco':
            full_data_dir = f'~/../../scratch/ngocbach/COCO/{train_or_test}_fine'
        else:
            data_dir_name = f'ImageNet100/{train_or_test}_fine' if preprocessor.data_str == 'imagenet' \
                else f'{train_or_test}_fine'
            full_data_dir = os.path.join(data_dir, data_dir_name)

        if train_fine_predictions is not None:
            print(f'get error detector loader for {train_or_test}')
            datasets[train_or_test] = ErrorDetectorImageFolder(
                root=full_data_dir,
                preprocessor=preprocessor,
                fine_predictions=train_fine_predictions if train_or_test == 'train' else test_fine_predictions,
                coarse_predictions=train_coarse_predictions if train_or_test == 'train' else test_coarse_predictions,
                transform=get_dataset_transforms(
                    data=preprocessor.data_str,
                    train_or_test=train_or_test,
                    model_name=model_name))
        elif binary_label is not None:
            datasets[train_or_test] = BinaryImageFolder(root=full_data_dir,
                                                        transform=get_dataset_transforms(data=preprocessor.data_str,
                                                                                         train_or_test=train_or_test),
                                                        l=binary_label,
                                                        preprocessor=preprocessor,
                                                        evaluation=evaluation)
        elif combined:
            datasets[train_or_test] = CombinedImageFolderWithName(preprocessor=preprocessor,
                                                                  root=full_data_dir,
                                                                  transform=get_dataset_transforms(
                                                                      data=preprocessor.data_str,
                                                                      train_or_test=train_or_test,
                                                                      error_fixing=error_fixing,
                                                                      model_name=model_name,
                                                                      weight=weights))
        else:
            datasets[train_or_test] = IndividualImageFolderWithName(
                root=full_data_dir,
                transform=get_dataset_transforms(data=preprocessor.data_str,
                                                 train_or_test=train_or_test))

    return datasets


def get_loaders(preprocessor: data_preprocessor.FineCoarseDataPreprocessor,
                datasets: typing.Dict[str, torchvision.datasets.ImageFolder],
                batch_size: int,
                subset_indices: typing.Sequence = None,
                evaluation: bool = None,
                train_eval_split: float = None,
                label: label.Label = None,
                get_fraction_of_example_with_label: typing.Dict[label.Label, float] = None,
                debug: bool = False,
                binary_error_model: bool = False
                ) -> typing.Dict[str, torch.utils.data.DataLoader]:
    """
    Instantiates and returns train and test torch data loaders

    Parameters
    ----------
        :param binary_error_model:
        :param label:
        :param debug:
        :param preprocessor:
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
            get_fraction_of_example_with_label=get_fraction_of_example_with_label)

    for split in ['train', 'test'] + (['train_eval'] if train_eval_split is not None else []):
        relevant_dataset = datasets[split if split != 'train_eval' else 'train']

        if not debug and (subset_indices is None or split != 'train'):
            loader_dataset = relevant_dataset
        else:
            loader_dataset = torch.utils.data.Subset(dataset=relevant_dataset,
                                                     indices=subset_indices if not debug else [0])

        if debug:
            loader_dataset = torch.utils.data.Subset(dataset=relevant_dataset,
                                                     indices=[0])
        elif split == 'train' and train_eval_split is not None:
            loader_dataset = torch.utils.data.Subset(dataset=relevant_dataset,
                                                     indices=train_indices)
        elif split == 'train_eval' and train_eval_split is not None:
            loader_dataset = torch.utils.data.Subset(dataset=relevant_dataset,
                                                     indices=train_eval_indices)

        # define sampler for binary model only, and on train dataset only
        if (label is not None or binary_error_model) and train_eval_split and split == 'train':
            train_true_data = preprocessor.train_true_fine_data if label.g.g_str == 'fine' \
                else preprocessor.train_true_coarse_data
            num_example_of_l = preprocessor.fine_counts[label.index] if label.g.g_str == 'fine' \
                else preprocessor.coarse_counts[label.index]
            weight = [0, 0]
            weight[1] = 1 / num_example_of_l
            weight[0] = 1 / (len(train_true_data) - num_example_of_l)
            samples_weight = np.array(
                [weight[idx]
                 for idx in np.where(train_true_data[train_indices] == label.index, 1, 0)])
            samples_weight = torch.from_numpy(samples_weight)
            samples_weight = samples_weight.double()
            sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))

            loaders[split] = torch.utils.data.DataLoader(
                dataset=loader_dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=4,
            )
            continue

        loaders[split] = torch.utils.data.DataLoader(
            dataset=loader_dataset,
            batch_size=batch_size,
            shuffle=split == 'train' and (evaluation is None or not evaluation),
            num_workers=4,
        )

    return loaders


def get_one_hot_encoding(input_arr: np.array) -> np.array:
    return np.eye(np.max(input_arr) + 1)[input_arr].T
