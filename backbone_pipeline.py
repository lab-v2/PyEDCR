import numpy as np
import torch.utils.data
import typing
import inspect
import importlib.util
import sys

import models
import utils
import data_preprocessing
import config

# Use the configuration variables:
batch_size = config.batch_size
scheduler_gamma = config.scheduler_gamma
num_epochs = config.num_epochs
ltn_num_epochs = config.ltn_num_epochs
vit_model_names = config.vit_model_names

combined_results_path = config.combined_results_path
individual_results_path = config.individual_results_path
binary_results_path = config.binary_results_path

scheduler_step_size = config.scheduler_step_size


def save_prediction_files(data_str: str,
                          test: bool,
                          fine_tuners: typing.Union[models.FineTuner, typing.Dict[str, models.FineTuner]],
                          combined: bool,
                          lrs: typing.Union[str, float, typing.Dict[str, typing.Union[str, float]]],
                          fine_prediction: np.array,
                          coarse_prediction: np.array,
                          epoch: int = None,
                          loss: str = 'BCE',
                          fine_ground_truths: np.array = None,
                          coarse_ground_truths: np.array = None,
                          fine_lower_predictions: typing.Dict[int, list] = {},
                          coarse_lower_predictions: typing.Dict[int, list] = {}):
    """
    Saves prediction files and optional ground truth files.

    :param data_str:
    :param coarse_lower_predictions:
    :param fine_lower_predictions:
    :param test: True for test data, False for training data.
    :param fine_tuners: A single FineTuner object (for combined models) or a
                       dictionary of FineTuner objects (for individual models).
    :param combined: Whether the model are individual or combine one.
    :param lrs: The learning rate(s) used during training.
    :param fine_prediction: NumPy array of fine-grained predictions.
    :param coarse_prediction: NumPy array of coarse-grained predictions.
    :param epoch: The epoch number (optional).
    :param loss: The loss function used during training (optional).
    :param fine_ground_truths: NumPy array of true fine-grained labels (optional).
    :param coarse_ground_truths: NumPy array of true coarse-grained labels (optional).
    """
    epoch_str = f'_e{epoch}' if epoch is not None else ''
    test_str = 'test' if test else 'train'

    if combined:
        for g_str in data_preprocessing.DataPreprocessor.granularities_str:
            if config.get_ground_truth:
                break
            prediction = fine_prediction if g_str == 'fine' else coarse_prediction
            np.save(models.get_filepath(data_str=data_str,
                                        model_name=fine_tuners,
                                        combined=True,
                                        test=test,
                                        granularity=g_str,
                                        loss=loss,
                                        lr=lrs,
                                        pred=True,
                                        epoch=epoch),
                    prediction)

            lower_predictions = fine_lower_predictions if g_str == 'fine' else coarse_lower_predictions
            for lower_prediction_index, lower_prediction_values in lower_predictions.items():
                np.save(models.get_filepath(data_str=data_str,
                                            model_name=fine_tuners,
                                            combined=True,
                                            test=test,
                                            granularity=g_str,
                                            loss=loss,
                                            lr=lrs,
                                            pred=True,
                                            epoch=epoch,
                                            lower_prediction_index=lower_prediction_index),
                        lower_prediction_values)

        if data_str == 'imagenet':
            data_path_str = 'data/ImageNet100/'
        elif data_str == 'openimage':
            data_path_str = 'scratch/ngocbach/OpenImage/' if not config.running_on_sol \
                else '/scratch/ngocbach/OpenImage/'
        else:
            data_path_str = 'data/'

        if fine_ground_truths is not None:
            np.save(f"{data_path_str}{test_str}_fine/{test_str}_true_fine.npy",
                    fine_ground_truths)
            np.save(f"{data_path_str}{test_str}_coarse/{test_str}_true_coarse.npy",
                    coarse_ground_truths)
    else:
        np.save(f"{individual_results_path}_{test_str}_{fine_tuners['fine']}"
                f"_pred_lr{lrs['fine']}_{epoch_str}_fine_individual.npy",
                fine_prediction)
        np.save(f"{individual_results_path}_{test_str}_{fine_tuners['coarse']}"
                f"_pred_lr{lrs['coarse']}_{epoch_str}_coarse_individual.npy",
                coarse_prediction)

        # if fine_ground_truths is not None:
        #     np.save(f"{combined_results_path}{test_str}_true_fine.npy",
        #             fine_ground_truths)
        #     np.save(f"{combined_results_path}{test_str}_true_fine.npy",
        #             fine_ground_truths)


def save_binary_prediction_files(data_str: str,
                                 test: bool,
                                 fine_tuner: typing.Union[models.FineTuner, typing.Dict[str, models.FineTuner]],
                                 lr: typing.Union[str, float],
                                 predictions: np.array,
                                 l: data_preprocessing.Label,
                                 epoch: int = None,
                                 loss: str = 'BCE',
                                 ground_truths: np.array = None,
                                 evaluation: bool = False):
    """
    Saves prediction files and optional ground truth files.

    :param evaluation:
    :param l:
    :param ground_truths:
    :param predictions:
    :param test: True for test data, False for training data.
    :param fine_tuner: A single FineTuner object (for combined models).
    :param lr: The learning rate used during training.
    :param epoch: The epoch number (optional).
    :param loss: The loss function used during training (optional).
    """
    test_str = 'test' if test else 'train'

    np.save(models.get_filepath(model_name=fine_tuner,
                                l=l,
                                test=test,
                                loss=loss,
                                lr=lr,
                                pred=True,
                                epoch=epoch,
                                data_str=data_str),
            predictions)
    try:
        if data_str == 'imagenet':
            preprocessor = data_preprocessing.DataPreprocessor(data_str)
            for id, label_str in preprocessor.fine_grain_mapping_dict.items():
                if label_str == l.l_str:
                    np.save(f"data/ImageNet100/{test_str}_{l.g.g_str}/{id}/binary_true.npy",
                            ground_truths)
                    break

        elif data_str == 'openimage':
            np.save(f"scratch/ngocbach/OpenImage/{test_str}_{l.g.g_str}/{l}/binary_true.npy",
                    ground_truths)
        else:
            np.save(f"data/{test_str}_{l.g.g_str}/{l}/binary_true.npy",
                    ground_truths)
    except FileNotFoundError:
        print('cannot save binary model')
    except:
        print('do not know error')

    if not evaluation:
        torch.save(fine_tuner.state_dict(),
                   f"models/binary_models/binary_{l}_{fine_tuner}_lr{lr}_loss_{loss}_e{epoch}.pth")


def load_module_from_file(module_name, filepath):
    """
    Dynamically loads a module from the specified file path.

    Args:
    - module_name (str): The name of the module.
    - filepath (str): The path to the .py file to load.

    Returns:
    - module: The loaded module.
    """
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def find_subclasses_in_module(module, parent_class):
    """
    Finds all subclasses of a specified class within a given module.

    Args:
    - module (module): The module to inspect.
    - parent_class (type): The parent class to find subclasses for.

    Returns:
    - list[type]: A list of subclasses of the parent_class found in the module.
    """
    return [cls for name, cls in inspect.getmembers(module, inspect.isclass) if
            issubclass(cls, parent_class) and cls is not parent_class]


def initiate(data_str: str,
             lr: typing.Union[str, float],
             preprocessor: data_preprocessing.DataPreprocessor = None,
             model_name: str = 'vit_b_16',
             weights: str = 'DEFAULT',
             combined: bool = True,
             l: data_preprocessing.Label = None,
             pretrained_path: str = None,
             debug: bool = False,
             error_indices: typing.Sequence = None,
             evaluation: bool = False,
             train_eval_split: float = None,
             get_fraction_of_example_with_label: typing.Dict[data_preprocessing.Label, float] = None,
             train_fine_predictions: np.array = None,
             train_coarse_predictions: np.array = None,
             test_fine_predictions: np.array = None,
             test_coarse_predictions: np.array = None,
             ):
    """
    Initializes models, datasets, and devices for training.

    :param train_coarse_predictions:
    :param train_fine_predictions:
    :param preprocessor:
    :param data_str:
    :param get_fraction_of_example_with_label:
    :param train_eval_split:
    :param weights:
    :param model_name:
    :param l:
    :param evaluation:
    :param error_indices:
    :param lr: List of learning rates for the models.
    :param combined: Whether the model are individual or combine one.
    :param pretrained_path: Path to a pretrained model (optional).
    :param debug: True to force CPU usage for debugging.
    :return: A tuple containing:
             - fine_tuners: A list of VITFineTuner model objects.
             - loaders: A dictionary of data loaders for train, val, and test.
             - devices: A list of torch.device objects for model placement.
             - num_fine_grain_classes: The number of fine-grained classes.
             - num_coarse_grain_classes: The number of coarse-grained classes.
    """
    print(f'Model: {model_name}\nLearning rate: {lr}')

    if preprocessor is None:
        preprocessor = data_preprocessing.DataPreprocessor(data_str=data_str)

    datasets = data_preprocessing.get_datasets(
        preprocessor=preprocessor,
        combined=combined,
        binary_label=l,
        evaluation=evaluation,
        error_fixing=error_indices is not None,
        model_name=model_name,
        weights=weights,
        train_fine_predictions=train_fine_predictions,
        train_coarse_predictions=train_coarse_predictions,
        test_fine_predictions=test_fine_predictions,
        test_coarse_predictions=test_coarse_predictions
    )

    device = torch.device('cpu') if debug else (
        torch.device('mps' if utils.is_local() and torch.backends.mps.is_available() else
                     ("cuda" if torch.cuda.is_available() else 'cpu')))
    devices = [device]
    print(f'Using {device}')

    available_models = find_subclasses_in_module(module=models,
                                                 parent_class=models.FineTuner)

    model_class = next(curr_model_class for curr_model_class in available_models
                       if curr_model_class.__name__.split('FineTuner')[0].lower().
                       startswith(model_name[:3]))

    if train_fine_predictions is not None:
        results_path = binary_results_path
        fine_tuners = None
        # if pretrained_path is None:
        #     fine_tuners = [models.ErrorDetector(model_name=model_name,
        #                                         num_classes=preprocessor.num_fine_grain_classes + preprocessor.
        #                                         num_coarse_grain_classes,
        #                                         preprocessor=preprocessor)]
        # else:
        #     fine_tuners = [models.ErrorDetector.from_pretrained(
        #         model_name=model_name,
        #         num_classes=preprocessor.num_fine_grain_classes + preprocessor.num_coarse_grain_classes,
        #         pretrained_path=pretrained_path,
        #         device=device,
        #         preprocessor=preprocessor
        #     )]
    else:
        if combined:
            results_path = combined_results_path if l is None else binary_results_path
            num_classes = preprocessor.num_fine_grain_classes + preprocessor.num_coarse_grain_classes \
                if l is None else 2

            if pretrained_path is not None:
                print(f'Loading pretrained model from {pretrained_path}')
                if model_name == 'tresnet_m':
                    fine_tuners = [model_class.from_pretrained(model_name=model_name,
                                                               num_classes=num_classes,
                                                               pretrained_path=pretrained_path,
                                                               preprocessor=preprocessor,
                                                               device=device)]
                else:
                    fine_tuners = [model_class.from_pretrained(model_name=model_name,
                                                               num_classes=num_classes,
                                                               pretrained_path=pretrained_path,
                                                               device=device)]
            else:
                fine_tuners = [model_class(model_name=model_name,
                                           num_classes=num_classes)]

        else:
            results_path = individual_results_path

            if torch.cuda.device_count() < 2:
                raise ValueError("This setup requires at least 2 GPUs.")

            devices = [torch.device("cuda:0"), torch.device("cuda:1")]

            fine_tuners = ([models.VITFineTuner(model_name=model_name,
                                                num_classes=preprocessor.num_fine_grain_classes),
                            models.VITFineTuner(model_name=model_name,
                                                num_classes=preprocessor.num_coarse_grain_classes)])

    utils.create_directory(results_path)
    loaders = data_preprocessing.get_loaders(preprocessor=preprocessor,
                                             datasets=datasets,
                                             batch_size=batch_size,
                                             evaluation=evaluation,
                                             subset_indices=error_indices,
                                             train_eval_split=train_eval_split,
                                             label=l,
                                             get_fraction_of_example_with_label=get_fraction_of_example_with_label,
                                             debug=debug)

    print(f"Total number of train images: {len(loaders['train'].dataset)}\n"
          f"Total number of eval images: {len(loaders['train_eval'].dataset) if train_eval_split else 0}\n"
          f"Total number of test images: {len(loaders['test'].dataset)}")

    assert error_indices is None or len(loaders['train'].dataset) == len(error_indices)
    return preprocessor, fine_tuners, loaders, devices
