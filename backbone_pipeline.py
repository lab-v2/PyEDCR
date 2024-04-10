import numpy as np
import torch.utils.data
import typing

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


# original_prediction_weight = config.original_prediction_weight


# vit_model_names = [f'vit_{vit_model_name}' for vit_model_name in ['b_16']]


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

        data_path_str = 'ImageNet100/' if data_str == 'imagenet' else ''

        if fine_ground_truths is not None:
            np.save(f"data/{data_path_str}{test_str}_fine/{test_str}_true_fine.npy",
                    fine_ground_truths)
            np.save(f"data/{data_path_str}{test_str}_coarse/{test_str}_true_coarse.npy",
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


def save_binary_prediction_files(test: bool,
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
                                epoch=epoch),
            predictions)

    np.save(f"data/{test_str}_{l.g.g_str}/{l}/binary_true.npy",
            ground_truths)

    if not evaluation:
        torch.save(fine_tuner.state_dict(),
                   f"models/binary_models/binary_{l}_{fine_tuner}_lr{lr}_loss_{loss}_e{epoch}.pth")


def initiate(data_str: str,
             lrs: typing.List[typing.Union[str, float]],
             model_names: typing.List[str] = ['vit_b_16'],
             weights: typing.List[str] = ['DEFAULT'],
             combined: bool = True,
             l: data_preprocessing.Label = None,
             pretrained_path: str = None,
             debug: bool = False,
             error_indices: typing.Sequence = None,
             evaluation: bool = False,
             train_eval_split: float = None,
             get_indices: bool = False,
             get_fraction_of_example_with_label: typing.Dict[data_preprocessing.Label, float] = None,
             print_counts: bool = True):
    """
    Initializes models, datasets, and devices for training.

    :param data_str:
    :param print_counts:
    :param get_fraction_of_example_with_label:
    :param get_indices:
    :param train_eval_split:
    :param weights:
    :param model_names:
    :param l:
    :param evaluation:
    :param error_indices:
    :param lrs: List of learning rates for the models.
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
    print(f'Models: {model_names}\n'
          f'Learning rates: {lrs}')

    preprocessor = data_preprocessing.DataPreprocessor(data_str=data_str)

    datasets = data_preprocessing.get_datasets(preprocessor=preprocessor,
                                               combined=combined,
                                               binary_label=l,
                                               evaluation=evaluation,
                                               error_fixing=error_indices is not None,
                                               model_names=model_names,
                                               weights=weights)


    device = torch.device('cpu') if debug else (
        torch.device('mps' if utils.is_local() and torch.backends.mps.is_available() else
                     ("cuda" if torch.cuda.is_available() else 'cpu')))
    devices = [device]
    print(f'Using {device}')

    num_fine_grain_classes, num_coarse_grain_classes = None, None

    if l is not None:
        results_path = binary_results_path
        fine_tuners = [models.VITFineTuner(vit_model_name=vit_model_name,
                                           num_classes=2)
                       for vit_model_name in model_names]
    else:
        if combined:
            num_classes = preprocessor.num_fine_grain_classes + preprocessor.num_coarse_grain_classes

            if pretrained_path is not None:
                print(f'Loading pretrained model from {pretrained_path}')
                fine_tuners = [models.VITFineTuner.from_pretrained(vit_model_name=model_name,
                                                                   num_classes=num_classes,
                                                                   pretrained_path=pretrained_path,
                                                                   device=device) if model_name.startswith('vit')
                               else models.DINOV2FineTuner.from_pretrained(dino_v2_model_name=model_name,
                                                                           num_classes=num_classes,
                                                                           pretrained_path=pretrained_path,
                                                                           device=device)
                               for model_name in model_names]
            else:
                fine_tuners = [models.VITFineTuner(vit_model_name=model_name,
                                                   weights=weight,
                                                   num_classes=num_classes) if model_name.startswith('vit')
                               else (models.EfficientNetV2FineTuner(efficient_net_v2_model_name=model_name,
                                                                    num_classes=num_classes)
                                     if model_name.startswith('efficient_net_v2') else
                                     models.DINOV2FineTuner(dino_v2_model_name=model_name,
                                                            num_classes=num_classes))
                               for model_name, weight in zip(model_names, weights)]

            results_path = combined_results_path
        else:
            num_gpus = torch.cuda.device_count()

            if num_gpus < 2:
                raise ValueError("This setup requires at least 2 GPUs.")

            devices = [torch.device("cuda:0"), torch.device("cuda:1")]

            fine_tuners = ([models.VITFineTuner(vit_model_name=vit_model_name,
                                                num_classes=preprocessor.num_fine_grain_classes)
                            for vit_model_name in model_names] +
                           [models.VITFineTuner(vit_model_name=vit_model_name,
                                                num_classes=preprocessor.num_coarse_grain_classes)
                            for vit_model_name in model_names])
            results_path = individual_results_path

    utils.create_directory(results_path)
    loaders = data_preprocessing.get_loaders(preprocessor=preprocessor,
                                             datasets=datasets,
                                             batch_size=batch_size,
                                             evaluation=evaluation,
                                             subset_indices=error_indices,
                                             train_eval_split=train_eval_split,
                                             get_indices=get_indices,
                                             get_fraction_of_example_with_label=get_fraction_of_example_with_label)

    print(f"Total number of train images: {len(loaders['train'].dataset)}\n"
          f"Total number of eval images: {len(loaders['train_eval'].dataset) if train_eval_split else 0}\n"
          f"Total number of test images: {len(loaders['test'].dataset)}")

    assert error_indices is None or len(loaders['train'].dataset) == len(error_indices)

    if l is None:
        return preprocessor, fine_tuners, loaders, devices, num_fine_grain_classes, num_coarse_grain_classes
    else:
        positive_class_weight = preprocessor.get_imbalance_weight(l=l,
                                                                  train_images_num=len(loaders['train'].dataset),
                                                                  evaluation=evaluation)

        return preprocessor, fine_tuners, loaders, devices, positive_class_weight
