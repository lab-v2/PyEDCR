import numpy as np
import torch.utils.data
import typing

import models
import utils
import data_preprocessing

original_prediction_weight = 1 / (len(data_preprocessing.fine_grain_classes_str) +
                                  len(data_preprocessing.coarse_grain_classes_str))
binary_results_path = fr'binary_results'
combined_results_path = fr'combined_results'
individual_results_path = fr'individual_results'
batch_size = 64
scheduler_gamma = 0.9
ltn_num_epochs = 5


# vit_model_names = [f'vit_{vit_model_name}' for vit_model_name in ['b_16']]


def save_prediction_files(test: bool,
                          fine_tuners: typing.Union[models.FineTuner, dict[str, models.FineTuner]],
                          combined: bool,
                          lrs: typing.Union[str, float, dict[str, typing.Union[str, float]]],
                          fine_prediction: np.array,
                          coarse_prediction: np.array,
                          epoch: int = None,
                          loss: str = 'BCE',
                          fine_ground_truths: np.array = None,
                          coarse_ground_truths: np.array = None,
                          fine_lower_predictions: dict[int, list] = {},
                          coarse_lower_predictions: dict[int, list] = {}):
    """
    Saves prediction files and optional ground truth files.

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
        for g_str in data_preprocessing.granularities_str:
            prediction = fine_prediction if g_str == 'fine' else coarse_prediction
            np.save(models.get_filepath(model_name=fine_tuners,
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
                np.save(models.get_filepath(model_name=fine_tuners,
                                            combined=True,
                                            test=test,
                                            granularity=g_str,
                                            loss=loss,
                                            lr=lrs,
                                            pred=True,
                                            epoch=epoch,
                                            lower_prediction_index=lower_prediction_index),
                        lower_prediction_values)

        if fine_ground_truths is not None:
            np.save(f"data/{test_str}_fine/{test_str}_true_fine.npy",
                    fine_ground_truths)
            np.save(f"data/{test_str}_coarse/{test_str}_true_coarse.npy",
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
                                 fine_tuner: typing.Union[models.FineTuner, dict[str, models.FineTuner]],
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


def get_imbalance_weight(l: data_preprocessing.Label,
                         train_images_num: int,
                         evaluation: bool = False) -> list[float]:
    g_ground_truth = data_preprocessing.train_true_fine_data if l.g.g_str == 'fine' \
        else data_preprocessing.train_true_coarse_data
    positive_examples_num = np.sum(np.where(g_ground_truth == l.index, 1, 0))
    negative_examples_num = train_images_num - positive_examples_num

    positive_class_weight = train_images_num / positive_examples_num

    if not evaluation:
        print(f'\nl={l}:\n'
              f'weight of positive class: {positive_class_weight}')

    return positive_class_weight


def initiate(lrs: list[typing.Union[str, float]],
             vit_model_names: list[str] = ['vit_b_16'],
             combined: bool = True,
             l: data_preprocessing.Label = None,
             pretrained_path: str = None,
             debug: bool = False,
             indices: typing.Sequence = None,
             evaluation: bool = None):
    """
    Initializes models, datasets, and devices for training.

    :param vit_model_names:
    :param l:
    :param evaluation:
    :param indices:
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
    print(f'Models: {vit_model_names}\n'
          f'Learning rates: {lrs}')

    datasets = data_preprocessing.get_datasets(combined=combined,
                                               binary_label=l)

    device = torch.device('cpu') if debug else (
        torch.device('mps' if torch.backends.mps.is_available() else
                     ("cuda" if torch.cuda.is_available() else 'cpu')))
    devices = [device]
    print(f'Using {device}')

    num_fine_grain_classes, num_coarse_grain_classes = None, None

    if l is not None:
        results_path = binary_results_path
        fine_tuners = [models.VITFineTuner(vit_model_name=vit_model_name,
                                           num_classes=2)
                       for vit_model_name in vit_model_names]
    else:
        num_fine_grain_classes = len(data_preprocessing.fine_grain_classes_str)
        num_coarse_grain_classes = len(data_preprocessing.coarse_grain_classes_str)

        if combined:
            num_classes = num_fine_grain_classes + num_coarse_grain_classes

            if pretrained_path is not None:
                print(f'Loading pretrained model from {pretrained_path}')
                fine_tuners = [models.VITFineTuner.from_pretrained(vit_model_name=vit_model_name,
                                                                   classes_num=num_classes,
                                                                   pretrained_path=pretrained_path,
                                                                   device=device)
                               for vit_model_name in vit_model_names]
            else:
                fine_tuners = [models.VITFineTuner(vit_model_name=vit_model_name,
                                                   num_classes=num_classes)
                               for vit_model_name in vit_model_names]

            results_path = combined_results_path
        else:
            num_gpus = torch.cuda.device_count()

            if num_gpus < 2:
                raise ValueError("This setup requires at least 2 GPUs.")

            devices = [torch.device("cuda:0"), torch.device("cuda:1")]

            fine_tuners = ([models.VITFineTuner(vit_model_name=vit_model_name,
                                                num_classes=num_fine_grain_classes)
                            for vit_model_name in vit_model_names] +
                           [models.VITFineTuner(vit_model_name=vit_model_name,
                                                num_classes=num_coarse_grain_classes)
                            for vit_model_name in vit_model_names])
            results_path = individual_results_path

    utils.create_directory(results_path)
    loaders = data_preprocessing.get_loaders(datasets=datasets,
                                             batch_size=batch_size,
                                             indices=indices,
                                             evaluation=evaluation)

    train_images_num = len(loaders['train'].dataset)
    test_images_num = len(loaders['test'].dataset)

    print(f"Total number of train images: {train_images_num}\n"
          f"Total number of test images: {test_images_num}")

    if l is None:
        return fine_tuners, loaders, devices, num_fine_grain_classes, num_coarse_grain_classes
    else:
        positive_class_weight = get_imbalance_weight(l=l,
                                                     train_images_num=train_images_num,
                                                     evaluation=evaluation)

        return fine_tuners, loaders, devices, positive_class_weight
