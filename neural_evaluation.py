import os
import utils

if utils.is_local():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import torch.utils.data
import numpy as np
import typing

import data_preprocessing
import models
import neural_metrics
import backbone_pipeline
import config


def evaluate_individual_models(preprocessor: data_preprocessing.DataPreprocessor,
                               fine_tuners: typing.List[models.FineTuner],
                               loaders: typing.Dict[str, torch.utils.data.DataLoader],
                               devices: typing.List[torch.device],
                               test: bool) -> (typing.List[int], typing.List[int], float,):
    loader = loaders[f'test' if test else f'train']
    fine_fine_tuner, coarse_fine_tuner = fine_tuners

    device_1, device_2 = devices
    fine_fine_tuner.to(device_1)
    coarse_fine_tuner.to(device_2)

    fine_fine_tuner.eval()
    coarse_fine_tuner.eval()

    fine_prediction = []
    coarse_prediction = []

    true_fine_data = []
    true_coarse_data = []

    name_list = []

    print(f'Started testing...')

    with (torch.no_grad()):
        if utils.is_local():
            from tqdm import tqdm
            gen = tqdm(enumerate(loader),
                       total=len(loader))
        else:
            gen = enumerate(loader)

        for i, data in gen:
            batch_examples, batch_true_fine_data, batch_names, batch_true_coarse_data = \
                data[0], data[1].to(device_1), data[2], data[3].to(device_2)

            pred_fine_data = fine_fine_tuner(batch_examples.to(device_1))
            pred_coarse_data = coarse_fine_tuner(batch_examples.to(device_2))

            predicted_fine = torch.max(pred_fine_data, 1)[1]
            predicted_coarse = torch.max(pred_coarse_data, 1)[1]

            true_fine_data += batch_true_fine_data.tolist()
            true_coarse_data += batch_true_coarse_data.tolist()

            fine_prediction += predicted_fine.tolist()
            coarse_prediction += predicted_coarse.tolist()

            name_list += batch_names

    fine_accuracy, coarse_accuracy = (
        neural_metrics.get_and_print_metrics(preprocessor=preprocessor,
                                             pred_fine_data=fine_prediction,
                                             pred_coarse_data=coarse_prediction,
                                             loss='Cross Entropy',
                                             true_fine_data=true_fine_data,
                                             true_coarse_data=true_coarse_data,
                                             combined=False,
                                             test=test))

    return (true_fine_data, true_coarse_data, fine_prediction, coarse_prediction,
            fine_accuracy, coarse_accuracy)


def evaluate_combined_model(preprocessor: data_preprocessing.DataPreprocessor,
                            fine_tuner: models.FineTuner,
                            loaders: typing.Dict[str, torch.utils.data.DataLoader],
                            loss: str,
                            device: torch.device,
                            split: str,
                            print_results: bool = True,
                            lower_predictions_indices: typing.List[int] = []) -> \
        (typing.List[int], typing.List[int], typing.List[int], typing.List[int], float, float):
    loader = loaders[split]
    fine_tuner.to(device)
    fine_tuner.eval()

    fine_predictions = []
    coarse_predictions = []

    fine_lower_predictions = {lower_predictions_index: [] for lower_predictions_index in lower_predictions_indices}
    coarse_lower_predictions = {lower_predictions_index: [] for lower_predictions_index in lower_predictions_indices}

    fine_ground_truths = []
    coarse_ground_truths = []
    fine_accuracy, coarse_accuracy = None, None

    print(utils.blue_text(f'Evaluating {fine_tuner} on {split} using {device}...'))

    with torch.no_grad():
        if utils.is_local():
            from tqdm import tqdm
            gen = tqdm(enumerate(loader), total=len(loader))
        else:
            gen = enumerate(loader)

        for i, data in gen:
            X, Y_true_fine, Y_true_coarse = data[0].to(device), data[1].to(device), data[3].to(device)

            fine_ground_truths += Y_true_fine.tolist()
            coarse_ground_truths += Y_true_coarse.tolist()

            if config.get_ground_truth:
                continue

            Y_pred = fine_tuner(X)
            Y_pred_fine = Y_pred[:, :len(preprocessor.fine_grain_classes_str)]
            Y_pred_coarse = Y_pred[:, len(preprocessor.fine_grain_classes_str):]

            sorted_probs_fine = torch.sort(Y_pred_fine, descending=True)[1]
            predicted_fine = sorted_probs_fine[:, 0]

            sorted_probs_coarse = torch.sort(Y_pred_coarse, descending=True)[1]
            predicted_coarse = sorted_probs_coarse[:, 0]

            fine_predictions += predicted_fine.tolist()
            coarse_predictions += predicted_coarse.tolist()

            for lower_predictions_index in lower_predictions_indices:
                curr_lower_prediction_fine = sorted_probs_fine[:, lower_predictions_index - 1]
                curr_lower_prediction_coarse = sorted_probs_coarse[:, lower_predictions_index - 1]

                fine_lower_predictions[lower_predictions_index] += curr_lower_prediction_fine.tolist()
                coarse_lower_predictions[lower_predictions_index] += curr_lower_prediction_coarse.tolist()

    if print_results:
        fine_accuracy, coarse_accuracy = (
            neural_metrics.get_and_print_metrics(preprocessor=preprocessor,
                                                 pred_fine_data=fine_predictions,
                                                 pred_coarse_data=coarse_predictions,
                                                 loss=loss,
                                                 true_fine_data=fine_ground_truths,
                                                 true_coarse_data=coarse_ground_truths,
                                                 test=split == 'test'))

    return (fine_ground_truths, coarse_ground_truths, fine_predictions, coarse_predictions,
            fine_lower_predictions, coarse_lower_predictions, fine_accuracy, coarse_accuracy)


def evaluate_binary_model(l: data_preprocessing.Label,
                          fine_tuner: models.FineTuner,
                          loaders: typing.Dict[str, torch.utils.data.DataLoader],
                          loss: str,
                          device: torch.device,
                          split: str,
                          print_results: bool = True) -> \
        (typing.List[int], typing.List[int], typing.List[int], typing.List[int], float, float):
    loader = loaders[split]
    fine_tuner.to(device)
    fine_tuner.eval()

    predictions = []
    ground_truths = []
    accuracy = 0

    print(utils.blue_text(f'Evaluating binary {fine_tuner} with l={l} on {split} using {device}...'))

    with torch.no_grad():
        if utils.is_local():
            from tqdm import tqdm
            gen = tqdm(enumerate(loader), total=len(loader))
        else:
            gen = enumerate(loader)

        for i, data in gen:
            X, Y = data[0].to(device), data[1].to(device)

            Y_pred = fine_tuner(X)
            sorted_probs = torch.sort(Y_pred, descending=True)[1]
            predicted = sorted_probs[:, 0]

            ground_truths += Y.tolist()
            predictions += predicted.tolist()

    if print_results:
        accuracy, f1, precision, recall = neural_metrics.get_and_print_binary_metrics(pred_data=predictions,
                                                                                      loss=loss,
                                                                                      true_data=ground_truths,
                                                                                      test=split == 'test')

    return ground_truths, predictions, accuracy


def run_combined_evaluating_pipeline(data_str: str,
                                     model_name: str,
                                     split: str,
                                     lr: typing.Union[str, float],
                                     loss: str,
                                     num_epochs: int,
                                     pretrained_path: str = None,
                                     pretrained_fine_tuner: models.FineTuner = None,
                                     save_files: bool = True,
                                     debug: bool = utils.is_debug_mode(),
                                     print_results: bool = True,
                                     indices: np.array = None,
                                     lower_predictions_indices: typing.List[int] = []):
    """
    Evaluates a pre-trained combined VITFineTuner model on test or validation data.\

    :param data_str:
    :param model_name:
    :param num_epochs:
    :param lower_predictions_indices:
    :param split:
    :param indices:
    :param print_results:
    :param pretrained_fine_tuner:
    :param lr: List of learning rates used during training.
    :param loss: The loss function used during training.
    :param pretrained_path: Path to a pre-trained model (optional).
    :param save_files: Whether to save predictions and ground truth labels
    :param debug: True to force CPU usage for debugging.

    :return: A tuple containing:
             - fine_ground_truths: NumPy array of fine-grained ground truth labels.
             - coarse_ground_truths: NumPy array of coarse-grained ground truth labels.
             - fine_predictions: NumPy array of fine-grained predictions.
             - coarse_predictions: NumPy array of coarse-grained predictions.
             - fine_accuracy: Fine-grained accuracy score.
             - coarse_accuracy: Coarse-grained accuracy score.
    """
    preprocessor, fine_tuners, loaders, devices = backbone_pipeline.initiate(data_str=data_str,
                                                                             model_name=model_name,
                                                                             lr=lr,
                                                                             combined=True,
                                                                             pretrained_path=pretrained_path,
                                                                             debug=debug,
                                                                             error_indices=indices,
                                                                             evaluation=True)

    (fine_ground_truths, coarse_ground_truths, fine_predictions, coarse_predictions,
     fine_lower_predictions, coarse_lower_predictions, fine_accuracy, coarse_accuracy) = (
        evaluate_combined_model(
            preprocessor=preprocessor,
            fine_tuner=fine_tuners[0] if pretrained_fine_tuner is None else pretrained_fine_tuner,
            loaders=loaders,
            loss=loss,
            device=devices[0],
            split=split,
            print_results=print_results,
            lower_predictions_indices=lower_predictions_indices))

    if save_files:
        backbone_pipeline.save_prediction_files(data_str=data_str,
                                                test=split == 'test',
                                                fine_tuners=fine_tuners[0],
                                                combined=True,
                                                lrs=lr[0],
                                                loss=loss,
                                                fine_prediction=fine_predictions,
                                                coarse_prediction=coarse_predictions,
                                                fine_ground_truths=fine_ground_truths,
                                                coarse_ground_truths=coarse_ground_truths,
                                                epoch=num_epochs,
                                                fine_lower_predictions=fine_lower_predictions,
                                                coarse_lower_predictions=coarse_lower_predictions)

    return fine_predictions, coarse_predictions


def run_binary_evaluating_pipeline(data_str: str,
                                   model_name: str,
                                   l: data_preprocessing.Label,
                                   split: str,
                                   lr: typing.Union[str, float],
                                   loss: str,
                                   num_epochs: int,
                                   pretrained_path: str = None,
                                   pretrained_fine_tuner: models.FineTuner = None,
                                   save_files: bool = True,
                                   debug: bool = utils.is_debug_mode(),
                                   print_results: bool = True):
    preprocessor, fine_tuners, loaders, devices, weight = (backbone_pipeline.initiate(data_str=data_str,
                                                                                      model_name=model_name,
                                                                                      l=l,
                                                                                      lr=lr,
                                                                                      pretrained_path=pretrained_path,
                                                                                      debug=debug,
                                                                                      evaluation=True))

    fine_tuner = fine_tuners[0] if pretrained_fine_tuner is None else pretrained_fine_tuner

    ground_truths, predictions, accuracy = evaluate_binary_model(l=l,
                                                                 fine_tuner=fine_tuner,
                                                                 loaders=loaders,
                                                                 loss=loss,
                                                                 device=devices[0],
                                                                 split=split,
                                                                 print_results=print_results)

    if save_files:
        backbone_pipeline.save_binary_prediction_files(test=split == 'test',
                                                       fine_tuner=fine_tuner,
                                                       lr=lr[0],
                                                       epoch=num_epochs,
                                                       l=l,
                                                       predictions=predictions,
                                                       ground_truths=ground_truths,
                                                       evaluation=True)

    return predictions, accuracy


def evaluate_binary_models_from_files(data_str: str,
                                      g_str: str,
                                      test: bool,
                                      lr: typing.Union[str, float],
                                      num_epochs: int,
                                      model_name: str = 'vit_b_16',
                                      loss: str = 'BCE'):
    preprocessor = data_preprocessing.DataPreprocessor(data_str)
    g_ground_truth = preprocessor.train_true_fine_data if g_str == 'fine' \
        else preprocessor.train_true_coarse_data
    print(f'gt shape : {g_ground_truth.shape}')

    for l in preprocessor.get_labels(g=preprocessor.granularities[g_str]).values():
        l_file = models.get_filepath(data_str=data_str,
                                     model_name=model_name,
                                     l=l,
                                     test=test,
                                     loss=loss,
                                     lr=lr,
                                     pred=True,
                                     epoch=num_epochs)
        if os.path.exists(l_file):
            predictions = np.load(l_file)
            print(f'l_pred shape : {predictions.shape}')
            # print(f'{l}:{np.sum(np.where(predictions == 1, 1, 0))}')
            ground_truths = np.where(g_ground_truth == l.index, 1, 0)
            neural_metrics.get_and_print_binary_metrics(pred_data=predictions,
                                                        loss=loss,
                                                        true_data=ground_truths,
                                                        test=test)


if __name__ == '__main__':
    # data_str = 'military_vehicles'
    data_str = 'imagenet'
    # main_model_name = 'vit_b_16'
    main_model_name = new_model_name = 'dinov2_vitl14'
    pretrained_path = 'models/dinov2_vitl14_lr1e-06_BCE.pth'
    # lr = 0.0001
    lr = 0.000001

    # evaluate_binary_models_from_files(model_name='vit_b_16',
    #                                   g_str='fine',
    #                                   test=False,
    #                                   lrs=0.0001,
    #                                   num_epochs=10,
    #                                   loss='BCE'
    #                                   )

    # l = data_preprocessing.fine_grain_labels[data_preprocessing.fine_grain_classes_str[1]]
    # run_binary_evaluating_pipeline(model_name='vit_b_16',
    #                                l=l,
    #                                split='train',
    #                                lrs=[0.0001],
    #                                loss='BCE',
    #                                num_epochs=5,
    #                                pretrained_path=
    #                                f'models/binary_models/binary_{l}_vit_b_16_lr0.0001_loss_BCE_e5.pth')

    # evaluate_binary_models_from_files(g_str='fine',
    #                                   test=False,
    #                                   lrs=0.0001,
    #                                   num_epochs=10)

    run_combined_evaluating_pipeline(data_str=data_str,
                                     model_name=main_model_name,
                                     split='train',
                                     lr=lr,
                                     loss='BCE',
                                     num_epochs=2,
                                     pretrained_path=pretrained_path,
                                     save_files=True)
    #
    # run_combined_evaluating_pipeline(test=True,
    #                                  lrs=[0.0001],
    #                                  loss='BCE',
    #                                  pretrained_path='models/vit_b_16_BCE_lr0.0001.pth')
