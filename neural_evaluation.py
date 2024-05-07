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
    split = 'test' if test else 'train'
    loader = loaders[split]
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

    fine_accuracy, coarse_accuracy, fine_f1, coarse_f1 = (
        neural_metrics.get_and_print_metrics(preprocessor=preprocessor,
                                             pred_fine_data=fine_prediction,
                                             pred_coarse_data=coarse_prediction,
                                             loss='Cross Entropy',
                                             true_fine_data=true_fine_data,
                                             true_coarse_data=true_coarse_data,
                                             combined=False,
                                             split=split))

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

    fine_ground_truths = torch.Tensor([]).to(device)
    coarse_ground_truths = torch.Tensor([]).to(device)
    fine_accuracy, coarse_accuracy, fine_f1, coarse_f1 = None, None, None, None

    print(utils.blue_text(f'Evaluating {fine_tuner} on {split} using {device}...'))

    total_loss = 0

    with torch.no_grad():
        if utils.is_local():
            from tqdm import tqdm
            gen = tqdm(enumerate(loader), total=len(loader))
        else:
            from tqdm import tqdm
            gen = tqdm(enumerate(loader), total=len(loader))

        for i, data in gen:
            X, Y_true_fine, Y_true_coarse = data[0].to(device), data[1].to(device), data[3].to(device)

            Y_true_fine_one_hot = torch.nn.functional.one_hot(Y_true_fine,
                                                              num_classes=len(preprocessor.fine_grain_classes_str))
            Y_true_coarse_one_hot = torch.nn.functional.one_hot(Y_true_coarse,
                                                                num_classes=len(preprocessor.coarse_grain_classes_str))

            Y_true = torch.cat(tensors=[Y_true_fine_one_hot, Y_true_coarse_one_hot], dim=1).float()
            fine_ground_truths = torch.cat([fine_ground_truths, Y_true_fine])
            coarse_ground_truths = torch.cat([coarse_ground_truths, Y_true_coarse])

            if config.get_ground_truth:
                continue

            Y_pred = fine_tuner(X)

            Y_pred_fine = Y_pred[:, :len(preprocessor.fine_grain_classes_str)]
            Y_pred_coarse = Y_pred[:, len(preprocessor.fine_grain_classes_str):]

            criterion = torch.nn.CrossEntropyLoss()
            batch_total_loss = criterion(Y_pred, Y_true)
            total_loss += batch_total_loss.item()

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
        fine_accuracy, coarse_accuracy, fine_f1, coarse_f1 = (
            neural_metrics.get_and_print_metrics(preprocessor=preprocessor,
                                                 pred_fine_data=fine_predictions,
                                                 pred_coarse_data=coarse_predictions,
                                                 loss=loss,
                                                 true_fine_data=fine_ground_truths.tolist(),
                                                 true_coarse_data=coarse_ground_truths,
                                                 split=split))

    return (fine_ground_truths, coarse_ground_truths, fine_predictions, coarse_predictions,
            fine_lower_predictions, coarse_lower_predictions, fine_accuracy, coarse_accuracy, fine_f1, coarse_f1,
            total_loss)


def evaluate_binary_model(fine_tuner: models.FineTuner,
                          loaders: typing.Dict[str, torch.utils.data.DataLoader],
                          loss: str,
                          device: torch.device,
                          split: str,
                          l: data_preprocessing.Label = None,
                          print_results: bool = True,
                          preprocessor: data_preprocessing.DataPreprocessor = None,
                          error_fine_prediction: np.array = None,
                          error_coarse_prediction: np.array = None, ) -> \
        (typing.List[int], typing.List[int], typing.List[int], typing.List[int], float, float):
    loader = loaders[split]
    fine_tuner.to(device)
    fine_tuner.eval()

    predictions = []
    ground_truths = []
    accuracy = 0
    f1 = 0

    print(utils.blue_text(f'Evaluating binary {fine_tuner} with '
                          f'l={l if l is not None else ""} on {split} using {device}...'))

    with torch.no_grad():
        from tqdm import tqdm
        gen = tqdm(enumerate(loader), total=len(loader))

        for i, data in gen:
            if l is not None:
                X, Y = data[0].to(device), data[1].to(device)
                Y_pred = fine_tuner(X)
                sorted_probs = torch.sort(Y_pred, descending=True)[1]
                predicted = sorted_probs[:, 0]
                ground_truths += Y.tolist()
                predictions += predicted.tolist()
            else:
                # binary error model
                X, Y_pred_fine, Y_pred_coarse, E_true = [b.to(device) for b in data]

                # Y_pred_fine_one_hot = torch.nn.functional.one_hot(Y_pred_fine, num_classes=len(
                #     preprocessor.fine_grain_classes_str))
                # Y_pred_coarse_one_hot = torch.nn.functional.one_hot(Y_pred_coarse, num_classes=len(
                #     preprocessor.coarse_grain_classes_str))
                # Y_pred = torch.cat(tensors=[Y_pred_fine_one_hot, Y_pred_coarse_one_hot], dim=1).float()
                #
                # E_pred = torch.where(E_pred > 0.5, 1, 0)

                if error_fine_prediction is not None:
                    corr_pred_fine = np.where(np.array(Y_pred_fine) == error_fine_prediction, 1, 0)
                    corr_pred_coarse = np.where(np.array(Y_pred_coarse) == error_coarse_prediction, 1, 0)
                    E_pred = torch.tensor(1 - corr_pred_fine * corr_pred_coarse)
                    ground_truths += E_true.tolist()
                    predictions += E_pred.tolist()

    if print_results:
        accuracy, f1, precision, recall = neural_metrics.get_individual_metrics(pred_data=predictions,
                                                                                true_data=ground_truths)

    return ground_truths, predictions, accuracy, f1


def run_combined_evaluating_pipeline(data_str: str,
                                     model_name: str,
                                     split: str,
                                     lr: typing.Union[str, float],
                                     loss: str,
                                     num_epochs: int,
                                     pretrained_path: str = None,
                                     pretrained_fine_tuner: models.FineTuner = None,
                                     save_files: bool = True,
                                     debug: bool = False,
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
     fine_lower_predictions, coarse_lower_predictions, fine_accuracy, coarse_accuracy, fine_f1, coarse_f1,
     total_loss) = (
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
                                                lrs=lr,
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
                                   print_results: bool = True,
                                   train_eval_split: float = None):
    if split == 'train_eval' and train_eval_split is None:
        raise ValueError(f'{split} mode do not have value for train and train eval')
    preprocessor, fine_tuners, loaders, devices = (backbone_pipeline.initiate(data_str=data_str,
                                                                              model_name=model_name,
                                                                              l=l,
                                                                              lr=lr,
                                                                              pretrained_path=pretrained_path,
                                                                              debug=debug,
                                                                              train_eval_split=train_eval_split,
                                                                              evaluation=True))

    fine_tuner = fine_tuners[0] if pretrained_fine_tuner is None else pretrained_fine_tuner

    ground_truths, predictions, accuracy, f1 = evaluate_binary_model(l=l,
                                                                     fine_tuner=fine_tuner,
                                                                     loaders=loaders,
                                                                     loss=loss,
                                                                     device=devices[0],
                                                                     split=split,
                                                                     print_results=print_results)

    if save_files and split != 'train_eval':
        backbone_pipeline.save_binary_prediction_files(test=split == 'test',
                                                       fine_tuner=fine_tuner,
                                                       lr=lr,
                                                       epoch=num_epochs,
                                                       l=l,
                                                       predictions=predictions,
                                                       ground_truths=ground_truths,
                                                       evaluation=True,
                                                       data_str=data_str)

    return predictions, accuracy, f1


def evaluate_binary_models_from_files(data_str: str,
                                      g_str: str,
                                      test: bool,
                                      lr: typing.Union[str, float],
                                      num_epochs: int,
                                      l: data_preprocessing.Label,
                                      model_name: str = 'vit_b_16',
                                      loss: str = 'BCE'):
    preprocessor = data_preprocessing.DataPreprocessor(data_str)
    if not test:
        g_ground_truth = preprocessor.train_true_fine_data if g_str == 'fine' \
            else preprocessor.train_true_coarse_data
    else:
        g_ground_truth = preprocessor.test_true_fine_data if g_str == 'fine' \
            else preprocessor.test_true_coarse_data
    print(f'gt shape : {g_ground_truth.shape}')

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
        ground_truths = np.where(g_ground_truth == l.index, 1, 0)
        accuracy, f1, precision, recall = neural_metrics.get_and_print_binary_metrics(pred_data=predictions,
                                                                                      loss=loss,
                                                                                      true_data=ground_truths,
                                                                                      test=test)
        return accuracy, f1, precision, recall

    else:
        print(f'{l_file} is not exist')
        return None


def get_error_metric(main_train_fine_prediction: np.array,
                     main_train_coarse_prediction: np.array,
                     main_test_fine_prediction: np.array,
                     main_test_coarse_prediction: np.array,
                     additional_train_fine_prediction: np.array,
                     additional_train_coarse_prediction: np.array,
                     additional_test_fine_prediction: np.array,
                     additional_test_coarse_prediction: np.array,
                     data_str: str,
                     model_name: str,
                     ):
    preprocessor, fine_tuners, loaders, devices = (backbone_pipeline.initiate(
        data_str=data_str,
        model_name=model_name,
        lr=lr,
        evaluation=True,
        train_fine_predictions=main_train_fine_prediction,
        train_coarse_predictions=main_train_coarse_prediction,
        test_fine_predictions=main_test_fine_prediction,
        test_coarse_predictions=main_test_coarse_prediction,
    ))

    for split in ['train', 'test']:
        loader = loaders[split]
        device = torch.device('cpu')

        # Get error from main and additional (a.k.a model use to derive error) model
        error_fine_prediction = np.where(main_train_fine_prediction == additional_train_fine_prediction
                                         if split == 'train'
                                         else main_test_fine_prediction == additional_test_fine_prediction
                                         , 0, 1)
        error_coarse_prediction = np.where(main_train_coarse_prediction == additional_train_coarse_prediction
                                           if split == 'train'
                                           else main_test_coarse_prediction == additional_test_coarse_prediction
                                           , 0, 1)
        error_prediction = np.where((error_fine_prediction == 1) | (error_coarse_prediction == 1), 1, 0)

        error_ground_truth = []
        print(utils.blue_text(f'Getting ground truth of {data_str} using {device} on {split}...'))

        with torch.no_grad():
            from tqdm import tqdm
            gen = tqdm(enumerate(loader), total=len(loader))

            for i, data in gen:
                X, Y_pred_fine, Y_pred_coarse, E_true = [b.to(device) for b in data]
                error_ground_truth += E_true.tolist()

        accuracy, f1, precision, recall = neural_metrics.get_individual_metrics(pred_data=error_prediction,
                                                                                true_data=np.array(error_ground_truth))

        print(
            utils.blue_text(
                f'{split} dataset: accuracy: {accuracy}, f1: {f1}, precision: {precision}, recall: {recall}'))


if __name__ == '__main__':
    data_str = 'imagenet'
    main_model_name = new_model_name = 'dinov2_vits14'
    lr = 0.000001
    num_epoch_main = 8

    main_test_fine_prediction = np.load(
        f'combined_results/{main_model_name}_test_fine_pred_BCE_lr{lr}_e{num_epoch_main - 1}.npy')
    main_test_coarse_prediction = np.load(
        f'combined_results/{main_model_name}_test_coarse_pred_BCE_lr{lr}_e{num_epoch_main - 1}.npy')
    additional_test_fine_prediction = np.load(
        f'combined_results/{data_str}_{main_model_name}_test_fine_pred_BCE_lr{lr}_e29_additional.npy')
    additional_test_coarse_prediction = np.load(
        f'combined_results/{data_str}_{main_model_name}_test_coarse_pred_BCE_lr{lr}_e29_additional.npy')

    main_train_fine_prediction = np.load(
        f'combined_results/{main_model_name}_train_fine_pred_BCE_lr{lr}_e{num_epoch_main - 1}.npy')
    main_train_coarse_prediction = np.load(
        f'combined_results/{main_model_name}_train_coarse_pred_BCE_lr{lr}_e{num_epoch_main - 1}.npy')
    additional_train_fine_prediction = np.load(
        f'combined_results/{data_str}_{main_model_name}_train_fine_pred_BCE_lr{lr}_e29_additional.npy')
    additional_train_coarse_prediction = np.load(
        f'combined_results/{data_str}_{main_model_name}_train_coarse_pred_BCE_lr{lr}_e29_additional.npy')

    get_error_metric(main_train_fine_prediction=main_train_fine_prediction,
                     main_train_coarse_prediction=main_train_coarse_prediction,
                     additional_train_fine_prediction=additional_train_fine_prediction,
                     additional_train_coarse_prediction=additional_train_coarse_prediction,
                     main_test_fine_prediction=main_test_fine_prediction,
                     main_test_coarse_prediction=main_test_coarse_prediction,
                     additional_test_fine_prediction=additional_test_fine_prediction,
                     additional_test_coarse_prediction=additional_test_coarse_prediction,
                     data_str=data_str,
                     model_name=main_model_name, )

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

    # run_combined_evaluating_pipeline(data_str=data_str,
    #                                  model_name=main_model_name,
    #                                  split='train',
    #                                  lr=lr,
    #                                  loss='BCE',
    #                                  num_epochs=0,
    #                                  pretrained_path=pretrained_path,
    #                                  print_results=True,
    #                                  save_files=True)
    #
    # run_combined_evaluating_pipeline(test=True,
    #                                  lrs=[0.0001],
    #                                  loss='BCE',
    #                                  pretrained_path='models/vit_b_16_BCE_lr0.0001.pth')
