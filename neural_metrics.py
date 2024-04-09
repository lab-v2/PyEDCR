import typing
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import utils
import data_preprocessing


def get_change_str(change: typing.Union[float, str]):
    return '' if change == '' else (utils.red_text(f'({round(change, 2)}%)') if change < 0
                                    else utils.green_text(f'(+{round(change, 2)}%)'))


def print_num_inconsistencies(preprocessor: data_preprocessing.DataPreprocessor,
                              pred_fine_data: np.array,
                              pred_coarse_data: np.array,
                              prior: bool = True):
    """
    Prints the number of inconsistencies between fine and coarse predictions.

    :param preprocessor:
    :param pred_fine_data: NumPy array of predictions at the fine granularity.
    :param pred_coarse_data: NumPy array of predictions at the coarse granularity.
    :param prior:
    """
    inconsistencies = preprocessor.get_num_inconsistencies(fine_labels=pred_fine_data,
                                                           coarse_labels=pred_coarse_data)

    print(f"Total {'prior' if prior else 'post'} inconsistencies "
          f"{utils.red_text(inconsistencies)}/{utils.red_text(len(pred_fine_data))} "
          f'({utils.red_text(round(inconsistencies / len(pred_fine_data) * 100, 2))}%)\n')


def get_binary_metrics(pred_data: np.array,
                       true_data: np.array):
    """
    Calculates and returns performance metrics for fine and coarse granularities.
    :return: A tuple containing the accuracy, F1, precision, and recall metrics
             for both fine and coarse granularities.
    """
    accuracy = accuracy_score(y_true=pred_data,
                              y_pred=true_data)
    f1 = f1_score(y_true=pred_data,
                  y_pred=true_data,
                  labels=[0, 1],
                  average='macro')
    precision = precision_score(y_true=pred_data,
                                y_pred=true_data,
                                labels=[0, 1],
                                average='macro')
    recall = recall_score(y_true=pred_data,
                          y_pred=true_data,
                          labels=[0, 1],
                          average='macro')

    return accuracy, f1, precision, recall


def get_individual_metrics(pred_data: np.array,
                           true_data: np.array,
                           labels=None):
    accuracy = accuracy_score(y_true=true_data,
                              y_pred=pred_data)
    f1 = f1_score(y_true=true_data,
                  y_pred=pred_data,
                  labels=labels,
                  average='macro')
    precision = precision_score(y_true=true_data,
                                y_pred=pred_data,
                                labels=labels,
                                average='macro')
    recall = recall_score(y_true=true_data,
                          y_pred=pred_data,
                          labels=labels,
                          average='macro')

    return accuracy, f1, precision, recall


def get_metrics(preprocessor: data_preprocessing.DataPreprocessor,
                pred_fine_data: np.array,
                pred_coarse_data: np.array,
                true_fine_data: np.array,
                true_coarse_data: np.array):
    """
    Calculates and returns performance metrics for fine and coarse granularities.
    :param preprocessor:
    :param pred_fine_data: NumPy array of predictions at the fine granularity.
    :param pred_coarse_data: NumPy array of predictions at the coarse granularity.
    :param true_fine_data: NumPy array of true labels at the fine granularity.
    :param true_coarse_data: NumPy array of true labels at the coarse granularity.
    :return: A tuple containing the accuracy, F1, precision, and recall metrics
             for both fine and coarse granularities.
    """
    fine_accuracy, fine_f1, fine_precision, fine_recall = (
        get_individual_metrics(pred_data=pred_fine_data,
                               true_data=true_fine_data,
                               labels=range(len(preprocessor.fine_grain_classes_str))))

    coarse_accuracy, coarse_f1, coarse_precision, coarse_recall = (
        get_individual_metrics(pred_data=pred_coarse_data,
                               true_data=true_coarse_data,
                               labels=range(len(preprocessor.coarse_grain_classes_str))))

    return (fine_accuracy, fine_f1, fine_precision, fine_recall,
            coarse_accuracy, coarse_f1, coarse_precision, coarse_recall)


def get_and_print_binary_metrics(pred_data: np.array,
                                 loss: str,
                                 true_data: np.array,
                                 test: bool,
                                 prior: bool = True,
                                 model_name: str = '',
                                 lr: typing.Union[str, float] = ''):
    """
    Calculates, prints, and returns accuracy metrics for fine and coarse granularities.

    :param true_data:
    :param pred_data:
    :param loss: The loss function used during training.
    :param test: True for test data, False for training data.
    :param prior:
    :param model_name: The name of the model (optional).
    :param lr: The learning rate used during training (optional).
    :return: fine_accuracy, coarse_accuracy
    """
    accuracy, f1, precision, recall = get_binary_metrics(pred_data=pred_data,
                                                         true_data=true_data)
    prior_str = 'prior' if prior else 'post'
    test_str = 'Test' if test else 'Train'

    print('#' * 100 + '\n' + (f'Main model name: {utils.blue_text(model_name)} ' if model_name != '' else '') +
          f"with {utils.blue_text(loss)} loss on the {utils.blue_text('test' if test else 'train')} dataset\n" +
          (f'with lr={utils.blue_text(lr)}\n' if lr != '' else '') +
          f'\n{test_str} {prior_str} accuracy: {utils.green_text(round(accuracy * 100, 2))}%'
          f' {test_str} {prior_str} macro f1: {utils.green_text(round(f1 * 100, 2))}%'
          f'\n {test_str} {prior_str}  macro precision: '
          f'{utils.green_text(round(precision * 100, 2))}%'
          f',  {test_str} {prior_str} macro recall: {utils.green_text(round(recall * 100, 2))}%\n'
          )

    return accuracy, f1, precision, recall


def get_and_print_metrics(preprocessor: data_preprocessing.DataPreprocessor,
                          pred_fine_data: np.array,
                          pred_coarse_data: np.array,
                          loss: str,
                          true_fine_data: np.array,
                          true_coarse_data: np.array,
                          test: bool,
                          prior: bool = True,
                          combined: bool = True,
                          model_name: str = '',
                          lr: typing.Union[str, float] = '',
                          print_inconsistencies: bool = True,
                          original_pred_fine_data: np.array = None,
                          original_pred_coarse_data: np.array = None):
    """
    Calculates, prints, and returns accuracy metrics for fine and coarse granularities.

    :param preprocessor:
    :param original_pred_coarse_data:
    :param original_pred_fine_data:
    :param print_inconsistencies:
    :param pred_fine_data: NumPy array of predictions at the fine granularity.
    :param pred_coarse_data: NumPy array of predictions at the coarse granularity.
    :param loss: The loss function used during training.
    :param true_fine_data: NumPy array of true labels at the fine granularity.
    :param true_coarse_data: NumPy array of true labels at the coarse granularity.
    :param test: True for test data, False for training data.
    :param prior:
    :param combined: Whether the model are individual or combine one.
    :param model_name: The name of the model (optional).
    :param lr: The learning rate used during training (optional).
    :return: fine_accuracy, coarse_accuracy
    """
    (fine_accuracy, fine_f1, fine_precision, fine_recall,
     coarse_accuracy, coarse_f1, coarse_precision, coarse_recall) = get_metrics(preprocessor=preprocessor,
                                                                                pred_fine_data=pred_fine_data,
                                                                                pred_coarse_data=pred_coarse_data,
                                                                                true_fine_data=true_fine_data,
                                                                                true_coarse_data=true_coarse_data)
    prior_str = 'prior' if prior else 'post'
    combined_str = 'combined' if combined else 'individual'

    fine_accuracy_change_str = ''
    fine_f1_change_str = ''
    fine_precision_change_str = ''
    fine_recall_change_str = ''
    coarse_accuracy_change_str = ''
    coarse_f1_change_str = ''
    coarse_precision_change_str = ''
    coarse_recall_change_str = ''

    if original_pred_fine_data is not None and original_pred_coarse_data is not None:
        (original_fine_accuracy, original_fine_f1, original_fine_precision, original_fine_recall,
         original_coarse_accuracy, original_coarse_f1, original_coarse_precision, original_coarse_recall) = get_metrics(
            preprocessor=preprocessor,
            pred_fine_data=original_pred_fine_data,
            pred_coarse_data=original_pred_coarse_data,
            true_fine_data=true_fine_data,
            true_coarse_data=true_coarse_data)
        fine_accuracy_change_str = fine_accuracy - original_fine_accuracy
        fine_f1_change_str = fine_f1 - original_fine_f1
        fine_precision_change_str = fine_precision - original_fine_precision
        fine_recall_change_str = fine_recall - original_fine_recall
        coarse_accuracy_change_str = coarse_accuracy - original_coarse_accuracy
        coarse_f1_change_str = coarse_f1 - original_coarse_f1
        coarse_precision_change_str = coarse_precision - original_coarse_precision
        coarse_recall_change_str = coarse_recall - original_coarse_recall

    print('#' * 100 + '\n' + (f'Main model name: {utils.blue_text(model_name)} ' if model_name != '' else '') +
          f"with {utils.blue_text(loss)} loss on the {utils.blue_text('test' if test else 'train')} dataset\n" +
          (f'with lr={utils.blue_text(lr)}\n' if lr != '' else '') +
          f'\nFine-grain {prior_str} {combined_str} accuracy: {utils.green_text(round(fine_accuracy * 100, 2))}%'
          f' {get_change_str(fine_accuracy_change_str)}, '
          f'fine-grain {prior_str} {combined_str} macro f1: {utils.green_text(round(fine_f1 * 100, 2))}%'
          f' {get_change_str(fine_f1_change_str)}'
          f'\nFine-grain {prior_str} {combined_str} macro precision: '
          f'{utils.green_text(round(fine_precision * 100, 2))}% {get_change_str(fine_precision_change_str)}'
          f', fine-grain {prior_str} {combined_str} macro recall: {utils.green_text(round(fine_recall * 100, 2))}%'
          f' {get_change_str(fine_recall_change_str)}\n'
          f'\nCoarse-grain {prior_str} {combined_str} accuracy: '
          f'{utils.green_text(round(coarse_accuracy * 100, 2))}% {get_change_str(coarse_accuracy_change_str)}'
          f', coarse-grain {prior_str} {combined_str} macro f1: '
          f'{utils.green_text(round(coarse_f1 * 100, 2))}% {get_change_str(coarse_f1_change_str)}'
          f'\nCoarse-grain {prior_str} {combined_str} macro precision: '
          f'{utils.green_text(round(coarse_precision * 100, 2))}% {get_change_str(coarse_precision_change_str)}'
          f', coarse-grain {prior_str} {combined_str} macro recall: '
          f'{utils.green_text(round(coarse_recall * 100, 2))}% {get_change_str(coarse_recall_change_str)}\n'
          )

    if print_inconsistencies:
        print_num_inconsistencies(preprocessor=preprocessor,
                                  pred_fine_data=pred_fine_data,
                                  pred_coarse_data=pred_coarse_data,
                                  prior=prior)

    return fine_accuracy, coarse_accuracy


def get_and_print_post_epoch_binary_metrics(epoch: int,
                                            num_epochs: int,
                                            train_ground_truths: np.array,
                                            train_predictions: np.array,
                                            total_running_loss: float
                                            ):
    training_accuracy = accuracy_score(y_true=train_ground_truths,
                                       y_pred=train_predictions)
    training_f1 = f1_score(y_true=train_ground_truths,
                           y_pred=train_predictions,
                           labels=[0, 1],
                           average='macro')

    print(f'\nEpoch {epoch + 1}/{num_epochs} done'
          f'\nMean loss across epochs: {round(total_running_loss / (epoch + 1), 2)}'
          f'\npost-epoch training accuracy: {round(training_accuracy * 100, 2)}%'
          f', post-epoch f1: {round(training_f1 * 100, 2)}%\n')

    return training_accuracy, training_f1


def get_and_print_post_epoch_metrics(preprocessor: data_preprocessing.DataPreprocessor,
                                     epoch: int,
                                     num_epochs: int,
                                     train_fine_ground_truth: np.array,
                                     train_fine_prediction: np.array,
                                     train_coarse_ground_truth: np.array,
                                     train_coarse_prediction: np.array,
                                     # running_fine_loss: float = None,
                                     # running_coarse_loss: float = None,
                                     # running_total_loss: float = None
                                     ):
    training_fine_accuracy = accuracy_score(y_true=train_fine_ground_truth,
                                            y_pred=train_fine_prediction)
    training_coarse_accuracy = accuracy_score(y_true=train_coarse_ground_truth,
                                              y_pred=train_coarse_prediction)
    training_fine_f1 = f1_score(y_true=train_fine_ground_truth,
                                y_pred=train_fine_prediction,
                                labels=range(preprocessor.num_fine_grain_classes),
                                average='macro')
    training_coarse_f1 = f1_score(y_true=train_coarse_ground_truth,
                                  y_pred=train_coarse_prediction,
                                  labels=range(preprocessor.num_coarse_grain_classes),
                                  average='macro')

    # loss_str = (f'Training epoch total fine loss: {round(running_fine_loss / num_batches, 2)}'
    #             f'\ntraining epoch total coarse loss: {round(running_coarse_loss / num_batches, 2)}') \
    #     if running_fine_loss is not None else
    #     f'Training epoch total loss: {round(running_total_loss / num_batches, 2)}'

    print(f'\nEpoch {epoch + 1}/{num_epochs} done,\n'
          # f'{loss_str}'
          f'\nLast batch training fine accuracy: {round(training_fine_accuracy * 100, 2)}%'
          f', last batch fine f1: {round(training_fine_f1 * 100, 2)}%'
          f'\nLast batch training coarse accuracy: {round(training_coarse_accuracy * 100, 2)}%'
          f', last batch coarse f1: {round(training_coarse_f1 * 100, 2)}%\n')

    return training_fine_accuracy, training_coarse_accuracy


def print_post_batch_metrics(batch_num: int,
                             num_batches: int,
                             batch_fine_grain_loss: float = None,
                             batch_coarse_grain_loss: float = None,
                             batch_total_loss: float = None):
    if batch_num > 0 and batch_num % 10 == 0:
        if batch_fine_grain_loss is not None:
            print(f'\nCompleted batch num {batch_num}/{num_batches}, '
                  f'batch fine-grain loss: {round(batch_fine_grain_loss, 2)}, '
                  f'batch coarse-grain loss: {round(batch_coarse_grain_loss, 2)}')
        else:
            print(f'\nCompleted batch num {batch_num}/{num_batches}, batch total loss: {round(batch_total_loss, 2)}')
