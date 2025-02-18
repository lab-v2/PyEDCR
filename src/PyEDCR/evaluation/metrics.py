import numpy as np
import typing
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score

from src.PyEDCR.data_processing import data_preprocessor
from src.PyEDCR.utils import utils
from src.PyEDCR.classes import granularity, label


def get_l_detection_rule_support(edcr,
                                 test: bool,
                                 l: label.Label) -> float:
    if l not in edcr.error_detection_rules:
        return 0

    test_or_train = 'test' if test else 'train'

    N_l = np.sum(edcr.get_where_label_is_l(pred=True, test=test, l=l))
    r_l = edcr.error_detection_rules[l]
    where_l_detection_rule_body_is_satisfied = (
        r_l.get_where_body_is_satisfied(
            pred_fine_data=edcr.pred_data[test_or_train]['original'][edcr.preprocessor.granularities['fine']],
            pred_coarse_data=edcr.pred_data[test_or_train]['original'][edcr.preprocessor.granularities['coarse']]))
    num_predicted_l_and_any_conditions_satisfied = np.sum(where_l_detection_rule_body_is_satisfied)
    s_l = num_predicted_l_and_any_conditions_satisfied / N_l

    assert s_l <= 1

    return s_l


def get_l_detection_rule_confidence(edcr,
                                    test: bool,
                                    l: label.Label) -> float:
    if l not in edcr.error_detection_rules:
        return 0

    test_or_train = 'test' if test else 'train'

    r_l = edcr.error_detection_rules[l]
    where_l_detection_rule_body_is_satisfied = (
        r_l.get_where_body_is_satisfied(
            pred_fine_data=edcr.pred_data[test_or_train]['original'][edcr.preprocessor.granularities['fine']],
            pred_coarse_data=edcr.pred_data[test_or_train]['original'][edcr.preprocessor.granularities['coarse']]))
    where_l_fp = edcr.get_where_fp_l(test=test, l=l)
    where_head_and_body_is_satisfied = where_l_detection_rule_body_is_satisfied * where_l_fp

    num_where_l_detection_rule_body_is_satisfied = np.sum(where_l_detection_rule_body_is_satisfied)

    if num_where_l_detection_rule_body_is_satisfied == 0:
        return 0

    c_l = np.sum(where_head_and_body_is_satisfied) / num_where_l_detection_rule_body_is_satisfied
    return c_l


def get_l_detection_rule_theoretical_precision_increase(edcr,
                                                        test: bool,
                                                        l: label.Label) -> float:
    s_l = edcr.get_l_detection_rule_support(test=test, l=l)

    if s_l == 0:
        return 0

    c_l = edcr.get_l_detection_rule_confidence(test=test, l=l)
    p_l = edcr.get_l_precision_and_recall(l=l, test=test, stage='original')[0]

    return s_l / (1 - s_l) * (c_l + p_l - 1)


def get_g_detection_rule_theoretical_precision_increase(edcr,
                                                        test: bool,
                                                        g: granularity.Granularity):
    precision_increases = [edcr.get_l_detection_rule_theoretical_precision_increase(test=test, l=l)
                           for l in edcr.preprocessor.get_labels(g).values()]
    return np.mean(precision_increases)


def get_l_detection_rule_theoretical_recall_decrease(edcr,
                                                     test: bool,
                                                     l: label.Label) -> float:
    c_l = edcr.get_l_detection_rule_confidence(test=test, l=l)
    s_l = edcr.get_l_detection_rule_support(test=test, l=l)
    p_l, r_l = edcr.get_l_precision_and_recall(l=l, test=test, stage='original')

    return (1 - c_l) * s_l * r_l / p_l


def get_g_detection_rule_theoretical_recall_decrease(edcr,
                                                     test: bool,
                                                     g: granularity.Granularity):
    recall_decreases = [edcr.get_l_detection_rule_theoretical_recall_decrease(test=test, l=l)
                        for l in edcr.preprocessor.get_labels(g).values()]
    return np.mean(recall_decreases)


def get_l_correction_rule_confidence(edcr,
                                     test: bool,
                                     l: label.Label,
                                     pred_fine_data: np.array = None,
                                     pred_coarse_data: np.array = None
                                     ) -> float:
    if l not in edcr.error_correction_rules:
        return 0

    test_or_train = 'test' if test else 'train'

    r_l = edcr.error_correction_rules[l]
    where_l_correction_rule_body_is_satisfied = (
        r_l.get_where_body_is_satisfied(
            fine_data=edcr.pred_data[test_or_train]['post_correction'][edcr.preprocessor.granularities['fine']]
            if pred_fine_data is None else pred_fine_data,
            coarse_data=edcr.pred_data[test_or_train]['post_correction'][edcr.preprocessor.granularities['coarse']]
            if pred_coarse_data is None else pred_coarse_data))
    where_l_gt = edcr.get_where_label_is_l(pred=False, test=test, l=l)

    where_head_and_body_is_satisfied = where_l_correction_rule_body_is_satisfied * where_l_gt
    num_where_l_correction_rule_body_is_satisfied = np.sum(where_l_correction_rule_body_is_satisfied)

    if num_where_l_correction_rule_body_is_satisfied == 0:
        return 0

    c_l = np.sum(where_head_and_body_is_satisfied) / num_where_l_correction_rule_body_is_satisfied
    return c_l


def get_l_correction_rule_support(edcr,
                                  test: bool,
                                  l: label.Label,
                                  pred_fine_data: np.array = None,
                                  pred_coarse_data: np.array = None
                                  ) -> float:
    if l not in edcr.error_correction_rules:
        return 0

    test_or_train = 'test' if test else 'train'

    N_l = np.sum(edcr.get_where_label_is_l(pred=True, test=test, l=l, stage='post_correction')
                 if (pred_fine_data is None and pred_coarse_data is None)
                 else edcr.get_where_label_is_l_in_data(l=l,
                                                        test_pred_fine_data=pred_fine_data,
                                                        test_pred_coarse_data=pred_coarse_data))
    if N_l == 0:
        return 0

    r_l = edcr.error_correction_rules[l]
    where_rule_body_is_satisfied = (
        r_l.get_where_body_is_satisfied(
            fine_data=edcr.pred_data[test_or_train]['post_correction'][edcr.preprocessor.granularities['fine']]
            if pred_fine_data is None else pred_fine_data,
            coarse_data=edcr.pred_data[test_or_train]['post_correction'][
                edcr.preprocessor.granularities['coarse']]
            if pred_coarse_data is None else pred_coarse_data))

    s_l = np.sum(where_rule_body_is_satisfied) / N_l

    return s_l


def get_l_correction_rule_theoretical_precision_increase(edcr,
                                                         test: bool,
                                                         l: label.Label) -> float:
    c_l = edcr.get_l_correction_rule_confidence(test=test, l=l)
    s_l = edcr.get_l_correction_rule_support(test=test, l=l)
    p_l_prior_correction = edcr.get_l_precision_and_recall(l=l,
                                                           test=test,
                                                           stage='post_correction')[0]

    return s_l * (c_l - p_l_prior_correction) / (1 + s_l)


def evaluate_and_print_g_detection_rule_precision_increase(edcr,
                                                           test: bool,
                                                           g: granularity.Granularity,
                                                           threshold: float = 1e-5):
    original_g_precisions = edcr.get_g_precision_and_recall(g=g, test=test, stage='original')[0]
    post_detection_g_precisions = edcr.get_g_precision_and_recall(g=g, test=test, stage='post_detection')[0]

    original_g_mean_precision = np.mean(list(original_g_precisions.values()))
    post_detection_mean_precision = np.mean(list(post_detection_g_precisions.values()))

    precision_diff = post_detection_mean_precision - original_g_mean_precision
    detection_rule_theoretical_precision_increase = (
        edcr.get_g_detection_rule_theoretical_precision_increase(test=test, g=g))
    precision_theory_holds = abs(detection_rule_theoretical_precision_increase - precision_diff) < threshold
    precision_theory_holds_str = utils.green_text('The theory holds!') if precision_theory_holds else (
        utils.red_text('The theory does not hold!'))

    print('\n' + '#' * 20 + f'post detection {g}-grain precision results' + '#' * 20)

    print(f'{g}-grain new mean precision: {post_detection_mean_precision}, '
          f'{g}-grain old mean precision: {original_g_mean_precision}, '
          f'diff: {utils.blue_text(precision_diff)}\n'
          f'theoretical precision increase: {utils.blue_text(detection_rule_theoretical_precision_increase)}\n'
          f'{precision_theory_holds_str}'
          )


def evaluate_and_print_g_detection_rule_recall_decrease(edcr,
                                                        test: bool,
                                                        g: granularity.Granularity,
                                                        threshold: float = 1e-5):
    original_g_recalls = edcr.get_g_precision_and_recall(g=g, test=test, stage='original')[1]
    post_detection_recalls = edcr.get_g_precision_and_recall(g=g, test=test, stage='post_detection')[1]

    original_g_mean_recall = np.mean(list(original_g_recalls.values()))
    post_detection_g_mean_recall = np.mean(list(post_detection_recalls.values()))
    recall_diff = post_detection_g_mean_recall - original_g_mean_recall

    detection_rule_theoretical_recall_decrease = (
        edcr.get_g_detection_rule_theoretical_recall_decrease(test=test, g=g))
    recall_theory_holds = abs(abs(detection_rule_theoretical_recall_decrease) - abs(recall_diff)) < threshold
    recall_theory_holds_str = utils.green_text('The theory holds!') if recall_theory_holds else (
        utils.red_text('The theory does not hold!'))

    print('\n' + '#' * 20 + f'post detection {g}-grain recall results' + '#' * 20)

    print(f'{g}-grain new mean recall: {post_detection_g_mean_recall}, '
          f'{g}-grain old mean recall: {original_g_mean_recall}, '
          f'diff: {utils.blue_text(recall_diff)}\n'
          f'theoretical recall decrease: -{utils.blue_text(detection_rule_theoretical_recall_decrease)}\n'
          f'{recall_theory_holds_str}')



def get_change_str(change: typing.Union[float, str]):
    return '' if change == '' else (utils.red_text(f'({round(change, 2)}%)') if change < 0
                                    else utils.green_text(f'(+{round(change, 2)}%)'))


def print_num_inconsistencies(preprocessor: data_preprocessor.FineCoarseDataPreprocessor,
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
    inconsistencies, _ = preprocessor.get_num_inconsistencies(fine_labels=pred_fine_data,
                                                              coarse_labels=pred_coarse_data)

    print(f"Total {'prior' if prior else 'post'} inconsistencies "
          f"{utils.red_text(inconsistencies)}/{utils.red_text(len(pred_fine_data))} "
          f'({utils.red_text(round(inconsistencies / len(pred_fine_data) * 100, 2))}%)\n')

    # if current_num_test_inconsistencies is not None and original_test_inconsistencies is not None:
    #     print(f'Recovered inconsistencies: '
    #           f'{round(current_num_test_inconsistencies / original_test_inconsistencies[1] * 100, 2)}%'
    #           )


def get_individual_metrics(pred_data: np.array,
                           true_data: np.array,
                           labels: list = None,
                           binary: bool = False):
    accuracy = accuracy_score(y_true=true_data,
                              y_pred=pred_data)
    balanced_accuracy = balanced_accuracy_score(y_true=true_data,
                                                y_pred=pred_data)
    f1 = f1_score(y_true=true_data,
                  y_pred=pred_data,
                  labels=labels,
                  average='macro'
                  )
    precision = precision_score(y_true=true_data,
                                y_pred=pred_data,
                                labels=labels,
                                average='macro'
                                )
    recall = recall_score(y_true=true_data,
                          y_pred=pred_data,
                          labels=labels,
                          average='macro'
                          )

    # if len(labels) > 3: for idx in labels: precision_per_class = precision_score(y_true=true_data,
    # y_pred=pred_data, labels=labels, average=None)[idx] recall_per_class = recall_score(y_true=true_data,
    # y_pred=pred_data, labels=labels, average=None)[idx] print(f'class {idx} has precision {precision_per_class} and
    # recall {recall_per_class}')

    if not binary:
        return accuracy, f1, precision, recall

    return accuracy, balanced_accuracy, f1, precision, recall


def get_metrics(preprocessor: data_preprocessor.FineCoarseDataPreprocessor,
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
                               labels=list(range(len(preprocessor.fine_grain_classes_str)))))

    coarse_accuracy, coarse_f1, coarse_precision, coarse_recall = (
        get_individual_metrics(pred_data=pred_coarse_data,
                               true_data=true_coarse_data,
                               labels=list(range(len(preprocessor.coarse_grain_classes_str)))))

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
    accuracy, f1, precision, recall = get_individual_metrics(pred_data=pred_data,
                                                             true_data=true_data,
                                                             labels=[0, 1])
    prior_str = 'prior' if prior else 'post'
    test_str = 'Test' if test else 'Train'

    print('#' * 100 + '\n' + (f'Main model name: {utils.blue_text(model_name)} ' if model_name != '' else '') +
          f"with {utils.blue_text(loss)} loss on the {utils.blue_text('test' if test else 'train')} dataset\n" +
          (f'with lr={utils.blue_text(lr)}\n' if lr != '' else '') +
          f'\n{test_str} {prior_str} accuracy: {utils.green_text(round(accuracy * 100, 2))}%'
          f' {test_str} {prior_str} macro f1: {utils.green_text(round(f1 * 100, 2))}%'
          f'\n {test_str} {prior_str}  macro precision: '
          f'{utils.green_text(round(precision * 100, 2))}%'
          f',  {test_str} {prior_str} macro recall: {utils.green_text(round(recall * 100, 2))}%\n')

    return accuracy, f1, precision, recall


def get_and_print_metrics(preprocessor: data_preprocessor.FineCoarseDataPreprocessor,
                          pred_fine_data: np.array,
                          pred_coarse_data: np.array,
                          loss: str,
                          true_fine_data: np.array,
                          true_coarse_data: np.array,
                          split: str,
                          prior: bool = True,
                          combined: bool = True,
                          model_name: str = '',
                          lr: typing.Union[str, float] = '',
                          print_inconsistencies: bool = True,
                          original_pred_fine_data: np.array = None,
                          original_pred_coarse_data: np.array = None):
    """
    Calculates, prints, and returns accuracy metrics for fine and coarse granularities.

    :param split: The dataset split (e.g., 'test', 'train').
    :param preprocessor: The preprocessor object.
    :param original_pred_coarse_data: The original predictions at the coarse granularity.
    :param original_pred_fine_data: The original predictions at the fine granularity.
    :param print_inconsistencies: Whether to print the number of inconsistencies.
    :param pred_fine_data: NumPy array of predictions at the fine granularity.
    :param pred_coarse_data: NumPy array of predictions at the coarse granularity.
    :param loss: The loss function used during training.
    :param true_fine_data: NumPy array of true labels at the fine granularity.
    :param true_coarse_data: NumPy array of true labels at the coarse granularity.
    :param prior: Whether the model is prior or post.
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
          f"with {utils.blue_text(loss)} loss on the {split} dataset\n" +
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

    return fine_accuracy, coarse_accuracy, fine_f1, coarse_f1


def get_and_print_post_epoch_binary_metrics(epoch: int,
                                            num_epochs: int,
                                            train_ground_truths: np.array,
                                            train_predictions: np.array,
                                            total_running_loss: float):
    accuracy, f1, _, _ = get_individual_metrics(pred_data=train_predictions,
                                                true_data=train_ground_truths,
                                                labels=[1])

    print(f'\nEpoch {epoch + 1}/{num_epochs} done'
          f'\nMean loss across epochs: {round(total_running_loss / (epoch + 1), 2)}'
          f'\npost-epoch training accuracy: {round(accuracy * 100, 2)}%'
          f', post-epoch f1: {round(f1 * 100, 2)}%\n')

    return accuracy, f1


def get_and_print_post_metrics(preprocessor: data_preprocessor.FineCoarseDataPreprocessor,
                               train_fine_ground_truth: np.array,
                               train_fine_prediction: np.array,
                               train_coarse_ground_truth: np.array,
                               train_coarse_prediction: np.array,
                               curr_epoch: int = None,
                               total_num_epochs: int = None,
                               curr_batch_num: int = None,
                               total_batch_num: int = None):
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
    post_str = f'Epoch {curr_epoch + 1}/{total_num_epochs} done' if curr_epoch is not None \
        else f'Batch {curr_batch_num}/{total_batch_num} done'

    print(f'\n{post_str}'
          f'\nTraining fine accuracy: {round(training_fine_accuracy * 100, 2)}%'
          f', training fine f1: {round(training_fine_f1 * 100, 2)}%'
          f'\nLast batch training coarse accuracy: {round(training_coarse_accuracy * 100, 2)}%'
          f', last batch coarse f1: {round(training_coarse_f1 * 100, 2)}%\n')

    return training_fine_accuracy, training_coarse_accuracy



def print_post_batch_binary_metrics(batch_num: int,
                                    num_batches: int,
                                    train_ground_truths: np.array,
                                    train_predictions: np.array,
                                    batch_total_loss: float = None):
    if batch_num > 0 and batch_num % 5 == 0:
        accuracy, f1, _, _ = get_individual_metrics(pred_data=train_predictions,
                                                    true_data=train_ground_truths,
                                                    labels=[0, 1])
        print(f'\nCompleted batch num {batch_num}/{num_batches}, current accuracy: {accuracy:.2f},'
              f'current f1: {f1:.2f}, batch total loss: {batch_total_loss:.2f}')
