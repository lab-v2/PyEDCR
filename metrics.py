import numpy as np

import data_preprocessing
import utils


def get_l_detection_rule_support(edcr,
                                 test: bool,
                                 l: data_preprocessing.Label) -> float:
    if l not in edcr.error_detection_rules:
        return 0

    test_or_train = 'test' if test else 'train'

    N_l = np.sum(edcr.get_where_label_is_l(pred=True, test=test, l=l))
    r_l = edcr.error_detection_rules[l]
    where_l_detection_rule_body_is_satisfied = (
        r_l.get_where_body_is_satisfied(
            pred_fine_data=edcr.pred_data[test_or_train]['original'][data_preprocessing.granularities['fine']],
            pred_coarse_data=edcr.pred_data[test_or_train]['original'][data_preprocessing.granularities['coarse']]))
    num_predicted_l_and_any_conditions_satisfied = np.sum(where_l_detection_rule_body_is_satisfied)
    s_l = num_predicted_l_and_any_conditions_satisfied / N_l

    assert s_l <= 1

    return s_l


def get_l_detection_rule_confidence(edcr,
                                    test: bool,
                                    l: data_preprocessing.Label) -> float:
    if l not in edcr.error_detection_rules:
        return 0

    test_or_train = 'test' if test else 'train'

    r_l = edcr.error_detection_rules[l]
    where_l_detection_rule_body_is_satisfied = (
        r_l.get_where_body_is_satisfied(
            pred_fine_data=edcr.pred_data[test_or_train]['original'][data_preprocessing.granularities['fine']],
            pred_coarse_data=edcr.pred_data[test_or_train]['original'][data_preprocessing.granularities['coarse']]))
    where_l_fp = edcr.get_where_fp_l(test=test, l=l)
    where_head_and_body_is_satisfied = where_l_detection_rule_body_is_satisfied * where_l_fp

    num_where_l_detection_rule_body_is_satisfied = np.sum(where_l_detection_rule_body_is_satisfied)

    if num_where_l_detection_rule_body_is_satisfied == 0:
        return 0

    c_l = np.sum(where_head_and_body_is_satisfied) / num_where_l_detection_rule_body_is_satisfied
    return c_l


def get_l_detection_rule_theoretical_precision_increase(edcr,
                                                        test: bool,
                                                        l: data_preprocessing.Label) -> float:
    s_l = edcr.get_l_detection_rule_support(test=test, l=l)

    if s_l == 0:
        return 0

    c_l = edcr.get_l_detection_rule_confidence(test=test, l=l)
    p_l = edcr.get_l_precision_and_recall(l=l, test=test, stage='original')[0]

    return s_l / (1 - s_l) * (c_l + p_l - 1)


def get_g_detection_rule_theoretical_precision_increase(edcr,
                                                        test: bool,
                                                        g: data_preprocessing.Granularity):
    precision_increases = [edcr.get_l_detection_rule_theoretical_precision_increase(test=test, l=l)
                           for l in data_preprocessing.get_labels(g).values()]
    return np.mean(precision_increases)


def get_l_detection_rule_theoretical_recall_decrease(edcr,
                                                     test: bool,
                                                     l: data_preprocessing.Label) -> float:
    c_l = edcr.get_l_detection_rule_confidence(test=test, l=l)
    s_l = edcr.get_l_detection_rule_support(test=test, l=l)
    p_l, r_l = edcr.get_l_precision_and_recall(l=l, test=test, stage='original')

    return (1 - c_l) * s_l * r_l / p_l


def get_g_detection_rule_theoretical_recall_decrease(edcr,
                                                     test: bool,
                                                     g: data_preprocessing.Granularity):
    recall_decreases = [edcr.get_l_detection_rule_theoretical_recall_decrease(test=test, l=l)
                        for l in data_preprocessing.get_labels(g).values()]
    return np.mean(recall_decreases)


def get_l_correction_rule_confidence(edcr,
                                     test: bool,
                                     l: data_preprocessing.Label,
                                     pred_fine_data: np.array = None,
                                     pred_coarse_data: np.array = None
                                     ) -> float:
    if l not in edcr.error_correction_rules:
        return 0

    test_or_train = 'test' if test else 'train'

    r_l = edcr.error_correction_rules[l]
    where_l_correction_rule_body_is_satisfied = (
        r_l.get_where_body_is_satisfied(
            fine_data=edcr.pred_data[test_or_train]['post_correction'][data_preprocessing.granularities['fine']]
            if pred_fine_data is None else pred_fine_data,
            coarse_data=edcr.pred_data[test_or_train]['post_correction'][data_preprocessing.granularities['coarse']]
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
                                  l: data_preprocessing.Label,
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
            fine_data=edcr.pred_data[test_or_train]['post_correction'][data_preprocessing.granularities['fine']]
            if pred_fine_data is None else pred_fine_data,
            coarse_data=edcr.pred_data[test_or_train]['post_correction'][
                data_preprocessing.granularities['coarse']]
            if pred_coarse_data is None else pred_coarse_data))

    s_l = np.sum(where_rule_body_is_satisfied) / N_l

    return s_l


def get_l_correction_rule_theoretical_precision_increase(edcr,
                                                         test: bool,
                                                         l: data_preprocessing.Label) -> float:
    c_l = edcr.get_l_correction_rule_confidence(test=test, l=l)
    s_l = edcr.get_l_correction_rule_support(test=test, l=l)
    p_l_prior_correction = edcr.get_l_precision_and_recall(l=l,
                                                           test=test,
                                                           stage='post_correction')[0]

    return s_l * (c_l - p_l_prior_correction) / (1 + s_l)


def get_g_correction_rule_theoretical_precision_increase(edcr,
                                                         test: bool,
                                                         g: data_preprocessing.Granularity):
    precision_increases = [edcr.get_l_correction_rule_theoretical_precision_increase(test=test, l=l)
                           for l in data_preprocessing.get_labels(g).values()]
    return np.mean(precision_increases)


def evaluate_and_print_g_detection_rule_precision_increase(edcr,
                                                           test: bool,
                                                           g: data_preprocessing.Granularity,
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
                                                        g: data_preprocessing.Granularity,
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


def get_l_correction_rule_theoretical_recall_increase(edcr,
                                                      test: bool,
                                                      l: data_preprocessing.Label,
                                                      CC_l: set) -> float:
    POS_CC_l = edcr.get_POS_l_CC(test=test, l=l, CC=CC_l)
    denominator = np.sum(edcr.get_where_label_is_l(pred=False, test=True, l=l, stage='original'))

    return POS_CC_l / denominator
