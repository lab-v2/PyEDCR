import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import multiprocessing as mp
import multiprocessing.managers
import typing
import warnings

warnings.filterwarnings('ignore')

import vit_pipeline
import utils
import data_preprocessing
import context_handlers

figs_folder = 'figs/'

main_model_name = 'vit_b_16'
# main_lr = 0.0001
epochs_num = 20

secondary_model_name = 'vit_l_16'
secondary_lr = 0.0001


def get_binary_condition_values(example_index: int,
                                fine_cla_datas: np.array,
                                coarse_cla_datas: np.array):
    res = []
    for fine_i, fine_cls in enumerate(fine_cla_datas[:, example_index].astype(int)):
        for coarse_i, coarse_cls in enumerate(coarse_cla_datas[:, example_index].astype(int)):
            pred = int(fine_cls & coarse_cls)
            consistent = int(pred & (data_preprocessing.fine_to_course_idx[fine_i] == coarse_i))
            res += [pred, consistent]

    return res


def get_assign_values_for_example(example_index: int,
                                  train_conditions_datas: np.array) -> list[int]:
    # train_conditions_datas.shape = (classes, examples)
    n_classes, n_examples = train_conditions_datas.shape

    return [int(train_conditions_datas[row_index, example_index]) for row_index in range(n_classes)]


def get_scores(y_true: np.array,
               y_pred: np.array):
    # try:
        y_actual = y_true
        y_hat = y_pred
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(y_hat)):
            if y_actual[i] == y_hat[i] == 1:
                TP += 1
            if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
                FP += 1
            if y_actual[i] == y_hat[i] == 0:
                TN += 1
            if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
                FN += 1
        # print(f"TP:{TP}, FP:{FP}, TN:{TN}, FN:{FN}")

        # pre = precision_score(y_true, y_pred)
        # rec = recall_score(y_true, y_pred)
        # f1 = f1_score(y_true, y_pred)
        # return [pre, rec, f1]
    # except:
        pre = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        f1micro = f1_score(y_true, y_pred, average='micro')

        return [pre, f1, f1micro]


def rearrange_values(n_classes: int,
                     pred_assign_condition_values: list[list]) -> list:
    all_values = [[] for _ in range(n_classes)]

    for value in pred_assign_condition_values:
        for class_index, class_values in enumerate(all_values):
            # pred, corr, tp, fp, cond1, cond2 ... condn
            each_items = []
            pred, corr = value[:2]

            # pred, corr
            for pred_or_corr in [pred, corr]:
                each_items.append(int(pred_or_corr == class_index))

            # tp
            each_items.append(int(each_items[0] == 1 and each_items[1] == 1))

            # fp
            each_items.append(int(each_items[0] == 1 and each_items[1] == 0))

            each_items.extend(value[2:])
            class_values.append(each_items)

    return all_values


def DetUSMPosRuleSelect(i: int,
                        all_charts: list):
    chart = all_charts[i]
    chart = np.array(chart)
    rule_indexs = [i for i in range(4, len(chart[0]))]
    each_sum = np.sum(chart, axis=0)
    tpi = each_sum[2]
    fpi = each_sum[3]
    pi = tpi * 1.0 / (tpi + fpi)

    pb_scores = []
    for ri in rule_indexs:
        posi = np.sum(chart[:, 1] * chart[:, ri], axis=0)
        bodyi = np.sum(chart[:, ri], axis=0)
        score = posi * 1.0 / bodyi
        if score > pi:
            pb_scores.append((score, ri))
    pb_scores = sorted(pb_scores)
    cci = []
    ccn = pb_scores
    for (score, ri) in pb_scores:

        cii = 0
        for (cs, ci) in cci:
            cii = cii | chart[:, ci]
        POScci = np.sum(cii * chart[:, 1], axis=0)
        BODcci = np.sum(cii, axis=0)
        POSccij = np.sum((cii | chart[:, ri]) * chart[:, 1], axis=0)
        BODccij = np.sum((cii | chart[:, ri]), axis=0)

        cni = 0
        cnij = 0
        for (cs, ci) in ccn:
            cni = (cni | chart[:, ci])
            if ci == ri:
                continue
            cnij = (cnij | chart[:, ci])
        POScni = np.sum(cni * chart[:, 1], axis=0)
        BODcni = np.sum(cni, axis=0)
        POScnij = np.sum(cnij * chart[:, 1], axis=0)
        BODcnij = np.sum(cnij, axis=0)

        a = POSccij * 1.0 / (BODccij + 0.001) - POScci * 1.0 / (BODcci + 0.001)
        b = POScnij * 1.0 / (BODcnij + 0.001) - POScni * 1.0 / (BODcni + 0.001)
        if a >= b:
            cci.append((score, ri))
        else:
            ccn.remove((score, ri))

    cii = 0
    for (cs, ci) in cci:
        cii = cii | chart[:, ci]
    POScci = np.sum(cii * chart[:, 1], axis=0)
    BODcci = np.sum(cii, axis=0)
    new_pre = POScci * 1.0 / (BODcci + 0.001)
    if new_pre < pi:
        cci = []
    cci = [c[1] for c in cci]
    # print(f"class{count}, cci:{cci}, new_pre:{new_pre}, pre:{pi}")

    return cci


def GreedyNegRuleSelect(i: int,
                        epsilon: float,
                        all_values: list):
    class_values = np.array(all_values[i])
    num_examples, num_values = class_values.shape

    condition_indices = list(range(4, num_values))
    each_sum = np.sum(class_values, axis=0)
    tp_i = each_sum[2]
    fp_i = each_sum[3]
    p_i = tp_i * 1.0 / (tp_i + fp_i)
    r_i = tp_i * 1.0 / each_sum[1]
    n_i = each_sum[0]
    q_i = epsilon * n_i * p_i / r_i
    # print(f"class{count}, q_i:{q_i}")

    all_class_tps = class_values[:, 2]
    all_class_fps = class_values[:, 3]

    DC_i = []
    DC_star = []

    for condition_index in condition_indices:
        condition_index_values = class_values[:, condition_index]

        neg_i = np.sum(all_class_tps * condition_index_values)

        if neg_i < q_i:
            DC_star.append(condition_index)

    with context_handlers.WrapTQDM(total=len(DC_star)) as progress_bar:
        while len(DC_star):
            best_score = -1
            best_index = -1

            for condition_index_from_DC_star in DC_star:
                tem_cond = 1

                # maximizing pos for c_best
                for condition_index in DC_i:
                    condition_index_values = class_values[:, condition_index]
                    tem_cond &= condition_index_values

                tem_cond &= class_values[:, condition_index_from_DC_star]

                pos_i_score = np.sum(all_class_fps * tem_cond)

                if best_score < pos_i_score:
                    best_score = pos_i_score
                    best_index = condition_index_from_DC_star

            DC_i.append(best_index)
            DC_star.remove(best_index)

            tem_cond = 1
            for condition_index in DC_i:
                tem_cond &= class_values[:, condition_index]

            tmp_DC_star = []

            for condition_index_from_DC_star in DC_star:
                tem = tem_cond & class_values[:, condition_index_from_DC_star]
                neg_i = np.sum(all_class_tps * tem)

                if neg_i < q_i:
                    tmp_DC_star.append(condition_index_from_DC_star)

            DC_star = tmp_DC_star

            if utils.is_local():
                time.sleep(0.1)
                progress_bar.update(1)

        # print(f"class:{i}, DC_i:{DC_i}")

    return DC_i


# def ruleForNPCorrection_worker_EC():
#     pass


def ruleForNPCorrection_worker(i: int,
                               class_values: list,
                               epsilon: float,
                               all_values: list[list],
                               main_granularity: str,
                               run_positive_rules: bool,
                               total_results: multiprocessing.managers.ListProxy,
                               shared_index: multiprocessing.managers.ValueProxy,
                               error_detections: multiprocessing.managers.DictProxy,
                               possible_test_consistency_constraints: dict[str, set]):
    classes = data_preprocessing.fine_grain_classes if main_granularity == 'fine' \
        else data_preprocessing.coarse_grain_classes
    curr_class = classes[i]
    class_values = np.array(class_values)

    DCi = GreedyNegRuleSelect(i=i,
                              epsilon=epsilon,
                              all_values=all_values)
    neg_i = 0
    pos_i = 0

    pred_i_for_all_examples = np.copy(class_values[:, 0])
    tem_cond = np.zeros_like(pred_i_for_all_examples)

    for cc in DCi:
        tem_cond |= class_values[:, cc]

    recovered = set()

    if np.sum(tem_cond) > 0:
        for example_index, example_values in enumerate(class_values):
            if tem_cond[example_index] and pred_i_for_all_examples[example_index]:
                neg_i += 1
                pred_i_for_all_examples[example_index] = 0

                condition_values = example_values[4:]

                if main_granularity == 'coarse':
                    fine_grain_condition_values = condition_values[:len(data_preprocessing.fine_grain_classes)]
                    fine_grain_prediction = data_preprocessing.fine_grain_classes[
                        np.argmax(fine_grain_condition_values)]
                    derived_coarse_grain_prediction = data_preprocessing.fine_to_coarse[fine_grain_prediction]

                    if derived_coarse_grain_prediction != curr_class:
                        recovered = recovered.union({fine_grain_prediction})
                else:
                    coarse_grain_condition_values = condition_values[len(data_preprocessing.fine_grain_classes):
                                                                     len(data_preprocessing.fine_grain_classes) +
                                                                     len(data_preprocessing.coarse_grain_classes)]
                    coarse_grain_prediction = data_preprocessing.coarse_grain_classes[
                        np.argmax(coarse_grain_condition_values)]
                    derived_coarse_grain_prediction = data_preprocessing.fine_to_coarse[curr_class]

                    if coarse_grain_prediction != derived_coarse_grain_prediction:
                        recovered = recovered.union({coarse_grain_prediction})

    if curr_class in possible_test_consistency_constraints:
        all_possible_constraints = len(possible_test_consistency_constraints[curr_class])
        error_detections[curr_class] = round(len(recovered) / all_possible_constraints * 100, 2)

    CCi = DetUSMPosRuleSelect(i=i,
                              all_charts=all_values) if run_positive_rules else []
    tem_cond = np.zeros_like(class_values[:, 0])

    for cc in CCi:
        tem_cond |= class_values[:, cc]

    if np.sum(tem_cond) > 0:
        for example_index, cv in enumerate(class_values):
            if tem_cond[example_index] and not pred_i_for_all_examples[example_index]:
                pos_i += 1
                pred_i_for_all_examples[example_index] = 1
                total_results[example_index] = i

    scores_cor = get_scores(class_values[:, 1], pred_i_for_all_examples)

    if not utils.is_local():
        shared_index.value += 1
        print(f'Completed {shared_index.value}/{len(all_values)}')

    return scores_cor + [neg_i,
                         pos_i,
                         len(DCi),
                         len(CCi)]


def ruleForNPCorrectionMP(all_values: list[list],
                          test_pred_granularity: np.array,
                          train_true_granularity: np.array,
                          train_pred_granularity: np.array,
                          main_granularity: str,
                          epsilon: float,
                          possible_test_consistency_constraints: dict[str, set],
                          run_positive_rules: bool = True):
    manager = mp.Manager()
    shared_results = manager.list(train_pred_granularity)
    error_detections = manager.dict({})
    shared_index = manager.Value('i', 0)

    args_list = [(i,
                  class_values,
                  epsilon,
                  all_values,
                  main_granularity,
                  run_positive_rules,
                  shared_results,
                  shared_index,
                  error_detections,
                  possible_test_consistency_constraints)
                 for i, class_values in enumerate(all_values)]

    n_classes = len(all_values)
    cpu_count = mp.cpu_count()

    # Create a pool of processes and map the function with arguments
    processes_num = min(n_classes, cpu_count)

    with mp.Pool(processes_num) as pool:
        print(f'Num of processes: {processes_num}')
        results = pool.starmap(func=ruleForNPCorrection_worker,
                               iterable=args_list)

    shared_results = np.array(list(shared_results))
    error_detections_values = np.array(list(dict(error_detections).values()))

    error_detections_mean = np.mean(error_detections_values)
    print(f'Mean error detections found for {main_granularity}-grain: {error_detections_mean}')

    results = [item for sublist in results for item in sublist]

    results.extend(get_scores(y_true=train_true_granularity,
                              y_pred=shared_results))
    posterior_acc = accuracy_score(y_true=train_true_granularity,
                                   y_pred=shared_results)

    # retrieve_error_detection_rule(error_detections)

    return results, posterior_acc, shared_results, error_detections_mean


def ruleForNPCorrection(all_charts: list,
                        true_data,
                        pred_data,
                        epsilon: float,
                        run_positive_rules: bool = True):
    results = []
    total_results = np.copy(pred_data)
    print(len(all_charts))

    for i, chart in enumerate(all_charts):
        with ((context_handlers.TimeWrapper(s=f'Class {i} is done'))):
            chart = np.array(chart)
            NCi = GreedyNegRuleSelect(i=i,
                                      epsilon=epsilon,
                                      all_values=all_charts)
            neg_i_count = 0
            pos_i_count = 0

            predict_result = np.copy(chart[:, 0])
            tem_cond = np.zeros_like(chart[:, 0])

            for cc in NCi:
                tem_cond |= chart[:, cc]

            if np.sum(tem_cond) > 0:
                for ct, cv in enumerate(chart):
                    if tem_cond[ct] and predict_result[ct]:
                        neg_i_count += 1
                        predict_result[ct] = 0

            CCi = DetUSMPosRuleSelect(i=i,
                                      all_charts=all_charts) if run_positive_rules else []
            tem_cond = np.zeros_like(chart[:, 0])

            for cc in CCi:
                tem_cond |= chart[:, cc]

            if np.sum(tem_cond) > 0:
                for ct, cv in enumerate(chart):
                    if tem_cond[ct] and not predict_result[ct]:
                        pos_i_count += 1
                        predict_result[ct] = 1
                        total_results[ct] = i

            scores_cor = get_scores(chart[:, 1], predict_result)
            results.extend(scores_cor + [neg_i_count,
                                         pos_i_count,
                                         len(NCi),
                                         len(CCi)])

    results.extend(get_scores(true_data, total_results))
    posterior_acc = accuracy_score(true_data, total_results)

    return results, posterior_acc, total_results, None


def plot(df: pd.DataFrame,
         classes: list,
         col_num: int,
         x_values: pd.Series,
         main_granularity: str,
         main_model_name: str,
         main_lr: float,
         # secondary_model_name: str,
         # secondary_lr: float,
         folder: str):
    average_precision = pd.Series(data=0,
                                  index=x_values.index)
    average_recall = pd.Series(data=0,
                               index=x_values.index)
    average_f1_score = pd.Series(data=0,
                                 index=x_values.index)

    for i in range(len(classes)):
        df_i = df.iloc[1:, 2 + i * col_num:2 + (i + 1) * col_num]

        added_str = f'.{i}' if i else ''
        precision_i = df_i[f'pre']
        recall_i = df_i[f'recall']
        f1_score_i = df_i[f'F1']

        average_precision += float(precision_i)
        average_recall += float(recall_i)
        average_f1_score += float(f1_score_i)

        plt.plot(x_values,
                 precision_i,
                 label='precision')
        plt.plot(x_values,
                 recall_i,
                 label='recall')
        plt.plot(x_values,
                 f1_score_i,
                 label='f1')

        plt.title(f'{main_granularity.capitalize()}-grain class #{i}-{classes[i]}, '
                  f'Main: {main_model_name}, lr: {main_lr}'
                  # f', secondary: {secondary_model_name}, lr: {secondary_lr}'
                  )
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.savefig(f'{folder}/cls{i}.png')
        plt.clf()
        plt.cla()

    plt.plot(x_values,
             average_precision / len(classes),
             label='average precision')
    plt.plot(x_values,
             average_recall / len(classes),
             label='average recall')
    plt.plot(x_values,
             average_f1_score / len(classes),
             label='average f1')

    plt.title(f'{main_granularity.capitalize()}-grain main: {main_model_name}, lr: {main_lr}, '
              # f'Secondary: {secondary_model_name}, lr: {secondary_lr}'
              )
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig(f'{folder}/average.png')
    plt.clf()
    plt.cla()


def retrieve_error_detection_rule(error_detections):
    for coarse_grain_label, coarse_grain_label_data in error_detections.items():
        for fine_grain_label in coarse_grain_label_data.keys():
            print(f'error <- predicted_coarse_grain = {coarse_grain_label} '
                  f'and predicted_fine_grain = {fine_grain_label}')


def get_one_hot_encoding(arr: np.array) -> np.array:
    return np.eye(np.max(arr) + 1)[arr].T


def get_possible_consistency_constraints(pred_fine_data: np.array,
                                         pred_coarse_data: np.array) -> dict[str, dict[str]]:
    possible_consistency_constraints = {}

    for fine_prediction_index, coarse_prediction_index in zip(pred_fine_data, pred_coarse_data):
        fine_prediction = data_preprocessing.fine_grain_classes[fine_prediction_index]
        coarse_prediction = data_preprocessing.coarse_grain_classes[coarse_prediction_index]

        if data_preprocessing.fine_to_coarse[fine_prediction] != coarse_prediction:
            if coarse_prediction not in possible_consistency_constraints:
                possible_consistency_constraints[coarse_prediction] = {fine_prediction}
            else:
                possible_consistency_constraints[coarse_prediction] = (
                    possible_consistency_constraints[coarse_prediction].union({fine_prediction}))

            if fine_prediction not in possible_consistency_constraints:
                possible_consistency_constraints[fine_prediction] = {coarse_prediction}
            else:
                possible_consistency_constraints[fine_prediction] = (
                    possible_consistency_constraints[fine_prediction].union({coarse_prediction}))

    return possible_consistency_constraints


def load_priors(test_pred_fine_path: str,
                test_pred_coarse_path: str,
                test_true_fine_path: str,
                test_true_coarse_path: str,
                train_pred_fine_path: str,
                train_pred_coarse_path: str,
                train_true_fine_path: str,
                train_true_coarse_path: str,
                main_lr,
                loss: str,
                combined: bool) -> (np.array, np.array):
    # loss_str = f'{loss}_' if (loss == 'soft_marginal' and combined) else ''

    # if combined:
    #     main_model_fine_path = f'{main_model_name}_{loss_str}test_fine_pred_lr{main_lr}_e{epochs_num - 1}.npy'
    #     main_model_coarse_path = f'{main_model_name}_{loss_str}test_coarse_pred_lr{main_lr}_e{epochs_num - 1}.npy'
    # else:
    #     main_model_fine_path = (f'{main_model_name}_{loss_str}test_pred_lr{main_lr}_e{epochs_num - 1}'
    #                             f'_fine_individual.npy')
    #     main_model_coarse_path = (f'{main_model_name}_{loss_str}test_pred_lr{main_lr}_e{epochs_num - 1}'
    #                               f'_coarse_individual.npy')
    #
    # path = vit_pipeline.combined_results_path if combined else vit_pipeline.individual_results_path

    # secondary_model_fine_path = (f'{secondary_model_name}_test_fine_pred_lr{secondary_lr}'
    #                              f'_e9.npy')
    # secondary_model_coarse_path = (f'{secondary_model_name}_test_coarse_pred_lr{secondary_lr}'
    #                                f'_e9.npy')

    # main_fine_data = np.load(os.path.join(path, main_model_fine_path))
    # main_coarse_data = np.load(os.path.join(path, main_model_coarse_path))

    # secondary_fine_data = np.load(os.path.join(vit_pipeline.combined_results_path, secondary_model_fine_path))
    # secondary_coarse_data = np.load(os.path.join(vit_pipeline.combined_results_path, secondary_model_coarse_path))

    # secondary_prior_fine_acc = accuracy_score(y_true=true_fine_data, y_pred=secondary_fine_data)
    # secondary_prior_coarse_acc = accuracy_score(y_true=true_coarse_data, y_pred=secondary_coarse_data)

    test_pred_fine_data = np.load(test_pred_fine_path)
    test_pred_coarse_data = np.load(test_pred_coarse_path)

    test_true_fine_data = np.load(test_true_fine_path)
    test_true_coarse_data = np.load(test_true_coarse_path)

    train_pred_fine_data = np.load(train_pred_fine_path)
    train_pred_coarse_data = np.load(train_pred_coarse_path)

    train_true_fine_data = np.load(train_true_fine_path)
    train_true_coarse_data = np.load(train_true_coarse_path)

    print(utils.blue_text('\n' + '#' * 50 + 'Train metrics' + '#' * 50))
    vit_pipeline.get_and_print_metrics(pred_fine_data=train_pred_fine_data,
                                       pred_coarse_data=train_pred_coarse_data,
                                       loss=loss,
                                       true_fine_data=train_true_fine_data,
                                       true_coarse_data=train_true_coarse_data,
                                       combined=combined,
                                       model_name=main_model_name,
                                       lr=main_lr)

    print(utils.blue_text('\n' + '#' * 50 + 'Test metrics' + '#' * 50))
    vit_pipeline.get_and_print_metrics(pred_fine_data=test_pred_fine_data,
                                       pred_coarse_data=test_pred_coarse_data,
                                       loss=loss,
                                       true_fine_data=test_true_fine_data,
                                       true_coarse_data=test_true_coarse_data,
                                       combined=combined,
                                       model_name=main_model_name,
                                       lr=main_lr)

    possible_test_consistency_constraints = (
        get_possible_consistency_constraints(pred_fine_data=test_pred_fine_data,
                                             pred_coarse_data=test_pred_coarse_data))

    # for coarse_prediction, fine_grain_inconsistencies in consistency_constraints_for_main_model.items():
    #     assert len(set(data_preprocessing.coarse_to_fine[coarse_prediction]).
    #                intersection(fine_grain_inconsistencies)) == 0

    # print([f'{k}: {len(v)}' for k, v in consistency_constraints_for_main_model.items()])

    return (test_pred_fine_data,
            test_pred_coarse_data,
            test_true_fine_data,
            test_true_coarse_data,
            train_pred_fine_data,
            train_pred_coarse_data,
            train_true_fine_data,
            train_true_coarse_data,
            possible_test_consistency_constraints)


def get_conditions_from_train(train_fine_data: np.array,
                              train_coarse_data: np.array,
                              # secondary_fine_data: np.array
                              ) -> dict[str, dict[str, np.array]]:
    condition_datas = {}

    for main_or_secondary in ['main',
                              # 'secondary'
                              ]:
        if main_or_secondary not in condition_datas:
            condition_datas[main_or_secondary] = {}

        # take_conditions_from = main_fine_data if main_or_secondary == 'main' else secondary_fine_data

        one_hot_encodings = {}
        for granularity in data_preprocessing.granularities:
            train_granularity_data = train_fine_data if granularity == 'fine' else train_coarse_data
            one_hot_encodings[granularity] = get_one_hot_encoding(train_granularity_data)

        # concatenated one-hot encoding with fine first and then coarse

        condition_datas[main_or_secondary] = np.concatenate([one_hot_encodings[granularity]
                                                             for granularity in data_preprocessing.granularities],
                                                            axis=0)

        # derived_coarse = np.array([data_preprocessing.fine_to_course_idx[fine_grain_prediction]
        #                            for fine_grain_prediction in train_fine_data])

        # condition_datas[main_or_secondary]['fine_to_coarse'] = get_one_hot_encoding(derived_coarse)

    return condition_datas


def run_EDCR_for_granularity(combined: bool,
                             main_lr: typing.Union[str, float],
                             main_granularity: str,
                             test_pred_granularity: np.array,
                             train_pred_granularity: np.array,
                             train_true_granularity: np.array,
                             train_condition_datas: dict[str, dict[str, np.array]],
                             multiprocessing: bool,
                             possible_test_consistency_constraints: dict[str, set]) -> np.array:
    with context_handlers.TimeWrapper():
        classes = data_preprocessing.fine_grain_classes if main_granularity == 'fine' else \
            data_preprocessing.coarse_grain_classes

        examples_num = train_true_granularity.shape[0]

        pred_assign_condition_values = [[train_pred_granularity[example_index], train_true_granularity[example_index]] +
                                        ((
                                             get_assign_values_for_example(example_index=example_index,
                                                                           train_conditions_datas=train_condition_datas[
                                                                               'main'])
                                             # +
                                             # (get_binary_condition_values(example_index=example_index,
                                             #                              fine_cla_datas=train_condition_datas['main']['fine'],
                                             #                              coarse_cla_datas=train_condition_datas['main']['coarse'])
                                             #  if consistency_constraints else [])
                                             # +
                                             # get_unary_condition_values(example_index=example_index,
                                             #                            cla_datas=condition_datas['main']['fine_to_coarse'])
                                         ) if combined else [])
                                        +
                                        (
                                            (
                                                    get_assign_values_for_example(example_index=example_index,
                                                                                  train_conditions_datas=
                                                                                  train_condition_datas['secondary'][
                                                                                      'fine'])
                                                    +
                                                    get_assign_values_for_example(example_index=example_index,
                                                                                  train_conditions_datas=
                                                                                  train_condition_datas['secondary'][
                                                                                      'coarse'])
                                                # +
                                                # (get_binary_condition_values(example_index=example_index,
                                                #                              fine_cla_datas=condition_datas['secondary']['fine'],
                                                #                              coarse_cla_datas=condition_datas['secondary']['coarse'])
                                                #  if consistency_constraints else [])
                                                # +
                                                # get_unary_condition_values(example_index=example_index,
                                                #                            cla_datas=condition_datas['secondary']['fine_to_coarse'])
                                            )
                                            if not combined else [])
                                        for example_index in range(examples_num)]

        all_values = rearrange_values(n_classes=len(classes),
                                      pred_assign_condition_values=pred_assign_condition_values)

        results = []
        result0 = [0]

        print(f'Started EDCR pipeline for the {main_granularity}-grain main classes...'
              # f', secondary: {secondary_model_name}, lr: {secondary_lr}\n'
              )

        for class_index, class_values in enumerate(all_values):
            class_values = np.array(class_values)
            result0.extend(get_scores(class_values[:, 1], class_values[:, 0]))
            result0.extend([0, 0, 0, 0])

        result0.extend(get_scores(train_true_granularity, train_pred_granularity))
        results.append(result0)

        # posterior_acc = 0
        total_results = np.zeros_like(train_pred_granularity)

        epsilons = [0.002 * i for i in range(1, 2, 1)]

        for epsilon in epsilons:
            result, posterior_acc, total_results, error_detections_mean = ruleForNPCorrectionMP(
                all_values=all_values,
                test_pred_granularity=test_pred_granularity,
                train_true_granularity=train_true_granularity,
                train_pred_granularity=train_pred_granularity,
                main_granularity=main_granularity,
                epsilon=epsilon,
                possible_test_consistency_constraints=possible_test_consistency_constraints
            ) if multiprocessing else ruleForNPCorrection(all_charts=all_values,
                                                          true_data=train_true_granularity,
                                                          pred_data=train_pred_granularity,
                                                          epsilon=epsilon)

            results.append([epsilon] + result)

        # prior_acc = main_prior_fine_acc if main_granularity == 'fine' else main_prior_coarse_acc
        # print(f'\nSaved plots for main: {main_granularity}-grain {main_model_name}, lr={main_lr}'
        #       # f', secondary: {secondary_model_name}, lr={secondary_lr}'
        #       f'\nPrior acc:{prior_acc}, post acc: {posterior_acc}')

        col = ['pre', 'recall', 'F1', 'NSC', 'PSC', 'NRC', 'PRC']
        df = pd.DataFrame(results, columns=['epsilon'] + col * len(classes) + ['acc', 'macro-F1', 'micro-F1'])

        # df.to_csv(results_file)
        # df = pd.read_csv(results_file)

        folder = (f'{figs_folder}/main_{main_granularity}_{main_model_name}_lr{main_lr}'
                  # f'_secondary_{secondary_model_name}_lr{secondary_lr}'
                  )
        utils.create_directory(folder)

        # plot(df=df,
        #      classes=classes,
        #      col_num=len(col),
        #      x_values=df['epsilon'][1:],
        #      main_granularity=main_granularity,
        #      main_model_name=main_model_name,
        #      main_lr=main_lr,
        #      folder=folder)

        np.save(f'{folder}/results.npy', total_results)

        # Save the DataFrame to an Excel file
        df.to_excel(f'{folder}/results.xlsx')

        print(f'\nCompleted {main_granularity}-grain EDCR run'
              # f'saved error detections and corrections to {folder}\n'
              )

    return total_results, error_detections_mean


def run_EDCR_pipeline(test_pred_fine_path: str,
                      test_pred_coarse_path: str,
                      test_true_fine_path: str,
                      test_true_coarse_path: str,
                      train_pred_fine_path: str,
                      train_pred_coarse_path: str,
                      train_true_fine_path: str,
                      train_true_coarse_path: str,
                      main_lr: typing.Union[str, float],
                      combined: bool,
                      loss: str,
                      consistency_constraints: bool,
                      multiprocessing: bool = True):
    (test_pred_fine_data,
     test_pred_coarse_data,
     test_true_fine_data,
     test_true_coarse_data,
     train_pred_fine_data,
     train_pred_coarse_data,
     train_true_fine_data,
     train_true_coarse_data,
     possible_test_consistency_constraints) = load_priors(test_pred_coarse_path=test_pred_coarse_path,
                                                          test_pred_fine_path=test_pred_fine_path,
                                                          test_true_fine_path=test_true_fine_path,
                                                          test_true_coarse_path=test_true_coarse_path,
                                                          train_pred_fine_path=train_pred_fine_path,
                                                          train_pred_coarse_path=train_pred_coarse_path,
                                                          train_true_fine_path=train_true_fine_path,
                                                          train_true_coarse_path=train_true_coarse_path,
                                                          main_lr=main_lr,
                                                          loss=loss,
                                                          combined=combined)
    train_condition_datas = get_conditions_from_train(train_fine_data=train_pred_fine_data,
                                                      train_coarse_data=train_pred_coarse_data,
                                                      # secondary_fine_data=secondary_fine_data
                                                      )
    pipeline_results = {}
    error_detections = []

    for main_granularity in data_preprocessing.granularities:
        if main_granularity == 'fine':
            test_pred_granularity = test_pred_fine_data
            train_pred_granularity = train_pred_fine_data
            train_true_granularity = train_true_fine_data
        else:
            test_pred_granularity = test_pred_coarse_data
            train_pred_granularity = train_pred_coarse_data
            train_true_granularity = train_true_coarse_data

        res = (
            run_EDCR_for_granularity(combined=combined,
                                     main_lr=main_lr,
                                     main_granularity=main_granularity,
                                     test_pred_granularity=test_pred_granularity,
                                     train_pred_granularity=train_pred_granularity,
                                     train_true_granularity=train_true_granularity,
                                     train_condition_datas=train_condition_datas,
                                     multiprocessing=multiprocessing,
                                     possible_test_consistency_constraints=possible_test_consistency_constraints))
        pipeline_results[main_granularity] = res[0]
        if multiprocessing:
            error_detections += [res[1]]

    if multiprocessing:
        error_detections = np.mean(np.array(error_detections))
        print(utils.green_text(f'Mean error detections found {np.mean(error_detections)}'))

    vit_pipeline.get_and_print_metrics(pred_fine_data=pipeline_results['fine'],
                                       pred_coarse_data=pipeline_results['coarse'],
                                       loss=loss,
                                       true_fine_data=train_true_fine_data,
                                       true_coarse_data=train_true_coarse_data,
                                       prior=False,
                                       combined=combined,
                                       model_name=main_model_name,
                                       lr=main_lr)


if __name__ == '__main__':
    combined = True
    print(utils.red_text(f'combined={combined}\n' + '#' * 100 + '\n'))

    test_pred_fine_path = 'combined_results/vit_b_16_test_fine_pred_lr0.0001_e19.npy'
    test_pred_coarse_path = 'combined_results/vit_b_16_test_coarse_pred_lr0.0001_e19.npy'

    test_true_fine_path = 'combined_results/test_true_fine.npy'
    test_true_coarse_path = 'combined_results/test_true_coarse.npy'

    train_pred_fine_path = 'combined_results/train_vit_b_16_fine_pred_lr0.0001.npy'
    train_pred_coarse_path = 'combined_results/train_vit_b_16_coarse_pred_lr0.0001.npy'

    train_true_fine_path = 'combined_results/train_true_fine.npy'
    train_true_coarse_path = 'combined_results/train_true_coarse.npy'

    run_EDCR_pipeline(test_pred_fine_path=test_pred_fine_path,
                      test_pred_coarse_path=test_pred_coarse_path,
                      test_true_fine_path=test_true_fine_path,
                      test_true_coarse_path=test_true_coarse_path,
                      train_pred_fine_path=train_pred_fine_path,
                      train_pred_coarse_path=train_pred_coarse_path,
                      train_true_fine_path=train_true_fine_path,
                      train_true_coarse_path=train_true_coarse_path,
                      main_lr=0.0001,
                      combined=combined,
                      loss='soft_marginal',
                      consistency_constraints=True,
                      multiprocessing=True)
