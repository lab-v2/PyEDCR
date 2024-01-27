import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import multiprocessing as mp
import multiprocessing.managers
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


def get_unary_condition_values(example_index: int,
                               cla_datas: np.array):
    return [int(cls[example_index]) for cls in cla_datas]


def get_scores(y_true: np.array,
               y_pred: np.array):
    try:
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

        pre = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return [pre, rec, f1]
    except:
        pre = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        f1micro = f1_score(y_true, y_pred, average='micro')

        return [pre, f1, f1micro]


def generate_chart(n_classes: int,
                   charts: list) -> list:
    all_charts = [[] for _ in range(n_classes)]

    for data in charts:
        for count, jj in enumerate(all_charts):
            # pred, corr, tp, fp, cond1, cond2 ... condn
            each_items = []
            for d in data[:2]:
                if d == count:
                    each_items.append(1)
                else:
                    each_items.append(0)

            if each_items[0] == 1 and each_items[1] == 1:
                each_items.append(1)
            else:
                each_items.append(0)
            if each_items[0] == 1 and each_items[1] == 0:
                each_items.append(1)
            else:
                each_items.append(0)

            each_items.extend(data[2:])
            jj.append(each_items)

    return all_charts


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
                        all_charts: list):
    chart = all_charts[i]
    chart = np.array(chart)
    rule_indexs = [i for i in range(4, len(chart[0]))]
    each_sum = np.sum(chart, axis=0)
    tpi = each_sum[2]
    fpi = each_sum[3]
    pi = tpi * 1.0 / (tpi + fpi)
    ri = tpi * 1.0 / each_sum[1]
    ni = each_sum[0]
    quantity = epsilon * ni * pi / ri
    # print(f"class{count}, quantity:{quantity}")

    NCi = []
    NCn = []
    for rule in rule_indexs:
        negi_score = np.sum(chart[:, 2] * chart[:, rule])
        if negi_score < quantity:
            NCn.append(rule)

    with context_handlers.WrapTQDM(total=len(NCn)) as progress_bar:
        while NCn:
            best_score = -1
            best_index = -1
            for c in NCn:
                tem_cond = 0
                for cc in NCi:
                    tem_cond |= chart[:, cc]
                tem_cond |= chart[:, c]
                posi_score = np.sum(chart[:, 3] * tem_cond)
                if best_score < posi_score:
                    best_score = posi_score
                    best_index = c
            NCi.append(best_index)
            NCn.remove(best_index)
            tem_cond = 0
            for cc in NCi:
                tem_cond |= chart[:, cc]
            tmp_NCn = []
            for c in NCn:
                tem = tem_cond | chart[:, c]
                negi_score = np.sum(chart[:, 2] * tem)
                if negi_score < quantity:
                    tmp_NCn.append(c)
            NCn = tmp_NCn

            if utils.is_local():
                time.sleep(0.1)
                progress_bar.update(1)

        # print(f"class:{i}, NCi:{NCi}")

    return NCi


def ruleForNPCorrection_worker(i: int,
                               chart: list,
                               epsilon: float,
                               all_charts: list[list],
                               main_granularity: str,
                               run_positive_rules: bool,
                               total_results: multiprocessing.managers.ListProxy,
                               shared_index: multiprocessing.managers.ValueProxy,
                               error_detections: multiprocessing.managers.DictProxy,
                               consistency_constraints_for_main_model: dict[str, set]
                               ):
    chart = np.array(chart)
    NCi = GreedyNegRuleSelect(i=i,
                              epsilon=epsilon,
                              all_charts=all_charts)
    neg_i_count = 0
    pos_i_count = 0

    predict_result = np.copy(chart[:, 0])
    tem_cond = np.zeros_like(chart[:, 0])

    for cc in NCi:
        tem_cond |= chart[:, cc]

    classes = data_preprocessing.fine_grain_classes if main_granularity == 'fine' \
        else data_preprocessing.coarse_grain_classes
    curr_class = classes[i]

    recovered = set()

    if np.sum(tem_cond) > 0:
        for example_index, example_values in enumerate(chart):
            if tem_cond[example_index] and predict_result[example_index]:
                neg_i_count += 1
                predict_result[example_index] = 0

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

    if curr_class in consistency_constraints_for_main_model:
        all_possible_constraints = len(consistency_constraints_for_main_model[curr_class])
        error_detections[curr_class] = round(len(recovered) / all_possible_constraints * 100, 2)

    CCi = DetUSMPosRuleSelect(i=i,
                              all_charts=all_charts) if run_positive_rules else []
    tem_cond = np.zeros_like(chart[:, 0])

    for cc in CCi:
        tem_cond |= chart[:, cc]

    if np.sum(tem_cond) > 0:
        for example_index, cv in enumerate(chart):
            if tem_cond[example_index] and not predict_result[example_index]:
                pos_i_count += 1
                predict_result[example_index] = 1
                total_results[example_index] = i

    scores_cor = get_scores(chart[:, 1], predict_result)

    if not utils.is_local():
        shared_index.value += 1
        print(f'Completed {shared_index.value}/{len(all_charts)}')

    return scores_cor + [neg_i_count,
                         pos_i_count,
                         len(NCi),
                         len(CCi)]


def ruleForNPCorrectionMP(all_charts: list[list],
                          true_data: np.array,
                          pred_data: np.array,
                          main_granularity: str,
                          epsilon: float,
                          consistency_constraints_for_main_model: dict[str, set],
                          run_positive_rules: bool = True):
    manager = mp.Manager()
    shared_results = manager.list(pred_data)
    error_detections = manager.dict({})
    shared_index = manager.Value('i', 0)

    args_list = [(i,
                  chart,
                  epsilon,
                  all_charts,
                  main_granularity,
                  run_positive_rules,
                  shared_results,
                  shared_index,
                  error_detections,
                  consistency_constraints_for_main_model)
                 for i, chart in enumerate(all_charts)]

    # Create a pool of processes and map the function with arguments
    processes_num = min(len(all_charts), mp.cpu_count())

    with mp.Pool(processes_num) as pool:
        print(f'Num of processes: {processes_num}')
        results = pool.starmap(ruleForNPCorrection_worker, args_list)

    shared_results = np.array(list(shared_results))
    error_detections_values = np.array(list(dict(error_detections).values()))

    error_detections_mean = np.mean(error_detections_values)
    print(f'Mean error detections found for {main_granularity}-grain: {error_detections_mean}')

    results = [item for sublist in results for item in sublist]

    results.extend(get_scores(y_true=true_data,
                              y_pred=shared_results))
    posterior_acc = accuracy_score(y_true=true_data,
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
                                      all_charts=all_charts)
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
        precision_i = df_i[f'pre{added_str}']
        recall_i = df_i[f'recall{added_str}']
        f1_score_i = df_i[f'F1{added_str}']

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


def rearrange_for_condition_values(arr: np.array) -> np.array:
    return np.eye(np.max(arr) + 1)[arr].T


def load_priors(main_lr,
                loss: str,
                combined: bool) -> (np.array, np.array):
    loss_str = f'{loss}_' if (loss == 'soft_marginal' and combined) else ''
    if combined:
        main_model_fine_path = f'{main_model_name}_{loss_str}test_fine_pred_lr{main_lr}_e{epochs_num - 1}.npy'
        main_model_coarse_path = f'{main_model_name}_{loss_str}test_coarse_pred_lr{main_lr}_e{epochs_num - 1}.npy'
    else:
        main_model_fine_path = (f'{main_model_name}_{loss_str}test_pred_lr{main_lr}_e{epochs_num - 1}'
                                f'_fine_individual.npy')
        main_model_coarse_path = (f'{main_model_name}_{loss_str}test_pred_lr{main_lr}_e{epochs_num - 1}'
                                  f'_coarse_individual.npy')

    path = vit_pipeline.combined_results_path if combined else vit_pipeline.individual_results_path

    secondary_model_fine_path = (f'{secondary_model_name}_test_fine_pred_lr{secondary_lr}'
                                 f'_e9.npy')
    secondary_model_coarse_path = (f'{secondary_model_name}_test_coarse_pred_lr{secondary_lr}'
                                   f'_e9.npy')

    main_fine_data = np.load(os.path.join(path, main_model_fine_path))
    main_coarse_data = np.load(os.path.join(path, main_model_coarse_path))

    secondary_fine_data = np.load(os.path.join(vit_pipeline.combined_results_path, secondary_model_fine_path))
    secondary_coarse_data = np.load(os.path.join(vit_pipeline.combined_results_path, secondary_model_coarse_path))

    # secondary_prior_fine_acc = accuracy_score(y_true=true_fine_data, y_pred=secondary_fine_data)
    # secondary_prior_coarse_acc = accuracy_score(y_true=true_coarse_data, y_pred=secondary_coarse_data)

    vit_pipeline.get_and_print_metrics(fine_predictions=main_fine_data,
                                       coarse_predictions=main_coarse_data,
                                       loss=loss,
                                       combined=combined,
                                       model_name=main_model_name,
                                       lr=main_lr)

    consistency_constraints_for_main_model = {}

    for fine_prediction_index, coarse_prediction_index in zip(main_fine_data, main_coarse_data):
        fine_prediction = data_preprocessing.fine_grain_classes[fine_prediction_index]
        coarse_prediction = data_preprocessing.coarse_grain_classes[coarse_prediction_index]

        if data_preprocessing.fine_to_coarse[fine_prediction] != coarse_prediction:
            if coarse_prediction not in consistency_constraints_for_main_model:
                consistency_constraints_for_main_model[coarse_prediction] = {fine_prediction}
            else:
                consistency_constraints_for_main_model[coarse_prediction] = (
                    consistency_constraints_for_main_model[coarse_prediction].union({fine_prediction}))

            if fine_prediction not in consistency_constraints_for_main_model:
                consistency_constraints_for_main_model[fine_prediction] = {coarse_prediction}
            else:
                consistency_constraints_for_main_model[fine_prediction] = (
                    consistency_constraints_for_main_model[fine_prediction].union({coarse_prediction}))

    # for coarse_prediction, fine_grain_inconsistencies in consistency_constraints_for_main_model.items():
    #     assert len(set(data_preprocessing.coarse_to_fine[coarse_prediction]).
    #                intersection(fine_grain_inconsistencies)) == 0

    # print([f'{k}: {len(v)}' for k, v in consistency_constraints_for_main_model.items()])

    return (main_fine_data, main_coarse_data, secondary_fine_data, secondary_coarse_data,
            consistency_constraints_for_main_model)


def get_conditions_data(main_fine_data: np.array,
                        main_coarse_data: np.array,
                        secondary_fine_data: np.array) -> dict[str, dict[str, np.array]]:
    condition_datas = {}

    for main_or_secondary in ['main',
                              'secondary'
                              ]:
        take_conditions_from = main_fine_data if main_or_secondary == 'main' else secondary_fine_data

        for granularity in data_preprocessing.granularities:
            if main_or_secondary not in condition_datas:
                condition_datas[main_or_secondary] = {}

            cla_data = main_fine_data if granularity == 'fine' else main_coarse_data
            condition_datas[main_or_secondary][granularity] = rearrange_for_condition_values(cla_data)

        derived_coarse = np.array([data_preprocessing.fine_to_course_idx[fine_grain_prediction]
                                   for fine_grain_prediction in take_conditions_from])

        condition_datas[main_or_secondary]['fine_to_coarse'] = rearrange_for_condition_values(derived_coarse)

    return condition_datas


def run_EDCR_for_granularity(main_lr,
                             main_granularity: str,
                             main_fine_data: np.array,
                             main_coarse_data: np.array,
                             condition_datas: dict[str, dict[str, np.array]],
                             conditions_from_secondary: bool,
                             conditions_from_main: bool,
                             consistency_constraints: bool,
                             multiprocessing: bool,
                             consistency_constraints_for_main_model: dict[str, set]) -> np.array:
    with ((context_handlers.TimeWrapper())):
        if main_granularity == 'fine':
            classes = data_preprocessing.fine_grain_classes
            true_data = data_preprocessing.true_fine_data
            pred_data = main_fine_data
        else:
            classes = data_preprocessing.coarse_grain_classes
            true_data = data_preprocessing.true_coarse_data
            pred_data = main_coarse_data

        examples_num = true_data.shape[0]

        charts = [[pred_data[example_index], true_data[example_index]] +
                  ((
                           get_unary_condition_values(example_index=example_index,
                                                      cla_datas=condition_datas['main']['fine'])
                           +
                           get_unary_condition_values(example_index=example_index,
                                                      cla_datas=condition_datas['main']['coarse'])
                           +
                           (get_binary_condition_values(example_index=example_index,
                                                        fine_cla_datas=condition_datas['main']['fine'],
                                                        coarse_cla_datas=condition_datas['main']['coarse'])
                            if consistency_constraints else [])
                       # +
                       # get_unary_condition_values(example_index=example_index,
                       #                            cla_datas=condition_datas['main']['fine_to_coarse'])
                   ) if conditions_from_main else [])
                  +
                  (
                      (
                              get_unary_condition_values(example_index=example_index,
                                                         cla_datas=condition_datas['secondary']['fine'])
                              +
                              get_unary_condition_values(example_index=example_index,
                                                         cla_datas=condition_datas['secondary']['coarse'])
                          # +
                          # (get_binary_condition_values(example_index=example_index,
                          #                              fine_cla_datas=condition_datas['secondary']['fine'],
                          #                              coarse_cla_datas=condition_datas['secondary']['coarse'])
                          #  if consistency_constraints else [])
                          # +
                          # get_unary_condition_values(example_index=example_index,
                          #                            cla_datas=condition_datas['secondary']['fine_to_coarse'])
                      )
                      if conditions_from_secondary else [])
                  for example_index in range(examples_num)]

        all_charts = generate_chart(n_classes=len(classes),
                                    charts=charts)

        results = []
        result0 = [0]

        print(f'Started EDCR pipeline for the {main_granularity}-grain main classes...'
              # f', secondary: {secondary_model_name}, lr: {secondary_lr}\n'
              )

        for count, chart in enumerate(all_charts):
            chart = np.array(chart)
            result0.extend(get_scores(chart[:, 1], chart[:, 0]))
            result0.extend([0, 0, 0, 0])

        result0.extend(get_scores(true_data, pred_data))
        results.append(result0)

        # posterior_acc = 0
        total_results = np.zeros_like(pred_data)

        epsilons = [0.002 * i for i in range(1, 2, 1)]

        for epsilon in epsilons:
            result, posterior_acc, total_results, error_detections_mean = ruleForNPCorrectionMP(
                all_charts=all_charts,
                true_data=true_data,
                pred_data=pred_data,
                main_granularity=main_granularity,
                epsilon=epsilon,
                consistency_constraints_for_main_model=consistency_constraints_for_main_model
            ) if multiprocessing else ruleForNPCorrection(all_charts=all_charts,
                                                          true_data=true_data,
                                                          pred_data=pred_data,
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
        #      secondary_model_name=secondary_model_name,
        #      secondary_lr=secondary_lr,
        #      folder=folder)

        np.save(f'{folder}/results.npy', total_results)

        # Save the DataFrame to an Excel file
        df.to_excel(f'{folder}/results.xlsx')

        print(f'\nCompleted {main_granularity}-grain EDCR run'
              # f'saved error detections and corrections to {folder}\n'
              )

    return total_results, error_detections_mean


def run_EDCR_pipeline(main_lr,
                      combined: bool,
                      loss: str,
                      conditions_from_secondary: bool,
                      conditions_from_main: bool,
                      consistency_constraints: bool,
                      multiprocessing: bool = True):
    (main_fine_data, main_coarse_data, secondary_fine_data, secondary_coarse_data,
     consistency_constraints_for_main_model) = load_priors(main_lr=main_lr,
                                                           loss=loss,
                                                           combined=combined)
    condition_datas = get_conditions_data(main_fine_data=main_fine_data,
                                          main_coarse_data=main_coarse_data,
                                          secondary_fine_data=secondary_fine_data)
    pipeline_results = {}
    error_detections = []

    for main_granularity in data_preprocessing.granularities:
        res = (
            run_EDCR_for_granularity(main_lr=main_lr,
                                     main_granularity=main_granularity,
                                     main_fine_data=main_fine_data,
                                     main_coarse_data=main_coarse_data,
                                     condition_datas=condition_datas,
                                     conditions_from_secondary=conditions_from_secondary,
                                     conditions_from_main=conditions_from_main,
                                     consistency_constraints=consistency_constraints,
                                     multiprocessing=multiprocessing,
                                     consistency_constraints_for_main_model=consistency_constraints_for_main_model))
        pipeline_results[main_granularity] = res[0]
        if multiprocessing:
            error_detections += [res[1]]

    if multiprocessing:
        error_detections = np.mean(np.array(error_detections))
        print(utils.green_text(f'Mean error detections found {np.mean(error_detections)}'))

    vit_pipeline.get_and_print_metrics(fine_predictions=pipeline_results['fine'],
                                       coarse_predictions=pipeline_results['coarse'],
                                       loss=loss,
                                       prior=False,
                                       combined=combined,
                                       model_name=main_model_name,
                                       lr=main_lr)


if __name__ == '__main__':
    combined = False
    conditions_from_main = True
    print(utils.red_text(f'\nconditions_from_secondary={not conditions_from_main}, '
                         f'conditions_from_main={conditions_from_main}\n' +
                         f'combined={combined}\n' + '#' * 100 + '\n'))

    run_EDCR_pipeline(main_lr=0.0001,
                      combined=combined,
                      loss='soft_marginal',
                      conditions_from_secondary=not conditions_from_main,
                      conditions_from_main=conditions_from_main,
                      consistency_constraints=True,
                      multiprocessing=False)
