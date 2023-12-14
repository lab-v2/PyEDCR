import os
# import json
import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import multiprocessing as mp

import warnings

warnings.filterwarnings('ignore')

import vit_pipeline
import utils
import data_preprocessing
import context_handlers

figs_folder = 'figs/'
results_file = "rule_for_NPcorrection.csv"

main_model_name = 'vit_b_16'
main_lr = 0.0001

# secondary_model_name = 'vit_l_16'
# secondary_lr = 0.0001


def get_binary_condition_values(i: int,
                                fine_cla_datas: np.array,
                                coarse_cla_datas: np.array):
    res = []
    for fine_i, fine_cls in enumerate(fine_cla_datas[:, i].astype(int)):
        for coarse_i, coarse_cls in enumerate(coarse_cla_datas[:, i].astype(int)):
            pred = int(fine_cls & coarse_cls)
            consistent = int(pred & (data_preprocessing.fine_to_course_idx[fine_i] == coarse_i))
            res += [pred, consistent]

    return res


def get_unary_condition_values(i: int,
                               cla_datas: np.array):
    return [int(cls[i]) for cls in cla_datas]


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
                               run_positive_rules: bool,
                               total_results: list,
                               shared_index: mp.Value):
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
                          epsilon: float,
                          error_detections: dict,
                          corrections: dict,
                          run_positive_rules: bool = True):
    manager = mp.Manager()
    shared_results = manager.list(pred_data)
    shared_index = manager.Value('i', 0)

    # Create argument tuples for each process
    args_list = [(i, chart, epsilon, all_charts, run_positive_rules, shared_results, shared_index)
                 for i, chart in enumerate(all_charts)]

    # Create a pool of processes and map the function with arguments
    processes_num = mp.cpu_count()

    with mp.Pool(processes_num) as pool:
        print(f'Num of processes: {processes_num}')
        results = pool.starmap(ruleForNPCorrection_worker, args_list)

    # Assuming shared_results is a ListProxy object obtained from multiprocessing manager
    shared_results = np.array(list(shared_results))  # Convert ListProxy to a regular list

    results = [item for sublist in results for item in sublist]

    results.extend(get_scores(y_true=true_data,
                              y_pred=shared_results))
    posterior_acc = accuracy_score(y_true=true_data,
                                   y_pred=shared_results)

    return results, posterior_acc, shared_results, error_detections, corrections


def ruleForNPCorrection(all_charts: list,
                        true_data,
                        pred_data,
                        epsilon: float,
                        error_detections: dict,
                        corrections: dict,
                        run_positive_rules: bool = True):
    results = []
    total_results = np.copy(pred_data)
    print(len(all_charts))

    for i, chart in enumerate(all_charts):
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

    return results, posterior_acc, total_results, error_detections, corrections


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


def retrieve_error_detection_rule(best_coarse_main_model,
                                  main_model_name,
                                  main_lr,
                                  best_coarse_main_lr,
                                  secondary_model_name,
                                  best_coarse_secondary_model,
                                  secondary_lr,
                                  best_coarse_secondary_lr,
                                  error_detections):
    if (best_coarse_main_model == main_model_name and main_lr == best_coarse_main_lr
            and secondary_model_name == best_coarse_secondary_model and secondary_lr == best_coarse_secondary_lr):

        for coarse_grain_label, coarse_grain_label_data in error_detections.items():
            for fine_grain_label in coarse_grain_label_data.keys():
                print('Corrections: '
                      f'error <- predicted_coarse_grain = {coarse_grain_label} '
                      f'and predicted_fine_grain = {fine_grain_label}')


def rearrange_for_condition_values(arr: np.array) -> np.array:
    return np.eye(np.max(arr) + 1)[arr].T


def run_EDCR():
    main_model_fine_path = f'{main_model_name}_test_fine_pred_lr{main_lr}_e{vit_pipeline.num_epochs - 1}.npy'
    main_model_coarse_path = f'{main_model_name}_test_coarse_pred_lr{main_lr}_e{vit_pipeline.num_epochs - 1}.npy'

    # secondary_model_fine_path = (f'{secondary_model_name}_test_fine_pred_lr{secondary_lr}'
    #                              f'_e{vit_pipeline.num_epochs - 1}.npy')
    # secondary_model_coarse_path = (f'{secondary_model_name}_test_coarse_pred_lr{secondary_lr}'
    #                                f'_e{vit_pipeline.num_epochs - 1}.npy')

    main_fine_data = np.load(os.path.join(vit_pipeline.combined_results_path, main_model_fine_path))
    main_coarse_data = np.load(os.path.join(vit_pipeline.combined_results_path, main_model_coarse_path))

    # secondary_fine_data = np.load(os.path.join(vit_pipeline.combined_results_path, secondary_model_fine_path))
    # secondary_coarse_data = np.load(os.path.join(vit_pipeline.combined_results_path, secondary_model_coarse_path))

    true_fine_data = np.load(os.path.join(vit_pipeline.combined_results_path, 'test_fine_true.npy'))
    true_coarse_data = np.load(os.path.join(vit_pipeline.combined_results_path, 'test_coarse_true.npy'))

    main_prior_fine_acc = accuracy_score(y_true=true_fine_data, y_pred=main_fine_data)
    main_prior_coarse_acc = accuracy_score(y_true=true_coarse_data, y_pred=main_coarse_data)
    #
    # secondary_prior_fine_acc = accuracy_score(y_true=true_fine_data, y_pred=secondary_fine_data)
    # secondary_prior_coarse_acc = accuracy_score(y_true=true_coarse_data, y_pred=secondary_coarse_data)

    print(f'Main prior fine accuracy: {round(main_prior_fine_acc * 100, 2)}%, '
          f'main prior coarse accuracy: {round(main_prior_coarse_acc * 100, 2)}%\n'
          # f'Secondary fine accuracy: {round(secondary_prior_fine_acc * 100, 2)}%, '
          # f'secondary coarse accuracy: {round(secondary_prior_coarse_acc * 100, 2)}%\n'
          )

    condition_datas = {}

    for main_or_secondary in ['main',
                              # 'secondary'
                              ]:
        for granularity in data_preprocessing.granularities:
            if main_or_secondary not in condition_datas:
                condition_datas[main_or_secondary] = {}

            cla_data = eval(f'{main_or_secondary}_{granularity}_data')
            condition_datas[main_or_secondary][granularity] = rearrange_for_condition_values(cla_data)

        derived_coarse = np.array([data_preprocessing.fine_to_course_idx[fine_grain_prediction]
                                   for fine_grain_prediction in eval(f'{main_or_secondary}_fine_data')])

        condition_datas[main_or_secondary]['fine_to_coarse'] = rearrange_for_condition_values(derived_coarse)

    for main_granularity in data_preprocessing.granularities:
        classes = data_preprocessing.get_classes(main_granularity)

        true_data = eval(f'true_{main_granularity}_data')
        pred_data = eval(f'main_{main_granularity}_data')

        m = true_data.shape[0]

        charts = [[pred_data[i], true_data[i]] +
                  (
                      get_binary_condition_values(i=i,
                                                  fine_cla_datas=condition_datas['main']['fine'],
                                                  coarse_cla_datas=condition_datas['main']['coarse'])
                      +
                      get_unary_condition_values(i=i,
                                                 cla_datas=condition_datas['main']['fine'])
                      +
                      get_unary_condition_values(i=i,
                                                 cla_datas=condition_datas['main']['coarse'])
                      +
                      get_unary_condition_values(i=i,
                                                 cla_datas=condition_datas['main']['fine_to_coarse'])
                  )
                  # + (get_binary_condition_values(i=i,
                  #                              fine_cla_datas=condition_datas['secondary']['fine'],
                  #                              coarse_cla_datas=condition_datas['secondary']['coarse']) +
                  #  get_unary_condition_values(i=i,
                  #                             cla_datas=condition_datas['secondary']['fine']) +
                  #  get_unary_condition_values(i=i,
                  #                             cla_datas=condition_datas['secondary']['coarse']) +
                  #  get_unary_condition_values(i=i,
                  #                             cla_datas=condition_datas['secondary']['fine_to_coarse']))
                  for i in range(m)]

        all_charts = generate_chart(n_classes=len(classes),
                                    charts=charts)

        results = []
        result0 = [0]

        print(f'Started EDCR pipeline for main: {main_granularity}-grain  {main_model_name}, lr: {main_lr}, '
              # f'secondary: {secondary_model_name}, lr: {secondary_lr}\n'
              )

        for count, chart in enumerate(all_charts):
            chart = np.array(chart)
            result0.extend(get_scores(chart[:, 1], chart[:, 0]))
            result0.extend([0, 0, 0, 0])

        result0.extend(get_scores(true_data, pred_data))
        results.append(result0)

        posterior_acc = 0
        total_results = np.zeros_like(pred_data)

        epsilons = [0.002 * i for i in range(1, 2, 1)]

        error_detections = {}
        corrections = {}
        for e_num, epsilon in enumerate(epsilons):
            result, posterior_acc, total_results, error_detections, corrections = ruleForNPCorrectionMP(
                all_charts=all_charts,
                true_data=true_data,
                pred_data=pred_data,
                error_detections=error_detections,
                corrections=corrections,
                epsilon=epsilon)
            results.append([epsilon] + result)

        prior_acc = eval(f'main_prior_{main_granularity}_acc')
        print(f'\nSaved plots for main: {main_granularity}-grain {main_model_name}, lr={main_lr}'
              # f', secondary: {secondary_model_name}, lr={secondary_lr}'
              f'\nPrior acc:{prior_acc}, post acc: {posterior_acc}')

        col = ['pre', 'recall', 'F1', 'NSC', 'PSC', 'NRC', 'PRC']
        df = pd.DataFrame(results, columns=['epsilon'] + col * len(classes) + ['acc', 'macro-F1', 'micro-F1'])

        df.to_csv(results_file)
        df = pd.read_csv(results_file)

        folder = (f'{figs_folder}/main_{main_granularity}_{main_model_name}_lr{main_lr}'
                  # f'_secondary_{secondary_model_name}_lr{secondary_lr}'
                  )
        utils.create_directory(folder)

        plot(df=df,
             classes=classes,
             col_num=len(col),
             x_values=df['epsilon'][1:],
             main_granularity=main_granularity,
             main_model_name=main_model_name,
             main_lr=main_lr,
             # secondary_model_name=secondary_model_name,
             # secondary_lr=secondary_lr,
             folder=folder)

        np.save(f'{folder}/results.npy', total_results)

        # Save the DataFrame to an Excel file
        df.to_excel(f'{folder}/results.xlsx')

        # with open(f'{folder}/error_detections.json', 'w') as json_file:
        #     json.dump(error_detections, json_file)
        #
        # with open(f'{folder}/corrections.json', 'w') as json_file:
        #     json.dump(corrections, json_file)

        print(f'\nsaved error detections and corrections to {folder}\n')


if __name__ == '__main__':
    run_EDCR()
