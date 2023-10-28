import re
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import multiprocessing as mp
import itertools

import warnings
warnings.filterwarnings('ignore')

from vit_pipeline import vit_model_names, num_epochs, lrs
from utils import create_directory

data_dir = 'results'
true_data = np.load(os.path.join(data_dir, 'test_true_coarse.npy'))
results_folder = '.'
results_file = results_folder + "rule_for_NPcorrection.csv"

data_file_path = rf'data/WEO_Data_Sheet.xlsx'
dataframes_by_sheet = pd.read_excel(data_file_path, sheet_name=None)
fine_grain_results_df = dataframes_by_sheet['Fine-Grain Results']
fine_grain_classes = fine_grain_results_df['Class Name'].to_list()
coarse_grain_results_df = dataframes_by_sheet['Coarse-Grain Results']
coarse_grain_classes = coarse_grain_results_df['Class Name'].to_list()
n_classes = len(coarse_grain_classes)
figs_folder = 'figs/'


def rules1(i: int,
           cla_datas: np.array):
    rule_scores = []

    for cls in cla_datas:
        cls_i = int(cls[i])
        rule_scores += [cls_i, 1 - cls_i]

    return rule_scores


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


def generate_chart(charts: list) -> list:
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
    count = i
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
    count = i
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
    # print(f"class:{count}, NCi:{NCi}")
    return NCi


def ruleForNPCorrection(all_charts: list,
                        true_data,
                        pred_data,
                        epsilon: float,
                        run_positives: bool = True, ):
    results = []
    total_results = np.copy(pred_data)

    for count, chart in enumerate(all_charts):
        chart = np.array(chart)
        NCi = GreedyNegRuleSelect(i=count,
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

        CCi = DetUSMPosRuleSelect(i=count,
                                  all_charts=all_charts) if run_positives else []
        tem_cond = np.zeros_like(chart[:, 0])

        for cc in CCi:
            tem_cond |= chart[:, cc]

        if np.sum(tem_cond) > 0:
            for ct, cv in enumerate(chart):
                if tem_cond[ct]:
                    if not predict_result[ct]:
                        pos_i_count += 1
                        predict_result[ct] = 1
                        total_results[ct] = count

        scores_cor = get_scores(chart[:, 1], predict_result)
        results.extend(scores_cor + [neg_i_count,
                                     pos_i_count,
                                     len(NCi),
                                     len(CCi)])
    results.extend(get_scores(true_data, total_results))
    posterior_acc = accuracy_score(true_data, total_results)

    return results, posterior_acc, total_results


def plot(df: pd.DataFrame,
         n_classes: int,
         col_num: int,
         x_values: pd.Series,
         main_model_name: str,
         secondary_model_name: str,
         main_lr: float,
         secondary_lr: float,
         folder: str):
    for i in range(n_classes):
        df_i = df.iloc[1:, 2 + i * col_num:2 + (i + 1) * col_num]

        added_str = f'.{i}' if i else ''
        pre_i = df_i[f'pre{added_str}']
        rec_i = df_i[f'recall{added_str}']
        f1_i = df_i[f'F1{added_str}']

        plt.plot(x_values,
                 pre_i,
                 label='pre')
        plt.plot(x_values,
                 rec_i,
                 label='rec')
        plt.plot(x_values,
                 f1_i,
                 label='f1')

        plt.title(f'Class #{i}-{coarse_grain_classes[i]}, Main: {main_model_name}, lr: {main_lr}'
                  f'Secondary: {secondary_model_name}, lr: {secondary_lr}')
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.savefig(f'{folder}/cls{i}.png')
        plt.clf()
        plt.cla()


def run_EDCR(granularity: str,
             main_model_name: str,
             main_lr: float,
             secondary_model_name: str,
             secondary_lr: float,
             true_data: np.array,
             pred_data: np.array,
             prior_acc: float):
    suffix = '_coarse' if granularity == 'coarse' else ''
    filename = f"{data_dir}/{secondary_model_name}_test_pred_lr{secondary_lr}_e{num_epochs - 1}{suffix}.npy"

    try:
        cla_datas = np.load(filename)
    except FileNotFoundError:
        print(f'{filename} not found')
        return

    cla_datas = np.eye(np.max(cla_datas) + 1)[cla_datas].T
    epsilons = [0.003 * i for i in range(1, 100, 1)]

    m = true_data.shape[0]
    charts = [[pred_data[i], true_data[i]] + rules1(i=i, cla_datas=cla_datas) for i in range(m)]
    all_charts = generate_chart(charts)

    results = []
    result0 = [0]

    print(f'Started EDCR pipeline for {granularity}-grain with main {main_model_name}, lr: {main_lr}, '
          f'secondary: {secondary_model_name}, lr: {secondary_lr}\n')

    for count, chart in enumerate(all_charts):
        chart = np.array(chart)
        result0.extend(get_scores(chart[:, 1], chart[:, 0]))
        result0.extend([0, 0, 0, 0])

    result0.extend(get_scores(true_data, pred_data))
    results.append(result0)

    posterior_acc = 0
    total_results = np.zeros_like(pred_data)

    for epsilon in tqdm(epsilons, total=len(epsilons)):
        result, posterior_acc, total_results = ruleForNPCorrection(all_charts=all_charts,
                                                                   true_data=true_data,
                                                                   pred_data=pred_data,
                                                                   epsilon=epsilon)
        results.append([epsilon] + result)

    col = ['pre', 'recall', 'F1', 'NSC', 'PSC', 'NRC', 'PRC']
    df = pd.DataFrame(results, columns=['epsilon'] + col * n_classes + ['acc', 'macro-F1', 'micro-F1'])

    df.to_csv(results_file)
    df = pd.read_csv(results_file)

    folder = (f'{figs_folder}/main_{main_model_name}_lr{main_lr}'
              f'_secondary_{secondary_model_name}_lr{secondary_lr}_{granularity}')
    create_directory(folder)

    plot(df=df,
         n_classes=n_classes,
         col_num=len(col),
         x_values=df['epsilon'][1:],
         main_model_name=main_model_name,
         secondary_model_name=secondary_model_name,
         main_lr=main_lr,
         secondary_lr=secondary_lr,
         folder=folder)

    np.save(f'{folder}/results{suffix}.npy', total_results)

    print(f'Saved plots for {granularity}-grain with main: {main_model_name}, secondary: {secondary_model_name}\n'
          f'Prior acc:{prior_acc}, post acc: {posterior_acc}\n')



def handle_file(granularity: str,
                filename: str,
                data_dir: str,
                true_data: np.array):
    match = re.match(pattern=rf'(.+?)_test_pred_lr(.+?)_e{num_epochs - 1}_coarse.npy',
                     string=filename)

    if match:
        main_model_name = match.group(1)
        main_lr = match.group(2)
        pred_data = np.load(os.path.join(data_dir, filename))
        prior_acc = accuracy_score(true_data, pred_data)

        names_lrs = itertools.product([name for name in vit_model_names
                                       if name != main_model_name and name != 'vit_h_14'], lrs)
        iterable = [(granularity, main_model_name, main_lr, secondary_model_name, secondary_lr,
                     true_data, pred_data, prior_acc) for secondary_model_name, secondary_lr in names_lrs]

        with mp.Pool(processes=mp.cpu_count()) as pool:
            pool.starmap(func=run_EDCR,
                         iterable=iterable)


if __name__ == '__main__':
    granularity = 'fine'
    for filename in os.listdir(data_dir):
        handle_file(granularity=granularity,
                    filename=filename,
                    data_dir=data_dir,
                    true_data=true_data)
