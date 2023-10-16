import re
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from vision_models import vit_model_names, Plot
from scrape_train_test import create_directory
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


run_positives = True

results_folder = '.'
results_file = results_folder + "rule_for_NPcorrection.csv"

data_file_path = rf'data/WEO_Data_Sheet.xlsx'
dataframes_by_sheet = pd.read_excel(data_file_path, sheet_name=None)
fine_grain_results_df = dataframes_by_sheet['Fine-Grain Results']
fine_grain_classes = fine_grain_results_df['Class Name'].to_list()
n_classes = len(fine_grain_classes)
figs_folder = 'figs/'


def rules1(i: int):
    rule_scores = []

    for cls in cla_datas:
        cls_i = int(cls[i])
        rule_scores += [cls_i, 1-cls_i]

    return rule_scores


def get_scores(y_true, y_pred):
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


def generate_chart(charts):
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


def DetUSMPosRuleSelect(i, all_charts):
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


def GreedyNegRuleSelect(i, epsilon, all_charts):
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


def ruleForNPCorrection(all_charts, epsilon, run_positives):
    results = []
    total_results = np.copy(pred_data)

    for count, chart in enumerate(all_charts):
        chart = np.array(chart)
        NCi = GreedyNegRuleSelect(count, epsilon, all_charts)
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

        CCi = DetUSMPosRuleSelect(count, all_charts) if run_positives else []
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

    acc = accuracy_score(true_data, total_results)

    return results, acc


def plot(df: pd.DataFrame,
         n_classes: int,
         col_num: int,
         x_values: list[float],
         lr: float):
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

        plt.title(f'Class #{i}-{fine_grain_classes[i]}, Main: {main_model_name}, '
                  f'Secondary: {secondary_model_name}, LR: {lr}')
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.savefig(f'{figs_folder}/{main_model_name}->{secondary_model_name}_lr{lr}'
                    f'/cls{i}_{main_model_name}_to_{secondary_model_name}_lr{lr}.png')
        plt.clf()
        plt.cla()


if __name__ == '__main__':
    data_dir = '.'
    true_data = np.load('test_true.npy')

    for filename in os.listdir(data_dir):

        match = re.match(pattern=r'(.+?)_test_pred_lr(.+?)_e(\d+?).npy',
                         string=filename)

        if match:
            main_model_name = match.group(1)
            pred_data = np.load(filename)
            lr = match.group(2)
            epoch = match.group(3)

            if epoch == '0' and lr == '5e-05':

                for secondary_model_name in [name for name in vit_model_names.values() if f'vit_{name}' != main_model_name]:

                    try:
                        cla_datas = np.load(f"vit_{secondary_model_name}_test_pred_lr5e-05_e3.npy")
                    except FileNotFoundError:
                        continue

                    cla_datas = np.eye(np.max(cla_datas) + 1)[cla_datas].T

                    high_scores = [0.7, 0.8]
                    low_scores = [0.1, 0.2]
                    epsilons = [0.003 * i for i in range(1, 100, 1)]

                    m = true_data.shape[0]
                    charts = [[pred_data[i], true_data[i]] + rules1(i) for i in range(m)]
                    all_charts = generate_chart(charts)

                    results = []
                    result0 = [0]

                    print(f'Started EDCR pipeline for {main_model_name}->{secondary_model_name}')
                    for count, chart in enumerate(all_charts):
                        chart = np.array(chart)
                        result0.extend(get_scores(chart[:, 1], chart[:, 0]))
                        result0.extend([0, 0, 0, 0])
                    result0.extend(get_scores(true_data, pred_data))
                    results.append(result0)

                    accuracies = []

                    for ep in tqdm(epsilons, total=len(epsilons)):
                        result, acc = ruleForNPCorrection(all_charts, ep, run_positives=run_positives)
                        results.append([ep] + result)
                        accuracies += [acc]

                    with Plot():
                        plt.plot(epsilons, accuracies)
                        plt.title(f'Main: {main_model_name}, Secondary: {secondary_model_name}, '
                                  f'LR: {lr}')
                        plt.grid()
                        plt.tight_layout()
                        plt.ylabel('Accuracy')
                        plt.ylabel('Epsilon')

                    col = ['pre', 'recall', 'F1', 'NSC', 'PSC', 'NRC', 'PRC']
                    df = pd.DataFrame(results, columns=['epsilon'] + col * n_classes + ['acc', 'macro-F1', 'micro-F1'])

                    df.to_csv(results_file)
                    df = pd.read_csv(results_file)
                    create_directory(f'{figs_folder}/{main_model_name}->{secondary_model_name}_lr{lr}')

                    plot(df=df,
                         n_classes=n_classes,
                         col_num=len(col),
                         x_values=df['epsilon'][1:],
                         lr=lr)

                    print(f'Plotted {main_model_name}->{secondary_model_name}')