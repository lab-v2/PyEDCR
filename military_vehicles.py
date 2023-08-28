# %% Imports

# import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

# %% Data load

data_file_path = rf'data/WEO_Data_Sheet.xlsx'
dataframes_by_sheet = pd.read_excel(data_file_path, sheet_name=None)

fine_grain_results_df = dataframes_by_sheet['Fine-Grain Results']
fine_grain_classes = fine_grain_results_df['Class Name'].values

coarse_grain_results_df = dataframes_by_sheet['Coarse-Grain Results']
coarse_grain_classes = coarse_grain_results_df['Class Name'].values

zeros_and_ones_df = dataframes_by_sheet['1s_0s_Sheet']
image_names = zeros_and_ones_df['Image Name'].values
confidences = [i * 0.01 for i in range(1, 100, 10)]


def get_example_info(image_name: str) -> pd.Series:
    """
    :param image_name: The image name of the example to consider
    :return: A row of all the info about the example
    """

    return zeros_and_ones_df[zeros_and_ones_df['Image Name'] == image_name]


def get_example_fine_grain_one_hot_classes(image_name: str):
    """
    :param image_name: The image name of the example to consider
    :return: One-hot prediction vectors for all the classes on the image
    """

    return get_example_info(image_name)[fine_grain_classes].values


def get_class_name(cls: str,
                   ground_truth: bool) -> str:
    """
    :param cls: The image name of the example to consider
    :param ground_truth: Whether to get g_t data or not
    :return: A string of the class name
    """

    if ground_truth and cls == 'Air Defense':
        return 'Air Defence'

    return cls if cls != 'Self Propelled Artillery' else 'SPA'


def get_class_index(image_name: str,
                    ground_truth: bool,
                    granularity: str = 'fine') -> int:
    """
    :param image_name: The image name of the example to consider
    :param ground_truth: Whether to get ground truth data or not
    :param granularity: Fine or course label
    :return: A string of the class name
    """

    w_info = get_example_info(image_name)
    column_name_generator = lambda cls: get_class_name(cls, ground_truth) if ground_truth else (
        f"pred_{get_class_name(cls, ground_truth)}" if granularity == 'fine'
        else f"Exp 2 Prediction ({get_class_name(cls, ground_truth)})")
    classes = fine_grain_classes if granularity == 'fine' else coarse_grain_classes
    class_index = np.array([w_info[column_name_generator(cls)] for cls in classes]).argmax()

    return class_index


def get_fine_grain_predicted_index(image_name: str) -> int:
    """
    :param image_name: The image name of the example to consider
    :return: The fine grain predicted index
    """

    return get_class_index(image_name=image_name, ground_truth=False)


def get_fine_grain_true_index(image_name: str) -> int:
    """
    :param image_name: The image name of the example to consider
    :return: The fine grain ground truth index
    """

    return get_class_index(image_name=image_name, ground_truth=True)


def get_class(image_name: str,
              ground_truth: bool,
              granularity: str = 'fine') -> str:
    """
    :param image_name: The image name of the example to consider
    :param ground_truth: Whether to get ground truth data or not
    :param granularity: Fine or course label
    :return: A row of all the info about the example
    """

    class_index = get_class_index(image_name=image_name, ground_truth=ground_truth, granularity=granularity)
    classes = fine_grain_classes if granularity == 'fine' else coarse_grain_classes
    resulted_class = classes[class_index]

    return resulted_class


# %% Bowen's code

def generate_chart(charts: list[list[int]]) -> list[list[int]]:
    """
    :param initial data structure before processing:
    :return: fully encoded data structure for the algorithm (list of list of ints)
    appends true positive and false positive encodings into the data structure
    """
    all_charts = [[] for _ in range(len(fine_grain_classes))]
    for data in charts:
        for count, jj in enumerate(all_charts):
            # pred, corr, tp, fp, cond1, cond2 ... condn
            each_items = []
            for d in data[:2]:
                if d == count:
                    each_items.append(1)
                else:
                    each_items.append(0)

            # tp
            if each_items[0] == 1 and each_items[1] == 1:
                each_items.append(1)
            else:
                each_items.append(0)

            # fp
            if each_items[0] == 1 and each_items[1] == 0:
                each_items.append(1)
            else:
                each_items.append(0)
            each_items.extend(data[2:])
            jj.append(each_items)
    return all_charts


charts = []
number_of_samples = zeros_and_ones_df.shape[0]

high_scores = [0.55, 0.60]  # >0.95 is one rule, >0.98 is another rule, in total 4*24
low_scores = [0.05, 0.02]

def rules1(image_name: str) -> list[int]:
    """
    :param image_name:
    :return: encoded rules for the algorithm
    create an encoded version of the rules
    """
    rule_scores = []
    one_hot = get_example_fine_grain_one_hot_classes(image_name).ravel()

    for class_prediction in one_hot:
        for score in high_scores:
            if class_prediction > score:
                rule_scores.append(1)
            else:
                rule_scores.append(0)
        for score in low_scores:
            if class_prediction < score:
                rule_scores.append(1)
            else:
                rule_scores.append(0)
    return rule_scores

pred_data = [get_fine_grain_predicted_index(image_name) for image_name in image_names]
true_data = [get_fine_grain_true_index(image_name) for image_name in image_names]

for i in range(number_of_samples):
    """
    
    """
    image_name = image_names[i]

    tmp_charts = []
    tmp_charts.extend([get_fine_grain_predicted_index(image_name), get_fine_grain_true_index(image_name)])
    tmp_charts += rules1(image_name)
    charts.append(tmp_charts)

all_charts = generate_chart(charts)


# %%

def GreedyNegRuleSelect(i: int, epsilon: float, all_charts: list[list[int]]):
    count = i
    chart = all_charts[i]
    chart = np.array(chart)
    rule_indexs = [i for i in range(4, len(chart[0]))]
    each_sum = np.sum(chart, axis=0)
    tpi = each_sum[2]
    fpi = each_sum[3]
    pi = tpi * 1.0 / (tpi + fpi)
    NCi = []

    if each_sum[1] and tpi:
        ri = tpi * 1.0 / each_sum[1]
        ni = each_sum[0]
        quantity = epsilon * ni * pi / ri
        print(f"class{count}, quantity:{quantity}")

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
        print(f"class:{count}, NCi:{NCi}")

    return NCi


def GreedyNegRules(all_charts):
    epsilon = 0.01

    NCis = []
    for i, chart in enumerate(all_charts):
        NCi = GreedyNegRuleSelect(i, epsilon, all_charts)
        NCis += [NCi]

    return NCis


greedy_rules = GreedyNegRules(all_charts)

print(greedy_rules)


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
        ciij = 0
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
    print(f"class{count}, cci:{cci}, new_pre:{new_pre}, pre:{pi}")

    return cci


def DetUSMPosRules(all_charts):
    CCis = []

    for count, chart in enumerate(all_charts):
        CCi = DetUSMPosRuleSelect(count, all_charts)
        CCis += [CCi]

    return CCis


positive_rules = DetUSMPosRules(all_charts)
print("Positive rules:")
print(positive_rules)


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
        print(f"TP:{TP}, FP:{FP}, TN:{TN}, FN:{FN}")

        pre = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return [pre, rec, f1]
    except:
        pre = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        f1micro = f1_score(y_true, y_pred, average='micro')
        return [pre, f1, f1micro]


def ruleForNegativeCorrection(all_charts, epsilon):  # how to use
    results = []
    total_results = np.copy(pred_data)
    for count, chart in enumerate(all_charts):
        chart = np.array(chart)
        NCi = GreedyNegRuleSelect(count, epsilon, all_charts)
        negi_count = 0
        posi_count = 0

        predict_result = np.copy(chart[:, 0])
        tem_cond = 0
        for cc in NCi:
            tem_cond |= chart[:, cc]
        if np.sum(tem_cond) > 0:
            for ct, cv in enumerate(chart):
                if tem_cond[ct] and predict_result[ct]:
                    negi_count += 1
                    predict_result[ct] = 0

        CCi = []
        scores_cor = get_scores(chart[:, 1], predict_result)
        results.extend(scores_cor + [negi_count, posi_count, len(NCi), len(CCi)])
    results.extend(get_scores(true_data, total_results))
    return results


def ruleForNPCorrection(all_charts, epsilon):
    results = []
    total_results = np.copy(pred_data)
    for count, chart in enumerate(all_charts):
        chart = np.array(chart)
        NCi = GreedyNegRuleSelect(count, epsilon, all_charts)
        negi_count = 0
        posi_count = 0

        predict_result = np.copy(chart[:, 0])
        tem_cond = 0
        for cc in NCi:
            tem_cond |= chart[:, cc]
        if np.sum(tem_cond) > 0:
            for ct, cv in enumerate(chart):
                if tem_cond[ct] and predict_result[ct]:
                    negi_count += 1
                    predict_result[ct] = 0

        CCi = DetUSMPosRuleSelect(count, all_charts)
        tem_cond = 0
        rec_true = []
        rec_pred = []
        for cc in CCi:
            tem_cond |= chart[:, cc]
        if np.sum(tem_cond) > 0:
            for ct, cv in enumerate(chart):
                if tem_cond[ct]:
                    if not predict_result[ct]:
                        posi_count += 1
                        predict_result[ct] = 1
                        total_results[ct] = count
                else:
                    rec_true.append(cv[1])
                    rec_pred.append(cv[0])

        scores_cor = get_scores(chart[:, 1], predict_result)
        results.extend(scores_cor + [negi_count, posi_count, len(NCi), len(CCi)])
    results.extend(get_scores(true_data, total_results))
    return results


def ruleForPNCorrection(all_charts, epsilon):
    results = []
    total_results = np.copy(pred_data)
    for count, chart in enumerate(all_charts):
        chart = np.array(chart)
        negi_count = 0
        posi_count = 0

        predict_result = np.copy(chart[:, 0])
        CCi = DetUSMPosRuleSelect(count, all_charts)
        tem_cond = 0
        for cc in CCi:
            tem_cond |= chart[:, cc]
        if np.sum(tem_cond) > 0:
            for ct, cv in enumerate(chart):
                if tem_cond[ct]:
                    if not predict_result[ct]:
                        posi_count += 1
                        predict_result[ct] = 1
                        total_results[ct] = count

        NCi = GreedyNegRuleSelect(count, epsilon, all_charts)

        tem_cond = 0
        for cc in NCi:
            tem_cond |= chart[:, cc]
        if np.sum(tem_cond) > 0:
            for ct, cv in enumerate(chart):
                if tem_cond[ct] and predict_result[ct]:
                    negi_count += 1
                    predict_result[ct] = 0

        scores_cor = get_scores(chart[:, 1], predict_result)
        results.extend(scores_cor + [negi_count, posi_count, len(NCi), len(CCi)])
    results.extend(get_scores(true_data, total_results))
    return results


def PosNegRuleLearn(all_charts):
    epsilon = 0.01
    # pi = [[] for _ in range(6)]
    # CCall = [[] for _ in range(6)]
    pi = []
    CCall = []
    CCall_set = []
    for count, chart in enumerate(all_charts):
        chart = np.array(chart)
        NCi = GreedyNegRuleSelect(count, epsilon, all_charts)

        tem_cond = 0
        for cc in NCi:
            tem_cond |= chart[:, cc]
        if np.sum(tem_cond) > 0:
            for ct, cv in enumerate(chart):
                if tem_cond[ct] and cv[0]:
                    pi.append(ct)

        CCall.extend(NCi)
        CCall_set.extend(NCi)
    CCall_set = list(set(CCall_set))
    print(f"size of Neg PI:{len(pi)}")
    print(f"CCall:{CCall_set}")

    for count, chart in enumerate(all_charts):
        chart = np.array(chart)
        tmp_CCi = DetUSMPosRuleSelect(count, all_charts)
        CCi = []
        for i in tmp_CCi:
            if i in CCall_set:
                CCi.append(i)
        tem_cond = 0
        for cc in CCi:
            tem_cond |= chart[:, cc]
        if np.sum(tem_cond) > 0:
            for ct, cv in enumerate(chart):
                if tem_cond[ct] and not cv[0]:
                    pi.append(ct)
    print(f"size of Neg + pos PI:{len(pi)}")
    return pi


results = []
result0 = [0]
for count, chart in enumerate(all_charts):
    chart = np.array(chart)
    result0.extend(get_scores(chart[:, 1], chart[:, 0]))
    result0.extend([0, 0, 0, 0])
result0.extend(get_scores(true_data, pred_data))
results.append(result0)
epsilon = [0.001 * i for i in range(0, 101, 1)]
print(epsilon)

for ep in epsilon:
    # result = PosNegRuleLearn(all_charts, epsilon)
    result = ruleForNegativeCorrection(all_charts, ep)
    results.append([ep] + result)
    print(f"ep:{ep}\n{result}")
col = ['pre', 'recall', 'F1', 'NSC', 'PSC', 'NRC', 'PRC']
df = pd.DataFrame(results, columns=['epsilon'] + col * len(fine_grain_classes) + ['acc', 'macro-F1', 'micro-F1'])
df.to_csv("rule_for_Negativecorrection.csv")
#
results = [result0]
for ep in epsilon:
    # result = PosNegRuleLearn(all_charts, epsilon)
    result = ruleForNPCorrection(all_charts, ep)
    results.append([ep] + result)
    print(f"ep:{ep}\n{result}")
col = ['pre', 'recall', 'F1', 'NSC', 'PSC', 'NRC', 'PRC']
df = pd.DataFrame(results, columns=['epsilon'] + col * len(fine_grain_classes) + ['acc', 'macro-F1', 'micro-F1'])
df.to_csv("rule_for_NPcorrection.csv")
#
results = [result0]
for ep in epsilon:
    # result = PosNegRuleLearn(all_charts, epsilon)
    result = ruleForPNCorrection(all_charts, ep)
    results.append([ep] + result)
    print(f"ep:{ep}\n{result}")
df = pd.DataFrame(results, columns=['epsilon'] + col * len(fine_grain_classes) + ['acc', 'macro-F1', 'micro-F1'])
df.to_csv("rule_for_PNcorrection.csv")

