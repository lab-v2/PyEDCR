import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from plotting import plot

base_path0 = 'LRCN_F1_no_overlap_sequential/'
base_path1 = 'no_overlap_sequential_10/'
results_file = base_path0 + "rule_for_NPcorrection.csv"
epsilons = [0.001 * i for i in range(1, 100, 1)]

true_data = np.load(base_path0 + "test_true.npy", allow_pickle=True)
pred_data = np.load(base_path0 + "test_pred.npy", allow_pickle=True)

cla4_data = np.load(base_path1 + "test_out_cla4.npy", allow_pickle=True)
cla3_data = np.load(base_path1 + "test_out_cla3.npy", allow_pickle=True)
cla2_data = np.load(base_path1 + "test_out_cla2.npy", allow_pickle=True)
cla1_data = np.load(base_path1 + "test_out_cla1.npy", allow_pickle=True)
cla0_data = np.load(base_path1 + "test_out_cla0.npy", allow_pickle=True)

labels = set(true_data.flatten())
len_labels = len(labels)
n_classes = len_labels


def rules1(i: int):
    rule_scores = []

    for cls in cla_datas:
        rule_scores += [int(cls[i]), 1 - int(cls[i])]

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
    print(f"class{count}, cci:{cci}, new_pre:{new_pre}, pre:{pi}")
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
    print(f"class{count}, quantity:{quantity}")

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
    print(f"class:{count}, NCi:{NCi}")
    return NCi


def ruleForNPCorrection(all_charts, epsilon):
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

        CCi = DetUSMPosRuleSelect(count, all_charts)
        tem_cond = np.zeros_like(chart[:, 0])
        rec_true = []
        rec_pred = []

        for cc in CCi:
            tem_cond |= chart[:, cc]
        if np.sum(tem_cond) > 0:
            for ct, cv in enumerate(chart):
                if tem_cond[ct]:
                    if not predict_result[ct]:
                        pos_i_count += 1
                        predict_result[ct] = 1
                        total_results[ct] = count
                else:
                    rec_true.append(cv[1])
                    rec_pred.append(cv[0])

        scores_cor = get_scores(chart[:, 1], predict_result)
        results.extend(scores_cor + [neg_i_count, pos_i_count, len(NCi), len(CCi)])
    results.extend(get_scores(true_data, total_results))
    return results


if __name__ == '__main__':
    charts = []
    cla_datas = [cla0_data, cla1_data, cla2_data, cla3_data, cla4_data]  # neural network binary result

    high_scores = [0.8]
    low_scores = [0.2]

    m = true_data.shape[0]
    for i in range(m):
        charts.append([pred_data[i], true_data[i]] + rules1(i))

    all_charts = generate_chart(charts)

    results = []
    result0 = [0]
    for count, chart in enumerate(all_charts):
        chart = np.array(chart)
        result0.extend(get_scores(chart[:, 1], chart[:, 0]))
        result0.extend([0, 0, 0, 0])
    result0.extend(get_scores(true_data, pred_data))
    results.append(result0)

    for ep in epsilons:
        result = ruleForNPCorrection(all_charts, ep)
        results.append([ep] + result)
        print(f"ep:{ep}\n{result}")
    col = ['pre', 'recall', 'F1', 'NSC', 'PSC', 'NRC', 'PRC']
    df = pd.DataFrame(results, columns=['epsilon'] + col * n_classes + ['acc', 'macro-F1', 'micro-F1'])

    df.to_csv(results_file)

    df = pd.read_csv(results_file)
    plot(df=df, n=n_classes, epsilons=epsilons)