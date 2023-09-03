import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import matplotlib.pyplot as plt

data_file_path = rf'data/WEO_Data_Sheet.xlsx'
dataframes_by_sheet = pd.read_excel(data_file_path, sheet_name=None)

fine_grain_results_df = dataframes_by_sheet['Fine-Grain Results']
fine_grain_classes = fine_grain_results_df['Class Name'].values
n = len(fine_grain_classes)
coarse_grain_results_df = dataframes_by_sheet['Coarse-Grain Results']
coarse_grain_classes = coarse_grain_results_df['Class Name'].values

zeros_and_ones_df = dataframes_by_sheet['1s_0s_Sheet']
image_names = zeros_and_ones_df['Image Name'].values


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
    class_index = int(np.array([w_info[column_name_generator(cls)] for cls in classes]).argmax())

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
    :param charts: initial data structure before processing
    :return: fully encoded data structure for the algorithm (list of list of ints)
    appends true positive and false positive encodings into the data structure
    """
    all_charts = [[] for _ in range(n)]
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


def GreedyNegRuleSelect(i: int,
                        epsilon: float,
                        all_charts: list[list[int]]) -> list[int]:
    count = i
    chart = all_charts[i]
    chart = np.array(chart)
    rule_indexs = [i for i in range(4, len(chart[0]))]
    each_sum = np.sum(chart, axis=0)
    tpi = each_sum[2]
    fpi = each_sum[3]

    NCi = []
    positive_i = tpi + fpi


    if each_sum[1] and tpi:
        pi = tpi * 1.0 / positive_i
        ri = tpi * 1.0 / each_sum[1]
        ni = each_sum[0]
        quantity = epsilon * ni * pi / ri
        print(f"class{count}, quantity:{quantity}")

        NCn = []
        for rule in rule_indexs:
            neg_i_score = np.sum(chart[:, 2] * chart[:, rule])
            if neg_i_score < quantity:
                NCn.append(rule)

        while NCn:
            # argmax c \in NCn POS_{NCi + [c]}
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

            # add c_best to NCi
            NCi.append(best_index)

            # remove c_best to from NCn
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


def DetUSMPosRuleSelect(i: int,
                        all_charts: list[list[int]]):
    count = i
    chart = all_charts[i]
    chart = np.array(chart)
    rule_indices = [i for i in range(4, len(chart[0]))]
    each_sum = np.sum(chart, axis=0)
    tpi = each_sum[2]
    fpi = each_sum[3]
    pi = tpi * 1.0 / (tpi + fpi)

    pb_scores = []

    for ri in rule_indices:
        pos_i = np.sum(chart[:, 1] * chart[:, ri], axis=0)
        body_i = np.sum(chart[:, ri], axis=0)
        if body_i:
            score = pos_i * 1.0 / body_i

            if score > pi:
                pb_scores.append((score, ri))

    pb_scores = sorted(pb_scores)

    cci = []
    ccn = pb_scores

    for (score, ri) in pb_scores:
        cii = np.zeros_like(chart[:, ri])

        for (cs, ci) in cci:
            cii = cii | chart[:, ci]

        POScci = np.sum(cii * chart[:, 1], axis=0)
        BODcci = np.sum(cii, axis=0)
        bitwise_or = cii | chart[:, ri]
        POSccij = np.sum(bitwise_or * chart[:, 1], axis=0)
        BODccij = np.sum(bitwise_or, axis=0)

        cni = np.zeros_like(chart[:, ri])
        cnij = np.zeros_like(chart[:, ri])

        for (cs, ci) in ccn:
            cni |= chart[:, ci]
            if ci == ri:
                continue
            cnij |= chart[:, ci]

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

    cii = np.zeros_like(chart[:, 0])
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


def get_scores(y_true: np.array, y_pred: np.array) -> list[float]:
    try:
        pre = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        return [pre, rec, f1]
    except:
        pre = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        f1micro = f1_score(y_true, y_pred, average='micro')

        return [pre, f1, f1micro]


def ruleForNPCorrection(all_charts: list[list[int]],
                        epsilon: float):
    results = []
    total_results = np.copy(pred_data)
    for count, chart in enumerate(all_charts):
        chart = np.array(chart)
        NCi = GreedyNegRuleSelect(count, epsilon, all_charts)
        negi_count = 0
        posi_count = 0

        predict_result = np.copy(chart[:, 0])
        tem_cond = np.zeros_like(chart[:, 0])

        for cc in NCi:
            tem_cond |= chart[:, cc]
        if np.sum(tem_cond) > 0:
            for ct, cv in enumerate(chart):
                if tem_cond[ct] and predict_result[ct]:
                    negi_count += 1
                    predict_result[ct] = 0

        CCi = DetUSMPosRuleSelect(count, all_charts)
        tem_cond = np.zeros_like(chart[:, 0])

        for cc in CCi:
            tem_cond |= chart[:, cc]
        if np.sum(tem_cond) > 0:
            for ct, cv in enumerate(chart):
                if tem_cond[ct] and not predict_result[ct]:
                    posi_count += 1
                    predict_result[ct] = 1
                    total_results[ct] = count

        scores_cor = get_scores(chart[:, 1], predict_result)
        results.extend(scores_cor + [negi_count, posi_count, len(NCi), len(CCi)])

    results.extend(get_scores(true_data, total_results))
    return results


def plot(df: pd.DataFrame,
         n: int,
         epsilons: list[float]):
    for i in range(n):
        df_i = df.iloc[1:, 2 + i * 7:2 + (i + 1) * 7]

        pre_i = df_i.iloc[:, 0]
        rec_i = df_i.iloc[:, 1]
        f1_i = df_i.iloc[:, 1]

        plt.plot(epsilons[1:], pre_i[1:], label='pre')
        plt.plot(epsilons[1:], rec_i[1:], label='rec')
        plt.plot(epsilons[1:], f1_i[1:], label='f1')

        plt.title(f'cls - {i}')
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.show()


if __name__ == '__main__':

    # %% Data load

    true_data = np.load("inception_true.npy")
    assert np.all(true_data == np.load('vit_true.npy'))

    pred_data = np.load("inception_pred.npy")
    print(f'Inception accuracy: {accuracy_score(y_true=true_data, y_pred=pred_data)}')
    cla_datas = np.load('vit_pred.npy')
    print(f'VIT accuracy: {accuracy_score(y_true=true_data, y_pred=cla_datas)}')

    n = np.max(cla_datas) + 1
    cla_datas = np.eye(n)[cla_datas]

    charts = []

    m = true_data.shape[0]
    for i in range(m):
        tmp_charts = [pred_data[i], true_data[i]] + [int(cls[i]) for cls in cla_datas.T]
        charts.append(tmp_charts)

    all_charts = generate_chart(charts)

    results = []
    result0 = [0]

    for count, chart in enumerate(all_charts):
        chart = np.array(chart)
        result0.extend(get_scores(chart[:, 1], chart[:, 0]))
        result0.extend([0, 0, 0, 0])

    result0.extend(get_scores(true_data, pred_data))
    results.append(result0)

    epsilons = [0.0001 * i for i in range(0, 101, 1)]
    col = ['precision', 'recall', 'F1', 'NSC', 'PSC', 'NRC', 'PRC']
    results = [result0]

    for epsilon in epsilons:
        result = ruleForNPCorrection(all_charts, epsilon)
        results.append([epsilon] + result)
        print(f"ep:{epsilon}\n{result}")

    df = pd.DataFrame(results, columns=['epsilon'] + col * n + ['acc', 'macro-F1', 'micro-F1'])

    results_filename = "results.csv"
    df.to_csv(results_filename)

    df = pd.read_csv(results_filename)
    plot(df=df,
         n=n,
         epsilons=epsilons)
