import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot(df: pd.DataFrame,
         n: int,
         epsilons: list[float]):
    for i in range(n):
        df_i = df.iloc[1:, 2 + i * 7:2 + (i + 1) * 7]

        added_str = f'.{i}' if i else ''
        pre_i = df_i[f'pre{added_str}']
        rec_i = df_i[f'recall{added_str}']
        f1_i = df_i[f'F1{added_str}']

        plt.plot(epsilons, pre_i, label='pre')
        plt.plot(epsilons, rec_i, label='rec')
        plt.plot(epsilons, f1_i, label='f1')

        plt.title(f'class #{i}')
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.show()



if __name__ == '__main__':
    base_path0 = 'LRCN_F1_no_overlap_sequential/'
    results_file = base_path0 + "rule_for_NPcorrection.csv"
    df = pd.read_csv(results_file)
    cla_datas = np.load('vit_pred.npy')
    n = np.max(cla_datas) + 1
    epsilons = df['epsilon']
    plot(df=df, n=n, epsilons=epsilons)
