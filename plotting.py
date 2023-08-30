import pandas as pd
import matplotlib.pyplot as plt


def plot(df: pd.DataFrame,
         n: int,
         epsilons: list[float]):
    for i in range(n):
        df_i = df.iloc[1:, 2 + i * 7:2 + (i + 1) * 7]

        pre_i = df_i.iloc[:, 0]
        rec_i = df_i.iloc[:, 1]
        f1_i = df_i.iloc[:, 2]

        plt.plot(epsilons[1:], pre_i[1:], label='pre')
        plt.plot(epsilons[1:], rec_i[1:], label='rec')
        plt.plot(epsilons[1:], f1_i[1:], label='f1')

        plt.title(f'cls - {i}')
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.show()


if __name__ == '__main__':
    from reup import results_file, n_classes, epsilons

    df = pd.read_csv(results_file)
    plot(df=df, n=n_classes, epsilons=epsilons)
