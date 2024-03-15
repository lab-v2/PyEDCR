import matplotlib.pyplot as plt
import numpy as np

import data_preprocessing




def plot_per_class(epsilons,
                   ps,
                   rs,
                   folder: str):
    for g in data_preprocessing.granularities:
        # plot all label per granularity:
        for label in data_preprocessing.get_labels(g).values():
            plt.plot(epsilons, [ps[g]['initial'][e][label] for e in epsilons],
                     label='initial average precision')
            plt.plot(epsilons, [ps[g]['pre_correction'][e][label] for e in epsilons],
                     label='pre correction average precision')
            plt.plot(epsilons, [ps[g]['post_correction'][e][label] for e in epsilons],
                     label='post correction average precision')

            plt.plot(epsilons, [rs[g]['initial'][e][label] for e in epsilons],
                     label='initial average recall')
            plt.plot(epsilons, [rs[g]['pre_correction'][e][label] for e in epsilons],
                     label='pre correction average recall')
            plt.plot(epsilons, [rs[g]['post_correction'][e][label] for e in epsilons],
                     label='post correction average recall')

            plt.legend()
            plt.tight_layout()
            plt.grid()
            plt.title(f'{label}')
            plt.savefig(f'figs/{folder}/{label}.png')
            plt.clf()
            plt.cla()


def plot_all(epsilons,
             ps,
             rs,
             folder: str):
    for g in data_preprocessing.granularities:
        # plot average precision recall per granularity:

        plt.plot(epsilons, [np.mean(list(ps[g]['initial'][e].values())) for e in epsilons],
                 label='initial average precision')
        plt.plot(epsilons, [np.mean(list(ps[g]['pre_correction'][e].values())) for e in epsilons],
                 label='pre correction average precision')
        plt.plot(epsilons, [np.mean(list(ps[g]['post_correction'][e].values())) for e in epsilons],
                 label='post correction average precision')

        plt.plot(epsilons, [np.mean(list(rs[g]['initial'][e].values())) for e in epsilons],
                 label='initial average precision')
        plt.plot(epsilons, [np.mean(list(rs[g]['pre_correction'][e].values())) for e in epsilons],
                 label='pre correction average precision')
        plt.plot(epsilons, [np.mean(list(rs[g]['post_correction'][e].values())) for e in epsilons],
                 label='post correction average precision')

        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.title(f'average precision recall for {g}')
        plt.savefig(f'figs/{folder}/average_{g}.png')
        plt.clf()
        plt.cla()