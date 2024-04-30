import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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



def plot_3d_epsilons_ODD(images_per_class,
                         epsilons,
                         error_accuracies,
                         error_f1s,
                         consistency_error_accuracies,
                         consistency_error_f1s,
                         RCC_ratios):
    # Initialize 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plots
    scatter1 = ax.scatter(images_per_class, epsilons, error_accuracies,
                          color='r', label='Error Accuracies')
    scatter2 = ax.scatter(images_per_class, epsilons, error_f1s,
                          color='g', label='Error F1s')
    scatter3 = ax.scatter(images_per_class, epsilons, consistency_error_accuracies,
                          color='b', label='Consistency Error Accuracies')
    scatter4 = ax.scatter(images_per_class, epsilons, consistency_error_f1s,
                          color='y', label='Consistency Error F1s')
    scatter5 = ax.scatter(images_per_class, epsilons, [r * 100 for r in RCC_ratios],
                          color='m', label='RCC Ratios')

    # Labels and Legend
    ax.set_xlabel('Images per Class')
    ax.set_ylabel('Epsilons')
    ax.set_zlabel('Values')
    ax.legend(loc='center right',
              bbox_to_anchor=(0.1, 0.5))

    plt.show()
