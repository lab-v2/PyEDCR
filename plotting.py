import matplotlib.pyplot as plt
import scipy
import numpy as np
from matplotlib.patches import Patch
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


def plot_3d_epsilons_ODD(images_per_class: np.array,
                         epsilons: np.array,
                         error_accuracies: np.array,
                         error_f1s: np.array,
                         consistency_error_accuracies: np.array,
                         consistency_error_f1s: np.array,
                         RCC_ratios: np.array
                         ):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Creating a mesh grid
    xi = np.linspace(min(images_per_class), max(images_per_class), 100)
    yi = np.linspace(min(epsilons), max(epsilons), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Dictionary to store interpolated surfaces
    metrics = {
        # 'Error Accuracy': (error_accuracies, 'Reds'),
        'Error F1': (error_f1s, 'Greens'),
        # 'Consistency Error Accuracy': (consistency_error_accuracies, 'Blues'),
        # 'Consistency Error F1': (consistency_error_f1s, 'Oranges'),
        # 'RCC Ratio': ([r * 100 for r in RCC_ratios], 'Purples')  # Scaling RCC ratios for visualization
    }

    # Plot each metric as a surface
    for label, (values, cmap) in metrics.items():
        # Interpolate z values on created grid
        zi = scipy.interpolate.griddata(points=(images_per_class, epsilons),
                                        values=values,
                                        xi=(xi, yi),
                                        method='cubic')

        # Plot surface
        ax.plot_surface(xi, yi, zi, cmap=cmap, edgecolor='none', alpha=0.75, label=label)


    ax.set_ylim(ax.get_ylim()[::-1])

    # Adjust the view angle
    ax.view_init(
        # elev=30,
        azim=-30
    )  # Elevate 30°, rotate to 120°

    # Labels and Legend
    ax.set_xlabel('Images per Class')
    ax.set_ylabel('Epsilon')
    ax.set_zlabel('Values')

    # Since 3D legend is not directly supported, we use a workaround to show legends for surfaces
    legend_patches = [Patch(color=plt.get_cmap(name)(0.5), label=label) for label, (_, name) in metrics.items()]
    ax.legend(handles=legend_patches, loc='best', fontsize='15')

    plt.tight_layout()
    plt.show()
