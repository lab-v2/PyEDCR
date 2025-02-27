import typing
import matplotlib.pyplot as plt
import scipy
import numpy as np


def plot_3d_metrics(x_values: np.array,
                    y_values: np.array,
                    metrics: np.array):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Creating a mesh grid
    xi = np.linspace(min(x_values), max(x_values), 100)
    yi = np.linspace(min(y_values), max(y_values), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Plot each metric as a surface
    for label, (values, cmap, color) in metrics.items():
        # Interpolate z values on created grid
        zi = scipy.interpolate.griddata(points=(x_values, y_values),
                                        values=values,
                                        xi=(xi, yi),
                                        method='cubic')

        # Plot surface
        ax.plot_surface(xi, yi, zi, cmap=cmap, edgecolor='none', alpha=0.75, label=label)
        ax.scatter(x_values, y_values, values, color=color, label=f'{label} Data')

    ax.set_ylim(ax.get_ylim()[::-1])

    # Adjust the view angle
    ax.view_init(
        # elev=30,
        azim=-30
    )  # Elevate 30°, rotate to 120°

    # Labels and Legend
    ax.set_xlabel('Noise Ratio')
    ax.set_ylabel('Epsilon')
    ax.set_zlabel('Error f1 Score')

    # # Since 3D legend is not directly supported, we use a workaround to show legends for surfaces
    # legend_patches = [matplotlib.patches.Patch(color=plt.get_cmap(name)(0.5), label=label)
    #                   for label, (_, name) in metrics.items()]
    # ax.legend(handles=legend_patches, loc='best', fontsize='15')

    plt.tight_layout()
    plt.show()


def plot_2d_metrics(data_str: str,
                    x_values,
                    metrics: typing.Dict,
                    style_dict,
                    fontsize: int):
    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot each metric
    for metric_name, metric_values in metrics.items():
        color, linestyle = style_dict.get(metric_name, ('k', '-'))  # Default to black solid line if not specified
        plt.plot(x_values, metric_values, label=metric_name, color=color, linestyle=linestyle, linewidth=3)

    # Add labels and title
    models_dict = {'vit_b_16': 'VIT_b_16',
                   'dinov2_vits14': 'DINO V2 VIT14_s',
                   'dinov2_vitl14': 'DINO V2 VIT14_l',
                   'tresnet_m': 'Tresnet M',
                   'vit_l_16': 'VIT_l_16'}
    data_dict = {'military_vehicles': 'Military Vehicles',
                 'imagenet': 'ImageNet',
                 'openimage': 'OpenImage',
                 'coco': 'COCO'}

    plt.xlabel("Noise ratio", fontsize=fontsize)
    # plt.ylabel("Percentage (%)")
    # plt.title(f"Noise ratio experiments for {models_dict[model_name]} on {data_dict[data_str]} "
    #           f"with binary and secondary conditions")

    plt.xticks(np.arange(0.0, max(x_values) + 0.1, 0.1), fontsize=fontsize)
    plt.yticks(np.arange(0.0, 101, 10), fontsize=fontsize)

    # Add legend
    plt.legend(fontsize=fontsize)

    # Show the plot
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{data_str}_noise.png', format='png', dpi=600)
    plt.show()
