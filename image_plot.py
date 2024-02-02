import matplotlib.pyplot as plt
import os

def count_images_in_two_directories(directory1, directory2):
    # Dictionary to store the count of images in each subfolder
    image_counts = {}

    # Function to count images in a single directory
    def count_in_directory(directory):
        counts = {}
        for subfolder in os.listdir(directory):

            subfolder_path = os.path.join(directory, subfolder)
            if os.path.isdir(subfolder_path):
                count = sum(1 for file in os.listdir(subfolder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')))
                subfolder = 'MT-LB' if subfolder == 'MT_LB' else subfolder
                subfolder = 'SPA' if subfolder == 'Self Propelled Artillery' else subfolder
                subfolder = '2S19-MSTA' if subfolder == '2S19_MSTA' else subfolder

                counts[subfolder] = counts.get(subfolder, 0) + count
        return counts

    # Count images in both directories
    counts1 = count_in_directory(directory1)
    counts2 = count_in_directory(directory2)

    # Combine counts from both directories
    for subfolder in set(counts1.keys()).union(set(counts2.keys())):
        image_counts[subfolder] = counts1.get(subfolder, 0) + counts2.get(subfolder, 0)

    return image_counts


def plot_image_counts(counts):
    # Sort counts by values from smallest to largest
    sorted_counts = sorted(counts.items(), key=lambda x: x[1])
    classes, num_images = zip(*sorted_counts)  # Unzip into two lists

    plt.figure(figsize=(25, 25))
    plt.bar(classes, num_images, color='orange')  # Set color to orange
    plt.ylabel('Frequency', fontsize=63, labelpad=20)
    plt.xticks(rotation=90, fontsize=63)
    plt.yticks(fontsize=63)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)

    # Adjust plot margins
    plt.subplots_adjust(left=0.75)  # Adjust left margin if necessary

    plt.tight_layout()
    plt.grid()

    plt.savefig(fname='distplot_sorted_orange.png', format='png')
    plt.show()



dir1 = 'test_coarse'
dir2 = 'train_coarse'

count = count_images_in_two_directories(dir1, dir2)

plot_image_counts(count)