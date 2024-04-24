import json
import matplotlib.pyplot as plt
import os

f = open('data/ImageNet100/Labels.json')
labels = json.load(f)

labels = {k: v.split(',', 1)[0] for k, v in labels.items()}

train_folder = 'data/ImageNet100/train_fine'
test_folder = 'data/ImageNet100/test_fine'


def plot_image_distribution(train_directory, test_directory, labels):
    # Initialize a dictionary to hold the combined count of images per class
    combined_class_distribution = {}

    # Function to count images in a directory
    def count_images_in_directory(directory):
        class_counts = {}
        # Loop through each class directory in the main folder
        for class_folder in os.listdir(directory):
            class_path = os.path.join(directory, class_folder)
            if os.path.isdir(class_path):  # Check if it is a directory
                # Count the number of image files in the class directory
                num_images = len([name for name in os.listdir(class_path) if name.lower().endswith('.jpeg')])
                class_counts[class_folder] = num_images
        return class_counts

    # Count images in both train and test directories
    train_counts = count_images_in_directory(train_directory)
    test_counts = count_images_in_directory(test_directory)

    # Sum the counts from both directories
    for class_code in train_counts:
        combined_count = train_counts[class_code] + test_counts.get(class_code, 0)
        class_label = labels.get(class_code, class_code)  # Map the code to a label
        combined_class_distribution[class_label] = combined_count

    # Prepare to plot
    class_labels = list(combined_class_distribution.keys())
    counts = list(combined_class_distribution.values())

    # Create a vertical bar chart
    plt.figure(figsize=(15, 10))
    plt.bar(class_labels, counts, color='skyblue')
    plt.ylabel('Total Number of Images', fontsize=22.5)
    plt.xlabel('Classes', fontsize=22.5)
    plt.title('Distribution of Images', fontsize=22.5)
    plt.xticks(rotation=90, fontsize=22.5)
    plt.yticks(fontsize=22.5)
    plt.tight_layout()
    plt.show()


def count_subfolders_in_directories(directory_paths):
    """
    Counts the number of subfolders in each provided directory path.

    :param directory_paths: A list of directory paths to count subfolders in
    :return: A dictionary mapping directory paths to the number of subfolders
    """
    subfolder_counts = {}
    for directory_path in directory_paths:
        try:
            # List all the entries in the directory given by path
            entries = os.listdir(directory_path)
            # Count all directories within the given directory
            subfolder_count = sum(os.path.isdir(os.path.join(directory_path, entry)) for entry in entries)
            subfolder_counts[directory_path] = subfolder_count
        except FileNotFoundError:
            # If the directory does not exist, set the count to None
            subfolder_counts[directory_path] = None
        except NotADirectoryError:
            # If the path is not a directory, set the count to None
            subfolder_counts[directory_path] = None
    return subfolder_counts


if __name__ == '__main__':
    plot_image_distribution(train_folder, test_folder, labels)