import os
import numpy as np
import matplotlib.pyplot as plt
import re  # Import the regular expressions module

# Set the directory where your .npy files are located
data_dir = 'results/'

# Initialize dictionaries to store training accuracy data for each model
model_data = {}

# Loop through all files in the directory
for filename in os.listdir(data_dir):
    if filename.endswith(".npy") and "train" in filename:
        model_match = re.match(r'(.+?)_train', filename)

        if model_match:
            model_name = model_match.group(1)
            parts = filename.split('_')
            metric = next(part for part in parts if part in ['acc', 'loss'])  # Find 'acc' or 'loss'
            lr_match = re.search(r'lr(.+?)_', filename)

            if lr_match:
                lr_value = float(lr_match.group(1))
                num_epochs = int(parts[-1].split('.')[0].split('e')[-1]) + 1

                # Load the data from the .npy file
                data = np.load(os.path.join(data_dir, filename))

                # Store the data in the model_data dictionary
                if model_name not in model_data:
                    model_data[model_name] = {}
                if metric not in model_data[model_name]:
                    model_data[model_name][metric] = {}
                if lr_value not in model_data[model_name][metric]:
                    model_data[model_name][metric][lr_value] = {}

                model_data[model_name][metric][lr_value][num_epochs] = data[-1]
# Initialize dictionaries to store LR and epoch data for each model
lr_epochs_data = {}

# Loop through all files in the directory
for filename in os.listdir(data_dir):
    if filename.endswith(".npy") and "train" in filename:
        model_match = re.match(r'(.+?)_train', filename)

        if model_match:
            model_name = model_match.group(1)
            parts = filename.split('_')
            lr_match = re.search(r'lr(.+?)_', filename)

            if lr_match:
                lr_value = float(lr_match.group(1))
                num_epochs = int(parts[-1].split('.')[0].split('e')[-1]) + 1

                if model_name not in lr_epochs_data:
                    lr_epochs_data[model_name] = {
                        'lr_values': set(),
                        'num_epochs': set()
                    }

                lr_epochs_data[model_name]['lr_values'].add(lr_value)
                lr_epochs_data[model_name]['num_epochs'].add(num_epochs)


# Define a test function to check if LR and epoch counts are consistent for each model
def test_lr_epoch_consistency():
    for model_name, data in lr_epochs_data.items():
        lr_values = data['lr_values']
        num_epochs = data['num_epochs']

        if len(lr_values) != 1:
            print(f"Model '{model_name}' has inconsistent LR values: {lr_values}")
            return False

        if len(num_epochs) != 1:
            print(f"Model '{model_name}' has inconsistent epoch counts: {num_epochs}")
            return False

    return True


# Run the test function
if test_lr_epoch_consistency():
    print("LR and epoch counts are consistent for all models.")
else:
    print("LR and epoch counts are inconsistent for some models.")

# Now, create plots for training accuracy vs. epoch for each model
for model_name, model_metrics in model_data.items():
    for metric, lr_data in model_metrics.items():
        if metric == 'acc':  # Only plot accuracy data
            plt.figure(figsize=(10, 6))
            plt.title(f'{model_name} - Training Accuracy vs. Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')

            for lr_value, epoch_data in lr_data.items():
                # Sort the data based on the number of epochs
                sorted_epoch_data = sorted(epoch_data.items())
                epochs, accuracies = zip(*sorted_epoch_data)
                plt.plot(epochs, accuracies, label=f'lr={lr_value}')

            plt.legend()
            plt.grid()
            plt.show()
