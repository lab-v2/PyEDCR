import os
import numpy as np
import re
from sklearn.metrics import accuracy_score

# Specify the directory where your files are located
directory_path = '.'

# Initialize dictionaries to store loaded data
true_data_dict = {}
pred_data_dict = {}

# List all files in the directory
for filename in os.listdir(directory_path):
    if re.search(r'test_true|test_pred', filename):
        # Construct the full file path
        file_path = os.path.join(directory_path, filename)

        # Check if the filename contains 'test_true' or 'test_pred'
        if 'test_true' in filename:
            # Load the 'test_true' file with NumPy
            true_data_dict[filename] = np.load(file_path)
        elif 'test_pred' in filename:
            # Load the 'test_pred' file with NumPy
            pred_data_dict[filename] = np.load(file_path)

# Calculate accuracy for matching pairs
accuracy_scores = {}

for true_filename, true_data in true_data_dict.items():
    matching_pred_filename = true_filename.replace('test_true', 'test_pred')

    if matching_pred_filename in pred_data_dict:
        pred_data = pred_data_dict[matching_pred_filename]

        # Calculate accuracy score
        accuracy = accuracy_score(true_data, pred_data)
        if true_filename.split('_')[0] == 'vit':
            if accuracy > 0.1:
                accuracy_scores[true_filename] = accuracy
        else:
            accuracy_scores[true_filename] = accuracy
# Print accuracy scores for matching pairs
for true_filename, accuracy in accuracy_scores.items():
    print(f"Accuracy for {true_filename}: {accuracy}")