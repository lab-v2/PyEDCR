import typing
import torch
import torch.utils.data
from sklearn.metrics import accuracy_score

import models
import utils

best_coarse_main_model = 'vit_b_16'
best_coarse_main_lr = '5e-05'
best_coarse_secondary_model = 'vit_l_16'
best_coarse_secondary_lr = '1e-05'
best_coarse_folder = f'main_coarse_{best_coarse_main_model}_lr{best_coarse_main_lr}_secondary_{best_coarse_secondary_model}_lr{best_coarse_secondary_lr}'
best_coarse_results = np.load(rf'{EDCR_pipeline.figs_folder}{best_coarse_folder}/results.npy')
coarse_test_true = np.load(os.path.join(EDCR_pipeline.data_folder, f'test_true_coarse.npy'))

best_fine_main_model = 'vit_l_16'
best_fine_main_lr = '1e-06'
best_fine_secondary_model = 'vit_b_16'
best_fine_secondary_lr = '1e-06'
best_fine_folder = f'main_fine_{best_fine_main_model}_lr{best_fine_main_lr}_secondary_{best_fine_secondary_model}_lr{best_fine_secondary_lr}'
best_fine_results = np.load(rf'{EDCR_pipeline.figs_folder}{best_fine_folder}/results.npy')
fine_test_true = np.load(os.path.join(EDCR_pipeline.data_folder, f'test_true.npy'))

with open('fine_to_coarse.json', 'r') as json_file:
    image_fine_to_coarse = json.load(json_file)

image_fine_to_coarse = {int(fine): int(coarse) for batch_dict in image_fine_to_coarse for fine, coarse in batch_dict.items()}
image_fine_to_coarse

# def get_num_of_inconsistencies(coarse_results: np.array,
#                                fine_results: np.array) -> int:
#     num_of_inconsistencies = 0
#     for fine_example_num, fine_prediction_index in enumerate(fine_results):
#         coarse_example_num = image_fine_to_coarse[fine_example_num]
#         coarse_prediction_index = coarse_results[coarse_example_num]
#
#         fine_prediction = EDCR_pipeline.get_classes(granularity='fine')[fine_prediction_index]
#         coarse_prediction = EDCR_pipeline.get_classes(granularity='coarse')[coarse_prediction_index]
#         derived_coarse_prediction = EDCR_pipeline.fine_to_coarse[fine_prediction]
#
#         if derived_coarse_prediction != coarse_prediction:
#             num_of_inconsistencies += 1
#
#     return num_of_inconsistencies
#
# get_num_of_inconsistencies(coarse_results=coarse_test_true,
#                            fine_results=fine_test_true)