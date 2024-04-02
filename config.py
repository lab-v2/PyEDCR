import data_preprocessing

# Configuration for training process
g_fine, g_coarse = data_preprocessing.granularities.values()
fg_l, cg_l = list(data_preprocessing.fine_grain_labels.values()), list(data_preprocessing.coarse_grain_labels.values())

l_Air_Defense, l_BMD_coarse, l_BMP, l_BTR, l_MT_LB_coarse, l_SPA, l_Tank = cg_l

# Training Parameters
batch_size = 128
ltn_num_epochs = 1  # Number of epochs for LTN training (if applicable)
baseline_num_epochs = 20
lr = 0.0001  # Initial learning rate
scheduler_gamma = 0.9  # Learning rate scheduler gamma value
scheduler_step_size = 1  # Number of epochs after which to decay learning rate
loss = 'BCE'  # Loss function (e.g., 'BCE' for Binary Cross-Entropy)
beta = 0.1  # weight for LTN: higher beta -> higher weight to LTN
get_fraction_of_example_with_label = {l_Tank: 0.1}  # fraction of example to remove from a label in the train dataset

num_epochs = 20
K_train = None  # Number of example for training
K_test = None   # Number of example for testing

# Model and Loss Configuration
vit_model_names = [f'vit_{vit_model_name}' for vit_model_name in ['b_16']]
combined = True  # Whether to train a combined model for fine and coarse grain classification

# Additional Training Options
include_inconsistency_constraint = False  # Whether to include inconsistency constraint in loss
secondary_model_name = 'vit_b_16_soft_marginal'  # (Optional) Name of secondary model for additional tasks

# Data Paths
combined_results_path = 'combined_results'  # Path to save combined results
individual_results_path = 'individual_results'  # Path to save individual results (fine and coarse)

# Pre-trained Model Path
main_pretrained_path = "/home/ngocbach/PyEDCR/models/vit_b_16_BCE_lr0.0001.pth"  # Path to pre-trained model

# Data Preprocessing Information (assuming data_preprocessing.py exists)
original_prediction_weight = 1 / (len(data_preprocessing.fine_grain_classes_str) +
                                  len(data_preprocessing.coarse_grain_classes_str))  # Weight for original prediction in combined loss

# Config for PyEDCR
randomized = False