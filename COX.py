import os
from NeuralPyEDCR import simulate_for_values

data_str = 'COX'
main_model_name = 'vit_b_16'
secondary_model_name = 'vit_l_16'
main_lr = secondary_lr = binary_lr = 0.0001
original_num_epochs = 10
secondary_num_epochs = 20
binary_num_epochs = 10
number_of_fine_classes = 24

binary_l_strs = list({f.split(f'e{binary_num_epochs - 1}_')[-1].replace('.npy', '')
                      for f in os.listdir('binary_results')
                      if f.startswith(f'{data_str}_{binary_model_name}')})

simulate_for_values(total_number_of_points=1,
                    min_value=0.1,
                    max_value=0.1,
                    binary_l_strs=binary_l_strs,
                    binary_lr=binary_lr,
                    binary_num_epochs=binary_num_epochs,
                    multi_processing=True,
                    secondary_model_name=secondary_model_name,
                    secondary_model_loss='BCE',
                    secondary_num_epochs=secondary_num_epochs,
                    secondary_lr=secondary_lr,
                    maximize_ratio=True,
                    lists_of_fine_labels_to_take_out=[[]],
                    negated_conditions=False)
