from PyEDCR import EDCR
import data_preprocessing
import warnings
import numpy as np
import utils

# This will silence all warnings, including ones unrelated to your evaluation.
# Use this approach with caution!
warnings.filterwarnings('ignore')

K_train_slice = [(1, 10), (400, 410)]
K_test_slice = [(1, 10), (50, 60)]

edcr = EDCR.test(epsilon=0.1,
                 K_train=K_train_slice,
                 K_test=K_test_slice,
                 print_pred_and_true=True)
edcr.print_metrics(test=False, prior=True)
edcr.print_metrics(test=True, prior=True)

# get label 
label_fine = data_preprocessing.get_labels(data_preprocessing.granularities['fine'])
label_coarse = data_preprocessing.get_labels(data_preprocessing.granularities['coarse'])

print(utils.blue_text("=" * 50 + "test get_where_label_is_l" + "=" * 50))

# Test 1

label_30N6E = label_fine['2S19_MSTA']

print(f'label is: {label_30N6E}, granularity: {label_30N6E.g}, label_index: {label_30N6E.index}')

except_result_1 = 9

edcr.test_how_many_predicted_l(test=False,
                               l=label_30N6E,
                               expected_result=except_result_1)

# Test 2

label_30N6E = label_fine['30N6E']

print(f'label is: {label_30N6E}, granularity: {label_30N6E.g}, label_index: {label_30N6E.index}')

except_result_2 = 8

edcr.test_how_many_predicted_l(test=False,
                               l=label_30N6E,
                               expected_result=except_result_2)

print("Get_how_many_predicted_l method passed!")
