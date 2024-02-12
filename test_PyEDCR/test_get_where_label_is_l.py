from PyEDCR import EDCR
import data_preprocessing
import warnings
import numpy as np
import utils

# This will silence all warnings, including ones unrelated to your evaluation.
# Use this approach with caution!
warnings.filterwarnings('ignore')

edcr = EDCR(epsilon=0.1,
            check_mode=True
            )
edcr.print_metrics(test=False, prior=True)
edcr.print_metrics(test=True, prior=True)

# get label 
label_fine = data_preprocessing.get_labels(data_preprocessing.granularities[0])
label_coarse = data_preprocessing.get_labels(data_preprocessing.granularities[1])


print(utils.blue_text("=" * 50 + "test get_where_label_is_l" + "=" * 50))

# Test 1

label_30N6E = label_fine[1]

print(f'label is: {label_30N6E._l_str}, granularity: {label_30N6E.g}, label_index: {label_30N6E.index}')

get_where_label_is_train_fine_pred_30N6E = edcr.test_get_where_label_is_l(pred=True, 
                                                                          test=False, 
                                                                          l=label_30N6E)

except_result_1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0]

assert(np.all(get_where_label_is_train_fine_pred_30N6E == except_result_1))

# Test 2

label_BM_30 = label_fine[2]

print(f'label is: {label_BM_30._l_str}, granularity: {label_BM_30.g}, label_index: {label_BM_30.index}')

get_where_label_is_train_fine_pred_BM_30 = edcr.test_get_where_label_is_l(pred=True, 
                                                                          test=False, 
                                                                          l=label_BM_30)

except_result_2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1,
       1, 1, 1, 0, 1, 1, 1, 1]

assert(np.all(get_where_label_is_train_fine_pred_BM_30 == except_result_2))

# Test 3

label_air_defence = label_coarse[0]

print(f'label is: {label_air_defence._l_str}, granularity: {label_air_defence.g}, label_index: {label_air_defence.index}')

get_where_label_is_test_coarse_true_air_defence = edcr.test_get_where_label_is_l(pred=False, 
                                                                                test=True, 
                                                                                l=label_air_defence)

except_result_3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0]

assert(np.all(get_where_label_is_test_coarse_true_air_defence == except_result_3))

print("Get_where_label_is_l method passed!")