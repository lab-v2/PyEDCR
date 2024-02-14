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

# method name
method_str = "Get_where_train_fp_l"

print(utils.blue_text("=" * 50 + "test " + method_str + "=" * 50))

# Test 1

label_Rs_24 = label_fine[15]

print(f'label is: {label_Rs_24._l_str}, granularity: {label_Rs_24.g}, label_index: {label_Rs_24.index}')

except_result_1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0])

edcr.test_get_where_train_fp_l(l=label_Rs_24,
                                expected_result=except_result_1)

# Test 2

label_SPA = label_coarse[5]

print(f'label is: {label_SPA._l_str}, granularity: {label_SPA.g}, label_index: {label_SPA.index}')

except_result_2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0])

edcr.test_get_where_train_fp_l(l=label_SPA,
                                expected_result=except_result_2)

print(f"{method_str} method passed!")