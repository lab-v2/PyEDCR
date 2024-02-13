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

label_Tornado = label_fine[23]

print(f'label is: {label_Tornado._l_str}, granularity: {label_Tornado.g}, label_index: {label_Tornado.index}')

except_result_1 = 3

edcr.test_how_many_predicted_l(test=True, 
                            l=label_Tornado,
                            expected_result=except_result_1)

# Test 2

label_air_defence = label_coarse[0]

print(f'label is: {label_air_defence._l_str}, granularity: {label_air_defence.g}, label_index: {label_air_defence.index}')


except_result_2 = 10

edcr.test_how_many_predicted_l(test=False, 
                            l=label_air_defence,
                            expected_result=except_result_2)

print("Get_how_many_predicted_l method passed!")