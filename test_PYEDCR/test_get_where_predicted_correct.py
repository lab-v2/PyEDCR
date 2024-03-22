from PyEDCR import EDCR
import data_processing
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
method_str = "Get_where_predicted_correct"


print(utils.blue_text("=" * 50 + "test " + method_str + "=" * 50))

# Test 1

print(f'granularity: {data_preprocessing.granularities[1]}, train')

expect_result_train_coarse = [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
       1, 1, 1, 0, 1, 1, 1, 1]

edcr.test_get_where_predicted_correct(test=False, 
                                      g=data_preprocessing.granularities[1],
                                      expected_result=expect_result_train_coarse)

# Test 2

print(f'granularity: {data_preprocessing.granularities[0]}, train')

expect_result_train_fine = [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1,
       1, 1, 1, 0, 1, 1, 1, 1]

edcr.test_get_where_predicted_correct(test=False, 
                                      g=data_preprocessing.granularities[0],
                                      expected_result=expect_result_train_fine)

# Test 3

print(f'granularity: {data_preprocessing.granularities[1]}, test')

expect_result_test_coarse = [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0,
       1, 1, 0, 0, 0, 0, 1, 1]
edcr.test_get_where_predicted_correct(test=True, 
                                      g=data_preprocessing.granularities[1],
                                      expected_result=expect_result_test_coarse)

# Test 4

print(f'granularity: {data_preprocessing.granularities[0]}, test')

expect_result_test_fine = [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0,
       1, 1, 0, 0, 0, 0, 1, 1]

edcr.test_get_where_predicted_correct(test=True, 
                                      g=data_preprocessing.granularities[0],
                                      expected_result=expect_result_test_fine)

print(f"{method_str} method passed!")