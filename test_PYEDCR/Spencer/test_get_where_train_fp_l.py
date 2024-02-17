from PyEDCR import EDCR
import data_preprocessing
import warnings
import numpy as np
import utils

from test_PYEDCR.test import *


# This will silence all warnings, including ones unrelated to your evaluation.
# Use this approach with caution!
warnings.filterwarnings('ignore')

# method name
method_str = "get_where_train_fp_l"


# Test 1
def test_case_1():
    K= [(1, 5), (401, 405)]
    test = Test(epsilon=0.1, K_train=K, K_test=K, print_pred_and_true=False)
    test.run(method_str='get_where_label_is_l', pred=True, test=False, l=l_Tank, expected_output=0)


    edcr = EDCR.test(epsilon=0.1,
                     K_train=K_train_slice,
                     K_test=K_test_slice,
                     print_pred_and_true=True)


    case_number = 1

    print(utils.blue_text("=" * 50 + f"test {case_number} " + method_str + "=" * 50))
    test_label = label_fine['30N6E']

    print(f'label is: {test_label._l_str}, granularity: {test_label.g}, label_index: {test_label.index}')

    except_result_1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])

    edcr.test_get_where_train_fp_l(l=test_label,
                                    expected_result=except_result_1,
                                   print_result=True)

    print(f'Case {case_number} passed!')


def test_case_2():
    K_train_slice = [(351, 355), (471, 475), (4001, 4005), (4601, 4605), (6001, 6005)]
    K_test_slice = [(1, 10), (50, 60)]

    edcr = EDCR.test(epsilon=0.1,
                     K_train=K_train_slice,
                     K_test=K_test_slice,
                     print_pred_and_true=True)

    case_number = 2

    print(utils.blue_text("=" * 50 + f"test {case_number} " + method_str + "=" * 50))
    test_label = label_coarse['Air Defense']

    print(f'label is: {test_label._l_str}, granularity: {test_label.g}, label_index: {test_label.index}')

    except_result_2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0])

    edcr.test_get_where_train_fp_l(l=test_label,
                                   expected_result=except_result_2)

    print(f'Case {case_number} passed!')

def test_case_3():
    K_train_slice = [(6, 10), (2201, 2205), (3001, 3005), (3601, 3605), (3970, 3974), (7500, 7524)]

    edcr = EDCR.test(epsilon=0.1,
                     K_train=K_train_slice,
                     print_pred_and_true=True)

    case_number = 3

    print(utils.blue_text("=" * 50 + f"test {case_number} " + method_str + "=" * 50))
    test_label = label_fine['BM-30']

    print(f'label is: {test_label._l_str}, granularity: {test_label.g}, label_index: {test_label.index}')

    except_result_2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 1, 1, 0, 0, 0, 0, 0])

    edcr.test_get_where_train_fp_l(l=test_label,
                                   expected_result=except_result_2)

    print(f'Case {case_number} passed!')

if __name__ == '__main__':
    test_case_1()
    test_case_2()
    test_case_3()