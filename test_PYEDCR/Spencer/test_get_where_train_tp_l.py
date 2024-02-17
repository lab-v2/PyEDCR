from PyEDCR import EDCR
import data_preprocessing
import warnings
import numpy as np
import utils

from test_PYEDCR.test import *


# This will silence all warnings, including ones unrelated to your evaluation.
# Use this approach with caution!
warnings.filterwarnings('ignore')

# get label
label_fine = data_preprocessing.get_labels(data_preprocessing.granularities['fine'])
label_coarse = data_preprocessing.get_labels(data_preprocessing.granularities['coarse'])

# method name
method_str = "get_where_train_tp_l"


# Test 1
def test_case_1():
    K = [(1, 5), (401, 405)]
    test = Test(epsilon=0.1, K_train=K, K_test=None, method_str=method_str)
    case_number = 1

    print(utils.blue_text("=" * 50 + f"test {case_number} " + method_str + "=" * 50))

    test.run(l=l_30N6E, expected_output=np.array([0] * 5 + [1] * 3 + [0] + [1]))
    test.run(l=l_Air_Defense,expected_output=np.array([0] * 5 + [1] * 5))
    test.run(l=l_Tank, expected_output=np.array([0] * 10))
    test.run(l=l_SPA, expected_output=np.array([1] * 5 + [0] * 5))
    test.run(l=l_RS_24, expected_output=np.array([0] * 10))
    test.run(l=l_2S19_MSTA, expected_output=np.array([1] * 5 + [0] * 5))

    print(f'Case {case_number} passed!')


def test_case_2():
    K = [(351, 355), (471, 475), (4001, 4005), (4601, 4605), (6001, 6005)]
    test = Test(epsilon=0.1, K_train=K, K_test=None, method_str=method_str)
    case_number = 2

    print(utils.blue_text("=" * 50 + f"test {case_number} " + method_str + "=" * 50))

    test.run(l=l_Air_Defense, expected_output=np.array([1] * 5 + [0] * 5 + [1] * 10 + [0] * 5))
    test.run(l=l_SPA, expected_output=np.array([0] * 5 + [1] * 5 + [0] * 15))
    test.run(l=l_Tank, expected_output=np.array([0] * 20 + [1] * 5))
    test.run(l=l_MT_LB_coarse, expected_output=np.array([0] * 25))
    test.run(l=l_Pantsir_S1, expected_output=np.array([0] * 15 + [1] * 5 + [0] * 5))
    test.run(l=l_30N6E, expected_output=np.array([0] + [1] + [0] * 2 + [1] + [0] * 20))
    test.run(l=l_D_30, expected_output=np.array([0] * 25))
    test.run(l=l_RS_24, expected_output=np.array([0] * 25))
    test.run(l=l_BM_30, expected_output=np.array([0] * 5 + [1] * 3 + [0] + [1] + [0] * 15))
    test.run(l=l_Iskander, expected_output=np.array([0] * 10 + [1] * 5 + [0] * 10))
    test.run(l=l_T_64, expected_output=np.array([0] * 20 + [1] + [0] + [1] * 3))
    test.run(l=l_T_72, expected_output=np.array([0] * 25))


    print(f'Case {case_number} passed!')


def test_case_3():
    K = [(6, 10), (2201, 2205), (3001, 3005), (3601, 3605), (3970, 3974), (7500, 7524)]
    test = Test(epsilon=0.1, K_train=K, K_test=None, method_str=method_str)
    test.print_examples(test=False)
    case_number = 3

    print(utils.blue_text("=" * 50 + f"test {case_number} " + method_str + "=" * 50))

    test.run(l=l_SPA, expected_output=np.array([1] * 5 + [0] * 12 + [1] * 3 + [0] * 5 + [1] * 5 + [0] + [1] * 5 + [0] + [1] * 13))
    test.run(l=l_BTR, expected_output=np.array([0] * 6 + [1] * 11 + [0] * 33))
    test.run(l=l_BMP, expected_output=np.array([0] * 50))
    test.run(l=l_Air_Defense, expected_output=np.array([0] * 20 + [1] * 5 + [0] * 25))
    test.run(l=l_T_72, expected_output=np.array([0] * 50))
    test.run(l=l_2S19_MSTA, expected_output=np.array([0] + [1] * 4 + [0] * 45))
    # test.run(l=l_BRDM, expected_output=np.array()
    # test.run(l=l_BTR_60, expected_output=np.array()
    # test.run(l=l_BTR_70, expected_output=np.array()
    # test.run(l=l_BTR_80, expected_output=np.array()
    # test.run(l=l_D_30, expected_output=np.array()
    # test.run(l=l_Iskander, expected_output=np.array()
    # test.run(l=l_BM_30, expected_output=np.array()
    # test.run(l=l_Tornado, expected_output=np.array()

    print(f'Case {case_number} passed!')


if __name__ == '__main__':
    # test_case_1()
    # test_case_2()
    test_case_3()
