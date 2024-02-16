import numpy as np

from test_PYEDCR.test import *


def run_tests():
    K = [(0, 9)]
    test = Test(epsilon=0.1, K_train=K, K_test=K, print_pred_and_true=False)

    test.run(method_str='get_where_label_is_l', pred=True, test=False, l=l_Tank, expected_output=0)
    test.run(method_str='get_where_label_is_l', pred=True, test=False, l=l_Iskander, expected_output=0)
    test.run(method_str='get_where_label_is_l', pred=True, test=False, l=l_SPA, expected_output=1)
    test.run(method_str='get_where_label_is_l', pred=True, test=False, l=l_2S19_MSTA,
             expected_output=np.array([[1] * 6 + [0] + [1] * 3]))
    test.run(method_str='get_where_label_is_l', pred=False, test=False, l=l_Tank, expected_output=0)
    test.run(method_str='get_where_label_is_l', pred=False, test=False, l=l_Iskander, expected_output=0)
    test.run(method_str='get_where_label_is_l', pred=False, test=False, l=l_SPA, expected_output=1)
    test.run(method_str='get_where_label_is_l', pred=False, test=False, l=l_2S19_MSTA, expected_output=1)

    test.run(method_str='get_where_label_is_l', pred=True, test=True, l=l_Tank,
             expected_output=np.array([1] + [0] * 5 + [1] + [0] * 3))
    test.run(method_str='get_where_label_is_l', pred=True, test=True, l=l_Iskander, expected_output=0)
    test.run(method_str='get_where_label_is_l', pred=True, test=True, l=l_SPA,
             expected_output=np.array([0] + [1] * 5 + [0] + [1] * 3))
    test.run(method_str='get_where_label_is_l', pred=True, test=True, l=l_2S19_MSTA,
             expected_output=np.array([0] + [1] * 5 + [0] * 2 + [1] * 2))
    test.run(method_str='get_where_label_is_l', pred=False, test=True, l=l_Tank, expected_output=0)
    test.run(method_str='get_where_label_is_l', pred=False, test=True, l=l_Iskander, expected_output=0)
    test.run(method_str='get_where_label_is_l', pred=False, test=True, l=l_SPA, expected_output=1)
    test.run(method_str='get_where_label_is_l', pred=False, test=True, l=l_2S19_MSTA, expected_output=1)

    K = [(0,11)]
    test = Test(epsilon=0.1, K_train=K, K_test=K, print_pred_and_true=True)
    test.run(method_str='get_where_label_is_l', pred=True, test=False, l=l_Tank, expected_output=0)
    test.run(method_str='get_where_label_is_l', pred=True, test=False, l=l_Iskander, expected_output=0)
    test.run(method_str='get_where_label_is_l', pred=True, test=False, l=l_SPA,
             expected_output=np.array([1] * 11 + [0]))
    test.run(method_str='get_where_label_is_l', pred=True, test=False, l=l_2S19_MSTA,
             expected_output=np.array([1] * 11 + [0]))
    test.run(method_str='get_where_label_is_l', pred=False, test=False, l=l_Tank, expected_output=0)
    test.run(method_str='get_where_label_is_l', pred=False, test=False, l=l_Iskander, expected_output=0)
    test.run(method_str='get_where_label_is_l', pred=False, test=False, l=l_SPA, expected_output=1)
    test.run(method_str='get_where_label_is_l', pred=False, test=False, l=l_2S19_MSTA, expected_output=1)

    test.run(method_str='get_where_label_is_l', pred=True, test=True, l=l_Tank,
             expected_output=np.array([1] + [0] * 5 + [1] + [0] * 3))
    test.run(method_str='get_where_label_is_l', pred=True, test=True, l=l_Iskander, expected_output=0)
    test.run(method_str='get_where_label_is_l', pred=True, test=True, l=l_SPA,
             expected_output=np.array([0] + [1] * 5 + [0] + [1] * 3))
    test.run(method_str='get_where_label_is_l', pred=True, test=True, l=l_2S19_MSTA,
             expected_output=np.array([0] + [1] * 5 + [0] * 2 + [1] * 2))
    test.run(method_str='get_where_label_is_l', pred=False, test=True, l=l_Tank, expected_output=0)
    test.run(method_str='get_where_label_is_l', pred=False, test=True, l=l_Iskander, expected_output=0)
    test.run(method_str='get_where_label_is_l', pred=False, test=True, l=l_SPA, expected_output=1)
    test.run(method_str='get_where_label_is_l', pred=False, test=True, l=l_2S19_MSTA, expected_output=1)
    print('All tests passed!')


if __name__ == '__main__':
    run_tests()
