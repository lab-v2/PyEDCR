import numpy as np

from test import *


def run_tests():
    K = 10
    edcr = EDCR.test(epsilon=0.1, K=K, print_pred_and_true=True)
    edcr.test_get_where_label_is_l(pred=True, test=False, l=l_Tank, expected_result=0)
    edcr.test_get_where_label_is_l(pred=True, test=False, l=l_Iskander, expected_result=0)
    edcr.test_get_where_label_is_l(pred=True, test=False, l=l_SPA, expected_result=1)
    edcr.test_get_where_label_is_l(pred=True, test=False, l=l_2S19_MSTA, expected_result=1)
    edcr.test_get_where_label_is_l(pred=False, test=False, l=l_Tank, expected_result=0)
    edcr.test_get_where_label_is_l(pred=False, test=False, l=l_Iskander, expected_result=0)
    edcr.test_get_where_label_is_l(pred=False, test=False, l=l_SPA, expected_result=1)
    edcr.test_get_where_label_is_l(pred=False, test=False, l=l_2S19_MSTA, expected_result=1)

    edcr.test_get_where_label_is_l(pred=True, test=True, l=l_Tank,
                                   expected_result=np.array([1] + [0] * 5 + [1] + [0] * 3))
    edcr.test_get_where_label_is_l(pred=True, test=True, l=l_Iskander, expected_result=0)
    edcr.test_get_where_label_is_l(pred=True, test=True, l=l_SPA,
                                   expected_result=np.array([0] + [1] * 5 + [0] + [1] * 3))
    edcr.test_get_where_label_is_l(pred=True, test=True, l=l_2S19_MSTA,
                                   expected_result=np.array([0] + [1] * 5 + [0] * 2 + [1] * 2))
    edcr.test_get_where_label_is_l(pred=False, test=True, l=l_Tank, expected_result=0)
    edcr.test_get_where_label_is_l(pred=False, test=True, l=l_Iskander, expected_result=0)
    edcr.test_get_where_label_is_l(pred=False, test=True, l=l_SPA, expected_result=1)
    edcr.test_get_where_label_is_l(pred=False, test=True, l=l_2S19_MSTA, expected_result=1)

    K = 12
    edcr = EDCR.test(epsilon=0.1, K=K)
    edcr.test_get_where_label_is_l(pred=True, test=False, l=l_Tank, expected_result=0)
    edcr.test_get_where_label_is_l(pred=True, test=False, l=l_Iskander, expected_result=0)
    edcr.test_get_where_label_is_l(pred=True, test=False, l=l_SPA, expected_result=np.array([1] * 11 + [0]))
    edcr.test_get_where_label_is_l(pred=True, test=False, l=l_2S19_MSTA, expected_result=np.array([1] * 11 + [0]))
    edcr.test_get_where_label_is_l(pred=False, test=False, l=l_Tank, expected_result=0)
    edcr.test_get_where_label_is_l(pred=False, test=False, l=l_Iskander, expected_result=0)
    edcr.test_get_where_label_is_l(pred=False, test=False, l=l_SPA, expected_result=1)
    edcr.test_get_where_label_is_l(pred=False, test=False, l=l_2S19_MSTA, expected_result=1)

    # edcr.test_get_where_label_is_l(pred=True, test=True, l=l_Tank,
    #                                expected_result=np.array([1] + [0] * 5 + [1] + [0] * 3))
    # edcr.test_get_where_label_is_l(pred=True, test=True, l=l_Iskander, expected_result=0)
    # edcr.test_get_where_label_is_l(pred=True, test=True, l=l_SPA,
    #                                expected_result=np.array([0] + [1] * 5 + [0] + [1] * 3))
    # edcr.test_get_where_label_is_l(pred=True, test=True, l=l_2S19_MSTA,
    #                                expected_result=np.array([0] + [1] * 5 + [0] * 2 + [1] * 2))
    # edcr.test_get_where_label_is_l(pred=False, test=True, l=l_Tank, expected_result=0)
    # edcr.test_get_where_label_is_l(pred=False, test=True, l=l_Iskander, expected_result=0)
    # edcr.test_get_where_label_is_l(pred=False, test=True, l=l_SPA, expected_result=1)
    # edcr.test_get_where_label_is_l(pred=False, test=True, l=l_2S19_MSTA, expected_result=1)


if __name__ == '__main__':
    run_tests()
