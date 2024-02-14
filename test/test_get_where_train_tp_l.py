import numpy as np

from test import *


def run_tests():
    K = 10
    edcr = EDCR.test(epsilon=0.1, K=K)
    edcr.test_get_where_train_tp_l(l=l_2S19_MSTA, expected_result=1)
    K = 12
    edcr = EDCR.test(epsilon=0.1, K=K)
    edcr.test_get_where_train_tp_l(l=l_2S19_MSTA, expected_result=np.array([1]*11 + [0]))
    K = 20
    edcr = EDCR.test(epsilon=0.1, K=K)
    edcr.test_get_where_train_tp_l(l=l_2S19_MSTA, expected_result=np.array([1] * 11 + [0, 1, 1] * 3))


if __name__ == '__main__':
    run_tests()
