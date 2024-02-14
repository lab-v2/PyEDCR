import itertools

from test import *

def run_test_1():
    K = 10
    edcr = EDCR.test(epsilon=0.1, K=K, print_pred_and_true=True)
    print('hello')

    l = l_SPA
    C = {pred_Tornado, pred_BMP_1}
    edcr.test_get_POS_l_C(l=l, C=C, expected_result=1)

if __name__ == '__main__':
    run_test_1()