from test import *


def run_test_1():
    K = 10
    edcr = EDCR.test(epsilon=0.1, K=K, print_pred_and_true=True)
    C_l = {pred_Tornado, pred_BMP_1}

    edcr.test_get_NEG_l_C(l=l_SPA, C=C_l, expected_result=0)


if __name__ == '__main__':
    run_test_1()
