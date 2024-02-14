from test import *


def run_tests():
    K = 10
    edcr = EDCR.test(epsilon=0.1, K=K, print_pred_and_true=True)

    edcr.test_get_where_train_tp_l(l=l_2S19_MSTA, expected_result=1, print_result=True)


if __name__ == '__main__':
    run_tests()
