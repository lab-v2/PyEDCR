import numpy as np

from test import *


def run_test_1():
    K = 10
    edcr = EDCR.test(epsilon=0.1, K=K, print_pred_and_true=True)
    # train_pred_fine_data, train_pred_coarse_data = edcr.get_predictions(test=False)

    edcr.test_get_where_label_is_l(pred=True, test=False, l=l_Tank, expected_result=0)
    edcr.test_get_where_label_is_l(pred=True, test=False, l=l_SPA, expected_result=1)


if __name__ == '__main__':
    run_test_1()
