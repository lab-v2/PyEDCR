import numpy as np

from test import *


def run_test_1():
    K = 10
    edcr = EDCR.test(epsilon=0.1, K=K, print_pred_and_true=True)
    train_pred_fine_data, train_pred_coarse_data = edcr.get_predictions(test=False)

    edcr.test_get_where_any_conditions_satisfied(C={pred_Tornado, pred_BMP_1},
                                                 fine_data=train_pred_fine_data,
                                                 coarse_data=train_pred_coarse_data,
                                                 expected_result=np.zeros_like(train_pred_fine_data))

    edcr.test_get_where_any_conditions_satisfied(C={pred_Tornado, pred_BMP_1, pred_2S19_MSTA},
                                                 fine_data=train_pred_fine_data,
                                                 coarse_data=train_pred_coarse_data,
                                                 expected_result=np.ones_like(train_pred_fine_data))

    edcr.test_get_where_any_conditions_satisfied(C={pred_Tornado, pred_BMP_1, pred_SPA},
                                                 fine_data=train_pred_fine_data,
                                                 coarse_data=train_pred_coarse_data,
                                                 expected_result=np.ones_like(train_pred_fine_data))


if __name__ == '__main__':
    run_test_1()
