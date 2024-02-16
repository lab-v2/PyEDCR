from test_PYEDCR.test import *


def run_test_1():
    K = 10
    edcr = EDCR.test(epsilon=0.1, K=K, print_pred_and_true=True)

    CC_l = {(pred_Tornado, l_SPA), (pred_BMP_1, l_BMP)}

    edcr.test_BOD_CC(CC=CC_l, expected_result=0)
    edcr.test_CON_l_CC(l=l_SPA, CC=CC_l, expected_result=0)
    edcr.test_CON_l_CC(l=l_SPA, CC=CC_l, expected_result=0)

    CC_l = {(pred_Tornado, l_SPA), (pred_BMP_1, l_BMP), (pred_2S19_MSTA, l_SPA)}
    edcr.test_BOD_CC(CC=CC_l, expected_result=10)
    edcr.test_CON_l_CC(l=l_SPA, CC=CC_l, expected_result=1)
    edcr.test_CON_l_CC(l=l_BMP, CC=CC_l, expected_result=0)

    CC_l = set(itertools.product(pred_conditions.values(), fg_l + cg_l))
    edcr.test_BOD_CC(CC=CC_l, expected_result=K)
    edcr.test_CON_l_CC(l=l_SPA, CC=CC_l, expected_result=1)
    edcr.test_CON_l_CC(l=l_BMP, CC=CC_l, expected_result=0)
