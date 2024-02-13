import itertools

from test import *


# test case 1
# K = 10
# edcr = EDCR.test(epsilon=0.1, K=K, print_pred_and_true=True)
#
# CC_l = {(pred_Tornado, l_SPA), (pred_BMP_1, l_BMP)}
# edcr.test_BOD_l(l=l_SPA, CC=CC_l, expected_result=0)
# edcr.test_CON_l(l=l_SPA, CC=CC_l, expected_result=0)
# edcr.test_BOD_l(l=l_BMP, CC=CC_l, expected_result=0)
# edcr.test_CON_l(l=l_BMP, CC=CC_l, expected_result=0)
#
# CC_l = {(pred_Tornado, l_SPA), (pred_BMP_1, l_BMP), (pred_2S19_MSTA, l_SPA)}
# edcr.test_BOD_l(l=l_SPA, CC=CC_l, expected_result=10)
# edcr.test_CON_l(l=l_SPA, CC=CC_l, expected_result=1)
# edcr.test_BOD_l(l=l_BMP, CC=CC_l, expected_result=10)
# edcr.test_CON_l(l=l_BMP, CC=CC_l, expected_result=0)
#
# CC_l = set(itertools.product(pred_conditions.values(), fg_l + cg_l))
# edcr.test_BOD_l(l=l_SPA, CC=CC_l, expected_result=K)
# edcr.test_CON_l(l=l_SPA, CC=CC_l, expected_result=1)
# edcr.test_BOD_l(l=l_BMP, CC=CC_l, expected_result=K)
# edcr.test_CON_l(l=l_BMP, CC=CC_l, expected_result=0)

# K = 12
# edcr = EDCR.test(epsilon=0.1, K=K, print_pred_and_true=True)
#
# CC_l = {(pred_Tornado, l_SPA), (pred_BMP_1, l_BMP)}
# edcr.test_BOD_l(l=l_SPA, CC=CC_l, expected_result=0)
# edcr.test_CON_l(l=l_SPA, CC=CC_l, expected_result=0)
#
# CC_l = {(pred_Tornado, l_SPA), (pred_BMP_1, l_BMP), (pred_2S19_MSTA, l_SPA)}
# edcr.test_BOD_l(l=l_SPA, CC=CC_l, expected_result=11)
# edcr.test_CON_l(l=l_SPA, CC=CC_l, expected_result=1)
# edcr.test_BOD_l(l=l_BMP, CC=CC_l, expected_result=11)
# edcr.test_CON_l(l=l_BMP, CC=CC_l, expected_result=0)
#
# CC_l = {(pred_Tornado, l_SPA), (pred_BMP_1, l_BMP), (pred_2S19_MSTA, l_SPA), (pred_BTR_70, l_BTR)}
# edcr.test_BOD_l(l=l_SPA, CC=CC_l, expected_result=12)
# edcr.test_CON_l(l=l_SPA, CC=CC_l, expected_result=1)
# edcr.test_BOD_l(l=l_BMP, CC=CC_l, expected_result=12)
# edcr.test_CON_l(l=l_BMP, CC=CC_l, expected_result=0)
# #
# CC_l = set(itertools.product(pred_conditions.values(), fg_l + cg_l))
# edcr.test_BOD_l(l=l_SPA, CC=CC_l, expected_result=K)
# edcr.test_CON_l(l=l_SPA, CC=CC_l, expected_result=1)
# edcr.test_BOD_l(l=l_BMP, CC=CC_l, expected_result=K)
# edcr.test_CON_l(l=l_BMP, CC=CC_l, expected_result=0)

K = 20
edcr = EDCR.test(epsilon=0.1, K=K, print_pred_and_true=True)

CC_l = {(pred_Tornado, l_SPA), (pred_BMP_1, l_BMP)}
edcr.test_BOD_l(l=l_SPA, CC=CC_l, expected_result=0)
edcr.test_CON_l(l=l_SPA, CC=CC_l, expected_result=0)

CC_l = {(pred_Tornado, l_SPA), (pred_BMP_1, l_BMP), (pred_2S19_MSTA, l_SPA)}
edcr.test_BOD_l(l=l_SPA, CC=CC_l, expected_result=17)
edcr.test_CON_l(l=l_SPA, CC=CC_l, expected_result=1)

CC_l = {(pred_Tornado, l_SPA), (pred_BMP_1, l_BMP), (pred_2S19_MSTA, l_SPA), (pred_BTR_70, l_BTR)}
edcr.test_BOD_l(l=l_SPA, CC=CC_l, expected_result=18)
edcr.test_CON_l(l=l_SPA, CC=CC_l, expected_result=1)

CC_l = {(pred_Tornado, l_SPA), (pred_BMP_1, l_BMP), (pred_2S19_MSTA, l_SPA), (pred_BTR_70, l_BTR),
        (pred_RS_24, l_Air_Defense)}
edcr.test_BOD_l(l=l_SPA, CC=CC_l, expected_result=19)
edcr.test_CON_l(l=l_SPA, CC=CC_l, expected_result=1)

CC_l = {(pred_Tornado, l_SPA), (pred_BMP_1, l_BMP), (pred_2S19_MSTA, l_SPA), (pred_BTR_70, l_BTR),
        (pred_RS_24, l_Air_Defense), (pred_T_64, l_Tank)}
edcr.test_BOD_l(l=l_SPA, CC=CC_l, expected_result=20)
edcr.test_CON_l(l=l_SPA, CC=CC_l, expected_result=1)

CC_l = set(itertools.product(pred_conditions.values(), fg_l + cg_l))
edcr.test_BOD_l(l=l_SPA, CC=CC_l, expected_result=K)
edcr.test_CON_l(l=l_SPA, CC=CC_l, expected_result=1)
edcr.test_BOD_l(l=l_BMP, CC=CC_l, expected_result=K)
edcr.test_CON_l(l=l_BMP, CC=CC_l, expected_result=0)
