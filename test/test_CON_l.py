import itertools

import data_preprocessing
from PyEDCR import EDCR

g_fine, g_coarse = data_preprocessing.granularities.values()
fg_l, cg_l = list(data_preprocessing.fine_grain_labels.values()), list(data_preprocessing.coarse_grain_labels.values())
pred_conditions = {l: EDCR.PredCondition(l=l) for l in fg_l + cg_l}

l_Air_Defense, l_BMD, l_BMP, l_BTR, l_MT_LB, l_SPA, l_Tank = cg_l

(pred_2S19_MSTA, pred_30N6E, pred_BM_30, pred_BMD, pred_BMP_1, pred_BMP_2, pred_BMP_T15, pred_BRDM, pred_BTR_60,
 pred_BTR_70, pred_BTR_80, pred_D_30, pred_Iskander, pred_MT_LB, pred_Pantsir_S1, pred_RS_24, pred_T_14, pred_T_62,
 pred_T_64, pred_T_72, pred_T_80, pred_T_90, pred_TOS_1, pred_Tornado) = [pred_conditions[l] for l in fg_l]


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
#
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
