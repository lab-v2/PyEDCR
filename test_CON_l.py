import itertools

import data_preprocessing
from PyEDCR import EDCR

g_fine, g_coarse = data_preprocessing.granularities.values()
fg_l, cg_l = list(data_preprocessing.fine_grain_labels.values()), list(data_preprocessing.coarse_grain_labels.values())

# test case 1
l_tank, l_SPA, l_BMP = [data_preprocessing.coarse_grain_labels[l_str] for l_str in ['Tank',
                                                                                    'Self Propelled Artillery',
                                                                                    'BMP']]

l_Tornado, l_BMP_1, l_2S19_MSTA = [data_preprocessing.fine_grain_labels[l_str] for l_str in ['Tornado',
                                                                                             'BMP-1',
                                                                                             '2S19_MSTA']]

pred_conditions = {l: EDCR.PredCondition(l=l) for l in fg_l + cg_l}

pred_Tornado, pred_BMP_1, pred_2S19_MSTA = [pred_conditions[l] for l in [l_tank, l_SPA, l_BMP]]

# CC_l = {(pred_Tornado, l_SPA), (pred_BMP_1, l_BMP), (pred_2S19_MSTA, l_SPA)}

edcr = EDCR.test(epsilon=0.1, K=7000, print_pred_and_true=False)
# edcr.test_CON_l(l=l_SPA, CC=CC_l, expected_result=1)

# test case 2
CC_l = set(itertools.product(pred_conditions.values(), fg_l + cg_l))
# print(len(pred_conditions))
assert len(CC_l) == len(pred_conditions) * len(fg_l + cg_l)

edcr.test_CON_l(l=l_SPA, CC=CC_l, expected_result=1)


