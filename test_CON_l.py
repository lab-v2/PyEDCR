import data_preprocessing
from PyEDCR import EDCR

g_fine, g_coarse = data_preprocessing.granularities

# test case 1
l_tank = data_preprocessing.get_labels(g_coarse)[data_preprocessing.coarse_grain_classes_str.index('Tank')]
l_SPA = data_preprocessing.get_labels(g_coarse)[
    data_preprocessing.coarse_grain_classes_str.index('Self Propelled Artillery')]
l_BMP = data_preprocessing.get_labels(g_coarse)[data_preprocessing.coarse_grain_classes_str.index('BMP')]

l_Tornado = data_preprocessing.get_labels(g_fine)[data_preprocessing.fine_grain_classes_str.index('Tornado')]
l_BMP_1 = data_preprocessing.get_labels(g_fine)[data_preprocessing.fine_grain_classes_str.index('BMP-1')]
l_2S19_MSTA = data_preprocessing.get_labels(g_fine)[data_preprocessing.fine_grain_classes_str.index('2S19_MSTA')]

pred_Tornado = EDCR.PredCondition(l=l_Tornado)
pred_BMP_1 = EDCR.PredCondition(l=l_BMP_1)
pred_2S19_MSTA = EDCR.PredCondition(l=l_2S19_MSTA)

CC_l = {(pred_Tornado, l_SPA), (pred_BMP_1, l_BMP), (pred_2S19_MSTA, l_SPA)}

edcr = EDCR.test(epsilon=0.1, K=400, print_pred_and_true=False)
edcr.test_CON_l(l=l_SPA, CC=CC_l, expected_result=1)


