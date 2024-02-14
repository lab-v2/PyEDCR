from PyEDCR import EDCR
import data_preprocessing
import warnings
import numpy as np
import utils

# This will silence all warnings, including ones unrelated to your evaluation.
# Use this approach with caution!
warnings.filterwarnings('ignore')

# This is index for train fine true data:
# 2S19_MSTA: [0, 343]     30N6E: [344, 462]       BM-30: [463, 723]       BMD: [724, 1045]
# BMP-1: [1046, 1438]     BMP-2: [1439, 1828]     BMP-T15: [1829, 2169]   BRDM: [2170, 2558]
# BTR-60: [2559, 2949]    BTR-70: [2950, 3189]    BTR-80: [3190, 3602]    D-30: [3603, 3963]
# Iskander: [3964, 4215]  MT_LB: [4216, 4587]     Pantsir-S1: [4588, 4866]
# Rs-24: [4867, 5193]     T-14: [5194, 5525]      T-62: [5526, 5806]      T-64: [5807, 6181]
# T-72: [6182, 6605]      T-80: [6606, 6904]      T-90: [6905, 7198]      TOS-1: [7199, 7496]
# Tornado: [7497, 7822]

# This is index for test fine true data:
# 2S19_MSTA: [0, 58]      30N6E: [59, 79]     BM-30: [80, 219]        BMD: [220, 308]
# BMP-1: [309, 335]       BMP-2: [336, 400]       BMP-T15: [401, 484]     BRDM: [485, 548]
# BTR-60: [549, 709]      BTR-70: [710, 748]      BTR-80: [749, 824]      D-30: [825, 894]
# Iskander: [895, 999]    MT_LB: [1000, 1086]     Pantsir-S1: [1087, 1130]
# Rs-24: [1131, 1243]     T-14: [1244, 1303]      T-62: [1304, 1384]      T-64: [1385, 1410]
# T-72: [1411, 1443]      T-80: [1444, 1499]      T-90: [1500, 1556]      TOS-1: [1557, 1590]
# Tornado: [1591, 1620]

# Coarse_to_fine dictionary
# 'Air Defense': ['30N6E', 'Iskander', 'Pantsir-S1', 'Rs-24'],
# 'BMP': ['BMP-1', 'BMP-2', 'BMP-T15'],
# 'BTR': ['BRDM', 'BTR-60', 'BTR-70', 'BTR-80'],
# 'Tank': ['T-14', 'T-62', 'T-64', 'T-72', 'T-80', 'T-90'],
# 'Self Propelled Artillery': ['2S19_MSTA', 'BM-30', 'D-30', 'Tornado', 'TOS-1'],
# 'BMD': ['BMD'],
# 'MT_LB': ['MT_LB']

K_train_slice = [(5200, 5210), (5530, 5540)]
K_test_slice = [(1, 10), (50, 60)]

edcr = EDCR.test(epsilon=0.1,
                 K_train=K_train_slice,
                 K_test=K_test_slice,
                 print_pred_and_true=True)

# get label
label_fine = data_preprocessing.get_labels(data_preprocessing.granularities['fine'])
label_coarse = data_preprocessing.get_labels(data_preprocessing.granularities['coarse'])

# method name
method_str = "Get_POS_l_C"

print(utils.blue_text("=" * 50 + "test " + method_str + "=" * 50))

g_fine, g_coarse = data_preprocessing.granularities.values()
fg_l, cg_l = list(data_preprocessing.fine_grain_labels.values()), list(data_preprocessing.coarse_grain_labels.values())
pred_conditions = {l: EDCR.PredCondition(l=l) for l in fg_l + cg_l}

l_Air_Defense, l_BMD, l_BMP, l_BTR, l_MT_LB, l_SPA, l_Tank = cg_l


(pred_2S19_MSTA, pred_30N6E, pred_BM_30, pred_BMD, pred_BMP_1, pred_BMP_2, pred_BMP_T15, pred_BRDM, pred_BTR_60,
 pred_BTR_70, pred_BTR_80, pred_D_30, pred_Iskander, pred_MT_LB, pred_Pantsir_S1, pred_RS_24, pred_T_14, pred_T_62,
 pred_T_64, pred_T_72, pred_T_80, pred_T_90, pred_TOS_1, pred_Tornado) = [pred_conditions[l] for l in fg_l]


# Test 1: No condition is satisfied

CC = {(pred_BM_30, l_BMD), (pred_30N6E, l_Air_Defense)}

except_result_1 = 0

edcr.test_get_BOD_CC(CC=CC,
                     expected_result=except_result_1)

# Test 2: all example satisfy 2 condition

CC = {(pred_RS_24, l_Tank), (pred_T_62, l_BMD)}

except_result_1 = 2

edcr.test_get_BOD_CC(CC=CC,
                     expected_result=except_result_1)

print(f"{method_str} method passed!")
