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

K_train_slice = [(1, 10), (400, 410)]
K_test_slice = [(1, 10), (50, 60)]

edcr = EDCR.test(epsilon=0.1,
                 K_train=K_train_slice,
                 K_test=K_test_slice,
                 print_pred_and_true=True)
edcr.print_metrics(test=False, prior=True)
edcr.print_metrics(test=True, prior=True)

# get label 
label_fine = data_preprocessing.get_labels(data_preprocessing.granularities['fine'])
label_coarse = data_preprocessing.get_labels(data_preprocessing.granularities['coarse'])

print(utils.blue_text("=" * 50 + "test get_where_label_is_l" + "=" * 50))

# Test 1

label_30N6E = label_fine['2S19_MSTA']

print(f'label is: {label_30N6E}, granularity: {label_30N6E.g}, label_index: {label_30N6E.index}')

except_result_1 = 9

edcr.test_how_many_predicted_l(test=False,
                               l=label_30N6E,
                               expected_result=except_result_1)

# Test 2

label_30N6E = label_fine['30N6E']

print(f'label is: {label_30N6E}, granularity: {label_30N6E.g}, label_index: {label_30N6E.index}')

except_result_2 = 8

edcr.test_how_many_predicted_l(test=False,
                               l=label_30N6E,
                               expected_result=except_result_2)

print("Get_how_many_predicted_l method passed!")
