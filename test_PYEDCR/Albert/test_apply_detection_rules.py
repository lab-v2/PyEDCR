import data_processing
import warnings
import numpy as np
import utils
from test_PYEDCR.test import *
from PyEDCR import EDCR
import typing

# This will silence all warnings, including ones unrelated to your evaluation.
# Use this approach with caution
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

class TestApplyDetectionRules(Test):
    def __init__(self,
                 epsilon: float,
                 method_str: str,
                 K_train: list[(int, int)],
                 K_test: list[(int, int)],
                 ):
        super().__init__(epsilon, method_str, K_train, K_test)

    def set_rules(self,
                  rule_data: typing.Dict[data_preprocessing.Label, {EDCR._Condition}]):
        self.edcr.set_error_detection_rules(rule_data)


if __name__ == '__main__':
    method_str = 'apply_detection_rules'

    K_train_slice = [(1, 10), (400, 410)]
    K_test_slice = [(1, 10), (50, 60)]

    # Test 1
    test = TestApplyDetectionRules(epsilon=0.1,
                                   method_str=method_str,
                                   K_train=K_train_slice,
                                   K_test=K_test_slice)

    error_detection_rules_1 = {l_Tank: {pred_2S19_MSTA, pred_30N6E}}
    expected_output = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    test.set_rules(error_detection_rules_1)
    test.run(expected_output=expected_output,
             g=g_coarse)

    # Test 2
    test = TestApplyDetectionRules(epsilon=0.1,
                                   method_str=method_str,
                                   K_train=K_train_slice,
                                   K_test=K_test_slice)

    error_detection_rules_2 = {l_Air_Defense: {pred_2S19_MSTA, pred_30N6E}}
    expected_output = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1])

    test.set_rules(error_detection_rules_2)
    test.run(expected_output=expected_output,
             g=g_coarse)

    # Test 3
    test = TestApplyDetectionRules(epsilon=0.1,
                                   method_str=method_str,
                                   K_train=K_train_slice,
                                   K_test=K_test_slice)
    error_detection_rules_3 = {l_Air_Defense: {pred_2S19_MSTA, pred_30N6E},
                               l_Tank: {pred_2S19_MSTA, pred_30N6E}}
    expected_output = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1])

    test.set_rules(error_detection_rules_3)
    test.run(expected_output=expected_output,
             g=g_coarse)
