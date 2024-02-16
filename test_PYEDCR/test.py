import warnings

import numpy as np

warnings.filterwarnings('ignore')

import utils
import data_preprocessing
from PyEDCR import EDCR

g_fine, g_coarse = data_preprocessing.granularities.values()
fg_l, cg_l = list(data_preprocessing.fine_grain_labels.values()), list(data_preprocessing.coarse_grain_labels.values())
pred_conditions = {l: EDCR.PredCondition(l=l) for l in fg_l + cg_l}

pred_Air_Defense, pred_BMD_coarse, pred_BMP, pred_BTR, pred_MT_LB_coarse, pred_SPA, pred_Tank = \
    [pred_conditions[l] for l in cg_l]
l_Air_Defense, l_BMD_coarse, l_BMP, l_BTR, l_MT_LB_coarse, l_SPA, l_Tank = cg_l

(pred_2S19_MSTA, pred_30N6E, pred_BM_30, pred_BMD_fine, pred_BMP_1, pred_BMP_2, pred_BMP_T15, pred_BRDM, pred_BTR_60,
 pred_BTR_70, pred_BTR_80, pred_D_30, pred_Iskander, pred_MT_LB_fine, pred_Pantsir_S1, pred_RS_24, pred_T_14, pred_T_62,
 pred_T_64, pred_T_72, pred_T_80, pred_T_90, pred_TOS_1, pred_Tornado) = [pred_conditions[l] for l in fg_l]

(l_2S19_MSTA, l_30N6E, l_BM_30, l_BMD_fine, l_BMP_1, l_BMP_2, l_BMP_T15, l_BRDM, l_BTR_60, l_BTR_70, l_BTR_80, l_D_30,
 l_Iskander, l_MT_LB_fine, l_Pantsir_S1, l_RS_24, l_T_14, l_T_62, l_T_64, l_T_72, l_T_80, l_T_90, l_TOS_1, l_Tornado) \
    = fg_l

consistency_constraint = EDCR.ConsistencyCondition()


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


class Test:
    def __init__(self,
                 epsilon: float,
                 K_train: list[(int, int)] = None,
                 K_test: list[(int, int)] = None,
                 print_pred_and_true: bool = False):
        self.edcr = EDCR(main_model_name='vit_b_16',
                         combined=True,
                         loss='BCE',
                         lr=0.0001,
                         num_epochs=20,
                         epsilon=epsilon,
                         K_train=K_train,
                         K_test=K_test)
        if print_pred_and_true:
            fg = data_preprocessing.fine_grain_classes_str
            cg = data_preprocessing.coarse_grain_classes_str

            if K_train is not None:
                print(f'\nTaking {len(self.edcr.K_train)} / {self.edcr.T_train} train examples\n' +
                      '\n'.join([(
                        f'pred: {(fg[fine_prediction_index], cg[coarse_prediction_index])}, '
                        f'true: {(fg[fine_gt_index], cg[coarse_gt_index])}')
                        for fine_prediction_index, coarse_prediction_index, fine_gt_index, coarse_gt_index
                        in zip(*list(self.edcr.train_pred_data.values()),
                               *data_preprocessing.get_ground_truths(test=False, K=self.edcr.K_train))]))

            if K_test is not None:
                print(f'\nTaking {len(self.edcr.K_test)} / {self.edcr.T_test} test examples\n' +
                      '\n'.join([(
                        f'pred: {(fg[fine_prediction_index], cg[coarse_prediction_index])}, '
                        f'true: {(fg[fine_gt_index], cg[coarse_gt_index])}')
                        for fine_prediction_index, coarse_prediction_index, fine_gt_index, coarse_gt_index
                        in zip(*list(self.edcr.test_pred_data.values()),
                               *data_preprocessing.get_ground_truths(test=True, K=self.edcr.K_test))]))

    def run(self,
            method_str: str,
            expected_output,
            *method_args,
            **method_kwargs):
        method = getattr(self.edcr, method_str)
        output = method(*method_args, **method_kwargs)
        test_passed = np.all(output == expected_output) if isinstance(output, np.ndarray) \
            else output == expected_output

        if test_passed:
            print(utils.green_text(f'Test passed!'))
        else:
            print(utils.red_text('Test failed!'))
            print(f'Expected:\n{expected_output}')
            print(f'Actual:\n{output}')

        assert test_passed
