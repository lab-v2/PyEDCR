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

class TestApplyCorrectionRules(Test):
    def __init__(self,
                 epsilon: float,
                 K_train: list[(int, int)],
                 K_test: list[(int, int)],
                 ):
        method_str = 'apply_correction_rules'
        super().__init__(epsilon, method_str, K_train, K_test)

    def set_rules(self,
                  rule_data: typing.Dict[data_preprocessing.Label, {(EDCR._Condition, data_preprocessing.Label)}]):
        self.edcr.set_error_correction_rules(rule_data)

    def set_rules_for_edge_case_coarse(self):
        all_condition_class_pairs = {}
        for label in cg_l:
            all_condition_class_pairs[label] = set()
        for label in cg_l:
            for condition in [pred_conditions[l] for l in fg_l]:
                all_condition_class_pairs[label] = all_condition_class_pairs[label].union({(condition, label)})

        self.edcr.set_error_correction_rules(all_condition_class_pairs)

    def set_rules_for_edge_case_fine(self):
        all_condition_class_pairs = {}
        for label in fg_l:
            all_condition_class_pairs[label] = set()
        for label in fg_l:
            for condition in [pred_conditions[l] for l in cg_l]:
                all_condition_class_pairs[label] = all_condition_class_pairs[label].union({(condition, label)})

        self.edcr.set_error_correction_rules(all_condition_class_pairs)

    def print_examples(self,
                       test: bool = True):
        # for padding:
        N = 40
        K = self.edcr.K_test
        T = self.edcr.T_test
        source_str = 'test'
        old_data = self.edcr.original_test_pred_data.values()
        new_data = self.edcr.test_pred_data.values()

        print(f'\nTaking {len(K)} / {T} {source_str} examples\n')
        for i, (fine_prediction_index, coarse_prediction_index,
                fine_prediction_index_new, coarse_prediction_index_new,
                fine_gt_index, coarse_gt_index) \
                in enumerate(zip(*list(old_data),
                                 *list(new_data),
                                 *data_preprocessing.get_ground_truths(test=True, K=K))):
            # Calculate padding to ensure each column reaches the desired width
            pred_old_padding = N - len(
                f'pred_old: {(fg_str[fine_prediction_index], cg_str[coarse_prediction_index])}')
            pred_new_padding = N - len(
                f'pred_new: {(fg_str[fine_prediction_index_new], cg_str[coarse_prediction_index_new])}')
            true_padding = N - len(f'true: {(fg_str[fine_gt_index], cg_str[coarse_gt_index])}')

            if pred_new_padding != pred_old_padding:
                print(utils.blue_text(
                    f'pred_old:{(fg_str[fine_prediction_index], cg_str[coarse_prediction_index])}'
                    + ' ' * pred_old_padding +
                    f'pred_new:{(fg_str[fine_prediction_index_new], cg_str[coarse_prediction_index_new])}'
                    + ' ' * pred_new_padding +
                    f'true:{(fg_str[fine_gt_index], cg_str[coarse_gt_index])}' + ' ' * true_padding
                    + str(i)))
            else:
                print(
                    f'pred_old:{(fg_str[fine_prediction_index], cg_str[coarse_prediction_index])}'
                    + ' ' * pred_old_padding +
                    f'pred_new:{(fg_str[fine_prediction_index_new], cg_str[coarse_prediction_index_new])}'
                    + ' ' * pred_new_padding +
                    f'true:{(fg_str[fine_gt_index], cg_str[coarse_gt_index])}' + ' ' * true_padding
                    + str(i))

    def run(self,
            expected_result,
            *method_args,
            **method_kwargs):
        output = self.method(*method_args, **method_kwargs)
        test_passed = np.all(output == expected_result) if isinstance(output, np.ndarray) \
            else output == expected_result

        if test_passed:
            print(utils.green_text(f'Test passed!'))
            self.print_examples()
            # print(f'Expected:\n{expected_output}')
            # print(f'Actual:\n{output}')
        else:
            print(utils.red_text('Test failed!'))
            self.print_examples()
            print(f'Expected:\n{expected_result}')
            print(f'Actual:\n{output}')


def run_test_1(
        error_correction_rules_1: dict[data_preprocessing.Label, set[(EDCR._Condition, data_preprocessing.Label)]],
        expected_output_1: np.array):
    K_train_slice = [(1, 10), (400, 410)]
    K_test_slice = [(1250, 1259), (1310, 1319), (1450, 1459), (1500, 1509)]

    # Test 1
    test_1 = TestApplyCorrectionRules(epsilon=0.1,
                                      K_train=K_train_slice,
                                      K_test=K_test_slice)

    test_1.set_rules(error_correction_rules_1)
    test_1.run(expected_result=expected_output_1,
               g=g_coarse)


def run_edge_case_1():
    K_train_slice = [(1, 10), (400, 410)]
    K_test_slice = [(1250, 1259), (1310, 1319), (1450, 1459), (1500, 1509)]

    # Test 1
    test_1 = TestApplyCorrectionRules(epsilon=0.1,
                                      K_train=K_train_slice,
                                      K_test=K_test_slice)
    expected_output_1 = np.array([6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                                  6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                                  6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                                  6, 6, 6, 6, 6, 6, 6, 6, 6, 6, ])

    test_1.set_rules_for_edge_case_coarse()
    test_1.run(expected_result=expected_output_1,
               g=g_coarse)


def run_test_2(
        error_correction_rules_2: dict[data_preprocessing.Label, set[(EDCR._Condition, data_preprocessing.Label)]],
        expected_output_2: np.array):

    K_train_slice = [(1, 10), (400, 410)]
    K_test_slice = [(600, 609), (720, 729), (800, 809)]

    # Test 1
    test_2 = TestApplyCorrectionRules(epsilon=0.1,
                                      K_train=K_train_slice,
                                      K_test=K_test_slice)

    test_2.set_rules(error_correction_rules_2)
    test_2.run(expected_result=expected_output_2,
               g=g_coarse)


def run_edge_case_2():
    K_train_slice = [(1, 10), (400, 410)]
    K_test_slice = [(600, 609), (720, 729), (800, 809)]

    # Test 1
    test_1 = TestApplyCorrectionRules(epsilon=0.1,
                                      K_train=K_train_slice,
                                      K_test=K_test_slice)
    expected_output_1 = np.array([6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                                  6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                                  6, 6, 6, 6, 6, 6, 6, 6, 6, 6, ])

    test_1.set_rules_for_edge_case_coarse()
    test_1.run(expected_result=expected_output_1,
               g=g_coarse)


if __name__ == '__main__':
    # # Test 1:
    # error_correction_rules = {l_Air_Defense: {(pred_2S19_MSTA, l_Tank)}}
    # expected_output = np.array([6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    #                             6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    #                             6, 6, 6, 6, 6, 6, 6, 6, 6, 0,
    #                             6, 6, 6, 6, 6, 6, 6, 6, 6, 6, ])
    #
    # run_test_1(error_correction_rules_1=error_correction_rules,
    #            expected_output_1=expected_output)
    #
    # error_correction_rules = {l_Air_Defense: {(pred_2S19_MSTA, l_Tank)},
    #                           l_BTR: {(pred_2S19_MSTA, l_Tank)}}
    # expected_output = np.array([6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    #                             6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    #                             6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    #                             6, 6, 6, 6, 6, 6, 6, 6, 6, 6, ])
    #
    # run_test_1(error_correction_rules_1=error_correction_rules,
    #            expected_output_1=expected_output)
    #
    # error_correction_rules = {l_Air_Defense: {(pred_2S19_MSTA, l_Tank),
    #                                           (pred_T_14, l_Tank)},
    #                           }
    # expected_output = np.array([6, 6, 0, 0, 0, 0, 0, 0, 0, 0,
    #                             6, 6, 6, 6, 6, 6, 6, 6, 6, 0,
    #                             6, 6, 6, 6, 6, 6, 6, 6, 6, 0,
    #                             6, 0, 6, 6, 6, 6, 6, 6, 6, 6, ])
    #
    # run_test_1(error_correction_rules_1=error_correction_rules,
    #            expected_output_1=expected_output)
    #
    # error_correction_rules = {l_Air_Defense: {(pred_2S19_MSTA, l_Tank),
    #                                           (pred_T_14, l_Tank)},
    #                           l_Tank: {(pred_2S19_MSTA, l_Tank),
    #                                    (pred_T_14, l_Tank)},
    #                           l_BMP: {(pred_2S19_MSTA, l_Tank),
    #                                   (pred_T_14, l_Tank)}
    #                           }
    # expected_output = np.array([6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    #                             6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    #                             6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    #                             6, 6, 6, 6, 6, 6, 6, 6, 6, 6, ])
    #
    # run_test_1(error_correction_rules_1=error_correction_rules,
    #            expected_output_1=expected_output)

    # Edge case test 1

    # run_edge_case_1()

    # Test 2:
    error_correction_rules = {}
    expected_output = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                3, 6, 3, 3, 3, 2, 3, 3, 3, 2,
                                2, 3, 3, 3, 3, 3, 3, 3, 2, 3, ])

    run_test_2(error_correction_rules_2=error_correction_rules,
               expected_output_2=expected_output)

