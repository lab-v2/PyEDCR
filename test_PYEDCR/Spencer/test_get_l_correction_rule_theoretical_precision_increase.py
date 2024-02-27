from test_PYEDCR.test import *

method_str = 'get_l_correction_rule_theoretical_precision_increase'

def run_test_1():
    K = [(0, 9)]
    test = Test(epsilon=0.1, K_train=K, K_test=None, method_str=method_str)

    test.edcr.set_error_correction_rules({l_BTR_80: {(pred_BTR_80, l_BTR)},
                                          l_2S19_MSTA: {(pred_2S19_MSTA, l_SPA), (pred_T_72, l_SPA)}
                                          })

    test.print_examples(test=False)

    test.run(l=l_BTR_80, expected_output=0)
    test.run(l=l_2S19_MSTA, expected_output=0)

def run_test_2():
    K = [(2975, 2979), (1455, 1459), (3000, 3000), (3190, 3191), (3010, 3012)]
    test = Test(epsilon=0.1, K_train=K, K_test=None, method_str=method_str)

    test.edcr.set_error_correction_rules({l_BTR_80: {(pred_BTR_80, l_BTR)},
                                          l_2S19_MSTA: {(pred_2S19_MSTA, l_SPA), (pred_T_72, l_SPA)}
                                          })

    test.run(l=l_BTR_80, expected_output=1/15)

    test.print_examples(test=False)

def run_test_3():
    K = [(5219, 5219), (5250, 5254), (5270, 5281), (1829, 1831)]
    test = Test(epsilon=0.1, K_train=K, K_test=None, method_str=method_str)

    test.edcr.set_error_correction_rules({l_BMP_T15: {(pred_BMP_T15, l_BMP)}
                                          })

    test.run(l=l_BMP_T15, expected_output=2/7)

    test.print_examples(test=False)


if __name__ == '__main__':
    # run_test_1()
    # run_test_2()
    run_test_3()