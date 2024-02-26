from test_PYEDCR.test import *

method_str = 'get_l_correction_rule_theoretical_precision_increase'

def run_test_1():
    K = [(0, 9)]
    test = Test(epsilon=0.1, K_train=K, K_test=None, method_str=method_str)

    test.edcr.set_error_correction_rules({l_BTR_80: {(pred_BTR_80, l_BTR_80)},

                                          })

    test.run(l=l_BTR_80, expected_output=0)

if __name__ == '__main__':
    run_test_1()