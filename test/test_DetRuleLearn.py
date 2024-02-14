from test import *


def run_test_1():
    K = 10
    edcr = EDCR.test(epsilon=0.1, K=K, print_pred_and_true=True)

    edcr.test_DetRuleLearn(l=l_SPA, expected_result=set())


if __name__ == '__main__':
    run_test_1()
