import itertools

from test_PYEDCR.test import *

method_str = 'get_BOD_CC'

class TestGetBODCC(Test):
    def __init__(self,
                 epsilon: float,
                 K_train: list[(int, int)] = None,
                 K_test: list[(int, int)] = None):
        method_str = 'get_BOD_CC'
        super().__init__(epsilon=epsilon, method_str=method_str, K_train=K_train, K_test=K_test)

    def run(self,
            expected_output,
            *method_args,
            **method_kwargs):
        output = self.method(*method_args, **method_kwargs)
        test_passed = True if output[0] == expected_output[0] else False

        if test_passed:
            print(utils.green_text(f"Testing {self.method.__name__} "
                                   f"with args={(','.join(str(o) for o in method_args) if len(method_args) else '')}"
                                   + str({str(k): [str(v_i) for v_i in v] if isinstance(v, typing.Iterable) else str(v)
                                          for k, v in method_kwargs.items()}) +
                                   f" and expected_output={expected_output} has passed!"))
        else:
            print(utils.red_text(f'Test {self.method.__name__} has failed!'))
            self.print_examples()
            print(f'Expected:\n{expected_output}')
            print(f'Actual:\n{output}')

        assert test_passed

def run_test_1():
    K = [(0, 9)]
    test = TestGetBODCC(epsilon=0.1, K_train=K, K_test=None)

    CC_l_1 = {(pred_Tornado, l_SPA), (pred_BMP_1, l_BMP), (pred_BMP_1, l_SPA)}
    CC_l_2 = {(pred_T_72, l_SPA), (pred_BMP_2, l_BMP), (pred_BMP_1, l_SPA)}
    CC_l_3 = {(pred_2S19_MSTA, l_SPA), (pred_BMP_2, l_BMP), (pred_2S19_MSTA, l_Tank)}
    CC_l_4 = {(pred_2S19_MSTA, l_SPA), (pred_BMP_, l_BMP), (pred_2S19_MSTA, l_Tank)}



    case_number = 1
    test.print_examples(test=False)
    print(utils.blue_text("=" * 50 + f"test {case_number} " + method_str + "=" * 50))

    test.run(CC=CC_l_1, expected_output=(0, np.array([0] * 10)))
    test.run(CC=CC_l_2, expected_output=(1, np.array([0] * 10)))
    test.run(CC=CC_l_3, expected_output=(9, np.array([0] * 10)))


    # edcr.test_BOD_CC(CC=CC_l, expected_result=0)
    # edcr.test_CON_l_CC(l=l_SPA, CC=CC_l, expected_result=0)
    # edcr.test_CON_l_CC(l=l_SPA, CC=CC_l, expected_result=0)
    #
    # CC_l = {(pred_Tornado, l_SPA), (pred_BMP_1, l_BMP), (pred_2S19_MSTA, l_SPA)}
    # edcr.test_BOD_CC(CC=CC_l, expected_result=10)
    # edcr.test_CON_l_CC(l=l_SPA, CC=CC_l, expected_result=1)
    # edcr.test_CON_l_CC(l=l_BMP, CC=CC_l, expected_result=0)
    #
    # CC_l = set(itertools.product(pred_conditions.values(), fg_l + cg_l))
    # edcr.test_BOD_CC(CC=CC_l, expected_result=K)
    # edcr.test_CON_l_CC(l=l_SPA, CC=CC_l, expected_result=1)
    # edcr.test_CON_l_CC(l=l_BMP, CC=CC_l, expected_result=0)


def run_test_2():
    K = 12
    edcr = EDCR.test(epsilon=0.1, K=K, print_pred_and_true=True)

    CC_l = {(pred_Tornado, l_SPA), (pred_BMP_1, l_BMP)}
    edcr.test_BOD_CC(CC=CC_l, expected_result=0)
    edcr.test_BOD_CC(CC=CC_l, expected_result=0)

    CC_l = {(pred_Tornado, l_SPA), (pred_BMP_1, l_BMP), (pred_2S19_MSTA, l_SPA)}
    edcr.test_BOD_CC(CC=CC_l, expected_result=11)
    edcr.test_CON_l_CC(l=l_SPA, CC=CC_l, expected_result=1)
    edcr.test_CON_l_CC(l=l_BMP, CC=CC_l, expected_result=0)

    CC_l = {(pred_Tornado, l_SPA), (pred_BMP_1, l_BMP), (pred_2S19_MSTA, l_SPA), (pred_BTR_70, l_BTR)}
    edcr.test_BOD_CC(CC=CC_l, expected_result=12)
    edcr.test_CON_l_CC(l=l_SPA, CC=CC_l, expected_result=1)
    edcr.test_CON_l_CC(l=l_BMP, CC=CC_l, expected_result=0)

    CC_l = set(itertools.product(pred_conditions.values(), fg_l + cg_l))
    edcr.test_BOD_CC(CC=CC_l, expected_result=K)
    edcr.test_CON_l_CC(l=l_SPA, CC=CC_l, expected_result=1)
    edcr.test_CON_l_CC(l=l_BMP, CC=CC_l, expected_result=0)


def run_test_3():
    K = 20
    edcr = EDCR.test(epsilon=0.1, K=K, print_pred_and_true=True)

    CC_l = {(pred_Tornado, l_SPA), (pred_BMP_1, l_BMP)}
    edcr.test_BOD_CC(CC=CC_l, expected_result=0)
    edcr.test_CON_l_CC(l=l_SPA, CC=CC_l, expected_result=0)

    CC_l = {(pred_Tornado, l_SPA), (pred_BMP_1, l_BMP), (pred_2S19_MSTA, l_SPA)}
    edcr.test_BOD_CC(CC=CC_l, expected_result=17)
    edcr.test_CON_l_CC(l=l_SPA, CC=CC_l, expected_result=1)

    CC_l = {(pred_Tornado, l_SPA), (pred_BMP_1, l_BMP), (pred_2S19_MSTA, l_SPA), (pred_BTR_70, l_BTR)}
    edcr.test_BOD_CC(CC=CC_l, expected_result=18)
    edcr.test_CON_l_CC(l=l_SPA, CC=CC_l, expected_result=1)

    CC_l = {(pred_Tornado, l_SPA), (pred_BMP_1, l_BMP), (pred_2S19_MSTA, l_SPA), (pred_BTR_70, l_BTR),
            (pred_RS_24, l_Air_Defense)}
    edcr.test_BOD_CC(CC=CC_l, expected_result=19)
    edcr.test_CON_l_CC(l=l_SPA, CC=CC_l, expected_result=1)

    CC_l = {(pred_Tornado, l_SPA), (pred_BMP_1, l_BMP), (pred_2S19_MSTA, l_SPA), (pred_BTR_70, l_BTR),
            (pred_RS_24, l_Air_Defense), (pred_T_64, l_Tank)}
    edcr.test_BOD_CC(CC=CC_l, expected_result=20)
    edcr.test_CON_l_CC(l=l_SPA, CC=CC_l, expected_result=1)

    CC_l = set(itertools.product(pred_conditions.values(), fg_l + cg_l))
    edcr.test_BOD_CC(CC=CC_l, expected_result=K)
    edcr.test_CON_l_CC(l=l_SPA, CC=CC_l, expected_result=1)
    edcr.test_BOD_CC(CC=CC_l, expected_result=K)
    edcr.test_CON_l_CC(l=l_BMP, CC=CC_l, expected_result=0)


if __name__ == '__main__':
    run_test_1()
    # run_test_2()
    # run_test_3()
