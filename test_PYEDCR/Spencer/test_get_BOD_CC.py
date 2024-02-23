import itertools

from test_PYEDCR.test import *

method_str = 'get_BOD_CC'


class TestGetBODCC(Test):
    def __init__(self,
                 epsilon: float,
                 K_train: list[(int, int)] = None,
                 K_test: list[(int, int)] = None):
        super().__init__(epsilon=epsilon, method_str=method_str, K_train=K_train, K_test=K_test)

    def run(self,
            expected_output,
            *method_args,
            **method_kwargs):
        output = self.method(*method_args, **method_kwargs)
        test_passed = True if output[0] == expected_output[0] and (output[1] == expected_output[1]).all() else False

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
    CC_l_4 = {(pred_2S19_MSTA, l_SPA), (pred_T_72, l_SPA), (pred_2S19_MSTA, l_Tank), (pred_T_72, l_SPA)}
    CC_l_5 = {(pred_30N6E, l_SPA), (pred_T_72, l_BTR), (pred_2S19_MSTA, l_Tank)}
    CC_l_6 = {(pred_2S19_MSTA, l_BMP), (pred_T_72, l_BTR), (pred_2S19_MSTA, l_Tank)}
    CC_l_7 = {(pred_SPA, l_2S19_MSTA)}
    CC_l_8 = {(pred_SPA, l_2S19_MSTA), (pred_SPA, l_T_72), (pred_2S19_MSTA, l_SPA), (pred_T_72, l_SPA)}

    case_number = 1
    test.print_examples(test=False)
    print(utils.blue_text("=" * 50 + f"test {case_number} " + method_str + "=" * 50))

    test.run(CC=CC_l_1, expected_output=(0, np.array([0] * 10)))
    test.run(CC=CC_l_2, expected_output=(1, np.array([0] * 6 + [1] + [0] * 3)))
    test.run(CC=CC_l_3, expected_output=(9, np.array([1] * 6 + [0] + [1] * 3)))
    test.run(CC=CC_l_4, expected_output=(10, np.array([1] * 10)))
    test.run(CC=CC_l_5, expected_output=(0, np.array([0] * 10)))
    test.run(CC=CC_l_6, expected_output=(0, np.array([0] * 10)))
    test.run(CC=CC_l_7, expected_output=(9, np.array([1] * 6 + [0] + [1] * 3)))
    test.run(CC=CC_l_8, expected_output=(10, np.array([1] * 10)))


def run_test_2():
    K = [(351, 355), (471, 475), (4001, 4005), (4601, 4605), (6001, 6005)]
    test = TestGetBODCC(epsilon=0.1, K_train=K, K_test=None)

    CC_l_1 = {(pred_Pantsir_S1, l_Air_Defense)}
    CC_l_2 = {(pred_Pantsir_S1, l_Air_Defense), (pred_30N6E, l_Air_Defense), (pred_30N6E, l_SPA)}
    CC_l_3 = {(pred_Air_Defense, l_Pantsir_S1), (pred_BTR_60, l_BTR)}
    CC_l_4 = {(pred_Iskander, l_Air_Defense), (pred_T_72, l_Tank), (pred_Iskander, l_MT_LB_coarse), (pred_T_72, l_BMP)}
    CC_l_5 = {(pred_Tornado, l_Tank), (pred_T_90, l_T_80)}
    CC_l_6 = {(pred_2S19_MSTA, l_Tank), (pred_D_30, l_SPA), (pred_MT_LB_fine, l_MT_LB_coarse), (pred_BRDM, l_BTR)}
    CC_l_7 = {(pred_BM_30, l_SPA), (pred_BM_30, l_BTR), (pred_T_64, l_Tank)}
    CC_l_8 = {(pred_Air_Defense, l_RS_24), (pred_Air_Defense, l_Iskander), (pred_Tank, l_T_64)}
    CC_l_9 = {(pred_BM_30, l_SPA), (pred_BM_30, l_SPA), (pred_Tornado, l_SPA)}
    CC_l_10 = {}

    case_number = 2
    test.print_examples(test=False)
    print(utils.blue_text("=" * 50 + f"test {case_number} " + method_str + "=" * 50))

    test.run(CC=CC_l_1, expected_output=(6, np.array([1] + [0] * 14 + [1] * 5 + [0] * 5)))
    test.run(CC=CC_l_2, expected_output=(8, np.array([1] * 2 + [0] * 2 + [1] + [0] * 10 + [1] * 5 + [0] * 5)))
    test.run(CC=CC_l_3, expected_output=(6, np.array([1] + [0] * 14 + [1] * 5 + [0] * 5)))
    test.run(CC=CC_l_4, expected_output=(6, np.array([0] * 10 + [1] * 5 + [0] * 6 + [1] + [0] * 3)))
    test.run(CC=CC_l_5, expected_output=(0, np.array([0] * 25)))
    test.run(CC=CC_l_6, expected_output=(0, np.array([0] * 25)))
    test.run(CC=CC_l_7, expected_output=(8, np.array([0] * 5 + [1] * 3 + [0] + [1] + [0] * 10 + [1] + [0] + [1] * 3)))
    test.run(CC=CC_l_8, expected_output=(10, np.array([0] * 3 + [1] + [0] * 6 + [1] * 5 + [0] * 5 + [1] + [0] + [1] * 3)))
    test.run(CC=CC_l_9, expected_output=(5, np.array([0] * 5 + [1] * 5 + [0] * 15)))
    test.run(CC=CC_l_10, expected_output=(0, np.array([0] * 25)))

def run_test_3():
    K = [(6, 10), (2201, 2205), (3001, 3005), (3601, 3605), (3970, 3974), (7500, 7524)]
    test = TestGetBODCC(epsilon=0.1, K_train=K, K_test=None)

    CC_l_1 = {(pred_2S19_MSTA, l_SPA)}
    CC_l_2 = {(pred_TOS_1, l_BMP), (pred_TOS_1, l_SPA), (pred_T_72, l_Tank)}
    CC_l_3 = {(pred_Tornado, l_SPA), (pred_Tornado, l_BMP), (pred_Tornado, l_Air_Defense)}
    CC_l_4 = {(pred_SPA, l_2S19_MSTA), (pred_BTR, l_BTR_70)}
    CC_l_5 = {(pred_D_30, l_SPA), (pred_BTR_60, l_BTR), (pred_BTR_70, l_BTR), (pred_BTR_80, l_BTR)}
    CC_l_6 = {(pred_BM_30, l_SPA), (pred_2S19_MSTA, l_SPA)}
    CC_l_7 = {(pred_Tornado, l_BMP), (pred_Tornado, l_Air_Defense)}
    CC_l_8 = {(pred_Iskander, l_Air_Defense), (pred_Iskander, l_SPA)}
    CC_l_9 = {(pred_BTR, l_BRDM)}
    CC_l_10 = {(pred_Tornado, l_SPA), (pred_Tornado, l_BMP), (pred_Tornado, l_Air_Defense), (pred_BM_30, l_SPA),
               (pred_BTR_60, l_SPA), (pred_Iskander, l_SPA)}


    case_number = 3
    test.print_examples(test=False)
    print(utils.blue_text("=" * 50 + f"test {case_number} " + method_str + "=" * 50))

    test.run(CC=CC_l_1, expected_output=(4, np.array([0] + [1] * 4 + [0] * 45)))
    test.run(CC=CC_l_2, expected_output=(0, np.array([0] * 50)))
    test.run(CC=CC_l_3, expected_output=(19, np.array([0] * 26 + [1] * 2 + [0] + [1] * 2 + [0] + [1] * 11 + [0] * 3 +
                                                      [1] * 4)))
    test.run(CC=CC_l_4, expected_output=(7, np.array([0] + [1] * 4 + [0] * 7 + [1] * 3 + [0] * 35)))
    test.run(CC=CC_l_5, expected_output=(10, np.array([0] * 10 + [1] * 10 + [0] * 30)))
    test.run(CC=CC_l_6, expected_output=(8, np.array([0] + [1] * 4 + [0] * 20 + [1] + [0] * 2 + [1] + [0] * 14 + [1] *
                                                      2 + [0] * 5 )))
    test.run(CC=CC_l_7, expected_output=(2, np.array([0] * 30 + [1] + [0] * 5 + [1] + [0] * 13)))
    test.run(CC=CC_l_8, expected_output=(6, np.array([0] * 20 + [1] * 5 + [0] * 20 + [1] + [0] * 4)))
    test.run(CC=CC_l_9, expected_output=(4, np.array([0] * 6 + [1] * 4 + [0] * 40)))
    test.run(CC=CC_l_10, expected_output=(25, np.array([0] * 25 + [1] * 25)))


if __name__ == '__main__':
    run_test_1()
    run_test_2()
    run_test_3()
