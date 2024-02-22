import itertools

from test_PYEDCR.test import *


def run_test_1():
    K = [(0, 9)]
    test = Test(epsilon=0.1, K_train=K, K_test=None, method_str='get_CON_l_CC')

    CC_l_1 = {(pred_2S19_MSTA, l_SPA)}
    CC_l_2 = {(pred_2S19_MSTA, l_SPA), (pred_T_72, l_SPA)}
    CC_l_3 = {(pred_Iskander, l_Tank)}

    test.run(l=l_2S19_MSTA, CC=CC_l_1, expected_output=1.0)
    test.run(l=l_2S19_MSTA, CC=CC_l_2, expected_output=1.0)
    test.run(l=l_T_72, CC=CC_l_2, expected_output=0.0)
    test.run(l=l_T_80, CC=CC_l_3, expected_output=0.0)


def run_test_2():
    K = [(351, 355), (471, 475), (4001, 4005), (4601, 4605), (6001, 6005)]

    test = Test(epsilon=0.1, K_train=K, K_test=None, method_str='get_CON_l_CC')

    CC_l_1 = {(pred_Pantsir_S1, l_Air_Defense)}
    CC_l_2 = {(pred_Iskander, l_Air_Defense), (pred_T_72, l_Tank), (pred_BM_30, l_SPA)}
    CC_l_3 = {(pred_T_90, l_Tank), (pred_BMD_fine, l_BMD_coarse), (pred_30N6E, l_Air_Defense), (pred_T_64, l_Tank)}
    CC_l_4 = {(pred_Tank, l_T_90), (pred_BMD_coarse, l_BMD_fine), (pred_Air_Defense, l_30N6E), (pred_Tank, l_T_64)}
    CC_l_5 = {(pred_SPA, l_Tornado), (pred_Air_Defense, l_Iskander), (pred_Tank, l_T_64), (pred_Air_Defense, l_Pantsir_S1)}

    test.run(l=l_30N6E, CC=CC_l_1, expected_output=(1/6))
    test.run(l=l_Pantsir_S1, CC=CC_l_1, expected_output=(5/6))
    test.run(l=l_BM_30, CC=CC_l_2, expected_output=(4/10))
    test.run(l=l_T_72, CC=CC_l_2, expected_output=0.0)
    test.run(l=l_Iskander, CC=CC_l_2, expected_output=(5/10))
    test.run(l=l_T_64, CC=CC_l_3, expected_output=(4/6))
    test.run(l=l_30N6E, CC=CC_l_3, expected_output=(2/6))
    test.run(l=l_T_90, CC=CC_l_3, expected_output=0.0)
    test.run(l=l_Tank, CC=CC_l_4, expected_output=(4/6))
    test.run(l=l_Air_Defense, CC=CC_l_4, expected_output=(2/6))
    test.run(l=l_Air_Defense, CC=CC_l_5, expected_output=(11/16))
    test.run(l=l_Tank, CC=CC_l_5, expected_output=(4/16))
    test.run(l=l_SPA, CC=CC_l_5, expected_output=(1/16))


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
    # run_test_1()
    run_test_2()
    # run_test_3()
