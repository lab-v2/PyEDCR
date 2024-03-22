from test_PYEDCR.test import *


class TestWhereAnyConditionsSatisfied(Test):
    def __init__(self,
                 epsilon: float,
                 K_train: list[(int, int)] = None,
                 K_test: list[(int, int)] = None):
        method_str = 'get_where_any_conditions_satisfied'
        super().__init__(epsilon=epsilon, method_str=method_str, K_train=K_train, K_test=K_test)

    def run_edge_cases(self,
                       fine_data: np.array,
                       coarse_data: np.array):
        self.run(C=set(),
                 fine_data=fine_data,
                 coarse_data=coarse_data,
                 expected_output=0)

        self.run(C={pred_conditions[l] for l in fg_l},
                 fine_data=fine_data,
                 coarse_data=coarse_data,
                 expected_output=1)

        self.run(C={pred_conditions[l] for l in cg_l},
                 fine_data=fine_data,
                 coarse_data=coarse_data,
                 expected_output=1)

        self.run(C={pred_conditions[l] for l in fg_l + cg_l},
                 fine_data=fine_data,
                 coarse_data=coarse_data,
                 expected_output=1)



def run_tests():
    K = [(0, 9)]
    test = TestWhereAnyConditionsSatisfied(epsilon=0.1, K_train=K, K_test=K)

    train_pred_fine_data, train_pred_coarse_data = get_predictions(test=False)

    test.run(C={pred_Tornado, pred_BMP_1},
             fine_data=train_pred_fine_data,
             coarse_data=train_pred_coarse_data,
             expected_output=0)

    test.run(C={pred_Tornado, pred_BMP_1, pred_2S19_MSTA},
             fine_data=train_pred_fine_data,
             coarse_data=train_pred_coarse_data,
             expected_output=np.array([1] * 6 + [0] + [1] * 3))

    test.run(C={pred_SPA},
             fine_data=train_pred_fine_data,
             coarse_data=train_pred_coarse_data,
             expected_output=1)

    test.run(C={pred_Tornado, pred_BMP_1, pred_SPA},
             fine_data=train_pred_fine_data,
             coarse_data=train_pred_coarse_data,
             expected_output=1)

    test.run(C={pred_2S19_MSTA, pred_T_72},
             fine_data=train_pred_fine_data,
             coarse_data=train_pred_coarse_data,
             expected_output=1)

    test.run_edge_cases(fine_data=train_pred_fine_data,
                        coarse_data=train_pred_coarse_data)

    test_pred_fine_data, test_pred_coarse_data = get_predictions(test=True)

    test.run(C={pred_Tornado, pred_BMP_1},
             fine_data=test_pred_fine_data,
             coarse_data=test_pred_coarse_data,
             expected_output=0)

    test.run(C={pred_Tornado, pred_BMP_1, pred_2S19_MSTA},
             fine_data=test_pred_fine_data,
             coarse_data=test_pred_coarse_data,
             expected_output=np.array([0] + [1] * 5 + [0] * 2 + [1] * 2))

    test.run(C={pred_SPA},
             fine_data=test_pred_fine_data,
             coarse_data=test_pred_coarse_data,
             expected_output=np.array([0] + [1] * 5 + [0] * 1 + [1] * 3))

    test.run(C={pred_Tornado, pred_BMP_1, pred_SPA},
             fine_data=test_pred_fine_data,
             coarse_data=test_pred_coarse_data,
             expected_output=np.array([0] + [1] * 5 + [0] * 1 + [1] * 3))

    test.run(C={pred_2S19_MSTA, pred_T_72},
             fine_data=test_pred_fine_data,
             coarse_data=test_pred_coarse_data,
             expected_output=np.array([0] + [1] * 6 + [0] * 1 + [1] * 2))

    test.run_edge_cases(fine_data=test_pred_fine_data,
                        coarse_data=test_pred_coarse_data)

    K = [(500, 509), (1500, 1509)]

    test = TestWhereAnyConditionsSatisfied(epsilon=0.1, K_train=K, K_test=K)

    train_pred_fine_data, train_pred_coarse_data = get_predictions(test=False)

    test.run(C={pred_Tornado, pred_BMP_1},
             fine_data=train_pred_fine_data,
             coarse_data=train_pred_coarse_data,
             expected_output=0)

    test.run(C={pred_Tornado, pred_BMP_1, pred_2S19_MSTA},
             fine_data=train_pred_fine_data,
             coarse_data=train_pred_coarse_data,
             expected_output=0)

    test.run(C={pred_SPA},
             fine_data=train_pred_fine_data,
             coarse_data=train_pred_coarse_data,
             expected_output=np.array([1] * 10 + [0] * 10))

    test.run(C={pred_Tornado, pred_BMP_1, pred_SPA},
             fine_data=train_pred_fine_data,
             coarse_data=train_pred_coarse_data,
             expected_output=np.array([1] * 10 + [0] * 10))

    test.run(C={pred_2S19_MSTA, pred_T_72},
             fine_data=train_pred_fine_data,
             coarse_data=train_pred_coarse_data,
             expected_output=0)

    test.run_edge_cases(fine_data=train_pred_fine_data,
                        coarse_data=train_pred_coarse_data)

    test_pred_fine_data, test_pred_coarse_data = get_predictions(test=True)

    test.run(C={pred_Tornado, pred_BMP_1},
             fine_data=test_pred_fine_data,
             coarse_data=test_pred_coarse_data,
             expected_output=0)

    test.run(C={pred_Tornado, pred_BMP_1, pred_2S19_MSTA},
             fine_data=test_pred_fine_data,
             coarse_data=test_pred_coarse_data,
             expected_output=0)

    test.run(C={pred_SPA},
             fine_data=test_pred_fine_data,
             coarse_data=test_pred_coarse_data,
             expected_output=0)

    test.run(C={pred_Tornado, pred_BMP_1, pred_SPA},
             fine_data=test_pred_fine_data,
             coarse_data=test_pred_coarse_data,
             expected_output=0)

    test.run(C={pred_2S19_MSTA, pred_T_72},
             fine_data=test_pred_fine_data,
             coarse_data=test_pred_coarse_data,
             expected_output=np.array([0] * 12 + [1] + [0] * 7))

    test.run_edge_cases(fine_data=test_pred_fine_data,
                        coarse_data=test_pred_coarse_data)

    K = [(100, 104), (200, 204), (300, 304), (400, 404)]

    test = TestWhereAnyConditionsSatisfied(epsilon=0.1, K_train=K, K_test=K)

    train_pred_fine_data, train_pred_coarse_data = get_predictions(test=False)

    test.run(C={pred_Tornado, pred_BMP_1},
             fine_data=train_pred_fine_data,
             coarse_data=train_pred_coarse_data,
             expected_output=0)

    test.run(C={pred_Tornado, pred_BMP_1, pred_2S19_MSTA},
             fine_data=train_pred_fine_data,
             coarse_data=train_pred_coarse_data,
             expected_output=0)

    test.run(C={pred_SPA},
             fine_data=train_pred_fine_data,
             coarse_data=train_pred_coarse_data,
             expected_output=np.array([1] * 10 + [0] * 10))

    test.run(C={pred_Tornado, pred_BMP_1, pred_SPA},
             fine_data=train_pred_fine_data,
             coarse_data=train_pred_coarse_data,
             expected_output=np.array([1] * 10 + [0] * 10))

    test.run(C={pred_2S19_MSTA, pred_T_72},
             fine_data=train_pred_fine_data,
             coarse_data=train_pred_coarse_data,
             expected_output=0)

    test.run_edge_cases(fine_data=train_pred_fine_data,
                        coarse_data=train_pred_coarse_data)

    test_pred_fine_data, test_pred_coarse_data = get_predictions(test=True)

    test.run(C={pred_Tornado, pred_BMP_1},
             fine_data=test_pred_fine_data,
             coarse_data=test_pred_coarse_data,
             expected_output=0)

    test.run(C={pred_Tornado, pred_BMP_1, pred_2S19_MSTA},
             fine_data=test_pred_fine_data,
             coarse_data=test_pred_coarse_data,
             expected_output=0)

    test.run(C={pred_SPA},
             fine_data=test_pred_fine_data,
             coarse_data=test_pred_coarse_data,
             expected_output=0)

    test.run(C={pred_Tornado, pred_BMP_1, pred_SPA},
             fine_data=test_pred_fine_data,
             coarse_data=test_pred_coarse_data,
             expected_output=0)

    test.run(C={pred_2S19_MSTA, pred_T_72},
             fine_data=test_pred_fine_data,
             coarse_data=test_pred_coarse_data,
             expected_output=np.array([0] * 12 + [1] + [0] * 7))

    test.run_edge_cases(fine_data=test_pred_fine_data,
                        coarse_data=test_pred_coarse_data)


if __name__ == '__main__':
    run_tests()
