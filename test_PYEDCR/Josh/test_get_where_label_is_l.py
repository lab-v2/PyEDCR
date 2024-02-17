from test_PYEDCR.test import *

if __name__ == '__main__':
    method_str = 'get_where_label_is_l'

    K = [(0, 9)]
    test = Test(epsilon=0.1, method_str=method_str, K_train=K, K_test=K)
    test.run(pred=True, test=False, l=l_Tank, expected_output=0)
    test.run(pred=True, test=False, l=l_T_14, expected_output=0)
    test.run(pred=True, test=False, l=l_Iskander, expected_output=0)
    test.run(pred=True, test=False, l=l_SPA, expected_output=1)
    test.run(pred=True, test=False, l=l_2S19_MSTA, expected_output=np.array([1] * 6 + [0] + [1] * 3))
    test.run(pred=False, test=False, l=l_Tank, expected_output=0)
    test.run(pred=False, test=False, l=l_Iskander, expected_output=0)
    test.run(pred=False, test=False, l=l_T_14, expected_output=0)
    test.run(pred=False, test=False, l=l_SPA, expected_output=1)
    test.run(pred=False, test=False, l=l_2S19_MSTA, expected_output=1)
    test.run(pred=True, test=True, l=l_Tank,
             expected_output=np.array([1] + [0] * 5 + [1] + [0] * 3))
    test.run(pred=True, test=True, l=l_Iskander, expected_output=0)
    test.run(pred=True, test=True, l=l_T_14,
             expected_output=np.array([1] + [0] * 9))
    test.run(pred=True, test=True, l=l_SPA,
             expected_output=np.array([0] + [1] * 5 + [0] + [1] * 3))
    test.run(pred=True, test=True, l=l_2S19_MSTA,
             expected_output=np.array([0] + [1] * 5 + [0] * 2 + [1] * 2))
    test.run(pred=False, test=True, l=l_Tank, expected_output=0)
    test.run(pred=False, test=True, l=l_Iskander, expected_output=0)
    test.run(pred=False, test=True, l=l_T_14, expected_output=0)
    test.run(pred=False, test=True, l=l_SPA, expected_output=1)
    test.run(pred=False, test=True, l=l_2S19_MSTA, expected_output=1)

    K = [(0, 11)]
    test = Test(epsilon=0.1, method_str=method_str, K_train=K, K_test=K)
    test.run(pred=True, test=False, l=l_Tank, expected_output=0)
    test.run(pred=True, test=False, l=l_Iskander, expected_output=0)
    test.run(pred=True, test=False, l=l_SPA,
             expected_output=np.array([1] * 11 + [0]))
    test.run(pred=True, test=False, l=l_2S19_MSTA,
             expected_output=np.array([1] * 6 + [0] + [1] * 4 + [0]))
    test.run(pred=False, test=False, l=l_Tank, expected_output=0)
    test.run(pred=False, test=False, l=l_Iskander, expected_output=0)
    test.run(pred=False, test=False, l=l_SPA, expected_output=1)
    test.run(pred=False, test=False, l=l_2S19_MSTA, expected_output=1)
    test.run(pred=True, test=True, l=l_Tank,
             expected_output=np.array([1] + [0] * 5 + [1] + [0] * 5))
    test.run(pred=True, test=True, l=l_Iskander, expected_output=0)
    test.run(pred=True, test=True, l=l_SPA,
             expected_output=np.array([0] + [1] * 5 + [0] + [1] * 5))
    test.run(pred=True, test=True, l=l_2S19_MSTA,
             expected_output=np.array([0] + [1] * 5 + [0] * 2 + [1] * 4))
    test.run(pred=False, test=True, l=l_Tank, expected_output=0)
    test.run(pred=False, test=True, l=l_Iskander, expected_output=0)
    test.run(pred=False, test=True, l=l_SPA, expected_output=1)
    test.run(pred=False, test=True, l=l_2S19_MSTA, expected_output=1)

    K = [(100, 104), (120, 124)]

    test = Test(epsilon=0.1, method_str=method_str, K_train=K, K_test=K)
    test.run(pred=True, test=False, l=l_Tank, expected_output=0)
    test.run(pred=True, test=False, l=l_Iskander, expected_output=0)
    test.run(pred=True, test=False, l=l_SPA,
             expected_output=np.array([1] * 7 + [0, 1, 0]))
    test.run(pred=True, test=False, l=l_2S19_MSTA,
             expected_output=np.array([0] + [1] * 5 + [0] + [1] * 2 + [0]))
    test.run(pred=False, test=False, l=l_Tank, expected_output=0)
    test.run(pred=False, test=False, l=l_Iskander, expected_output=0)
    test.run(pred=False, test=False, l=l_SPA, expected_output=1)
    test.run(pred=False, test=False, l=l_2S19_MSTA, expected_output=1)
    test.run(pred=True, test=True, l=l_Tank,
             expected_output=np.array([0] * 5 + [1] + [0] * 4))
    test.run(pred=True, test=True, l=l_Iskander, expected_output=0)
    test.run(pred=True, test=True, l=l_SPA,
             expected_output=np.array([1] + [0] + [1] * 3 + [0] + [1] * 4))
    test.run(pred=True, test=True, l=l_2S19_MSTA, expected_output=0)
    test.run(pred=False, test=True, l=l_Tank, expected_output=0)
    test.run(pred=False, test=True, l=l_Iskander, expected_output=0)
    test.run(pred=False, test=True, l=l_SPA, expected_output=1)
    test.run(pred=False, test=True, l=l_2S19_MSTA, expected_output=0)
