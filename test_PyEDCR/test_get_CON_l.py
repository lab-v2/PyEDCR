import data_preprocessing
from PyEDCR import EDCR
import data_preprocessing
import warnings
import numpy as np
import utils

# This will silence all warnings, including ones unrelated to your evaluation.
# Use this approach with caution!
warnings.filterwarnings('ignore')

edcr = EDCR(epsilon=0.1,
            check_mode=True
            )
# edcr.print_metrics(test=False, prior=True)
# edcr.print_metrics(test=True, prior=True)

g_fine, g_coarse = data_preprocessing.granularities.values()
fg_l, cg_l = list(data_preprocessing.fine_grain_labels.values()), list(data_preprocessing.coarse_grain_labels.values())
pred_conditions = {l: EDCR.PredCondition(l=l) for l in fg_l + cg_l}

l_Air_Defense, l_BMD, l_BMP, l_BTR, l_MT_LB, l_SPA, l_Tank = cg_l


(pred_2S19_MSTA, pred_30N6E, pred_BM_30, pred_BMD, pred_BMP_1, pred_BMP_2, pred_BMP_T15, pred_BRDM, pred_BTR_60,
 pred_BTR_70, pred_BTR_80, pred_D_30, pred_Iskander, pred_MT_LB, pred_Pantsir_S1, pred_RS_24, pred_T_14, pred_T_62,
 pred_T_64, pred_T_72, pred_T_80, pred_T_90, pred_TOS_1, pred_Tornado) = [pred_conditions[l] for l in fg_l]

check_train_fine_pred = np.load("test_data/check_train_fine_pred.npy")
check_train_coarse_pred = np.load("test_data/check_train_coarse_pred.npy")

check_test_fine_pred = np.load("test_data/check_test_fine_pred.npy")
check_test_coarse_pred = np.load("test_data/check_test_coarse_pred.npy")

check_train_fine_true = np.load("test_data/check_train_fine_true.npy")
check_train_coarse_true = np.load("test_data/check_train_coarse_true.npy")

check_test_fine_true = np.load("test_data/check_test_fine_true.npy")
check_test_coarse_true = np.load("test_data/check_test_coarse_true.npy")

# Print data use for testing:

# Function to format prediction/ground truth pairs
def format_pair(pred_fine, pred_coarse, true_fine, true_coarse):
  max_pred_len = max([len(str(l)) for l in fg_l]) + max([len(str(l)) for l in cg_l]) + 10 
  max_true_len = max([len(str(l)) for l in fg_l]) + max([len(str(l)) for l in cg_l]) + 10
  pred_str = f"pred: ({fg_l[pred_fine]}, {cg_l[pred_coarse]})"
  true_str = f"true: ({fg_l[true_fine]}, {cg_l[true_coarse]})"
  pred_str = pred_str + " " * (max_pred_len - len(pred_str))
  true_str = true_str + " " * (max_true_len - len(true_str))
  return pred_str, true_str

# Print information in desired format
for i in range(len(check_train_fine_pred)):
  pred_fine, pred_coarse = check_train_fine_pred[i], check_train_coarse_pred[i]
  true_fine, true_coarse = check_train_fine_true[i], check_train_coarse_true[i]
  pred_str, true_str = format_pair(pred_fine, pred_coarse, true_fine, true_coarse)
  print(f'["{pred_str}", "{true_str}"]')

# method name
method_str = "Get_CON_l"

print(utils.blue_text("=" * 50 + "test " + method_str + "=" * 50))

# Test 1
CC_l = {(pred_BM_30, l_SPA), (pred_30N6E, l_Air_Defense)}
edcr.test_get_CON_l(l=l_Air_Defense, CC=CC_l, expected_result=7/17)

# Test 2
CC_l = {(pred_BM_30, l_SPA)}
edcr.test_get_CON_l(l=l_Air_Defense, CC=CC_l, expected_result=9/10)

# Test 3
CC_l = {(pred_30N6E, l_Air_Defense)}
edcr.test_get_CON_l(l=l_Air_Defense, CC=CC_l, expected_result=1)

print(f"{method_str} method passed!")