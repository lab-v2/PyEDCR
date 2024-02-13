import data_preprocessing

from PyEDCR import EDCR

g_fine, g_coarse = data_preprocessing.granularities.values()
fg_l, cg_l = list(data_preprocessing.fine_grain_labels.values()), list(data_preprocessing.coarse_grain_labels.values())
pred_conditions = {l: EDCR.PredCondition(l=l) for l in fg_l + cg_l}

l_Air_Defense, l_BMD, l_BMP, l_BTR, l_MT_LB, l_SPA, l_Tank = cg_l

(pred_2S19_MSTA, pred_30N6E, pred_BM_30, pred_BMD, pred_BMP_1, pred_BMP_2, pred_BMP_T15, pred_BRDM, pred_BTR_60,
 pred_BTR_70, pred_BTR_80, pred_D_30, pred_Iskander, pred_MT_LB, pred_Pantsir_S1, pred_RS_24, pred_T_14, pred_T_62,
 pred_T_64, pred_T_72, pred_T_80, pred_T_90, pred_TOS_1, pred_Tornado) = [pred_conditions[l] for l in fg_l]

CC_l = {(pred_Tornado, l_SPA), (pred_BMP_1, l_BMP)}

print(CC_l)
print(CC_l.union({(pred_Tornado, l_SPA)}))
print(CC_l.difference({(pred_Tornado, l_SPA)}))
