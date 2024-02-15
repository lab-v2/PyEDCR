# import vit_pipeline
# import EDCR_pipeline
# import utils

import data_preprocessing
import PyEDCR


def run():
    # vit_pipeline.run_combined_fine_tuning_pipeline(lrs=[0.0001],
    #                                                loss='soft_marginal')
    # vit_pipeline.run_individual_fine_tuning_pipeline()

    # vit_pipeline.run_combined_testing_pipeline(pretrained_path='vit_b_16_lr0.0001.pth')

    edcr = PyEDCR.EDCR(epsilon=0.1,
                       main_model_name='vit_b_16',
                       combined=True,
                       loss='BCE',
                       lr=0.0001,
                       num_epochs=20)
    # edcr.print_metrics(test=False, prior=True)
    edcr.print_metrics(test=True, prior=True)

    for g in data_preprocessing.granularities.values():
        edcr.DetCorrRuleLearn(g=g)

    # print([edcr.get_l_correction_rule_support_on_test(l=l) for l in
    #        list(data_preprocessing.fine_grain_labels.values()) +
    #        list(data_preprocessing.coarse_grain_labels.values())])

    for g in data_preprocessing.granularities:
        edcr.apply_detection_rules(g=g)

        # edcr.print_metrics(test=True, prior=False)
        # print(edcr.get_g_theoretical_precision_increase(g=data_preprocessing.granularities['fine']))
        # print(edcr.get_g_theoretical_precision_increase(g=data_preprocessing.granularities['coarse']))
        # print(edcr.get_theorem_1_condition_for_g(g=data_preprocessing.granularities['fine']))
        # print(edcr.get_theorem_1_condition_for_g(g=data_preprocessing.granularities['coarse']))

    for g in data_preprocessing.granularities:
        edcr.apply_correction_rules(g=g)

    for g in data_preprocessing.granularities:
        edcr.apply_reversion_rules(g=g)

    edcr.print_metrics(test=True, prior=False)
