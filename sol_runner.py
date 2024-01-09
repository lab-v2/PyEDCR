import vit_pipeline
import EDCR_pipeline
import itertools
import utils


def run():
    vit_pipeline.run_combined_fine_tuning_pipeline(lrs=[0.0001],
                                                   loss='BCE')

    # vit_pipeline.run_individual_fine_tuning_pipeline()

    # vit_pipeline.run_combined_testing_pipeline(pretrained_path='vit_b_16_lr0.0001.pth')

    # vit_pipeline.run_individual_fine_tuning_pipeline()

    # combined = True
    # conditions_from_main = False
    #
    # EDCR_pipeline.run_EDCR_pipeline(combined=combined,
    #                                 conditions_from_secondary=not conditions_from_main,
    #                                 conditions_from_main=conditions_from_main,
    #                                 consistency_constraints=True
    #                                 )
