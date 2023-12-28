import vit_pipeline
import EDCR_pipeline
import itertools


def run():
    # vit_pipeline.run_individual_fine_tuning_pipeline()

    # vit_pipeline.run_combined_testing_pipeline(pretrained_path='vit_b_16_lr0.0001.pth')
    for a, b, c in itertools.product([True, False], repeat=3):
        EDCR_pipeline.run_EDCR_pipeline(combined=True,
                                        conditions_from_secondary=a,
                                        conditions_from_main=b,
                                        consistency_constraints=c)
