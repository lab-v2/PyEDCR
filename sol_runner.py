import vit_pipeline
import EDCR_pipeline


def run():
    # vit_pipeline.run_individual_fine_tuning_pipeline()

    # vit_pipeline.run_combined_testing_pipeline(pretrained_path='vit_b_16_lr0.0001.pth')
    EDCR_pipeline.run_EDCR_pipeline(combined=True,
                                    conditions_from_secondary=True,
                                    conditions_from_main=False,
                                    consistency_constraints=True)
