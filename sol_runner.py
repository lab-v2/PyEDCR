import vit_pipeline
import EDCR_pipeline
import itertools
import utils


def run():
    # vit_pipeline.run_individual_fine_tuning_pipeline()
    # vit_pipeline.run_combined_testing_pipeline(pretrained_path='vit_b_16_lr0.0001.pth')

    for a, b in itertools.product([True, False], repeat=2):
        print(utils.red_text(f'\nconditions_from_secondary={a}, conditions_from_main={b}\n' +
                             '#' * 100 + '\n'))

        EDCR_pipeline.run_EDCR_pipeline(combined=True,
                                        conditions_from_secondary=a,
                                        conditions_from_main=b,
                                        consistency_constraints=True)
