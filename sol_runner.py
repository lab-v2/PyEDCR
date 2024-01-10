import vit_pipeline
import EDCR_pipeline
import itertools
import utils


def run():
    # vit_pipeline.run_combined_fine_tuning_pipeline(lrs=[3e-6],
    #                                                loss='soft_marginal')
    # vit_pipeline.run_individual_fine_tuning_pipeline()

    # vit_pipeline.run_combined_testing_pipeline(pretrained_path='vit_b_16_lr0.0001.pth')

    # vit_pipeline.run_individual_fine_tuning_pipeline()

    combined = True
    for a, b, c in itertools.product([True, False], repeat=3):
        if a or b:
            print(utils.red_text(f'\nconditions_from_secondary={a}, conditions_from_main={b}, mp={c}\n' +
                                 f'combined={combined}\n' + '#' * 100 + '\n'))
            EDCR_pipeline.run_EDCR_pipeline(combined=combined,
                                            loss='BCE',
                                            conditions_from_secondary=a,
                                            conditions_from_main=b,
                                            consistency_constraints=True,
                                            multiprocessing=c)
