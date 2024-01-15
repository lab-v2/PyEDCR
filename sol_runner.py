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

    loss = ['soft_marginal', 'BCE'][1]
    combined = True
    conditions_from_secondary = True

    for conditions_from_main in [True, False]:

        print(utils.red_text(f'\nconditions_from_secondary={conditions_from_secondary}, '
                             f'conditions_from_main={conditions_from_main}\n' +
                             f'combined={combined}\n' + '#' * 100 + '\n'))

        EDCR_pipeline.run_EDCR_pipeline(combined=combined,
                                        loss=loss,
                                        conditions_from_secondary=conditions_from_secondary,
                                        conditions_from_main=conditions_from_main,
                                        consistency_constraints=True,
                                        multiprocessing=True)
