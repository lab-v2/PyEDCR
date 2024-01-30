import vit_pipeline
import EDCR_pipeline
import utils


def run():
    # vit_pipeline.run_combined_fine_tuning_pipeline(lrs=[0.0001],
    #                                                loss='soft_marginal')
    # vit_pipeline.run_individual_fine_tuning_pipeline()

    # vit_pipeline.run_combined_testing_pipeline(pretrained_path='vit_b_16_lr0.0001.pth')


    combined = True
    losses = ['BCE', 'soft_marginal']
    lrs = [0.0001, 3e-6]

    for loss in losses:
        for main_lr in lrs:
            print('\n')
            for i in range(3):
                print(utils.red_text('#' * 150))
            print('\n')

            print(utils.red_text(f'combined={combined}\n' + '#' * 100 + '\n'))

            EDCR_pipeline.run_EDCR_pipeline(main_lr=main_lr,
                                            combined=combined,
                                            loss=loss,
                                            consistency_constraints=True,
                                            multiprocessing=True)
