import vit_pipeline
import EDCR_pipeline
import utils


def run():
    # vit_pipeline.run_combined_fine_tuning_pipeline(lrs=[0.0001],
    #                                                loss='soft_marginal')
    # vit_pipeline.run_individual_fine_tuning_pipeline()

    # vit_pipeline.run_combined_testing_pipeline(pretrained_path='vit_b_16_lr0.0001.pth')

    # combined = True
    # losses = ['BCE', 'soft_marginal']
    # lrs = [0.0001, 3e-6]
    #
    # for loss in losses:
    #     for main_lr in lrs:
    #         print('\n')
    #         for i in range(3):
    #             print(utils.red_text('#' * 150))
    #         print('\n')
    #
    #         print(utils.red_text(f'combined={combined}\n' + '#' * 100 + '\n'))
    #
    #         EDCR_pipeline.run_EDCR_pipeline(main_lr=main_lr,
    #                                         combined=combined,
    #                                         loss=loss,
    #                                         consistency_constraints=True,
    #                                         multiprocessing=True)

    combined = True
    print(utils.red_text(f'combined={combined}\n' + '#' * 100 + '\n'))

    test_pred_fine_path = 'combined_results/vit_b_16_test_fine_pred_lr0.0001_e19.npy'
    test_pred_coarse_path = 'combined_results/vit_b_16_test_coarse_pred_lr0.0001_e19.npy'

    test_true_fine_path = 'combined_results/test_true_fine.npy'
    test_true_coarse_path = 'combined_results/test_true_coarse.npy'

    train_pred_fine_path = 'combined_results/train_vit_b_16_fine_pred_lr0.0001.npy'
    train_pred_coarse_path = 'combined_results/train_vit_b_16_coarse_pred_lr0.0001.npy'

    train_true_fine_path = 'combined_results/train_true_fine.npy'
    train_true_coarse_path = 'combined_results/train_true_coarse.npy'

    epsilons = [0.2, 0.3, 0.7]

    consistency_constraints = True
    for epsilon in epsilons:
        print(utils.red_text('\n' + '#' * 50 + f'Epsilon: {epsilon}' + '#' * 50 + '\n'))
        EDCR_pipeline.run_EDCR_pipeline(test_pred_fine_path=test_pred_fine_path,
                                        test_pred_coarse_path=test_pred_coarse_path,
                                        test_true_fine_path=test_true_fine_path,
                                        test_true_coarse_path=test_true_coarse_path,
                                        train_pred_fine_path=train_pred_fine_path,
                                        train_pred_coarse_path=train_pred_coarse_path,
                                        train_true_fine_path=train_true_fine_path,
                                        train_true_coarse_path=train_true_coarse_path,
                                        main_lr=0.0001,
                                        combined=combined,
                                        loss='BCE',
                                        epsilon=epsilon,
                                        consistency_constraints=consistency_constraints,
                                        multiprocessing=True)
