import os
import utils

if utils.is_local():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import numpy as np
import typing

import experiment_config
import data_preprocessor
import PyEDCR


class NeuralPyEDCR(PyEDCR.EDCR):
    def __init__(self,
                 data_str: str,
                 main_model_name: str = None,
                 combined: bool = None,
                 loss: str = None,
                 lr: typing.Union[str, float] = None,
                 original_num_epochs: int = None,
                 epsilon: typing.Union[str, float] = None,
                 EDCR_num_epochs: int = 1,
                 neural_num_epochs: int = 1,
                 sheet_index: int = None,
                 K_train: typing.Union[typing.List[typing.Tuple[int]], np.ndarray] = None,
                 K_test: typing.List[typing.Tuple[int]] = None,
                 secondary_model_name: str = None,
                 secondary_model_loss: str = None,
                 secondary_num_epochs: int = None,
                 secondary_lr: float = None,
                 lower_predictions_indices: typing.List[int] = [],
                 binary_l_strs: typing.List[str] = [],
                 binary_model_name: str = None,
                 binary_num_epochs: int = None,
                 binary_lr: typing.Union[str, float] = None,
                 maximize_ratio: bool = True,
                 indices_of_fine_labels_to_take_out: typing.List[int] = [],
                 negated_conditions: bool = False):
        super().__init__(data_str=data_str,
                         main_model_name=main_model_name,
                         combined=combined,
                         loss=loss,
                         lr=lr,
                         original_num_epochs=original_num_epochs,
                         epsilon=epsilon,
                         sheet_index=sheet_index,
                         K_train=K_train,
                         K_test=K_test,
                         secondary_model_name=secondary_model_name,
                         secondary_model_loss=secondary_model_loss,
                         secondary_num_epochs=secondary_num_epochs,
                         lower_predictions_indices=lower_predictions_indices,
                         binary_l_strs=binary_l_strs,
                         binary_model_name=binary_model_name,
                         binary_num_epochs=binary_num_epochs,
                         binary_lr=binary_lr,
                         secondary_lr=secondary_lr,
                         maximize_ratio=maximize_ratio,
                         indices_of_fine_labels_to_take_out=indices_of_fine_labels_to_take_out,
                         negated_conditions=negated_conditions)
        self.EDCR_num_epochs = EDCR_num_epochs
        self.neural_num_epochs = neural_num_epochs

        # for g in data_preprocessing.DataPreprocessor.granularities.values():
        #     print(f"prediction train {g.g_str} is {self.pred_data['train']['original'][g]}")
        #     print(f"and its ground truth is {self.pred_data['train']['original'][g]}")


    def run_learning_pipeline(self,
                              multi_processing: bool = True):
        print('Started learning pipeline...\n')
        # self.print_metrics(test=False, prior=True)

        for EDCR_epoch in range(self.EDCR_num_epochs):
            for g in data_preprocessor.FineCoarseDataPreprocessor.granularities.values():
                self.learn_detection_rules(g=g,
                                           multi_processing=multi_processing)
                # self.apply_detection_rules(test=False,
                #                            g=g)

            # self.run_training_correction_model_pipeline(new_model_name=new_model_name,
            #                                             new_lr=new_lr)
            # self.print_metrics(test=False, prior=False, stage='post_detection')

            edcr_epoch_str = f'Finished EDCR epoch {EDCR_epoch + 1}/{self.EDCR_num_epochs}'

            print(utils.blue_text('\n' + '#' * 100 +
                                  '\n' + '#' * int((100 - len(edcr_epoch_str)) / 2) + edcr_epoch_str +
                                  '#' * (100 - int((100 - len(edcr_epoch_str)) / 2) - len(edcr_epoch_str)) +
                                  '\n' + '#' * 100 + '\n'))

        # self.learn_correction_rules(g=g)
        # self.learn_correction_rules_alt(g=g)

        print('\nRule learning completed\n')


def work_on_value(args):
    (epsilon_index,
     epsilon,
     data_str,
     main_model_name,
     main_lr,
     original_num_epochs,
     secondary_model_name,
     secondary_model_loss,
     secondary_num_epochs,
     secondary_lr,
     binary_l_strs,
     binary_model_name,
     binary_lr,
     binary_num_epochs,
     maximize_ratio,
     multi_processing,
     fine_labels_to_take_out,
     negated_conditions
     ) = args

    edcr = NeuralPyEDCR(data_str=data_str,
                        epsilon=epsilon,
                        sheet_index=epsilon_index,
                        main_model_name=main_model_name,
                        combined=True,
                        loss='BCE',
                        lr=main_lr,
                        original_num_epochs=original_num_epochs,
                        secondary_model_name=secondary_model_name,
                        secondary_model_loss=secondary_model_loss,
                        secondary_num_epochs=secondary_num_epochs,
                        secondary_lr=secondary_lr,
                        binary_l_strs=binary_l_strs,
                        binary_model_name=binary_model_name,
                        binary_lr=binary_lr,
                        binary_num_epochs=binary_num_epochs,
                        EDCR_num_epochs=1,
                        neural_num_epochs=1,
                        maximize_ratio=maximize_ratio,
                        indices_of_fine_labels_to_take_out=fine_labels_to_take_out,
                        negated_conditions=negated_conditions
                        )
    # edcr.learn_error_binary_model(binary_model_name=main_model_name,
    #                               binary_lr=new_lr)
    edcr.print_metrics(split='test',
                       prior=True)
    edcr.run_learning_pipeline(multi_processing=multi_processing)
    edcr.run_error_detection_application_pipeline(test=True,
                                                  print_results=False,
                                                  save_to_google_sheets=True)
    # edcr.apply_new_model_on_test()


def simulate_for_values(data_str: str,
                        main_model_name: str,
                        main_lr: typing.Union[float, str],
                        original_num_epochs: int,
                        binary_model_name: str,
                        multi_processing: bool = True,
                        secondary_model_name: str = None,
                        secondary_model_loss: str = None,
                        secondary_num_epochs: int = None,
                        secondary_lr: float = None,
                        binary_l_strs: typing.List[str] = [],
                        binary_lr: typing.Union[str, float] = None,
                        binary_num_epochs: int = None,
                        maximize_ratio: bool = True,
                        lists_of_fine_labels_to_take_out: typing.List[typing.List[int]] = [],
                        negated_conditions: bool = False):


    datas = [(i,
              None if maximize_ratio else 0.1,
              data_str,
              main_model_name,
              main_lr,
              original_num_epochs,
              secondary_model_name,
              secondary_model_loss,
              secondary_num_epochs,
              secondary_lr,
              binary_l_strs,
              binary_model_name,
              binary_lr,
              binary_num_epochs,
              maximize_ratio,
              multi_processing,
              fine_labels_to_take_out,
              negated_conditions
              ) for i, fine_labels_to_take_out in enumerate(lists_of_fine_labels_to_take_out)]

    # if not utils.is_debug_mode():
    #     processes_num = min([len(datas), 10 if utils.is_local() else 100])
    #     process_map(work_on_value,
    #                 datas,
    #                 max_workers=processes_num)
    # else:
    for data in datas:
        work_on_value(data)


def run_experiment(config: experiment_config.ExperimentConfig):
    binary_l_strs = list({f.split(f'e{config.binary_num_epochs - 1}_')[-1].replace('.npy', '')
                          for f in os.listdir('binary_results')
                          if f.startswith(f'{config.data_str}_{config.binary_model_name}')})

    # print(google_sheets_api.get_maximal_epsilon(tab_name=sheet_tab))

    # sheet_tab_name = google_sheets_api.get_sheet_tab_name(main_model_name=main_model_name,
    #                                                       data_str=data_str,
    #                                                       secondary_model_name=secondary_model_name,
    #                                                       binary=len(binary_l_strs) > 0
    #                                                       )
    # number_of_ratios = 10

    # lists_of_fine_labels_to_take_out = [list(range(i)) for i in range(number_of_fine_classes)]
    # lists_of_fine_labels_to_take_out = [[]]
    # lists_of_fine_labels_to_take_out = [list(range(number_of_fine_classes-1))]

    for (curr_secondary_model_name, curr_secondary_model_loss, curr_secondary_num_epochs, curr_secondary_lr) in \
            [(config.secondary_model_name, 'BCE', config.secondary_num_epochs, config.secondary_lr),
             # [None] * 4
             ]:
        for (curr_binary_l_strs, curr_binary_lr, curr_binary_num_epochs) in \
                [(binary_l_strs, config.binary_lr, config.binary_num_epochs),
                 # ([], None, None)
                 ]:
            for (lists_of_fine_labels_to_take_out, maximize_ratio, multi_processing) in \
                    [
                        ([[]], True, True),
                        # ([list(range(i)) for i in range(int(number_of_fine_classes / 2) + 1)], True, True)
                    ]:
                simulate_for_values(
                    data_str=config.data_str,
                    main_model_name=config.main_model_name,
                    main_lr=config.main_lr,
                    original_num_epochs=config.original_num_epochs,
                    binary_model_name=config.binary_model_name,
                    binary_l_strs=curr_binary_l_strs,
                    binary_lr=curr_binary_lr,
                    binary_num_epochs=curr_binary_num_epochs,
                    multi_processing=multi_processing,
                    secondary_model_name=curr_secondary_model_name,
                    secondary_model_loss=curr_secondary_model_loss,
                    secondary_num_epochs=curr_secondary_num_epochs,
                    secondary_lr=curr_secondary_lr,
                    maximize_ratio=maximize_ratio,
                    lists_of_fine_labels_to_take_out=lists_of_fine_labels_to_take_out,
                    negated_conditions=False
                )

    # (x_values, y_values, error_accuracies, error_f1s, error_MMCs, error_acc_f1s) = (
    #     google_sheets_api.get_values_from_columns(sheet_tab_name=sheet_tab_name,
    #                                               column_letters=['A', 'B', 'C', 'D', 'E', 'F']))
    #
    # plotting.plot_3d_metrics(x_values=x_values,
    #                          y_values=y_values,
    #                          metrics={'Error F1': (error_f1s, 'Greens', 'g')})

    # (x_values, balance_error_accuracies, error_f1s, constraint_f1s) = (
    #     google_sheets_api.get_values_from_columns(sheet_tab_name=sheet_tab_name,
    #                                               column_letters=['B', 'D', 'G', 'J']))

    # if data_str == 'military_vehicles':
    #     # Find the first occurrence where a > 0.5
    #     first_greater = np.argmax(x_values > 0.5)  # This gives the index of the first True in the condition
    #
    #     x_values[first_greater] = 0.5
    #
    # # Create a mask for values <= 0.5
    # mask = x_values <= 0.5
    #
    # x_values, balance_error_accuracies, error_f1s, constraint_f1s = \
    #     [a[mask] for a in [x_values, balance_error_accuracies, error_f1s, constraint_f1s]]
    #
    # plotting.plot_2d_metrics(data_str=data_str,
    #                          model_name=main_model_name,
    #                          x_values=x_values[1:],
    #                          metrics={'Balanced Error Accuracy': balance_error_accuracies[1:],
    #                                   'Error F1-Score': error_f1s[1:],
    #                                   'Constraints F1-Score': constraint_f1s[1:]},
    #                          style_dict={
    #                              'Balanced Error Accuracy': ('k', '-'),  # Black solid line
    #                              'Error F1-Score': ('k', ':'),  # Gray solid line
    #                              'Constraints F1-Score': ('k', '--')  # Black dotted line
    #                          },
    #                          fontsize=24)


def main():
    military_vehicles_config = experiment_config.ExperimentConfig(
        data_str='military_vehicles',
        main_model_name='vit_b_16',
        secondary_model_name='vit_l_16',
        main_lr=0.0001,
        secondary_lr=0.0001,
        binary_lr=0.0001,
        original_num_epochs=10,
        secondary_num_epochs=20,
        binary_num_epochs=10
    )

    # imagenet_config = data_preprocessing.ExperimentConfig(
    #     data_str='imagenet',
    #     main_model_name='dinov2_vits14',
    #     secondary_model_name='dinov2_vitl14',
    #     main_lr=0.000001,
    #     secondary_lr=0.000001,
    #     binary_lr=0.000001,
    #     original_num_epochs=8,
    #     secondary_num_epochs=2,
    #     binary_num_epochs=5
    # )
    #
    # openimage_config = data_preprocessing.ExperimentConfig(
    #     data_str='openimage',
    #     main_model_name='vit_b_16',
    #     secondary_model_name='dinov2_vits14',
    #     main_lr=0.0001,
    #     secondary_lr=0.000001,
    #     binary_lr=0.000001,
    #     original_num_epochs=20,
    #     secondary_num_epochs=20,
    #     binary_num_epochs=4
    # )

    run_experiment(config=military_vehicles_config)





if __name__ == '__main__':
    main()
