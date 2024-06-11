import os
import utils

if utils.is_local():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import numpy as np
import typing
from tqdm.contrib.concurrent import process_map
import itertools

import data_preprocessing
import PyEDCR
import backbone_pipeline
import combined_fine_tuning
import neural_evaluation
import google_sheets_api
import plotting


class NeuralPyEDCR(PyEDCR.EDCR):
    def __init__(self,
                 data_str: str,
                 main_model_name: str,
                 combined: bool,
                 loss: str,
                 lr: typing.Union[str, float],
                 original_num_epochs: int,
                 epsilon: typing.Union[str, float] = None,
                 EDCR_num_epochs: int = 1,
                 neural_num_epochs: int = 1,
                 sheet_index: int = None,
                 K_train: typing.Union[typing.List[typing.Tuple[int]], np.ndarray] = None,
                 K_test: typing.List[typing.Tuple[int]] = None,
                 include_inconsistency_constraint: bool = False,
                 secondary_model_name: str = None,
                 secondary_model_loss: str = None,
                 secondary_num_epochs: int = None,
                 secondary_lr: float = None,
                 lower_predictions_indices: typing.List[int] = [],
                 binary_l_strs: typing.List[str] = [],
                 binary_model_name: str = None,
                 binary_num_epochs: int = None,
                 binary_lr: typing.Union[str, float] = None,
                 num_train_images_per_class: int = None,
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
                         include_inconsistency_constraint=include_inconsistency_constraint,
                         secondary_model_name=secondary_model_name,
                         secondary_model_loss=secondary_model_loss,
                         secondary_num_epochs=secondary_num_epochs,
                         lower_predictions_indices=lower_predictions_indices,
                         binary_l_strs=binary_l_strs,
                         binary_model_name=binary_model_name,
                         binary_num_epochs=binary_num_epochs,
                         binary_lr=binary_lr,
                         secondary_lr=secondary_lr,
                         num_train_images_per_class=num_train_images_per_class,
                         maximize_ratio=maximize_ratio,
                         indices_of_fine_labels_to_take_out=indices_of_fine_labels_to_take_out,
                         negated_conditions=negated_conditions)
        self.EDCR_num_epochs = EDCR_num_epochs
        self.neural_num_epochs = neural_num_epochs

        relevant_predicted_indices = None

        # if 'correct' in experiment_name:
        #     train_pred_correct_mask = np.ones_like(self.pred_data['train']['original'][
        #                                                data_preprocessing.DataPreprocessor.granularities['fine']])
        #
        #     for g in data_preprocessing.DataPreprocessor.granularities.values():
        #         train_pred_correct_mask &= self.get_where_predicted_correct(test=False, g=g)
        #
        #     relevant_predicted_indices = np.where(train_pred_correct_mask == 1)[0]

        # if num_train_images_per_class is not None:
        #     example_indices = []
        #
        #     for i in range(len(self.preprocessor.fine_grain_classes_str)):
        #         i_indices_in_ground_truth = np.where(self.preprocessor.train_true_fine_data == i)[0]
        #         cls_idx = np.intersect1d(i_indices_in_ground_truth, relevant_predicted_indices)
        #         example_indices.extend(cls_idx[:num_train_images_per_class])
        #         # break
        #
        #     self.K_train = np.array(example_indices)
        #
        #     for g in data_preprocessing.DataPreprocessor.granularities.values():
        #         self.pred_data['train']['original'][g] = self.pred_data['train']['original'][g][self.K_train]

        # for g in data_preprocessing.DataPreprocessor.granularities.values():
        #     print(f"prediction train {g.g_str} is {self.pred_data['train']['original'][g]}")
        #     print(f"and its ground truth is {self.pred_data['train']['original'][g]}")

    def run_training_correction_model_pipeline(self,
                                               new_model_name: str,
                                               new_lr: float):

        perceived_examples_with_errors = set()
        for g in data_preprocessing.DataPreprocessor.granularities.values():
            perceived_examples_with_errors = perceived_examples_with_errors.union(set(
                np.where(self.get_predictions(test=False, g=g, stage='post_detection') == -1)[0]))

        perceived_examples_with_errors = np.array(list(perceived_examples_with_errors))

        print(utils.red_text(f'\nNumber of perceived train errors: {len(perceived_examples_with_errors)} / '
                             f'{self.T_train}\n'))

        preprocessor, fine_tuners, loaders, devices = (
            backbone_pipeline.initiate(
                data_str=self.data_str,
                model_name=new_model_name,
                preprocessor=self.preprocessor,
                lr=new_lr,
                combined=self.combined,
                error_indices=perceived_examples_with_errors,
                # train_eval_split=0.8
            ))

        if self.correction_model is None:
            self.correction_model = fine_tuners[0]

        combined_fine_tuning.fine_tune_combined_model(
            preprocessor=preprocessor,
            lr=new_lr,
            fine_tuner=self.correction_model,
            device=devices[0],
            loaders=loaders,
            loss=self.loss,
            save_files=False,
            evaluate_on_test_between_epochs=False,
            num_epochs=self.neural_num_epochs,
            data_str=data_str,
            model_name=main_model_name
            # debug=True
        )
        print('#' * 100)

        _, _, loaders, devices = backbone_pipeline.initiate(
            data_str=self.data_str,
            model_name=new_model_name,
            preprocessor=self.preprocessor,
            lr=new_lr,
            combined=self.combined,
            error_indices=perceived_examples_with_errors,
            evaluation=True,
        )

        evaluation_return_values = neural_evaluation.evaluate_combined_model(
            preprocessor=self.preprocessor,
            fine_tuner=self.correction_model,
            loaders=loaders,
            loss=self.loss,
            device=devices[0],
            split='train',
            print_results=True)

        new_fine_predictions, new_coarse_predictions = evaluation_return_values[2], evaluation_return_values[3]

        self.pred_data['train']['post_detection'][data_preprocessing.DataPreprocessor.granularities['fine']][
            perceived_examples_with_errors] = new_fine_predictions
        self.pred_data['train']['post_detection'][data_preprocessing.DataPreprocessor.granularities['coarse']][
            perceived_examples_with_errors] = new_coarse_predictions

    def apply_new_model_on_test(self,
                                print_results: bool = True):
        new_fine_predictions, new_coarse_predictions = (
            neural_evaluation.run_combined_evaluating_pipeline(data_str=self.data_str,
                                                               model_name=self.main_model_name,
                                                               split='test',
                                                               lr=self.lr,
                                                               loss=self.loss,
                                                               num_epochs=self.neural_num_epochs,
                                                               pretrained_fine_tuner=self.correction_model,
                                                               save_files=False,
                                                               print_results=False))

        for g in data_preprocessing.DataPreprocessor.granularities.values():
            old_test_g_predictions = self.get_predictions(test=True, g=g, stage='post_detection')
            new_test_g_predictions = new_fine_predictions if g.g_str == 'fine' else new_coarse_predictions

            self.pred_data['test']['post_detection'][g] = np.where(old_test_g_predictions == -1,
                                                                   new_test_g_predictions,
                                                                   old_test_g_predictions)
        if print_results:
            self.print_metrics(split='test', prior=False, stage='post_detection')

            where_fixed_initial_error = set()
            for g in data_preprocessing.DataPreprocessor.granularities.values():
                where_fixed_initial_error = where_fixed_initial_error.union(set(
                    np.where(self.get_where_predicted_correct(test=True, g=g, stage='post_detection') == 1)[0]
                ).intersection(set(np.where(self.get_where_predicted_incorrect(test=True, g=g) == 1)[0])))

            print(f'where_fixed_initial_error: {len(where_fixed_initial_error)}')

    def run_learning_pipeline(self,
                              multi_processing: bool = True):
        print('Started learning pipeline...\n')
        # self.print_metrics(test=False, prior=True)

        for EDCR_epoch in range(self.EDCR_num_epochs):
            for g in data_preprocessing.DataPreprocessor.granularities.values():
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

    def learn_error_binary_model(self,
                                 binary_model_name: str,
                                 binary_lr: typing.Union[float, str]):
        preprocessor, fine_tuners, loaders, devices = backbone_pipeline.initiate(
            data_str=self.data_str,
            model_name=binary_model_name,
            preprocessor=self.preprocessor,
            lr=binary_lr,
            train_fine_predictions=self.get_predictions(test=False, g=self.preprocessor.granularities['fine']),
            train_coarse_predictions=self.get_predictions(test=False, g=self.preprocessor.granularities['coarse']),
            test_fine_predictions=self.get_predictions(test=True, g=self.preprocessor.granularities['fine']),
            test_coarse_predictions=self.get_predictions(test=True, g=self.preprocessor.granularities['coarse'])
            # debug=True
        )

        combined_fine_tuning.fine_tune_combined_model(
            preprocessor=preprocessor,
            lr=binary_lr,
            fine_tuner=fine_tuners[0],
            device=devices[0],
            loaders=loaders,
            loss='error_BCE',
            save_files=False,
            evaluate_on_test_between_epochs=False,
            num_epochs=2,
            data_str=data_str,
            model_name=main_model_name
        )


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
     num_train_images_per_class,
     maximize_ratio,
     multi_processing,
     fine_labels_to_take_out,
     negated_conditions
     ) = args

    print('#' * 25 + f'num_train_images_per_class = {num_train_images_per_class}, eps = {epsilon}' + '#' * 50)
    edcr = NeuralPyEDCR(data_str=data_str,
                        epsilon=epsilon,
                        sheet_index=epsilon_index,
                        main_model_name=main_model_name,
                        combined=True,
                        loss='BCE',
                        lr=main_lr,
                        original_num_epochs=original_num_epochs,
                        include_inconsistency_constraint=False,
                        secondary_model_name=secondary_model_name,
                        secondary_model_loss=secondary_model_loss,
                        secondary_num_epochs=secondary_num_epochs,
                        secondary_lr=secondary_lr,
                        binary_l_strs=binary_l_strs,
                        binary_model_name=binary_model_name,
                        binary_lr=binary_lr,
                        binary_num_epochs=binary_num_epochs,
                        # lower_predictions_indices=lower_predictions_indices,
                        EDCR_num_epochs=1,
                        neural_num_epochs=1,
                        # num_train_images_per_class=num_train_images_per_class
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


def simulate_for_values(total_number_of_points: int = 10,
                        min_value: float = 0.1,
                        max_value: float = 0.3,
                        multi_processing: bool = True,
                        secondary_model_name: str = None,
                        secondary_model_loss: str = None,
                        secondary_num_epochs: int = None,
                        secondary_lr: float = None,
                        binary_l_strs: typing.List[str] = [],
                        binary_lr: typing.Union[str, float] = None,
                        binary_num_epochs: int = None,
                        num_train_images_per_class: typing.Sequence[int] = None,
                        only_from_missing_values: bool = False,
                        maximize_ratio: bool = True,
                        lists_of_fine_labels_to_take_out: typing.List[typing.List[int]] = [],
                        negated_conditions: bool = False):
    # all_values = {i: element for i, element
    #               in enumerate(itertools.product(train_labels_noise_ratios,
    #                                              lists_of_fine_labels_to_take_out
    #                                              # np.linspace(start=min_value,
    #                                              #             stop=max_value,
    #                                              #             num=total_number_of_points)
    #                                              ))
    #               }

    # if only_from_missing_values:
    #     first_values, second_values = google_sheets_api.get_values_from_columns(sheet_tab_name=sheet_tab_name,
    #                                                                             column_letters=['A', 'B'])
    #     if len(first_values) and len(second_values):
    #         last_first_value = first_values[-1]
    #         last_epsilon = second_values[-1]
    #         all_values = {i: (first_value, second_value) for i, (first_value, second_value) in
    #                       all_values.items()
    #                       if ((first_value == last_first_value and second_value > last_epsilon)
    #                           or (first_value > last_first_value))}

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
              None,
              maximize_ratio,
              multi_processing,
              fine_labels_to_take_out,
              negated_conditions
              ) for i, fine_labels_to_take_out in enumerate(lists_of_fine_labels_to_take_out)]

    if not utils.is_debug_mode():
        processes_num = min([len(datas), 10 if utils.is_local() else 100])
        process_map(work_on_value,
                    datas,
                    max_workers=processes_num)
    else:
        for data in datas:
            work_on_value(data)


if __name__ == '__main__':
    # data_str = 'military_vehicles'
    # main_model_name = binary_model_name = 'vit_b_16'
    # secondary_model_name = 'vit_l_16'
    # main_lr = secondary_lr = binary_lr = 0.0001
    # original_num_epochs = 10
    # secondary_num_epochs = 20
    # binary_num_epochs = 10
    # number_of_fine_classes = 24

    data_str = 'imagenet'
    main_model_name = binary_model_name = 'dinov2_vits14'
    secondary_model_name = 'dinov2_vitl14'
    # main_lr = 0.00001
    main_lr = secondary_lr = binary_lr = 0.000001
    original_num_epochs = 8
    secondary_num_epochs = 2
    binary_num_epochs = 5
    number_of_fine_classes = 42

    # data_str = 'openimage'
    # main_model_name = 'vit_b_16'
    # secondary_model_name = binary_model_name = 'dinov2_vits14'
    # main_lr = 0.0001
    # binary_lr = 0.000001
    # secondary_lr = 0.000001
    # original_num_epochs = 20
    # secondary_num_epochs = 20
    # binary_num_epochs = 4
    # number_of_fine_classes = 30

    binary_l_strs = list({f.split(f'e{binary_num_epochs - 1}_')[-1].replace('.npy', '')
                          for f in os.listdir('binary_results')
                          if f.startswith(f'{data_str}_{binary_model_name}')})

    # print(google_sheets_api.get_maximal_epsilon(tab_name=sheet_tab))

    sheet_tab_name = google_sheets_api.get_sheet_tab_name(main_model_name=main_model_name,
                                                          data_str=data_str,
                                                          secondary_model_name=secondary_model_name,
                                                          binary=len(binary_l_strs) > 0
                                                          )
    number_of_ratios = 10

    # lists_of_fine_labels_to_take_out = [list(range(i)) for i in range(number_of_fine_classes)]
    # lists_of_fine_labels_to_take_out = [[]]
    # lists_of_fine_labels_to_take_out = [list(range(number_of_fine_classes-1))]

    for (curr_secondary_model_name, curr_secondary_model_loss, curr_secondary_num_epochs, curr_secondary_lr) in \
            [(secondary_model_name, 'BCE', secondary_num_epochs, secondary_lr),
             # [None] * 4
             ]:
        for (curr_binary_l_strs, curr_binary_lr, curr_binary_num_epochs) in \
                [(binary_l_strs, binary_lr, binary_num_epochs),
                 # ([], None, None)
                 ]:
            for (lists_of_fine_labels_to_take_out, maximize_ratio, multi_processing) in \
                    [
                        # ([[]], True, True),
                     ([list(range(i)) for i in range(int(number_of_fine_classes / 2) + 1)], True, True)
                     ]:
                simulate_for_values(
                    total_number_of_points=1,
                    min_value=0.1,
                    max_value=0.1,
                    binary_l_strs=curr_binary_l_strs,
                    binary_lr=curr_binary_lr,
                    binary_num_epochs=curr_binary_num_epochs,
                    multi_processing=multi_processing,
                    secondary_model_name=curr_secondary_model_name,
                    secondary_model_loss=curr_secondary_model_loss,
                    secondary_num_epochs=curr_secondary_num_epochs,
                    secondary_lr=curr_secondary_lr,
                    # only_from_missing_values=True
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

    (x_values, balance_error_accuracies, error_f1s, constraint_f1s) = (
        google_sheets_api.get_values_from_columns(sheet_tab_name=sheet_tab_name,
                                                  column_letters=['B', 'D', 'G', 'J']))

    if data_str == 'military_vehicles':
        # Find the first occurrence where a > 0.5
        first_greater = np.argmax(x_values > 0.5)  # This gives the index of the first True in the condition

        x_values[first_greater] = 0.5

    # Create a mask for values <= 0.5
    mask = x_values <= 0.5

    x_values, balance_error_accuracies, error_f1s, constraint_f1s = \
        [a[mask] for a in [x_values, balance_error_accuracies, error_f1s, constraint_f1s]]


    plotting.plot_2d_metrics(data_str=data_str,
                             model_name=main_model_name,
                             x_values=x_values[1:],
                             metrics={'Balanced Error Accuracy': balance_error_accuracies[1:],
                                      'Error F1-Score': error_f1s[1:],
                                      'Constraints F1-Score': constraint_f1s[1:]},
                             style_dict={
                                 'Balanced Error Accuracy': ('k', '-'),  # Black solid line
                                 'Error F1-Score': ('k', ':'),  # Gray solid line
                                 'Constraints F1-Score': ('k', '--')  # Black dotted line
                             },
                             fontsize=24)
