import os
import utils

if utils.is_local():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import numpy as np
import typing
import multiprocessing as mp
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
                 epsilon: typing.Union[str, float],
                 EDCR_num_epochs: int,
                 neural_num_epochs: int,
                 sheet_index: int = None,
                 K_train: typing.Union[typing.List[typing.Tuple[int]], np.ndarray] = None,
                 K_test: typing.List[typing.Tuple[int]] = None,
                 include_inconsistency_constraint: bool = False,
                 secondary_model_name: str = None,
                 secondary_model_loss: str = None,
                 secondary_num_epochs: int = None,
                 lower_predictions_indices: typing.List[int] = [],
                 binary_l_strs: typing.List[str] = [],
                 binary_num_epochs: int = None,
                 binary_lr: typing.Union[str, float] = None,
                 experiment_name: str = None,
                 num_train_images_per_class: int = None):

        super(NeuralPyEDCR, self).__init__(data_str=data_str,
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
                                           binary_num_epochs=binary_num_epochs,
                                           binary_lr=binary_lr,
                                           experiment_name=experiment_name,
                                           num_train_images_per_class=num_train_images_per_class)
        self.EDCR_num_epochs = EDCR_num_epochs
        self.neural_num_epochs = neural_num_epochs

        relevant_predicted_indices = None

        if 'correct' in experiment_name:
            train_pred_correct_mask = np.ones_like(self.pred_data['train']['original'][
                                                       data_preprocessing.DataPreprocessor.granularities['fine']])

            for g in data_preprocessing.DataPreprocessor.granularities.values():
                train_pred_correct_mask &= self.get_where_predicted_correct(test=False, g=g)

            relevant_predicted_indices = np.where(train_pred_correct_mask == 1)[0]

        elif experiment_name == 'inconsistency':
            train_pred_inconsistency_mask = np.ones_like(self.pred_data['train']['original'][
                                                             data_preprocessing.DataPreprocessor.granularities['fine']])
            train_pred_inconsistency_mask &= self.get_where_predicted_inconsistently(test=False)

            relevant_predicted_indices = np.where(train_pred_inconsistency_mask == 1)[0]

        if num_train_images_per_class is not None:
            example_indices = []

            for i in range(len(self.preprocessor.fine_grain_classes_str)):
                i_indices_in_ground_truth = np.where(self.preprocessor.train_true_fine_data == i)[0]
                cls_idx = np.intersect1d(i_indices_in_ground_truth, relevant_predicted_indices)
                example_indices.extend(cls_idx[:num_train_images_per_class])
                # break

            self.K_train = np.array(example_indices)

            for g in data_preprocessing.DataPreprocessor.granularities.values():
                self.pred_data['train']['original'][g] = self.pred_data['train']['original'][g][self.K_train]

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
            evaluate_on_test=False,
            num_epochs=self.neural_num_epochs
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
            self.print_metrics(test=True, prior=False, stage='post_detection')

            where_fixed_initial_error = set()
            for g in data_preprocessing.DataPreprocessor.granularities.values():
                where_fixed_initial_error = where_fixed_initial_error.union(set(
                    np.where(self.get_where_predicted_correct(test=True, g=g, stage='post_detection') == 1)[0]
                ).intersection(set(np.where(self.get_where_predicted_incorrect(test=True, g=g) == 1)[0])))

            print(f'where_fixed_initial_error: {len(where_fixed_initial_error)}')

    def run_learning_pipeline(self,
                              new_model_name: str,
                              new_lr: float,
                              multi_process: bool = True):
        print('Started learning pipeline...\n')
        self.print_metrics(test=False, prior=True)

        for EDCR_epoch in range(self.EDCR_num_epochs):
            for g in data_preprocessing.DataPreprocessor.granularities.values():
                self.learn_detection_rules(g=g,
                                           multi_process=multi_process)
                self.apply_detection_rules(test=False, g=g)

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
            fine_predictions=self.get_predictions(test=False, g=self.preprocessor.granularities['fine']),
            coarse_predictions=self.get_predictions(test=False, g=self.preprocessor.granularities['coarse']),
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
            evaluate_on_test=False,
            num_epochs=2,
        )


def work_on_value(args):
    (epsilon_index,
     epsilon,
     data_str,
     main_model_name,
     main_lr,
     original_num_epochs,
     secondary_model_name,
     secondary_num_epochs,
     binary_l_strs,
     binary_lr,
     binary_num_epochs,
     new_model_name,
     new_lr,
     num_train_images_per_class,
     experiment_name
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
                        secondary_num_epochs=secondary_num_epochs,
                        binary_l_strs=binary_l_strs,
                        binary_lr=binary_lr,
                        binary_num_epochs=binary_num_epochs,
                        # lower_predictions_indices=lower_predictions_indices,
                        EDCR_num_epochs=1,
                        neural_num_epochs=1,
                        experiment_name=experiment_name,
                        # num_train_images_per_class=num_train_images_per_class
                        )
    # edcr.learn_error_binary_model(binary_model_name=main_model_name,
    #                               binary_lr=new_lr)
    edcr.print_metrics(test=True,
                       prior=True,
                       print_actual_errors_num=True)
    edcr.run_learning_pipeline(new_model_name=new_model_name,
                               new_lr=new_lr,
                               multi_process=True)
    edcr.run_error_detection_application_pipeline(test=True,
                                                  print_results=False,
                                                  save_to_google_sheets=True)
    # edcr.apply_new_model_on_test()


def simulate_for_values(total_number_of_points: int = 10,
                        min_value: float = 0.1,
                        max_value: float = 0.3,
                        multi_process: bool = True,
                        secondary_model_name: str = None,
                        secondary_num_epochs: int = None,
                        binary_l_strs: typing.List[str] = [],
                        binary_lr: typing.Union[str, float] = None,
                        binary_num_epochs: int = None,
                        num_train_images_per_class: typing.Sequence[int] = None,
                        experiment_name: str = None,
                        only_from_missing_values: bool = False):
    all_data_epsilon_values = {i: (image_value, epsilon) for i, (image_value, epsilon)
                               in enumerate(itertools.product(num_train_images_per_class,
                                                              np.linspace(start=min_value,
                                                                          stop=max_value,
                                                                          num=total_number_of_points)))}
    if only_from_missing_values:
        image_values, epsilons = google_sheets_api.get_values_from_columns(sheet_tab_name=sheet_tab_name,
                                                                           column_letters=['A', 'B'])
        if len(image_values) and len(epsilons):
            last_image_value = int(image_values[-1])
            last_epsilon = round(epsilons[-1], 2)
            all_data_epsilon_values = {i: (image_value, epsilon) for i, (image_value, epsilon) in
                                       all_data_epsilon_values.items()
                                       if ((int(image_value) == last_image_value and round(epsilon, 2) > last_epsilon)
                                           or (int(image_value) > last_image_value))}

    datas = [(i,
              round(epsilon, 3),
              data_str,
              main_model_name,
              main_lr,
              original_num_epochs,
              secondary_model_name,
              secondary_num_epochs,
              binary_l_strs,
              binary_lr,
              binary_num_epochs,
              new_model_name,
              new_lr,
              int(curr_num_train_images_per_class),
              experiment_name,
              ) for i, (curr_num_train_images_per_class, epsilon) in all_data_epsilon_values.items()]

    if multi_process:
        processes_num = min([len(datas), mp.cpu_count()])
        process_map(work_on_value,
                    datas,
                    max_workers=processes_num)
    else:
        for data in datas:
            work_on_value(data)


if __name__ == '__main__':
    data_str = 'military_vehicles'
    main_model_name = new_model_name = 'vit_b_16'
    main_lr = new_lr = binary_lr = 0.0001
    original_num_epochs = 20
    binary_num_epochs = 10
    sheet_tab_name = 'Copy of VIT_b_16 on Military Vehicles'
    max_num_train_images_per_class = 500

    # data_str = 'imagenet'
    # main_model_name = new_model_name = 'dinov2_vits14'
    # main_lr = new_lr = binary_lr = 0.000001
    # original_num_epochs = 8
    # binary_num_epochs = 5
    # sheet_tab_name = 'DINO V2 VIT14_s on ImageNet'
    # max_num_train_images_per_class = 1300

    binary_l_strs = list({f.split(f'e{binary_num_epochs - 1}_')[-1].replace('.npy', '')
                          for f in os.listdir('binary_results')
                          if f.startswith(f'{data_str}_{main_model_name}')})

    # secondary_model_name = 'vit_l_16_BCE'
    # secondary_model_name = 'dinov2_vitl14'

    # data_str = 'openimage'
    # main_model_name = new_model_name = 'tresnet_m'
    # main_lr = new_lr = 0.000001
    # original_num_epochs = 0

    # print(google_sheets_api.get_maximal_epsilon(tab_name=sheet_tab))

    simulate_for_values(
        total_number_of_points=100,
        min_value=0.1,
        max_value=0.3,
        # binary_l_strs=binary_l_strs,
        # binary_lr=binary_lr,
        # binary_num_epochs=binary_num_epochs,
        experiment_name='few correct',
        num_train_images_per_class=np.linspace(start=1,
                                               stop=1,
                                               num=1),
        # multi_process=True,
        # only_from_missing_values=True
    )

    # (images_per_class, epsilons, error_accuracies, error_f1s, consistency_error_accuracies,
    #  consistency_error_f1s, RCC_ratios) = (
    #     google_sheets_api.get_values_from_columns(sheet_tab_name=sheet_tab_name,
    #                                               column_letters=['A', 'B', 'C', 'D', 'E', 'F', 'H']))
    #
    # plotting.plot_3d_epsilons_ODD(images_per_class,
    #                               epsilons,
    #                               error_accuracies,
    #                               error_f1s,
    #                               consistency_error_accuracies,
    #                               consistency_error_f1s,
    #                               RCC_ratios)
