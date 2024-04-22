import os
import utils

if utils.is_local():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import numpy as np
import typing
import multiprocessing as mp
from tqdm.contrib.concurrent import process_map

import data_preprocessing
import PyEDCR
import backbone_pipeline
import combined_fine_tuning
import neural_evaluation
import google_sheets_api


data_str = 'openimage'
main_model_name = new_model_name = 'tresnet_m'
main_lr = new_lr = 0.000001
original_num_epochs = 0

# secondary_model_name = 'vit_l_16_BCE'
secondary_model_name = None

sheet_id = '1JVLylVDMcYZgabsO2VbNCJLlrj7DSlMxYhY6YwQ38ck'
sheet_tab = f"Tresnet M on OpenImage Errors"


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
                 epsilon_index: int = None,
                 K_train: typing.List[typing.Tuple[int]] = None,
                 K_test: typing.List[typing.Tuple[int]] = None,
                 include_inconsistency_constraint: bool = False,
                 secondary_model_name: str = None,
                 secondary_model_loss: str = None,
                 secondary_num_epochs: int = None,
                 lower_predictions_indices: typing.List[int] = [],
                 binary_models: typing.List[str] = []):
        super(NeuralPyEDCR, self).__init__(data_str=data_str,
                                           main_model_name=main_model_name,
                                           combined=combined,
                                           loss=loss,
                                           lr=lr,
                                           original_num_epochs=original_num_epochs,
                                           epsilon=epsilon,
                                           epsilon_index=epsilon_index,
                                           K_train=K_train,
                                           K_test=K_test,
                                           include_inconsistency_constraint=include_inconsistency_constraint,
                                           secondary_model_name=secondary_model_name,
                                           secondary_model_loss=secondary_model_loss,
                                           secondary_num_epochs=secondary_num_epochs,
                                           lower_predictions_indices=lower_predictions_indices,
                                           binary_l_strs=binary_models)
        self.EDCR_num_epochs = EDCR_num_epochs
        self.neural_num_epochs = neural_num_epochs

    def run_training_new_model_pipeline(self,
                                        new_model_name: str,
                                        new_lr: float):

        perceived_examples_with_errors = set()
        for g in data_preprocessing.DataPreprocessor.granularities.values():
            perceived_examples_with_errors = perceived_examples_with_errors.union(set(
                np.where(self.get_predictions(test=False, g=g, stage='post_detection') == -1)[0]))

        perceived_examples_with_errors = np.array(list(perceived_examples_with_errors))

        print(utils.red_text(f'\nNumber of perceived train errors: {len(perceived_examples_with_errors)} / '
                             f'{self.T_train}\n'))

        # new_model_name = 'efficientnet_v2_s'
        preprocessor, fine_tuners, loaders, devices, num_fine_grain_classes, num_coarse_grain_classes = (
            backbone_pipeline.initiate(
                data_str=self.data_str,
                model_name=new_model_name,
                # weights=['IMAGENET1K_SWAG_E2E_V1'],
                lr=new_lr,
                combined=self.combined,
                error_indices=perceived_examples_with_errors,
                print_counts=False
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

        _, _, loaders, devices, _, _ = backbone_pipeline.initiate(
            data_str=self.data_str,
            model_name=new_model_name,
            lr=new_lr,
            combined=self.combined,
            error_indices=perceived_examples_with_errors,
            evaluation=True,
            print_counts=False)

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
                                                               lr=[self.lr],
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
                              new_lr: float):
        print('Started learning pipeline...\n')
        self.print_metrics(test=False, prior=True)

        for EDCR_epoch in range(self.EDCR_num_epochs):
            for g in data_preprocessing.DataPreprocessor.granularities.values():
                self.learn_detection_rules(g=g)
                self.apply_detection_rules(test=False, g=g)

            # self.run_training_new_model_pipeline(new_model_name=new_model_name,
            #                                      new_lr=new_lr)
            # self.print_metrics(test=False, prior=False, stage='post_detection')

            edcr_epoch_str = f'Finished EDCR epoch {EDCR_epoch + 1}/{self.EDCR_num_epochs}'

            print(utils.blue_text('\n' + '#' * 100 +
                                  '\n' + '#' * int((100 - len(edcr_epoch_str)) / 2) + edcr_epoch_str +
                                  '#' * (100 - int((100 - len(edcr_epoch_str)) / 2) - len(edcr_epoch_str)) +
                                  '\n' + '#' * 100 + '\n'))

        # self.learn_correction_rules(g=g)
        # self.learn_correction_rules_alt(g=g)

        print('\nRule learning completed\n')


def work_on_epsilon(epsilon: typing.Tuple[int, float]):
    print('#' * 25 + f'eps = {epsilon}' + '#' * 50)
    edcr = NeuralPyEDCR(data_str=data_str,
                        epsilon=epsilon[1],
                        epsilon_index=epsilon[0],
                        main_model_name=main_model_name,
                        combined=True,
                        loss='BCE',
                        lr=main_lr,
                        original_num_epochs=original_num_epochs,
                        include_inconsistency_constraint=False,
                        secondary_model_name=secondary_model_name,
                        secondary_num_epochs=2,
                        # binary_models=data_preprocessing.fine_grain_classes_str,
                        # lower_predictions_indices=lower_predictions_indices,
                        EDCR_num_epochs=1,
                        neural_num_epochs=1)
    edcr.print_metrics(test=True,
                       prior=True,
                       print_actual_errors_num=True)
    edcr.run_learning_pipeline(new_model_name=new_model_name,
                               new_lr=new_lr)
    edcr.run_error_detection_application_pipeline(test=True, print_results=False)
    # edcr.apply_new_model_on_test()


if __name__ == '__main__':
    empty_row_indices, total_value_num = google_sheets_api.find_empty_rows_in_column(tab_name=sheet_tab,
                                                                                     column='A')

    total_number_of_points = 300
    min_value = 0.1
    max_value = 0.3

    values_to_complete = total_number_of_points - total_value_num

    epsilons_to_take = [round((eps - 1) / 1000, 3) for eps in empty_row_indices
                        + [total_number_of_points - val for val in list(range(values_to_complete))]]
    print(epsilons_to_take)

    epsilons = [(x, y) for x, y in
                [(i, round(epsilon, 3)) for i, epsilon in enumerate(np.linspace(start=min_value / 100,
                                                                                stop=max_value,
                                                                                num=total_number_of_points))]
                if y in epsilons_to_take
                ]
    #
    # # For multiprocessing
    #
    # processes_num = min([len(epsilons), mp.cpu_count(), 10])
    # process_map(work_on_epsilon,
    #             epsilons,
    #             max_workers=processes_num)

    # For normal

    # Loop through epsilons sequentially (no multiprocessing)
    for epsilon in epsilons:
        work_on_epsilon(epsilon)  # Call your work function

    # for EDCR_num_epochs in [1]:
    #     for neural_num_epochs in [1]:

    # for lower_predictions_indices in [[2], [2, 3], [2, 3, 4]]:
    # print('\n' + '#' * 100 + '\n' +
    #       utils.blue_text(
    #           f'EDCR_num_epochs = {EDCR_num_epochs}, neural_num_epochs = {neural_num_epochs}'
    #           # f'lower_predictions_indices = {lower_predictions_indices}'
    #       )
    #       + '\n' + '#' * 100 + '\n')
