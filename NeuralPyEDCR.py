import numpy as np
import typing

import data_preprocessing
import PyEDCR
import utils
import vit_pipeline
import context_handlers
import combined_fine_tuning
import neural_evaluation


class NeuralPyEDCR(PyEDCR.EDCR):
    def __init__(self,
                 main_model_name: str,
                 combined: bool,
                 loss: str,
                 lr: typing.Union[str, float],
                 original_num_epochs: int,
                 epsilon: typing.Union[str, float],
                 EDCR_num_epochs: int,
                 neural_num_epochs: int,
                 K_train: list[(int, int)] = None,
                 K_test: list[(int, int)] = None,
                 include_inconsistency_constraint: bool = False,
                 secondary_model_name: str = None,
                 lower_predictions_indices: list[int] = [],
                 binary_models: list[str] = []
                 ):
        super(NeuralPyEDCR, self).__init__(main_model_name=main_model_name,
                                           combined=combined,
                                           loss=loss,
                                           lr=lr,
                                           original_num_epochs=original_num_epochs,
                                           epsilon=epsilon,
                                           K_train=K_train,
                                           K_test=K_test,
                                           include_inconsistency_constraint=include_inconsistency_constraint,
                                           secondary_model_name=secondary_model_name,
                                           lower_predictions_indices=lower_predictions_indices,
                                           binary_l_strs=binary_models)
        self.EDCR_num_epochs = EDCR_num_epochs
        self.neural_num_epochs = neural_num_epochs

    def run_training_new_model_pipeline(self):

        perceived_examples_with_errors = set()
        for g in data_preprocessing.granularities.values():
            perceived_examples_with_errors = perceived_examples_with_errors.union(set(
                np.where(self.get_predictions(test=False, g=g, stage='post_detection') == -1)[0]))

        perceived_examples_with_errors = np.array(list(perceived_examples_with_errors))

        print(utils.red_text(f'\nNumber of perceived train errors: {len(perceived_examples_with_errors)} / '
                             f'{self.T_train}\n'))

        new_model_name = 'vit_b_16'
        fine_tuners, loaders, devices, num_fine_grain_classes, num_coarse_grain_classes = vit_pipeline.initiate(
            model_names=[new_model_name],
            # weights=['IMAGENET1K_SWAG_E2E_V1'],
            lrs=[self.lr],
            combined=self.combined,
            error_indices=perceived_examples_with_errors,
            get_indices=True
            # train_eval_split=0.8
        )

        if self.correction_model is None:
            self.correction_model = fine_tuners[0]

        with context_handlers.ClearSession():
            combined_fine_tuning.fine_tune_combined_model(
                lrs=[self.lr],
                fine_tuner=self.correction_model,
                device=devices[0],
                loaders=loaders,
                loss=self.loss,
                save_files=False,
                evaluate_on_test=False,
                num_epochs=self.neural_num_epochs
                # Y_original_fine=
                # self.pred_data['train']['mid_learning'][data_preprocessing.granularities['fine']][
                #     examples_with_errors],
                # Y_original_coarse=
                # self.pred_data['train']['mid_learning'][data_preprocessing.granularities['coarse']][
                #     examples_with_errors]
            )
            print('#' * 100)

        _, loaders, devices, _, _ = vit_pipeline.initiate(
            model_names=[new_model_name],
            lrs=[self.lr],
            combined=self.combined,
            error_indices=perceived_examples_with_errors,
            evaluation=True,
            get_indices=True)

        evaluation_return_values = neural_evaluation.evaluate_combined_model(
            fine_tuner=self.correction_model,
            loaders=loaders,
            loss=self.loss,
            device=devices[0],
            split='train',
            print_results=True)

        new_fine_predictions, new_coarse_predictions = evaluation_return_values[2], evaluation_return_values[3]

        self.pred_data['train']['post_detection'][data_preprocessing.granularities['fine']][
            perceived_examples_with_errors] = new_fine_predictions
        self.pred_data['train']['post_detection'][data_preprocessing.granularities['coarse']][
            perceived_examples_with_errors] = new_coarse_predictions

    def apply_new_model_on_test(self,
                                print_results: bool = True):
        new_fine_predictions, new_coarse_predictions = (
            neural_evaluation.run_combined_evaluating_pipeline(model_name='vit_b_16',
                                                               split='test',
                                                               lrs=[self.lr],
                                                               loss=self.loss,
                                                               num_epochs=self.neural_num_epochs,
                                                               pretrained_fine_tuner=self.correction_model,
                                                               save_files=False,
                                                               print_results=False))

        for g in data_preprocessing.granularities.values():
            old_test_g_predictions = self.get_predictions(test=True, g=g, stage='post_detection')
            new_test_g_predictions = new_fine_predictions if g.g_str == 'fine' else new_coarse_predictions

            self.pred_data['test']['post_detection'][g] = np.where(old_test_g_predictions == -1,
                                                                   new_test_g_predictions,
                                                                   old_test_g_predictions)
        if print_results:
            self.print_metrics(test=True, prior=False, stage='post_detection')

            where_fixed_initial_error = set()
            for g in data_preprocessing.granularities.values():
                where_fixed_initial_error = where_fixed_initial_error.union(set(
                    np.where(self.get_where_predicted_correct(test=True, g=g, stage='post_detection') == 1)[0]
                ).intersection(set(np.where(self.get_where_predicted_incorrect(test=True, g=g) == 1)[0])))

            print(f'where_fixed_initial_error: {len(where_fixed_initial_error)}')

    def run_learning_pipeline(self):
        print('Started learning pipeline...\n')
        self.print_metrics(test=False, prior=True)

        for EDCR_epoch in range(self.EDCR_num_epochs):
            for g in data_preprocessing.granularities.values():
                self.learn_detection_rules(g=g)
                self.apply_detection_rules(test=False, g=g)

            self.run_training_new_model_pipeline()
            # self.print_metrics(test=False, prior=False, stage='post_detection')

            edcr_epoch_str = f'Finished EDCR epoch {EDCR_epoch + 1}/{self.EDCR_num_epochs}'

            print(utils.blue_text('\n' + '#' * 100 +
                                  '\n' + '#' * int((100 - len(edcr_epoch_str)) / 2) + edcr_epoch_str +
                                  '#' * (100 - int((100 - len(edcr_epoch_str)) / 2) - len(edcr_epoch_str)) +
                                  '\n' + '#' * 100 + '\n'))

        # self.learn_correction_rules(g=g)
        # self.learn_correction_rules_alt(g=g)

        print('\nRule learning completed\n')


if __name__ == '__main__':
    epsilons = [0.2]

    for EDCR_num_epochs in [4]:
        for neural_num_epochs in [1]:
            # for lower_predictions_indices in [[2], [2, 3], [2, 3, 4]]:
            print('\n' + '#' * 100 + '\n' +
                  utils.blue_text(f'EDCR_num_epochs = {EDCR_num_epochs}, neural_num_epochs = {neural_num_epochs}'
                                  # f'lower_predictions_indices = {lower_predictions_indices}'
                                  )
                  + '\n' + '#' * 100 + '\n')
            for eps in epsilons:
                print('#' * 25 + f'eps = {eps}' + '#' * 50)
                edcr = NeuralPyEDCR(epsilon=eps,
                                    main_model_name='vit_b_16',
                                    combined=True,
                                    loss='BCE',
                                    lr=0.0001,
                                    original_num_epochs=20,
                                    include_inconsistency_constraint=False,
                                    # secondary_model_name='vit_l_16_BCE',
                                    binary_models=data_preprocessing.fine_grain_classes_str,
                                    # lower_predictions_indices=lower_predictions_indices,
                                    EDCR_num_epochs=EDCR_num_epochs,
                                    neural_num_epochs=neural_num_epochs)
                edcr.print_metrics(test=True, prior=True, print_actual_errors_num=True)
                edcr.run_learning_pipeline()
                edcr.run_error_detection_application_pipeline(test=True, print_results=False)
                edcr.apply_new_model_on_test()
