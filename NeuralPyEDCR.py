import numpy as np
import typing

import data_preprocessing
import PyEDCR
import utils
import vit_pipeline
import context_handlers


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
                 lower_predictions_indices: list[int] = []
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
                                           lower_predictions_indices=lower_predictions_indices)
        self.EDCR_num_epochs = EDCR_num_epochs
        self.neural_num_epochs = neural_num_epochs

    def run_training_new_model_pipeline(self):

        examples_with_errors = set()
        for g in data_preprocessing.granularities.values():
            examples_with_errors = examples_with_errors.union(set(
                np.where(self.get_predictions(test=False, g=g, stage='post_detection') == -1)[0]))

        examples_with_errors = np.array(list(examples_with_errors))

        print(utils.red_text(f'\nNumber of errors: {len(examples_with_errors)} / '
                             f'{self.get_predictions(test=False)[0].shape[0]}\n'))

        fine_tuners, loaders, devices, num_fine_grain_classes, num_coarse_grain_classes = vit_pipeline.initiate(
            lrs=[self.lr],
            combined=self.combined,
            debug=False,
            indices=examples_with_errors,
            num_epochs=self.neural_num_epochs
            # pretrained_path='models/vit_b_16_BCE_lr0.0001.pth'
            # train_eval_split=0.8
        )

        if self.correction_model is None:
            self.correction_model = fine_tuners[0]

        with context_handlers.ClearSession():
            vit_pipeline.fine_tune_combined_model(
                lrs=[self.lr],
                fine_tuner=self.correction_model,
                device=devices[0],
                loaders=loaders,
                num_fine_grain_classes=num_fine_grain_classes,
                num_coarse_grain_classes=num_coarse_grain_classes,
                loss=self.loss,
                save_files=False,
                debug=False,
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

        fine_tuners, loaders, devices, num_fine_grain_classes, num_coarse_grain_classes = vit_pipeline.initiate(
            lrs=[self.lr],
            combined=self.combined,
            debug=False,
            indices=examples_with_errors,
            evaluation=True,
            num_epochs=self.neural_num_epochs)

        evaluation_return_values = vit_pipeline.evaluate_combined_model(
            fine_tuner=self.correction_model,
            loaders=loaders,
            loss=self.loss,
            device=devices[0],
            split='train',
            print_results=False)

        self.pred_data['train']['post_detection'][data_preprocessing.granularities['fine']][
            examples_with_errors] = evaluation_return_values[2]
        self.pred_data['train']['post_detection'][data_preprocessing.granularities['coarse']][
            examples_with_errors] = evaluation_return_values[3]

    def apply_new_model_on_test(self,
                                print_results: bool = True):
        new_fine_predictions, new_coarse_predictions = (
            vit_pipeline.run_combined_evaluating_pipeline(split='test',
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
    epsilons = [0.1 * i for i in range(2, 3)]

    for eps in epsilons:
        print('#' * 25 + f'eps = {eps}' + '#' * 50)
        edcr = NeuralPyEDCR(epsilon=eps,
                            main_model_name='vit_b_16',
                            combined=True,
                            loss='BCE',
                            lr=0.0001,
                            original_num_epochs=20,
                            include_inconsistency_constraint=False,
                            secondary_model_name='vit_b_16_soft_marginal',
                            lower_predictions_indices=[2, 3, 4, 5],
                            EDCR_num_epochs=4,
                            neural_num_epochs=4)
        edcr.print_metrics(test=True, prior=True)
        edcr.run_learning_pipeline()
        edcr.run_error_detection_application_pipeline(test=True, print_results=False)
        edcr.apply_new_model_on_test()
        # edcr.run_error_correction_application_pipeline(test=test_bool)
        # edcr.apply_reversion_rules(g=gra)

        # precision_dict[gra]['initial'][epsilon] = edcr.original_test_precisions[gra]
        # recall_dict[gra]['initial'][epsilon] = edcr.original_test_recalls[gra]
        # precision_dict[gra]['pre_correction'][epsilon] = edcr.post_detection_test_precisions[gra]
        # recall_dict[gra]['pre_correction'][epsilon] = edcr.post_detection_test_recalls[gra]
        # precision_dict[gra]['post_correction'][epsilon] = edcr.post_correction_test_precisions[gra]
        # recall_dict[gra]['post_correction'][epsilon] = edcr.post_correction_test_recalls[gra]

    # folder = "experiment_1"
    #
    # if not os.path.exists(f'figs/{folder}'):
    #     os.mkdir(f'figs/{folder}')
    #
    # plot_per_class(ps=precision_dict,
    #                rs=recall_dict,
    #                folder="experiment_1")
    # plot_all(precision_dict, recall_dict, "experiment_1")
