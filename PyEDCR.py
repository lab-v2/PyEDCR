from __future__ import annotations

import typing
import numpy as np
import warnings

warnings.filterwarnings('ignore')

import utils
import data_preprocessing
import neural_metrics
import context_handlers
import symbolic_metrics
import conditions
import rules
import models


class EDCR:
    """
    Performs error detection and correction based on model predictions.

    This class aims to identify and rectify errors in predictions made by a
    specified neural network model. It utilizes prediction data from both
    fine-grained and coarse-grained model runs to enhance its accuracy.

    Attributes:
        main_model_name (str): Name of the primary model  used for predictions.
        combined (bool): Whether combined features (coarse and fine) were used during training.
        loss (str): Loss function used during training.
        lr: Learning rate used during training.
        epsilon: Value using for constraint in getting rules
    """

    def __init__(self,
                 data_str: str,
                 main_model_name: str,
                 combined: bool,
                 loss: str,
                 lr: typing.Union[str, float],
                 original_num_epochs: int,
                 epsilon: typing.Union[str, float],
                 K_train: list[(int, int)] = None,
                 K_test: list[(int, int)] = None,
                 include_inconsistency_constraint: bool = False,
                 secondary_model_name: str = None,
                 lower_predictions_indices: list[int] = [],
                 binary_l_strs: list[str] = []):
        self.data_str = data_str
        self.preprocessor = data_preprocessing.DataPreprocessor(data_str=data_str)
        self.main_model_name = main_model_name
        self.combined = combined
        self.loss = loss
        self.lr = lr
        self.num_epochs = original_num_epochs
        self.epsilon = epsilon
        self.secondary_model_name = secondary_model_name
        self.lower_predictions_indices = lower_predictions_indices
        self.binary_l_strs = binary_l_strs

        pred_paths: dict[str, dict] = {
            'test' if test else 'train': {g_str: models.get_filepath(model_name=main_model_name,
                                                                     combined=combined,
                                                                     test=test,
                                                                     granularity=g_str,
                                                                     loss=loss,
                                                                     lr=lr,
                                                                     pred=True,
                                                                     epoch=original_num_epochs)
                                          for g_str in data_preprocessing.DataPreprocessor.granularities_str}
            for test in [True, False]}

        self.K_train = data_preprocessing.expand_ranges(K_train) if K_train is not None \
            else data_preprocessing.expand_ranges([(0, np.load(pred_paths['train']['fine']).shape[0] - 1)])
        self.K_test = data_preprocessing.expand_ranges(K_test) if K_test is not None \
            else data_preprocessing.expand_ranges([(0, np.load(pred_paths['test']['fine']).shape[0] - 1)])
        self.T_train = np.load(pred_paths['train']['fine']).shape[0]
        self.T_test = np.load(pred_paths['test']['fine']).shape[0]

        self.pred_data = \
            {test_or_train: {'original': {g: np.load(pred_paths[test_or_train][g.g_str])[
                self.K_test if test_or_train == 'test' else self.K_train]
                                          for g in data_preprocessing.DataPreprocessor.granularities.values()},
                             'mid_learning': {g: np.load(pred_paths[test_or_train][g.g_str])[
                                 self.K_test if test_or_train == 'test' else self.K_train]
                                              for g in data_preprocessing.DataPreprocessor.granularities.values()},
                             'post_detection': {g: np.load(pred_paths[test_or_train][g.g_str])[
                                 self.K_test if test_or_train == 'test' else self.K_train]
                                                for g in data_preprocessing.DataPreprocessor.granularities.values()},
                             'post_correction': {
                                 g: np.load(pred_paths[test_or_train][g.g_str])[
                                     self.K_test if test_or_train == 'test' else self.K_train]
                                 for g in data_preprocessing.DataPreprocessor.granularities.values()}}
             for test_or_train in ['test', 'train']}

        self.condition_datas = {g: {conditions.PredCondition(l=l)
                                    for l in self.preprocessor.get_labels(g).values()}
                                for g in data_preprocessing.DataPreprocessor.granularities.values()}

        if self.secondary_model_name is not None:
            secondary_loss = secondary_model_name.split('_')[-1]
            pred_paths['secondary_model'] = {
                'test' if test else 'train': {g_str: models.get_filepath(model_name=main_model_name,
                                                                         combined=combined,
                                                                         test=test,
                                                                         granularity=g_str,
                                                                         loss=secondary_loss,
                                                                         lr=lr,
                                                                         pred=True,
                                                                         epoch=original_num_epochs)
                                              for g_str in data_preprocessing.DataPreprocessor.granularities_str}
                for test in [True, False]}

            self.pred_data['secondary_model'] = \
                {test_or_train: {g: np.load(pred_paths['secondary_model'][test_or_train][g.g_str])
                                 for g in data_preprocessing.DataPreprocessor.granularities.values()}
                 for test_or_train in ['test', 'train']}

            for g in data_preprocessing.DataPreprocessor.granularities.values():
                self.condition_datas[g] = self.condition_datas[g].union(
                    {conditions.PredCondition(l=l, secondary_model=True)
                     for l in self.preprocessor.get_labels(g).values()})

        for lower_prediction_index in self.lower_predictions_indices:
            lower_prediction_key = f'lower_{lower_prediction_index}'

            pred_paths[lower_prediction_key] = {
                'test' if test else 'train':
                    {g_str: models.get_filepath(model_name=main_model_name,
                                                combined=combined,
                                                test=test,
                                                granularity=g_str,
                                                loss=self.loss,
                                                lr=lr,
                                                pred=True,
                                                epoch=original_num_epochs,
                                                lower_prediction_index=lower_prediction_index)
                     for g_str in data_preprocessing.DataPreprocessor.granularities_str}
                for test in [True, False]}

            self.pred_data[lower_prediction_key] = \
                {test_or_train: {g: np.load(pred_paths[lower_prediction_key][test_or_train][g.g_str])
                                 for g in data_preprocessing.DataPreprocessor.granularities.values()}
                 for test_or_train in ['test', 'train']}

            for g in data_preprocessing.DataPreprocessor.granularities.values():
                self.condition_datas[g] = self.condition_datas[g].union(
                    {conditions.PredCondition(l=l, lower_prediction_index=lower_prediction_index)
                     for l in self.preprocessor.get_labels(g).values()})

        self.pred_data['binary'] = {}

        for l_str in binary_l_strs:
            l = self.preprocessor.fine_grain_labels[l_str]
            pred_paths[l] = {
                'test' if test else 'train': models.get_filepath(model_name=main_model_name,
                                                                 l=l,
                                                                 test=test,
                                                                 loss=loss,
                                                                 lr=lr,
                                                                 pred=True,
                                                                 epoch=10)

                for test in [True, False]}

            self.pred_data['binary'][l] = \
                {test_or_train: {g: np.load(pred_paths[l][test_or_train])
                                 for g in data_preprocessing.DataPreprocessor.granularities.values()}
                 for test_or_train in ['test', 'train']}

            for g in data_preprocessing.DataPreprocessor.granularities.values():
                self.condition_datas[g] = self.condition_datas[g].union(
                    {conditions.PredCondition(l=l, binary=True)})

        if include_inconsistency_constraint:
            for g in data_preprocessing.DataPreprocessor.granularities.values():
                self.condition_datas[g] = self.condition_datas[g].union({conditions.InconsistencyCondition()})

        self.CC_all = {g: set() for g in data_preprocessing.DataPreprocessor.granularities.values()}

        self.num_predicted_l = {'original': {g: {}
                                             for g in data_preprocessing.DataPreprocessor.granularities.values()},
                                'post_detection': {g: {}
                                                   for g in data_preprocessing.DataPreprocessor.granularities.values()},
                                'post_correction':
                                    {g: {} for g in data_preprocessing.DataPreprocessor.granularities.values()}}

        for g in data_preprocessing.DataPreprocessor.granularities.values():
            for l in self.preprocessor.get_labels(g).values():
                self.num_predicted_l['original'][g][l] = np.sum(self.get_where_label_is_l(pred=True,
                                                                                          test=True,
                                                                                          l=l,
                                                                                          stage='original'))

        actual_examples_with_errors = set()
        for g in data_preprocessing.DataPreprocessor.granularities.values():
            actual_examples_with_errors = actual_examples_with_errors.union(set(
                np.where(self.get_where_predicted_incorrect(test=True, g=g) == 1)[0]))

        self.actual_examples_with_errors = np.array(list(actual_examples_with_errors))

        self.error_detection_rules: dict[data_preprocessing.Label, rules.ErrorDetectionRule] = {}
        self.error_correction_rules: dict[data_preprocessing.Label, rules.ErrorCorrectionRule] = {}

        self.correction_model = None

        print(utils.blue_text(
            f"Num of fine conditions: "
            f"{len(self.condition_datas[data_preprocessing.DataPreprocessor.granularities['fine']])}\n"
            f"Num of coarse conditions: "
            f"{len(self.condition_datas[data_preprocessing.DataPreprocessor.granularities['coarse']])}\n"))

    def set_error_detection_rules(self, input_rules: typing.Dict[data_preprocessing.Label, {conditions.Condition}]):
        """
        Manually sets the error detection rule dictionary.

        :params rules: A dictionary mapping label instances to error detection rule objects.
        """
        error_detection_rules = {}
        for label, DC_l in input_rules.items():
            error_detection_rules[label] = rules.ErrorDetectionRule(label, DC_l)
        self.error_detection_rules = error_detection_rules

    @staticmethod
    def get_C_str(CC: set[conditions.Condition]) -> str:
        return '{' + ', '.join(str(obj) for obj in CC) + '}'

    def get_predictions(self,
                        test: bool,
                        g: data_preprocessing.Granularity = None,
                        stage: str = 'original',
                        secondary: bool = False,
                        lower_predictions: bool = False,
                        binary: bool = False) -> typing.Union[np.array, tuple[np.array]]:
        """Retrieves prediction data based on specified test/train mode.

        :param binary:
        :param lower_predictions:
        :param secondary:
        :param stage:
        :param g: The granularity level
        :param test: whether to get data from train or test set
        :return: Fine-grained and coarse-grained prediction data.
        """
        test_str = 'test' if test else 'train'

        if secondary:
            pred_data = self.pred_data['secondary_model'][test_str]
        elif lower_predictions:
            pred_data = {g: {f'lower_{lower_prediction_index}': self.pred_data[
                f'lower_{lower_prediction_index}'][test_str][g]
                             for lower_prediction_index in self.lower_predictions_indices}
                         for g in data_preprocessing.DataPreprocessor.granularities.values()}
        elif binary:
            pred_data = self
        else:
            pred_data = self.pred_data[test_str][stage]

        if g is not None:
            return pred_data[g]

        pred_fine_data, pred_coarse_data = [pred_data[g]
                                            for g in data_preprocessing.DataPreprocessor.granularities.values()]

        return pred_fine_data, pred_coarse_data

    def get_where_label_is_l(self,
                             pred: bool,
                             test: bool,
                             l: data_preprocessing.Label,
                             stage: str = 'original',
                             secondary: bool = False) -> np.array:
        """ Retrieves indices of instances where the specified label is present.

        :param secondary:
        :param stage:
        :param pred: True for prediction, False for ground truth
        :param test: whether to get data from train or test set
        :param l: The label to search for.
        :return: A boolean array indicating which instances have the given label.
        """
        data = self.get_predictions(test=test, g=l.g, stage=stage, secondary=secondary) if pred else (
            self.preprocessor.get_ground_truths(test=test, K=self.K_test if test else self.K_train, g=l.g))
        return np.where(data == l.index, 1, 0)

    @staticmethod
    def get_where_label_is_l_in_data(l: data_preprocessing.Label,
                                     test_pred_fine_data: np.array,
                                     test_pred_coarse_data: np.array) -> np.array:
        """ Retrieves indices of instances where the specified label is present.

        :param test_pred_coarse_data:
        :param test_pred_fine_data:
        :param l: The label to search for.
        :return: A boolean array indicating which instances have the given label.
        """
        data = test_pred_fine_data if l.g == data_preprocessing.DataPreprocessor.granularities['fine'] \
            else test_pred_coarse_data
        where_label_is_l = np.where(data == l.index, 1, 0)
        return where_label_is_l

    def print_metrics(self,
                      test: bool,
                      prior: bool,
                      print_inconsistencies: bool = True,
                      stage: str = 'original',
                      print_actual_errors_num: bool = False):

        """Prints performance metrics for given test/train data.

        Calculates and prints various metrics (accuracy, precision, recall, etc.)
        using appropriate true labels and prediction data based on the specified mode.

        :param print_actual_errors_num:
        :param stage:
        :param print_inconsistencies: whether to print the inconsistencies metric or not
        :param prior:
        :param test: whether to get data from train or test set
        """

        original_pred_fine_data, original_pred_coarse_data = None, None

        # if stage != 'original':
        #     original_pred_fine_data, original_pred_coarse_data = self.get_predictions(test=test, stage='original')

        pred_fine_data, pred_coarse_data = self.get_predictions(test=test, stage=stage)
        true_fine_data, true_coarse_data = self.preprocessor.get_ground_truths(test=test,
                                                                               K=self.K_test if test else self.K_train)

        neural_metrics.get_and_print_metrics(pred_fine_data=pred_fine_data,
                                             pred_coarse_data=pred_coarse_data,
                                             loss=self.loss,
                                             true_fine_data=true_fine_data,
                                             true_coarse_data=true_coarse_data,
                                             test=test,
                                             prior=prior,
                                             combined=self.combined,
                                             model_name=self.main_model_name,
                                             lr=self.lr,
                                             print_inconsistencies=print_inconsistencies,
                                             original_pred_fine_data=original_pred_fine_data,
                                             original_pred_coarse_data=original_pred_coarse_data)

        # Calculate boolean masks for each condition
        correct_coarse_incorrect_fine = np.logical_and(pred_coarse_data == true_coarse_data,
                                                       pred_fine_data != true_fine_data)
        incorrect_coarse_correct_fine = np.logical_and(pred_coarse_data != true_coarse_data,
                                                       pred_fine_data == true_fine_data)
        incorrect_both = np.logical_and(pred_coarse_data != true_coarse_data, pred_fine_data != true_fine_data)
        correct_both = np.logical_and(pred_coarse_data == true_coarse_data,
                                      pred_fine_data == true_fine_data)  # for completeness

        # Calculate total number of examples
        total_examples = len(pred_coarse_data)  # Assuming shapes are compatible

        fractions = {
            'correct_coarse_incorrect_fine': np.sum(correct_coarse_incorrect_fine) / total_examples,
            'incorrect_coarse_correct_fine': np.sum(incorrect_coarse_correct_fine) / total_examples,
            'incorrect_both': np.sum(incorrect_both) / total_examples,
            'correct_both': np.sum(correct_both) / total_examples
        }

        print(f"fraction of error associate with each type: \n",
              f"correct_coarse_incorrect_fine: {round(fractions['correct_coarse_incorrect_fine'], 2)} \n",
              f"incorrect_coarse_correct_fine: {round(fractions['incorrect_coarse_correct_fine'], 2)} \n",
              f"incorrect_both: {round(fractions['incorrect_both'], 2)} \n",
              f"correct_both: {round(fractions['correct_both'], 2)} \n")

        if print_actual_errors_num:
            print(utils.red_text(f'\nNumber of actual errors on test: {len(self.actual_examples_with_errors)} / '
                                 f'{self.T_test}\n'))

    def get_where_predicted_correct(self,
                                    test: bool,
                                    g: data_preprocessing.Granularity,
                                    stage: str = 'original') -> np.array:
        """Calculates true positive mask for given granularity and label.

        :param stage:
        :param test: whether to get data from train or test set
        :param g: The granularity level.
        :return: A mask with 1s for true positive instances, 0s otherwise.
        """
        ground_truth = self.preprocessor.get_ground_truths(test=test, K=self.K_test if test else self.K_train, g=g)
        return np.where(self.get_predictions(test=test, g=g, stage=stage) == ground_truth, 1, 0)

    def get_where_predicted_correct_in_data(self,
                                            g: data_preprocessing.Granularity,
                                            test_pred_fine_data: np.array,
                                            test_pred_coarse_data: np.array) -> np.array:
        """Calculates true positive mask for given granularity and label.

        :param test_pred_fine_data: 
        :param test_pred_coarse_data:
        :param g: The granularity level.
        :return: A mask with 1s for true positive instances, 0s otherwise.
        """
        ground_truth = self.preprocessor.get_ground_truths(test=True, K=self.K_test, g=g)
        prediction = test_pred_fine_data if g == data_preprocessing.DataPreprocessor.granularities['fine'] \
            else test_pred_coarse_data
        return np.where(prediction == ground_truth, 1, 0)

    def get_where_predicted_incorrect(self,
                                      test: bool,
                                      g: data_preprocessing.Granularity,
                                      stage: str = 'original') -> np.array:
        """Calculates false positive mask for given granularity and label.

        :param stage:
        :param test: whether to get data from train or test set
        :param g: The granularity level
        :return: A mask with 1s for false positive instances, 0s otherwise.
        """
        return 1 - self.get_where_predicted_correct(test=test, g=g, stage=stage)

    def get_where_predicted_incorrect_in_data(self,
                                              g: data_preprocessing.Granularity,
                                              test_pred_fine_data: np.array,
                                              test_pred_coarse_data: np.array) -> np.array:
        """Calculates false positive mask for given granularity and label.

        :param test_pred_coarse_data:
        :param test_pred_fine_data:
        :param g: The granularity level
        :return: A mask with 1s for false positive instances, 0s otherwise.
        """
        return 1 - self.get_where_predicted_correct_in_data(g=g,
                                                            test_pred_fine_data=test_pred_fine_data,
                                                            test_pred_coarse_data=test_pred_coarse_data)

    def get_where_tp_l(self,
                       test: bool,
                       l: data_preprocessing.Label,
                       stage: str = 'original', ) -> np.array:
        """ Retrieves indices of training instances where the true label is l and the model correctly predicted l.

        :param stage:
        :param test: whether to get data from train or test set
        :param l: The label to query.
        :return: A boolean array indicating which training instances satisfy the criteria.
        """
        return (self.get_where_label_is_l(pred=True, test=test, l=l, stage=stage) *
                self.get_where_predicted_correct(test=test, g=l.g, stage=stage))

    def get_where_tp_l_in_data(self,
                               l: data_preprocessing.Label,
                               test_pred_fine_data: np.array,
                               test_pred_coarse_data: np.array) -> np.array:
        """ Retrieves indices of training instances where the true label is l and the model correctly predicted l.

        :param test_pred_coarse_data:
        :param test_pred_fine_data:
        :param l: The label to query.
        :return: A boolean array indicating which training instances satisfy the criteria.
        """
        return (self.get_where_label_is_l_in_data(l=l,
                                                  test_pred_fine_data=test_pred_fine_data,
                                                  test_pred_coarse_data=test_pred_coarse_data) *
                self.get_where_predicted_correct_in_data(g=l.g,
                                                         test_pred_fine_data=test_pred_fine_data,
                                                         test_pred_coarse_data=test_pred_coarse_data))

    def get_where_fp_l(self,
                       test: bool,
                       l: data_preprocessing.Label,
                       stage: str = 'original') -> np.array:
        """ Retrieves indices of instances where the predicted label is l and the ground truth is not l.

        :param stage:
        :param test: whether to get data from train or test set
        :param l: The label to query.
        :return: A boolean array indicating which instances satisfy the criteria.
        """
        return (self.get_where_label_is_l(pred=True, test=test, l=l, stage=stage) *
                self.get_where_predicted_incorrect(test=test, g=l.g, stage=stage))

    def get_where_fp_l_in_data(self,
                               l: data_preprocessing.Label,
                               test_pred_fine_data: np.array,
                               test_pred_coarse_data: np.array) -> np.array:
        """ Retrieves indices of instances where the predicted label is l and the ground truth is not l.

        :param test_pred_coarse_data:
        :param test_pred_fine_data:
        :param l: The label to query.
        :return: A boolean array indicating which instances satisfy the criteria.
        """
        return (self.get_where_label_is_l_in_data(l=l,
                                                  test_pred_fine_data=test_pred_fine_data,
                                                  test_pred_coarse_data=test_pred_coarse_data) *
                self.get_where_predicted_incorrect_in_data(g=l.g,
                                                           test_pred_fine_data=test_pred_fine_data,
                                                           test_pred_coarse_data=test_pred_coarse_data))

    def get_l_precision_and_recall(self,
                                   test: bool,
                                   l: data_preprocessing.Label,
                                   stage: str = 'original'):
        t_p_l = np.sum(self.get_where_tp_l(test=test, l=l, stage=stage))
        f_p_l = np.sum(self.get_where_fp_l(test=test, l=l, stage=stage))
        N_l_gt = np.sum(self.get_where_label_is_l(test=test, pred=False, l=l, stage=stage))

        p_l = t_p_l / (t_p_l + f_p_l) if not (t_p_l == 0 and f_p_l == 0) else 0
        r_l = t_p_l / N_l_gt if not N_l_gt == 0 else 0

        return p_l, r_l

    def get_g_precision_and_recall(self,
                                   test: bool,
                                   g: data_preprocessing.Granularity,
                                   stage: str = 'original',
                                   test_pred_fine_data: np.array = None,
                                   test_pred_coarse_data: np.array = None) -> (dict[data_preprocessing.Label, float],
                                                                               dict[data_preprocessing.Label, float]):
        p_g = {}
        r_g = {}

        if test_pred_fine_data is None and test_pred_coarse_data is None:
            for l in self.preprocessor.get_labels(g).values():
                p_g[l], r_g[l] = self.get_l_precision_and_recall(test=test, l=l, stage=stage)
        else:
            for l in self.preprocessor.get_labels(g).values():
                t_p_l = np.sum(self.get_where_tp_l_in_data(l=l,
                                                           test_pred_fine_data=test_pred_fine_data,
                                                           test_pred_coarse_data=test_pred_coarse_data))
                f_p_l = np.sum(self.get_where_fp_l_in_data(l=l,
                                                           test_pred_fine_data=test_pred_fine_data,
                                                           test_pred_coarse_data=test_pred_coarse_data))
                N_l_gt = np.sum(self.get_where_label_is_l_in_data(l=l,
                                                                  test_pred_fine_data=test_pred_fine_data,
                                                                  test_pred_coarse_data=test_pred_coarse_data))

                p_g[l] = t_p_l / (t_p_l + f_p_l)
                r_g[l] = t_p_l / N_l_gt

        return p_g, r_g

    def get_NEG_l_C(self,
                    l: data_preprocessing.Label,
                    C: set[conditions.Condition],
                    stage: str = 'original') -> int:
        """Calculate the number of train samples that satisfy any of the conditions and are true positive.

        :param stage:
        :param C: A set of `Condition` objects.
        :param l: The label of interest.
        :return: The number of instances that is true negative and satisfying all conditions.
        """

        train_pred_fine_data, train_pred_coarse_data = self.get_predictions(test=False, stage=stage)
        secondary_train_pred_fine_data, secondary_train_pred_coarse_data = (
            self.get_predictions(test=False, secondary=True)) if self.secondary_model_name is not None else (None, None)
        lower_train_pred_fine_data, lower_train_pred_coarse_data = (
            self.get_predictions(test=False, lower_predictions=True)) \
            if len(self.lower_predictions_indices) else (None, None)
        binary_data = self.pred_data['binary']

        where_any_conditions_satisfied_on_train = (
            rules.Rule.get_where_any_conditions_satisfied(C=C,
                                                          fine_data=train_pred_fine_data,
                                                          coarse_data=train_pred_coarse_data,
                                                          secondary_fine_data=secondary_train_pred_fine_data,
                                                          secondary_coarse_data=secondary_train_pred_coarse_data,
                                                          lower_predictions_fine_data=lower_train_pred_fine_data,
                                                          lower_predictions_coarse_data=lower_train_pred_coarse_data,
                                                          binary_data=binary_data))
        where_train_tp_l = self.get_where_tp_l(test=False, l=l, stage=stage)
        NEG_l = np.sum(where_train_tp_l * where_any_conditions_satisfied_on_train)

        return NEG_l

    def get_BOD_l_C(self,
                    C: set[conditions.Condition]) -> int:
        """Calculate the number of train samples that satisfy any conditions for some set of condition.

        :param C: A set of `Condition` objects.
        :return: The number of instances that are false negative and satisfying some conditions.
        """
        train_pred_fine_data, train_pred_coarse_data = self.get_predictions(test=False)
        secondary_train_pred_fine_data, secondary_train_pred_coarse_data = (
            self.get_predictions(test=False, secondary=True)) if self.secondary_model_name is not None else (None, None)
        lower_train_pred_fine_data, lower_train_pred_coarse_data = (
            self.get_predictions(test=False, lower_predictions=True)) \
            if len(self.lower_predictions_indices) else (None, None)
        binary_data = self.pred_data['binary']

        where_any_conditions_satisfied_on_train = (
            rules.Rule.get_where_any_conditions_satisfied(C=C,
                                                          fine_data=train_pred_fine_data,
                                                          coarse_data=train_pred_coarse_data,
                                                          secondary_fine_data=secondary_train_pred_fine_data,
                                                          secondary_coarse_data=secondary_train_pred_coarse_data,
                                                          lower_predictions_fine_data=lower_train_pred_fine_data,
                                                          lower_predictions_coarse_data=lower_train_pred_coarse_data,
                                                          binary_data=binary_data
                                                          ))
        BOD_l = np.sum(where_any_conditions_satisfied_on_train)

        return BOD_l

    def get_POS_l_C(self,
                    l: data_preprocessing.Label,
                    C: set[conditions.Condition],
                    stage: str = 'original') -> int:
        """Calculate the number of train samples that satisfy any conditions for some set of condition
        and are false positive.

        :param stage:
        :param C: A set of `Condition` objects.
        :param l: The label of interest.
        :return: The number of instances that are false negative and satisfying some conditions.
        """
        where_was_wrong_with_respect_to_l = self.get_where_fp_l(test=False, l=l, stage=stage)
        train_pred_fine_data, train_pred_coarse_data = self.get_predictions(test=False, stage=stage)
        secondary_train_pred_fine_data, secondary_train_pred_coarse_data = (
            self.get_predictions(test=False, secondary=True)) if self.secondary_model_name is not None else (None, None)
        lower_train_pred_fine_data, lower_train_pred_coarse_data = (
            self.get_predictions(test=False, lower_predictions=True)) \
            if len(self.lower_predictions_indices) else (None, None)
        binary_data = self.pred_data['binary']

        where_any_conditions_satisfied_on_train = (
            rules.Rule.get_where_any_conditions_satisfied(C=C,
                                                          fine_data=train_pred_fine_data,
                                                          coarse_data=train_pred_coarse_data,
                                                          secondary_fine_data=secondary_train_pred_fine_data,
                                                          secondary_coarse_data=secondary_train_pred_coarse_data,
                                                          lower_predictions_fine_data=lower_train_pred_fine_data,
                                                          lower_predictions_coarse_data=lower_train_pred_coarse_data,
                                                          binary_data=binary_data))
        POS_l = np.sum(where_was_wrong_with_respect_to_l * where_any_conditions_satisfied_on_train)

        return POS_l

    def DetRuleLearn(self,
                     l: data_preprocessing.Label) -> set[conditions.Condition]:
        """Learns error detection rules for a specific label and granularity. These rules capture conditions
        that, when satisfied, indicate a higher likelihood of prediction errors for a given label.

        :param l: The label of interest.
        :return: A set of `Condition` representing the learned error detection rules.
        """
        DC_l = set()
        stage = 'original' if self.correction_model is None else 'post_detection'
        N_l = np.sum(self.get_where_label_is_l(pred=True, test=False, l=l, stage=stage))

        if N_l:
            other_g_str = 'fine' if str(l.g) == 'coarse' else 'coarse'
            other_g = data_preprocessing.DataPreprocessor.granularities[other_g_str]

            P_l, R_l = self.get_l_precision_and_recall(test=False, l=l, stage=stage)
            q_l = self.epsilon * N_l * P_l / R_l

            DC_star = {cond for cond in self.condition_datas[other_g] if self.get_NEG_l_C(l=l,
                                                                                          C={cond},
                                                                                          stage=stage) <= q_l}

            while DC_star:
                best_score = -1
                best_cond = None

                for cond in DC_star:
                    POS_l = self.get_POS_l_C(l=l, C=DC_l.union({cond}), stage=stage)
                    if POS_l >= best_score:
                        best_score = POS_l
                        best_cond = cond

                DC_l = DC_l.union({best_cond})
                DC_star = {cond for cond in self.condition_datas[other_g].difference(DC_l)
                           if self.get_NEG_l_C(l=l, C=DC_l.union({cond}), stage=stage) <= q_l}

        return DC_l

    def learn_detection_rules(self,
                              g: data_preprocessing.Granularity):
        self.CC_all[g] = set()  # in this use case where the conditions are fine and coarse predictions
        granularity_labels = self.preprocessor.get_labels(g).values()

        print(f'\nLearning {g}-grain error detection rules...')
        with context_handlers.WrapTQDM(total=len(granularity_labels)) as progress_bar:
            for l in granularity_labels:
                DC_l = self.DetRuleLearn(l=l)

                if len(DC_l):
                    self.error_detection_rules[l] = rules.ErrorDetectionRule(l=l, DC_l=DC_l)

                for cond_l in DC_l:
                    if not (isinstance(cond_l, conditions.PredCondition) and (not cond_l.secondary_model)
                            and (cond_l.lower_prediction_index is None) and (cond_l.l == l)):
                        self.CC_all[g] = self.CC_all[g].union({(cond_l, l)})

                if utils.is_local():
                    progress_bar.update(1)

    def apply_detection_rules(self,
                              test: bool,
                              g: data_preprocessing.Granularity):
        """Applies error detection rules to test predictions for a given granularity. If a rule is satisfied for
        a particular label, the prediction data for that label is modified with a value of -1,
        indicating a potential error.

        :params g: The granularity of the predictions to be processed.
        """
        stage = 'original' if self.correction_model is None else 'post_detection'
        pred_fine_data, pred_coarse_data = self.get_predictions(test=test, stage=stage)

        secondary_pred_fine_data, secondary_pred_coarse_data = (
            self.get_predictions(test=test, secondary=True) if self.secondary_model_name is not None else None, None)
        lower_train_pred_fine_data, lower_train_pred_coarse_data = (
            self.get_predictions(test=False, lower_predictions=True)) \
            if len(self.lower_predictions_indices) else (None, None)
        binary_data = self.pred_data['binary']

        altered_pred_granularity_data = self.get_predictions(test=test, g=g, stage=stage)

        # self.pred_data['test' if test else 'train']['mid_learning'][g] = altered_pred_granularity_data

        error_ground_truths = self.get_where_predicted_incorrect(test=test, g=g)
        error_predictions = np.zeros_like(error_ground_truths)

        for rule_g_l in {l: rule_l for l, rule_l in self.error_detection_rules.items() if l.g == g}.values():
            altered_pred_data_l = rule_g_l(pred_fine_data=pred_fine_data,
                                           pred_coarse_data=pred_coarse_data,
                                           secondary_pred_fine_data=secondary_pred_fine_data,
                                           secondary_pred_coarse_data=secondary_pred_coarse_data,
                                           lower_predictions_fine_data=lower_train_pred_fine_data,
                                           lower_predictions_coarse_data=lower_train_pred_coarse_data,
                                           binary_data=binary_data)
            altered_pred_granularity_data = np.where(altered_pred_data_l == -1, -1, altered_pred_granularity_data)

            error_predictions |= np.where(altered_pred_data_l == -1, 1, 0)

        self.pred_data['test' if test else 'train']['post_detection'][g] = altered_pred_granularity_data

        error_accuracy, error_f1, error_precision, error_recall = neural_metrics.get_individual_metrics(
            pred_data=error_predictions,
            true_data=error_ground_truths,
            labels=[0, 1])

        print(utils.blue_text(f'{g}-grain:\n'
                              f'Train error accuracy: {error_accuracy}, Train error f1: {error_f1}\n'
                              f'Train error precision: {error_precision}, Train error recall: {error_recall}'))

        # error_mask = np.where(self.test_pred_data['post_detection'][g] == -1, -1, 0)

        # for l in data_preprocessing.get_labels(g).values():
        #     self.num_predicted_l['post_detection'][g][l] = np.sum(self.get_where_label_is_l(pred=True,
        #                                                                                     test=True,
        #                                                                                     l=l,
        #                                                                                     stage='post_detection'))
        #
        # return error_mask

    def print_how_many_not_assigned(self,
                                    test: bool,
                                    g: data_preprocessing.Granularity,
                                    stage: str):
        test_or_train = 'test' if test else 'train'
        print(f'\nNum not assigned in {test_or_train} {stage} {g}-grain predictions: ' +
              utils.red_text(f"{np.sum(np.where(self.get_predictions(test=test, g=g, stage=stage) == -1, 1, 0))}\n"))

    def run_error_detection_application_pipeline(self,
                                                 test: bool,
                                                 print_results: bool = True):
        for g in data_preprocessing.DataPreprocessor.granularities.values():
            self.apply_detection_rules(test=test, g=g)

            if print_results:
                symbolic_metrics.evaluate_and_print_g_detection_rule_precision_increase(edcr=self, test=test, g=g)
                symbolic_metrics.evaluate_and_print_g_detection_rule_recall_decrease(edcr=self, test=test, g=g)
                self.print_how_many_not_assigned(test=test, g=g, stage='post_detection')

        if print_results:
            self.print_metrics(test=test, prior=False, stage='post_detection', print_inconsistencies=False)


if __name__ == '__main__':
    # precision_dict, recall_dict = (
    #     {g: {'initial': {}, 'pre_correction': {}, 'post_correction': {}} for g in data_preprocessing.granularities},
    #     {g: {'initial': {}, 'pre_correction': {}, 'post_correction': {}} for g in data_preprocessing.granularities})

    epsilons = [0.1 * i for i in range(2, 3)]
    test_bool = True

    for eps in epsilons:
        print('#' * 25 + f'eps = {eps}' + '#' * 50)
        edcr = EDCR(data_str='imagenet',
                    epsilon=eps,
                    main_model_name='vit_b_16',
                    combined=True,
                    loss='BCE',
                    lr=0.0001,
                    original_num_epochs=20,
                    include_inconsistency_constraint=False,
                    secondary_model_name='vit_b_16_soft_marginal',
                    lower_predictions_indices=[2, 3, 4, 5])
        # edcr.print_metrics(test=test_bool, prior=True)
        # edcr.run_learning_pipeline(EDCR_epoch_num=20)
        # edcr.run_error_detection_application_pipeline(test=test_bool, print_results=False)
        # edcr.apply_new_model_on_test()
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
