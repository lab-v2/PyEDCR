from __future__ import annotations

import typing
import numpy as np
import multiprocessing as mp
import multiprocessing.managers
import warnings
import random

warnings.filterwarnings('ignore')

import utils
import data_preprocessing
import vit_pipeline
import context_handlers
import metrics
import conditions
import rules

randomized: bool = False


class EDCR:
    """
    Performs error detection and correction based on model predictions.

    This class aims to identify and rectify errors in predictions made by a
    specified neural network model. It utilizes prediction data from both
    fine-grained and coarse-grained model runs to enhance its accuracy.

    Attributes:
        main_model_name (str): Name of the primary model used for predictions.
        combined (bool): Whether combined features (coarse and fine) were used during training.
        loss (str): Loss function used during training.
        lr: Learning rate used during training.
        epsilon: Value using for constraint in getting rules
    """

    def __init__(self,
                 main_model_name: str,
                 combined: bool,
                 loss: str,
                 lr: typing.Union[str, float],
                 num_epochs: int,
                 epsilon: typing.Union[str, float],
                 K_train: list[(int, int)] = None,
                 K_test: list[(int, int)] = None,
                 include_inconsistency_constraint: bool = False,
                 secondary_model_name: str = None,
                 second_predictions: bool = False):
        self.main_model_name = main_model_name
        self.combined = combined
        self.loss = loss
        self.lr = lr
        self.epsilon = epsilon
        self.secondary_model_name = secondary_model_name
        self.second_predictions = second_predictions

        pred_paths: dict[str, dict] = {
            'test' if test else 'train': {g_str: vit_pipeline.get_filepath(model_name=main_model_name,
                                                                           combined=combined,
                                                                           test=test,
                                                                           granularity=g_str,
                                                                           loss=loss,
                                                                           lr=lr,
                                                                           pred=True,
                                                                           epoch=num_epochs)
                                          for g_str in data_preprocessing.granularities_str}
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
                                          for g in data_preprocessing.granularities.values()},
                             'mid_learning': {g: np.load(pred_paths[test_or_train][g.g_str])[
                                 self.K_test if test_or_train == 'test' else self.K_train]
                                              for g in data_preprocessing.granularities.values()},
                             'post_detection': {g: np.load(pred_paths[test_or_train][g.g_str])[
                                 self.K_test if test_or_train == 'test' else self.K_train]
                                                for g in data_preprocessing.granularities.values()},
                             'post_correction': {
                                 g: np.load(pred_paths[test_or_train][g.g_str])[
                                     self.K_test if test_or_train == 'test' else self.K_train]
                                 for g in data_preprocessing.granularities.values()}}
             for test_or_train in ['test', 'train']}

        self.condition_datas = {g: {conditions.PredCondition(l=l)
                                    for l in data_preprocessing.get_labels(g).values()}
                                for g in data_preprocessing.granularities.values()}

        if secondary_model_name is not None:
            secondary_loss = secondary_model_name.split(f'{main_model_name}_')[1]
            pred_paths['secondary_model'] = {
                'test' if test else 'train': {g_str: vit_pipeline.get_filepath(model_name=main_model_name,
                                                                               combined=combined,
                                                                               test=test,
                                                                               granularity=g_str,
                                                                               loss=secondary_loss,
                                                                               lr=lr,
                                                                               pred=True,
                                                                               epoch=num_epochs)
                                              for g_str in data_preprocessing.granularities_str}
                for test in [True, False]}

            self.pred_data['secondary_model'] = \
                {test_or_train: {g: np.load(pred_paths['secondary_model'][test_or_train][g.g_str])
                                 for g in data_preprocessing.granularities.values()}
                 for test_or_train in ['test', 'train']}

            for g in data_preprocessing.granularities.values():
                self.condition_datas[g] = self.condition_datas[g].union(
                    {conditions.PredCondition(l=l, secondary_model=True)
                     for l in data_preprocessing.get_labels(g).values()})

        if second_predictions:
            pred_paths['second_predictions'] = {
                'test' if test else 'train': {g_str: vit_pipeline.get_filepath(model_name=main_model_name,
                                                                               combined=combined,
                                                                               test=test,
                                                                               granularity=g_str,
                                                                               loss=self.loss,
                                                                               lr=lr,
                                                                               pred=True,
                                                                               epoch=num_epochs,
                                                                               second_predictions=True)
                                              for g_str in data_preprocessing.granularities_str}
                for test in [True, False]}

            self.pred_data['second_predictions'] = \
                {test_or_train: {g: np.load(pred_paths['second_predictions'][test_or_train][g.g_str])
                                 for g in data_preprocessing.granularities.values()}
                 for test_or_train in ['test', 'train']}

            for g in data_preprocessing.granularities.values():
                self.condition_datas[g] = self.condition_datas[g].union(
                    {conditions.PredCondition(l=l, second_predictions=True)
                     for l in data_preprocessing.get_labels(g).values()})

        if include_inconsistency_constraint:
            for g in data_preprocessing.granularities.values():
                self.condition_datas[g] = self.condition_datas[g].union({conditions.InconsistencyCondition()})

        self.CC_all = {g: set() for g in data_preprocessing.granularities.values()}

        self.num_predicted_l = {'original': {g: {} for g in data_preprocessing.granularities.values()},
                                'post_detection': {g: {} for g in data_preprocessing.granularities.values()},
                                'post_correction': {g: {} for g in data_preprocessing.granularities.values()}}

        for g in data_preprocessing.granularities.values():
            for l in data_preprocessing.get_labels(g).values():
                self.num_predicted_l['original'][g][l] = np.sum(self.get_where_label_is_l(pred=True,
                                                                                          test=True,
                                                                                          l=l,
                                                                                          stage='original'))

        self.error_detection_rules: dict[data_preprocessing.Label, rules.ErrorDetectionRule] = {}
        self.error_correction_rules: dict[data_preprocessing.Label, rules.ErrorCorrectionRule] = {}

        self.correction_model = None

        print(f"Num of fine conditions: {len(self.condition_datas[data_preprocessing.granularities['fine']])}\n"
              f"Num of coarse conditions: {len(self.condition_datas[data_preprocessing.granularities['coarse']])}\n")

    def set_error_detection_rules(self, input_rules: typing.Dict[data_preprocessing.Label, {conditions.Condition}]):
        """
        Manually sets the error detection rule dictionary.

        :params rules: A dictionary mapping label instances to error detection rule objects.
        """
        error_detection_rules = {}
        for label, DC_l in input_rules.items():
            error_detection_rules[label] = rules.ErrorDetectionRule(label, DC_l)
        self.error_detection_rules = error_detection_rules

    def set_error_correction_rules(self,
                                   input_rules: typing.Dict[
                                       data_preprocessing.Label, {(conditions.Condition, data_preprocessing.Label)}]):
        """
        Manually sets the error correction rule dictionary.

        :params rules: A dictionary mapping label instances to error detection rule objects.
        """
        error_correction_rules = {}
        for label, CC_l in input_rules.items():
            error_correction_rules[label] = rules.ErrorCorrectionRule(label, CC_l=CC_l)
        self.error_correction_rules = error_correction_rules

    @staticmethod
    def get_C_str(CC: set[conditions.Condition]) -> str:
        return '{' + ', '.join(str(obj) for obj in CC) + '}'

    @staticmethod
    def get_CC_str(CC: set[(conditions.Condition, data_preprocessing.Label)]) -> str:
        return ('{' + ', '.join(['(' + ', '.join(item_repr) + ')' for item_repr in
                                 [[str(obj) for obj in item] for item in CC]]) + '}')

    def get_predictions(self,
                        test: bool,
                        g: data_preprocessing.Granularity = None,
                        stage: str = 'original',
                        secondary: bool = False,
                        second_predictions: bool = False) -> typing.Union[np.array, tuple[np.array]]:
        """Retrieves prediction data based on specified test/train mode.

        :param second_predictions:
        :param secondary:
        :param stage:
        :param g: The granularity level
        :param test: whether to get data from train or test set
        :return: Fine-grained and coarse-grained prediction data.
        """
        test_str = 'test' if test else 'train'

        if secondary:
            pred_data = self.pred_data['secondary_model'][test_str]
        elif second_predictions:
            pred_data = self.pred_data['second_predictions'][test_str]
        else:
            pred_data = self.pred_data[test_str][stage]

        if g is not None:
            return pred_data[g]

        pred_fine_data, pred_coarse_data = [pred_data[g] for g in data_preprocessing.granularities.values()]

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
            data_preprocessing.get_ground_truths(test=test, K=self.K_test if test else self.K_train, g=l.g))
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
        data = test_pred_fine_data if l.g == data_preprocessing.granularities['fine'] else test_pred_coarse_data
        where_label_is_l = np.where(data == l.index, 1, 0)
        return where_label_is_l

    def print_metrics(self,
                      test: bool,
                      prior: bool,
                      print_inconsistencies: bool = True,
                      stage: str = 'original'):

        """Prints performance metrics for given test/train data.

        Calculates and prints various metrics (accuracy, precision, recall, etc.)
        using appropriate true labels and prediction data based on the specified mode.

        :param stage:
        :param print_inconsistencies: whether to print the inconsistencies metric or not
        :param prior:
        :param test: whether to get data from train or test set
        """

        original_pred_fine_data, original_pred_coarse_data = None, None

        # if stage != 'original':
        #     original_pred_fine_data, original_pred_coarse_data = self.get_predictions(test=test, stage='original')

        pred_fine_data, pred_coarse_data = self.get_predictions(test=test, stage=stage)
        true_fine_data, true_coarse_data = data_preprocessing.get_ground_truths(test=test, K=self.K_test) if test \
            else data_preprocessing.get_ground_truths(test=test, K=self.K_train)

        vit_pipeline.get_and_print_metrics(pred_fine_data=pred_fine_data,
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
        ground_truth = data_preprocessing.get_ground_truths(test=test, K=self.K_test if test else self.K_train, g=g)
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
        ground_truth = data_preprocessing.get_ground_truths(test=True, K=self.K_test, g=g)
        prediction = test_pred_fine_data if g == data_preprocessing.granularities['fine'] else test_pred_coarse_data
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
            for l in data_preprocessing.get_labels(g).values():
                p_g[l], r_g[l] = self.get_l_precision_and_recall(test=test, l=l, stage=stage)
        else:
            for l in data_preprocessing.get_labels(g).values():
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
        second_train_pred_fine_data, second_train_pred_coarse_data = (
            self.get_predictions(test=False, second_predictions=True)) if self.second_predictions else (None, None)

        where_any_conditions_satisfied_on_train = (
            rules.Rule.get_where_any_conditions_satisfied(C=C,
                                                          fine_data=train_pred_fine_data,
                                                          coarse_data=train_pred_coarse_data,
                                                          secondary_fine_data=secondary_train_pred_fine_data,
                                                          secondary_coarse_data=secondary_train_pred_coarse_data,
                                                          second_predictions_fine_data=second_train_pred_fine_data,
                                                          second_predictions_coarse_data=second_train_pred_coarse_data))
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
        second_train_pred_fine_data, second_train_pred_coarse_data = (
            self.get_predictions(test=False, second_predictions=True)) if self.second_predictions else (None, None)

        where_any_conditions_satisfied_on_train = (
            rules.Rule.get_where_any_conditions_satisfied(C=C,
                                                          fine_data=train_pred_fine_data,
                                                          coarse_data=train_pred_coarse_data,
                                                          secondary_fine_data=secondary_train_pred_fine_data,
                                                          secondary_coarse_data=secondary_train_pred_coarse_data,
                                                          second_predictions_fine_data=second_train_pred_fine_data,
                                                          second_predictions_coarse_data=second_train_pred_coarse_data
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
        second_train_pred_fine_data, second_train_pred_coarse_data = (
            self.get_predictions(test=False, second_predictions=True)) if self.second_predictions else (None, None)

        where_any_conditions_satisfied_on_train = (
            rules.Rule.get_where_any_conditions_satisfied(C=C,
                                                          fine_data=train_pred_fine_data,
                                                          coarse_data=train_pred_coarse_data,
                                                          secondary_fine_data=secondary_train_pred_fine_data,
                                                          secondary_coarse_data=secondary_train_pred_coarse_data,
                                                          second_predictions_fine_data=second_train_pred_fine_data,
                                                          second_predictions_coarse_data=second_train_pred_coarse_data
                                                          ))
        POS_l = np.sum(where_was_wrong_with_respect_to_l * where_any_conditions_satisfied_on_train)

        return POS_l

    def get_BOD_CC(self,
                   CC: set[(conditions.Condition, data_preprocessing.Label)]) -> (int, np.array):
        """Calculate the number of train samples that satisfy the body of the 2nd rule for some set of condition
        class pair.

        :param CC: A set of `Condition`-`Class` pair.
        :return: The number of instances that satisfy the body of the 2nd rule and the boolean array it.
        """
        train_fine_pred_data, train_coarse_pred_data = self.get_predictions(test=False)
        where_any_pair_is_satisfied_in_train_pred = np.zeros_like(train_fine_pred_data)

        for cond, l_prime in CC:
            where_predicted_l_prime_in_train = self.get_where_label_is_l(pred=True, test=False, l=l_prime)
            where_condition_is_satisfied_in_train_pred = cond(train_fine_pred_data, train_coarse_pred_data)
            where_pair_is_satisfied = where_predicted_l_prime_in_train * where_condition_is_satisfied_in_train_pred
            where_any_pair_is_satisfied_in_train_pred |= where_pair_is_satisfied

        BOD_l = np.sum(where_any_pair_is_satisfied_in_train_pred)

        return BOD_l, where_any_pair_is_satisfied_in_train_pred

    def get_POS_l_CC(self,
                     test: bool,
                     l: data_preprocessing.Label,
                     CC: set[(conditions.Condition, data_preprocessing.Label)]) -> int:
        """Calculate the number of samples that satisfy the body of the 2nd rule and head
        (ground truth is l) for a label l and some set of condition class pair.

        :param test:
        :param CC:
        :param l: The label of interest.
        :return: The number of instances that satisfy the body of the 2nd rule and the boolean array it.
        """
        where_ground_truths_is_l = self.get_where_label_is_l(pred=False, test=test, l=l)

        fine_pred_data, coarse_pred_data = self.get_predictions(test=test)
        where_any_pair_is_satisfied = np.zeros_like(fine_pred_data)

        for cond, l_prime in CC:
            where_predicted_l_prime = self.get_where_label_is_l(pred=True, test=test, l=l_prime)
            where_condition_is_satisfied = cond(fine_pred_data, coarse_pred_data)
            where_pair_is_satisfied = where_predicted_l_prime * where_condition_is_satisfied
            where_any_pair_is_satisfied |= where_pair_is_satisfied

        POS_l_CC = np.sum(where_any_pair_is_satisfied * where_ground_truths_is_l)

        return POS_l_CC

    def get_CON_l_CC(self,
                     test: bool,
                     l: data_preprocessing.Label,
                     CC: set[(conditions.Condition, data_preprocessing.Label)]) -> float:
        """Calculate the ratio of number of samples that satisfy the rule body and head with the ones
        that only satisfy the body, given a set of condition class pairs.

        :param test:
        :param CC: A set of `Condition` - `Label` pairs.
        :param l: The label of interest.
        :return: ratio as defined above
        """

        BOD_CC, where_any_pair_is_satisfied_in_train_pred = self.get_BOD_CC(CC=CC)
        POS_l_CC = self.get_POS_l_CC(test=test, l=l, CC=CC)
        CON_l_CC = POS_l_CC / BOD_CC if BOD_CC else 0

        return CON_l_CC

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
            other_g = data_preprocessing.granularities[other_g_str]

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

    def _CorrRuleLearn(self,
                       l: data_preprocessing.Label,
                       CC_all: set[(conditions.Condition, data_preprocessing.Label)],
                       shared_index: mp.managers.ValueProxy) -> \
            (data_preprocessing.Label, [tuple[conditions.Condition, data_preprocessing.Label]]):
        """Learns error correction rules for a specific label and granularity. These rules associate conditions
        with alternative labels that are more likely to be correct when those conditions are met.

        :param l: The label of interest.
        :param CC_all: A set of all condition-label pairs to consider for rule learning.
        :return: A set of condition-label pairs.
        """
        CC_l = set()

        CC_l_prime = CC_all
        CC_sorted = sorted(CC_l_prime, key=lambda c_l: self.get_CON_l_CC(test=False, l=l, CC={c_l}))

        with context_handlers.WrapTQDM(total=len(CC_sorted)) as progress_bar:
            for cond_and_l in CC_sorted:
                a = self.get_CON_l_CC(test=False, l=l, CC=CC_l.union({cond_and_l})) - self.get_CON_l_CC(test=False,
                                                                                                        l=l, CC=CC_l)
                b = (self.get_CON_l_CC(test=False, l=l, CC=CC_l_prime.difference({cond_and_l})) -
                     self.get_CON_l_CC(test=False, l=l, CC=CC_l_prime))

                # randomized algorithm
                a_prime = max(a, 0)
                b_prime = max(b, 0)
                P = a_prime / (a_prime + b_prime) if not (a_prime == 0 and b_prime == 0) else 1

                # if a >= b:
                if ((not randomized) and a >= b) or (randomized and (random.random() < P)):
                    CC_l = CC_l.union({cond_and_l})
                else:
                    CC_l_prime = CC_l_prime.difference({cond_and_l})

                if utils.is_local():
                    progress_bar.update(1)

        assert CC_l_prime == CC_l

        p_l = self.get_l_precision_and_recall(test=False, l=l)[0]
        CON_CC_l = self.get_CON_l_CC(test=False, l=l, CC=CC_l)

        print(f'\n{l}: len(CC_l)={len(CC_l)}/{len(CC_all)}, CON_l_CC={CON_CC_l}, '
              f'p_l={p_l}\n')

        # if CON_CC_l <= p_l:
        #     CC_l = set()

        if not utils.is_local():
            shared_index.value += 1
            print(f'Completed {shared_index.value}/{len(data_preprocessing.get_labels(l.g).values())}')

        return l, CC_l

    def learn_detection_rules(self,
                              g: data_preprocessing.Granularity):
        self.CC_all[g] = set()  # in this use case where the conditions are fine and coarse predictions
        granularity_labels = data_preprocessing.get_labels(g).values()

        print(f'\nLearning {g}-grain error detection rules...')
        with context_handlers.WrapTQDM(total=len(granularity_labels)) as progress_bar:
            for l in granularity_labels:
                DC_l = self.DetRuleLearn(l=l)

                if len(DC_l):
                    self.error_detection_rules[l] = rules.ErrorDetectionRule(l=l, DC_l=DC_l)

                for cond_l in DC_l:
                    if not (isinstance(cond_l, conditions.PredCondition) and (not cond_l.secondary_model)
                            and (not cond_l.second_predictions) and cond_l.l == l):
                        self.CC_all[g] = self.CC_all[g].union({(cond_l, l)})

                if utils.is_local():
                    progress_bar.update(1)

    def learn_correction_rules(self,
                               g: data_preprocessing.Granularity):

        granularity_labels = data_preprocessing.get_labels(g).values()

        print(f'\nLearning {g}-grain error correction rules...')
        processes_num = min(len(granularity_labels), mp.cpu_count())

        manager = mp.Manager()
        shared_index = manager.Value('i', 0)

        iterable = [(l, self.CC_all[g],
                     shared_index
                     ) for l in granularity_labels]

        with mp.Pool(processes_num) as pool:
            CC_ls = pool.starmap(func=self._CorrRuleLearn,
                                 iterable=iterable)

        for l, CC_l in CC_ls:
            if len(CC_l):
                self.error_correction_rules[l] = rules.ErrorCorrectionRule(l=l, CC_l=CC_l)
            else:
                print(utils.red_text('\n' + '#' * 10 + f' {l} does not have an error correction rule!\n'))

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
        second_train_pred_fine_data, second_train_pred_coarse_data = (
            self.get_predictions(test=test, second_predictions=True)) if self.second_predictions else (None, None)

        altered_pred_granularity_data = self.get_predictions(test=test, g=g, stage=stage)

        # self.pred_data['test' if test else 'train']['mid_learning'][g] = altered_pred_granularity_data

        for rule_g_l in {l: rule_l for l, rule_l in self.error_detection_rules.items() if l.g == g}.values():
            altered_pred_data_l = rule_g_l(pred_fine_data=pred_fine_data,
                                           pred_coarse_data=pred_coarse_data,
                                           secondary_pred_fine_data=secondary_pred_fine_data,
                                           secondary_pred_coarse_data=secondary_pred_coarse_data,
                                           second_predictions_fine_data=second_train_pred_fine_data,
                                           second_predictions_coarse_data=second_train_pred_coarse_data)
            altered_pred_granularity_data = np.where(altered_pred_data_l == -1, -1, altered_pred_granularity_data)

        self.pred_data['test' if test else 'train']['post_detection'][g] = altered_pred_granularity_data

        # error_mask = np.where(self.test_pred_data['post_detection'][g] == -1, -1, 0)

        # for l in data_preprocessing.get_labels(g).values():
        #     self.num_predicted_l['post_detection'][g][l] = np.sum(self.get_where_label_is_l(pred=True,
        #                                                                                     test=True,
        #                                                                                     l=l,
        #                                                                                     stage='post_detection'))
        #
        # return error_mask

    def evaluate_and_print_l_correction_rule_precision_increase(self,
                                                                test: bool,
                                                                l: data_preprocessing.Label,
                                                                previous_l_precision: float,
                                                                correction_rule_theoretical_precision_increase: float,
                                                                threshold: float = 1e-5
                                                                ):
        post_correction_l_precision = self.get_l_precision_and_recall(l=l, test=test, stage='post_correction')[0]

        precision_diff = post_correction_l_precision - previous_l_precision

        precision_theory_holds = abs(correction_rule_theoretical_precision_increase - precision_diff) < threshold
        precision_theory_holds_str = utils.green_text('The theory holds!') if precision_theory_holds else (
            utils.red_text('The theory does not hold!'))

        print(f'class {l} new precision: {post_correction_l_precision}, '
              f'class {l} old precision: {previous_l_precision}, '
              f'diff: {utils.blue_text(precision_diff)}\n'
              f'theoretical precision increase: {utils.blue_text(correction_rule_theoretical_precision_increase)}\n'
              f'{precision_theory_holds_str}'
              )

    def evaluate_and_print_l_correction_rule_recall_increase(self,
                                                             test: bool,
                                                             l: data_preprocessing.Label,
                                                             previous_l_recall: float,
                                                             correction_rule_theoretical_recall_increase: float,
                                                             threshold: float = 1e-5
                                                             ):
        post_correction_l_recall = self.get_l_precision_and_recall(l=l, test=test, stage='post_correction')[1]

        precision_diff = post_correction_l_recall - previous_l_recall

        precision_theory_holds = abs(correction_rule_theoretical_recall_increase - precision_diff) < threshold
        precision_theory_holds_str = utils.green_text('The theory holds!') if precision_theory_holds else (
            utils.red_text('The theory does not hold!'))

        print(f'class {l} new recall: {post_correction_l_recall}, '
              f'class {l} old recall: {previous_l_recall}, '
              f'diff: {utils.blue_text(precision_diff)}\n'
              f'theoretical recall increase: {utils.blue_text(correction_rule_theoretical_recall_increase)}\n'
              f'{precision_theory_holds_str}'
              )

    def print_how_many_not_assigned(self,
                                    test: bool,
                                    g: data_preprocessing.Granularity,
                                    stage: str):
        test_or_train = 'test' if test else 'train'
        print(f'\nNum not assigned in {test_or_train} {stage} {g}-grain predictions: ' +
              utils.red_text(f"{np.sum(np.where(self.get_predictions(test=test, g=g, stage=stage) == -1, 1, 0))}\n"))

    def apply_correction_rules(self,
                               test: bool,
                               g: data_preprocessing.Granularity):
        """Applies error correction rules to test predictions for a given granularity. If a rule is satisfied for a
        particular label, the prediction data for that label is corrected using the rule's logic.

        :param test:
        :param g: The granularity of the predictions to be processed.
        """
        # test_pred_fine_data, test_pred_coarse_data = self.get_predictions(test=True)
        # self.test_pred_data['post_correction'][g] = self.get_predictions(test=True, g=g)

        test_or_train = 'test' if test else 'train'
        g_l_rules = {l: rule_l for l, rule_l in self.error_correction_rules.items() if l.g == g}

        secondary_fine_data, secondary_coarse_data = self.get_predictions(test=test, secondary=True) \
            if self.secondary_model_name is not None else (None, None)
        second_train_pred_fine_data, second_train_pred_coarse_data = (
            self.get_predictions(test=test, second_predictions=True)) if self.second_predictions else (None, None)

        for l, rule_g_l in g_l_rules.items():
            previous_l_precision, previous_l_recall = self.get_l_precision_and_recall(l=l, test=test,
                                                                                      stage='post_correction')

            correction_rule_theoretical_precision_increase = (
                metrics.get_l_correction_rule_theoretical_precision_increase(edcr=self, test=test, l=l))
            correction_rule_theoretical_recall_increase = (
                metrics.get_l_correction_rule_theoretical_recall_increase(edcr=self, test=test, l=l,
                                                                          CC_l=self.error_correction_rules[l].C_l))

            fine_data, coarse_data = self.get_predictions(test=test, stage='post_correction')

            altered_pred_data_l = rule_g_l(pred_fine_data=fine_data, pred_coarse_data=coarse_data,
                                           secondary_pred_fine_data=secondary_fine_data,
                                           secondary_pred_coarse_data=secondary_coarse_data,
                                           second_predictions_fine_data=second_train_pred_fine_data,
                                           second_predictions_coarse_data=second_train_pred_coarse_data)

            self.pred_data[test_or_train]['post_correction'][g] = np.where(
                # (collision_array != 1) &
                (altered_pred_data_l == l.index),
                l.index,
                self.get_predictions(test=test, g=g, stage='post_correction'))

            # self.print_how_many_not_assigned(test=test, g=g, stage='post_correction')

            self.evaluate_and_print_l_correction_rule_precision_increase(
                test=test,
                l=l,
                previous_l_precision=previous_l_precision,
                correction_rule_theoretical_precision_increase=correction_rule_theoretical_precision_increase)

            self.evaluate_and_print_l_correction_rule_recall_increase(
                test=test,
                l=l,
                previous_l_recall=previous_l_recall,
                correction_rule_theoretical_recall_increase=correction_rule_theoretical_recall_increase)

            self.print_metrics(test=test, prior=False, stage='post_correction', print_inconsistencies=False)

        # collision_array = np.zeros_like(altered_pred_granularity_data)
        #
        # for l_1, altered_pred_data_l_1, in altered_pred_granularity_datas.items():
        #     for l_2, altered_pred_data_l_2 in altered_pred_granularity_datas.items():
        #         if l_1 != l_2:
        #             where_supposed_to_correct_to_l1 = np.where(altered_pred_data_l_1 == l_1.index, 1, 0)
        #             where_supposed_to_correct_to_l2 = np.where(altered_pred_data_l_2 == l_2.index, 1, 0)
        #             collision_array |= where_supposed_to_correct_to_l1 * where_supposed_to_correct_to_l2

        # for l, altered_pred_data_l in altered_pred_granularity_datas.items():

        for l in data_preprocessing.get_labels(g).values():
            self.num_predicted_l['post_correction'][g][l] = np.sum(self.get_where_label_is_l(pred=True,
                                                                                             test=True,
                                                                                             l=l,
                                                                                             stage='post_correction'))

        # return altered_pred_granularity_data

    def apply_reversion_rules(self,
                              test: bool,
                              g: data_preprocessing.Granularity):
        """Applies error reversion rules to recover a prediction for a given granularity. If the inference of detection
        and correction rules do not change the label, the prediction label for that example is set to be the original
        one.

        :param test:
        :param g: The granularity of the predictions to be processed.
        """
        test_or_train = 'test' if test else 'train'

        pred_granularity_data = self.get_predictions(test=test, g=g, stage='post_correction')

        self.pred_data[test_or_train][g] = np.where(pred_granularity_data == -1,
                                                    self.get_predictions(test=test, g=g, stage='original'),
                                                    pred_granularity_data)

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
            evaluation=True)

        (fine_ground_truths, coarse_ground_truths, fine_predictions, coarse_predictions,
         fine_accuracy, coarse_accuracy) = vit_pipeline.evaluate_combined_model(
            fine_tuner=self.correction_model,
            loaders=loaders,
            loss=self.loss,
            device=devices[0],
            split='train',
            print_results=False)

        self.pred_data['train']['post_detection'][data_preprocessing.granularities['fine']][
            examples_with_errors] = fine_predictions
        self.pred_data['train']['post_detection'][data_preprocessing.granularities['coarse']][
            examples_with_errors] = coarse_predictions

    def apply_new_model_on_test(self,
                                print_results: bool = True):
        new_fine_predictions, new_coarse_predictions = (
            vit_pipeline.run_combined_evaluating_pipeline(split='test',
                                                          lrs=[self.lr],
                                                          loss=self.loss,
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
            self.print_metrics(test=test_bool, prior=False, stage='post_detection')

    def run_learning_pipeline(self,
                              EDCR_epoch_num: int):
        print('Started learning pipeline...\n')
        self.print_metrics(test=False, prior=True)

        for EDCR_epoch in range(EDCR_epoch_num):
            for g in data_preprocessing.granularities.values():
                self.learn_detection_rules(g=g)
                self.apply_detection_rules(test=False, g=g)

            self.run_training_new_model_pipeline()
            # self.print_metrics(test=False, prior=False, stage='post_detection')

            edcr_epoch_str = f'Finished EDCR epoch {EDCR_epoch + 1}/{EDCR_epoch_num}'

            print(utils.blue_text('\n' + '#' * 100 +
                                  '\n' + '#' * int((100 - len(edcr_epoch_str)) / 2) + edcr_epoch_str +
                                  '#' * (100 - int((100 - len(edcr_epoch_str)) / 2) - len(edcr_epoch_str)) +
                                  '\n' + '#' * 100 + '\n'))

        # self.learn_correction_rules(g=g)
        # self.learn_correction_rules_alt(g=g)

        print('\nRule learning completed\n')

    def run_error_detection_application_pipeline(self,
                                                 test: bool,
                                                 print_results: bool = True):
        for g in data_preprocessing.granularities.values():
            self.apply_detection_rules(test=test, g=g)

            if print_results:
                metrics.evaluate_and_print_g_detection_rule_precision_increase(edcr=self, test=test, g=g)
                metrics.evaluate_and_print_g_detection_rule_recall_decrease(edcr=self, test=test, g=g)
                self.print_how_many_not_assigned(test=test, g=g, stage='post_detection')

        if print_results:
            self.print_metrics(test=test, prior=False, stage='post_detection', print_inconsistencies=False)

    def run_error_correction_application_pipeline(self,
                                                  test: bool):
        print('\n' + '#' * 50 + 'post correction' + '#' * 50)

        for g in data_preprocessing.granularities.values():
            self.apply_correction_rules(test=test, g=g)

        self.print_metrics(test=test, prior=False, stage='post_correction', print_inconsistencies=False)


if __name__ == '__main__':
    # precision_dict, recall_dict = (
    #     {g: {'initial': {}, 'pre_correction': {}, 'post_correction': {}} for g in data_preprocessing.granularities},
    #     {g: {'initial': {}, 'pre_correction': {}, 'post_correction': {}} for g in data_preprocessing.granularities})

    epsilons = [0.1 * i for i in range(2, 3)]
    test_bool = True

    for eps in epsilons:
        print('#' * 25 + f'eps = {eps}' + '#' * 50)
        edcr = EDCR(epsilon=eps,
                    main_model_name='vit_b_16',
                    combined=True,
                    loss='BCE',
                    lr=0.0001,
                    num_epochs=20,
                    include_inconsistency_constraint=False,
                    secondary_model_name='vit_b_16_soft_marginal',
                    second_predictions=True
                    )
        edcr.print_metrics(test=test_bool, prior=True)
        edcr.run_learning_pipeline(EDCR_epoch_num=3)
        edcr.run_error_detection_application_pipeline(test=test_bool, print_results=False)
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
