from __future__ import annotations

import abc
import typing
import numpy as np
import multiprocessing as mp
import multiprocessing.managers
import warnings
import matplotlib.pyplot as plt
import random

warnings.filterwarnings('ignore')

import utils
import data_preprocessing
import vit_pipeline
import context_handlers

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

    class _Condition(typing.Hashable, typing.Callable, abc.ABC):
        """Represents a condition that can be evaluated on an example.

        When treated as a function, it takes an example (e.g., image, data) as input
        and returns a value between 0 and 1 indicating whether the condition is satisfied.
        A value of 0 means the condition is not met, while a value of 1 means it is fully met.
        """

        @abc.abstractmethod
        def __init__(self, *args, **kwargs):
            pass

        @abc.abstractmethod
        def __call__(self, *args, **kwargs) -> typing.Union[bool, np.array]:
            pass

        @abc.abstractmethod
        def __hash__(self):
            pass

        @abc.abstractmethod
        def __eq__(self, other):
            pass

    class PredCondition(_Condition):
        """Represents a condition based on a model's prediction of a specific class.

        It evaluates to 1 if the model predicts the specified class for a given example,
        and 0 otherwise.
        """

        def __init__(self,
                     l: data_preprocessing.Label):
            """Initializes a PredCondition instance.

            :param l: The target Label for which the condition is evaluated.
            """
            self.l = l

        def __call__(self,
                     fine_data: np.array,
                     coarse_data: np.array) -> np.array:
            granularity_data = fine_data if self.l.g == data_preprocessing.granularities['fine'] else coarse_data
            return np.where(granularity_data == self.l.index, 1, 0)

        def __str__(self) -> str:
            return f'pred_{self.l}'

        def __hash__(self):
            return self.l.__hash__()

        def __eq__(self, other):
            return self.__hash__() == other.__hash__()

    class ConsistencyCondition(_Condition):
        def __init__(self):
            pass

        def __call__(self,
                     fine_data: np.array,
                     coarse_data: np.array) -> np.array:
            values = []
            for fine_prediction_index, coarse_prediction_index in zip(fine_data, coarse_data):
                values += [data_preprocessing.fine_to_course_idx[fine_prediction_index] == coarse_prediction_index]

            return np.array(values)

        def __hash__(self):
            return hash('ConsistencyCondition')

        def __eq__(self, other):
            return self.__hash__() == other.__hash__()

    class Rule(typing.Callable, typing.Sized, abc.ABC):
        """Represents a rule for evaluating predictions based on conditions and labels.

        :param l: The label associated with the rule.
        :param C_l: The set of conditions that define the rule.
        """

        def __init__(self,
                     l: data_preprocessing.Label,
                     C_l: set[typing.Union[EDCR._Condition, tuple[EDCR._Condition, data_preprocessing.Label]]]):
            self.l = l
            self.C_l = C_l

        def get_where_predicted_l(self,
                                  data: np.array,
                                  l_prime: data_preprocessing.Label = None) -> np.array:
            return np.where(data == (self.l.index if l_prime is None else l_prime.index), 1, 0)

        @abc.abstractmethod
        def __call__(self,
                     pred_fine_data: np.array,
                     pred_coarse_data: np.array) -> np.array:
            pass

        @abc.abstractmethod
        def get_where_body_is_satisfied(self,
                                        pred_fine_data: np.array,
                                        pred_coarse_data: np.array):
            pass

        def __len__(self):
            return len(self.C_l)

    class ErrorDetectionRule(Rule):
        def __init__(self,
                     l: data_preprocessing.Label,
                     DC_l: set[EDCR._Condition]):
            """Construct a detection rule for evaluating predictions based on conditions and labels.

            :param l: The label associated with the rule.
            :param DC_l: The set of conditions that define the rule.
            """
            super().__init__(l=l, C_l=DC_l)
            assert all(cond.l != self.l for cond in {cond_prime for cond_prime in self.C_l
                                                     if isinstance(cond_prime, EDCR.PredCondition)})

        def get_where_body_is_satisfied(self,
                                        pred_fine_data: np.array,
                                        pred_coarse_data: np.array) -> np.array:
            test_pred_granularity_data = pred_fine_data if self.l.g == data_preprocessing.granularities['fine'] \
                else pred_coarse_data
            where_predicted_l = self.get_where_predicted_l(data=test_pred_granularity_data)
            where_any_conditions_satisfied = EDCR.get_where_any_conditions_satisfied(C=self.C_l,
                                                                                     fine_data=pred_fine_data,
                                                                                     coarse_data=pred_coarse_data)
            where_body_is_satisfied = where_predicted_l * where_any_conditions_satisfied

            return where_body_is_satisfied

        def __call__(self,
                     pred_fine_data: np.array,
                     pred_coarse_data: np.array) -> np.array:
            """Infer the detection rule based on the provided prediction data.

            :param pred_fine_data: The fine-grained prediction data.
            :param pred_coarse_data: The coarse-grained prediction data.
            :return: modified prediction contains -1 at examples that have errors for a specific granularity as
            derived from Label l.
            """
            test_pred_granularity_data = pred_fine_data if self.l.g == data_preprocessing.granularities['fine'] \
                else pred_coarse_data
            where_predicted_l_and_any_conditions_satisfied = (
                self.get_where_body_is_satisfied(pred_fine_data=pred_fine_data,
                                                 pred_coarse_data=pred_coarse_data))
            altered_pred_data = np.where(where_predicted_l_and_any_conditions_satisfied == 1, -1,
                                         test_pred_granularity_data)

            return altered_pred_data

        def __str__(self) -> str:
            return '\n'.join(f'error_{self.l}(x) <- pred_{self.l}(x) ^ {cond}(x)' for cond in self.C_l)

    class ErrorCorrectionRule(Rule):
        def __init__(self,
                     l: data_preprocessing.Label,
                     CC_l: set[(EDCR._Condition, data_preprocessing.Label)]):
            """Construct a detection rule for evaluating predictions based on conditions and labels.

            :param l: The label associated with the rule.
            :param CC_l: The set of condition-class pair that define the rule.
            """
            C_l = {(cond, l_prime) for cond, l_prime in CC_l if isinstance(cond, EDCR.ConsistencyCondition)
                   or cond.l.g != l_prime.g}

            super().__init__(l=l, C_l=C_l)

        def get_where_body_is_satisfied(self,
                                        fine_data: np.array,
                                        coarse_data: np.array) -> np.array:
            test_pred_granularity_data = fine_data if self.l.g == data_preprocessing.granularities['fine'] \
                else coarse_data

            where_any_pair_satisfied = np.zeros_like(test_pred_granularity_data)

            for cond, l_prime in self.C_l:
                where_condition_satisfied = (
                    EDCR.get_where_any_conditions_satisfied(C={cond},
                                                            fine_data=fine_data,
                                                            coarse_data=coarse_data))
                where_predicted_l_prime = self.get_where_predicted_l(data=test_pred_granularity_data,
                                                                     l_prime=l_prime)
                where_pair_satisfied = where_condition_satisfied * where_predicted_l_prime

                where_any_pair_satisfied |= where_pair_satisfied

            return where_any_pair_satisfied

        def __call__(self,
                     fine_data: np.array,
                     coarse_data: np.array) -> np.array:
            """Infer the correction rule based on the provided prediction data.

            :param fine_data: The fine-grained prediction data.
            :param coarse_data: The coarse-grained prediction data.
            :return: new test prediction for a specific granularity as derived from Label l.
            """
            where_any_pair_satisfied = self.get_where_body_is_satisfied(fine_data=fine_data,
                                                                        coarse_data=coarse_data)

            altered_pred_data = np.where(where_any_pair_satisfied == 1, self.l.index, -1)

            return altered_pred_data

        def __str__(self) -> str:
            return '\n'.join(f'corr_{self.l}(x) <- {cond}(x) ^ pred_{l_prime}(x)' for (cond, l_prime) in self.C_l)

    def __init__(self,
                 main_model_name: str,
                 combined: bool,
                 loss: str,
                 lr: typing.Union[str, float],
                 num_epochs: int,
                 epsilon: typing.Union[str, float],
                 K_train: list[(int, int)] = None,
                 K_test: list[(int, int)] = None):
        self.main_model_name = main_model_name
        self.combined = combined
        self.loss = loss
        self.lr = lr
        self.epsilon = epsilon

        pred_paths = {'test' if test else 'train': {g_str: vit_pipeline.get_filepath(model_name=main_model_name,
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
            {test_or_train: {'original': {g: np.load(pred_paths[test_or_train][str(g)])[
                self.K_test if test_or_train == 'test' else self.K_train]
                                          for g in data_preprocessing.granularities.values()},
                             'post_detection': {g: np.load(pred_paths[test_or_train][str(g)])[
                                 self.K_test if test_or_train == 'test' else self.K_train]
                                                for g in data_preprocessing.granularities.values()},
                             'post_correction': {
                                 g: np.load(pred_paths[test_or_train][str(g)])[
                                     self.K_test if test_or_train == 'test' else self.K_train]
                                 for g in data_preprocessing.granularities.values()}}
             for test_or_train in ['test', 'train']}

        self.condition_datas = {g: {EDCR.PredCondition(l=l)
                                    for l in data_preprocessing.get_labels(g).values()}.union(
            {EDCR.ConsistencyCondition()}) for g in data_preprocessing.granularities.values()}

        self.CC_all = {g: set() for g in data_preprocessing.granularities.values()}
        # self.train_precisions = {}
        # self.train_recalls = {}
        #
        # for g in data_preprocessing.granularities.values():
        #     self.train_precisions[g], self.train_recalls[g] = self.get_g_precision_and_recall(g=g, test=False)

        self.post_detection_test_precisions = {}
        self.post_detection_test_recalls = {}

        self.post_correction_test_precisions = {}
        self.post_correction_test_recalls = {}

        self.num_predicted_l = {'original': {g: {} for g in data_preprocessing.granularities.values()},
                                'post_detection': {g: {} for g in data_preprocessing.granularities.values()},
                                'post_correction': {g: {} for g in data_preprocessing.granularities.values()}}

        for g in data_preprocessing.granularities.values():
            for l in data_preprocessing.get_labels(g).values():
                self.num_predicted_l['original'][g][l] = np.sum(self.get_where_label_is_l(pred=True,
                                                                                          test=True,
                                                                                          l=l,
                                                                                          stage='original'))

        self.error_detection_rules: dict[data_preprocessing.Label, EDCR.ErrorDetectionRule] = {}
        self.error_correction_rules: dict[data_preprocessing.Label, EDCR.ErrorCorrectionRule] = {}

    def set_error_detection_rules(self, rules: typing.Dict[data_preprocessing.Label, {_Condition}]):
        """
        Manually sets the error detection rule dictionary.

        :params rules: A dictionary mapping label instances to error detection rule objects.
        """
        error_detection_rules = {}
        for label, DC_l in rules.items():
            error_detection_rules[label] = EDCR.ErrorDetectionRule(label, DC_l)
        self.error_detection_rules = error_detection_rules

    def set_error_correction_rules(self,
                                   rules: typing.Dict[
                                       data_preprocessing.Label, {(_Condition, data_preprocessing.Label)}]):
        """
        Manually sets the error correction rule dictionary.

        :params rules: A dictionary mapping label instances to error detection rule objects.
        """
        error_correction_rules = {}
        for label, CC_l in rules.items():
            error_correction_rules[label] = EDCR.ErrorCorrectionRule(label, CC_l=CC_l)
        self.error_correction_rules = error_correction_rules

    @staticmethod
    def get_C_str(CC: set[_Condition]) -> str:
        return '{' + ', '.join(str(obj) for obj in CC) + '}'

    @staticmethod
    def get_CC_str(CC: set[(_Condition, data_preprocessing.Label)]) -> str:
        return ('{' + ', '.join(['(' + ', '.join(item_repr) + ')' for item_repr in
                                 [[str(obj) for obj in item] for item in CC]]) + '}')

    def get_predictions(self,
                        test: bool,
                        g: data_preprocessing.Granularity = None,
                        stage: str = 'original') -> typing.Union[np.array, tuple[np.array]]:
        """Retrieves prediction data based on specified test/train mode.

        :param stage:
        :param g: The granularity level
        :param test: whether to get data from train or test set
        :return: Fine-grained and coarse-grained prediction data.
        """
        pred_data = self.pred_data['test' if test else 'train'][stage]

        if g is not None:
            return pred_data[g]

        pred_fine_data, pred_coarse_data = [pred_data[g] for g in data_preprocessing.granularities.values()]

        return pred_fine_data, pred_coarse_data

    def get_where_label_is_l(self,
                             pred: bool,
                             test: bool,
                             l: data_preprocessing.Label,
                             stage: str = 'original') -> np.array:
        """ Retrieves indices of instances where the specified label is present.

        :param stage:
        :param pred: True for prediction, False for ground truth
        :param test: whether to get data from train or test set
        :param l: The label to search for.
        :return: A boolean array indicating which instances have the given label.
        """
        data = self.get_predictions(test=test, g=l.g, stage=stage) if pred else (
            data_preprocessing.get_ground_truths(test=test, K=self.K_test if test else self.K_train, g=l.g))
        return np.where(data == l.index, 1, 0)

    def get_where_label_is_l_in_data(self,
                                     l: data_preprocessing.Label,
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
                                           print_inconsistencies=print_inconsistencies)

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

        p_l = t_p_l / (t_p_l + f_p_l)
        r_l = t_p_l / N_l_gt

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

    @staticmethod
    def get_where_any_conditions_satisfied(C: set[_Condition],
                                           fine_data: np.array,
                                           coarse_data: np.array) -> np.array:
        """Checks where any given conditions are satisfied.

        :param fine_data: Data that used for Condition having FineGrainLabel l
        :param coarse_data: Data that used for Condition having CoarseGrainLabel l
        :param C: A set of `Condition` objects.
        :return: A NumPy array with True values if the example satisfy any conditions and False otherwise.
        """
        any_condition_satisfied = np.zeros_like(fine_data)

        for cond in C:
            any_condition_satisfied |= cond(fine_data=fine_data, coarse_data=coarse_data)

        return any_condition_satisfied

    def get_NEG_l_C(self,
                    l: data_preprocessing.Label,
                    C: set[_Condition]) -> int:
        """Calculate the number of train samples that satisfy any of the conditions and are true positive.

        :param C: A set of `Condition` objects.
        :param l: The label of interest.
        :return: The number of instances that is true negative and satisfying all conditions.
        """
        where_train_tp_l = self.get_where_tp_l(test=False, l=l)
        train_pred_fine_data, train_pred_coarse_data = self.get_predictions(test=False)
        where_any_conditions_satisfied_on_train = (
            self.get_where_any_conditions_satisfied(C=C,
                                                    fine_data=train_pred_fine_data,
                                                    coarse_data=train_pred_coarse_data))
        NEG_l = np.sum(where_train_tp_l * where_any_conditions_satisfied_on_train)

        return NEG_l

    def get_POS_l_C(self,
                    l: data_preprocessing.Label,
                    C: set[_Condition]) -> int:
        """Calculate the number of train samples that satisfy any conditions for some set of condition
        and are false positive.

        :param C: A set of `Condition` objects.
        :param l: The label of interest.
        :return: The number of instances that are false negative and satisfying some conditions.
        """
        where_was_wrong_with_respect_to_l = self.get_where_fp_l(test=False, l=l)
        train_pred_fine_data, train_pred_coarse_data = self.get_predictions(test=False)
        where_any_conditions_satisfied_on_train = (
            self.get_where_any_conditions_satisfied(C=C,
                                                    fine_data=train_pred_fine_data,
                                                    coarse_data=train_pred_coarse_data))
        POS_l = np.sum(where_was_wrong_with_respect_to_l * where_any_conditions_satisfied_on_train)

        return POS_l

    def get_BOD_CC(self,
                   CC: set[(_Condition, data_preprocessing.Label)]) -> (int, np.array):
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
                     l: data_preprocessing.Label,
                     where_any_pair_is_satisfied_in_train_pred: np.array) -> int:
        """Calculate the number of train samples that satisfy the body of the 2nd rule and head
        (ground truth is l) for a label l and some set of condition class pair.

        :param where_any_pair_is_satisfied_in_train_pred: A boolean mask examples satisfy any `Condition`-`Class` pair.
        :param l: The label of interest.
        :return: The number of instances that satisfy the body of the 2nd rule and the boolean array it.
        """
        where_train_ground_truths_is_l = self.get_where_label_is_l(pred=False, test=False, l=l)
        POS_l_CC = np.sum(where_any_pair_is_satisfied_in_train_pred * where_train_ground_truths_is_l)

        return POS_l_CC

    def get_CON_l_CC(self,
                     l: data_preprocessing.Label,
                     CC: set[(_Condition, data_preprocessing.Label)]) -> float:
        """Calculate the ratio of number of samples that satisfy the rule body and head with the ones
        that only satisfy the body, given a set of condition class pairs.

        :param CC: A set of `Condition` - `Label` pairs.
        :param l: The label of interest.
        :return: ratio as defined above
        """

        BOD_CC, where_any_pair_is_satisfied_in_train_pred = self.get_BOD_CC(CC=CC)
        POS_l_CC = (
            self.get_POS_l_CC(l=l,
                              where_any_pair_is_satisfied_in_train_pred=where_any_pair_is_satisfied_in_train_pred))
        CON_l_CC = POS_l_CC / BOD_CC if BOD_CC else 0

        return CON_l_CC

    def DetRuleLearn(self,
                     l: data_preprocessing.Label) -> set[_Condition]:
        """Learns error detection rules for a specific label and granularity. These rules capture conditions
        that, when satisfied, indicate a higher likelihood of prediction errors for a given label.

        :param l: The label of interest.
        :return: A set of `Condition` representing the learned error detection rules.
        """
        DC_l = set()
        N_l = np.sum(self.get_where_label_is_l(pred=True, test=False, l=l))
        g = l.g
        other_g_str = 'fine' if str(g) == 'coarse' else 'coarse'
        other_g = data_preprocessing.granularities[other_g_str]

        if N_l:
            P_l, R_l = self.get_l_precision_and_recall(test=False, l=l, stage='original')
            q_l = self.epsilon * N_l * P_l / R_l

            DC_star = {cond for cond in self.condition_datas[other_g] if self.get_NEG_l_C(l=l, C={cond}) <= q_l}

            while DC_star != set():
                best_score = -1
                best_cond = None

                for cond in DC_star:
                    POS_l_c = self.get_POS_l_C(l=l, C=DC_l.union({cond}))
                    if POS_l_c >= best_score:
                        best_score = POS_l_c
                        best_cond = cond

                DC_l = DC_l.union({best_cond})
                DC_star = {cond for cond in self.condition_datas[other_g].difference(DC_l)
                           if self.get_NEG_l_C(l=l, C=DC_l.union({cond})) <= q_l}

        return DC_l

    def _CorrRuleLearn(self,
                       l: data_preprocessing.Label,
                       CC_all: set[(_Condition, data_preprocessing.Label)],
                       shared_index: mp.managers.ValueProxy) -> \
            (data_preprocessing.Label, [tuple[_Condition, data_preprocessing.Label]]):
        """Learns error correction rules for a specific label and granularity. These rules associate conditions
        with alternative labels that are more likely to be correct when those conditions are met.

        :param l: The label of interest.
        :param CC_all: A set of all condition-label pairs to consider for rule learning.
        :return: A set of condition-label pairs.
        """
        CC_l = set()
        CC_l_prime = CC_all.copy()
        CC_sorted = sorted(CC_all, key=lambda c_l: self.get_CON_l_CC(l=l, CC={c_l}))

        with context_handlers.WrapTQDM(total=len(CC_sorted)) as progress_bar:
            for cond_and_l in CC_sorted:
                a = self.get_CON_l_CC(l=l, CC=CC_l.union({cond_and_l})) - self.get_CON_l_CC(l=l, CC=CC_l)
                b = (self.get_CON_l_CC(l=l, CC=CC_l_prime.difference({cond_and_l})) -
                     self.get_CON_l_CC(l=l, CC=CC_l_prime))

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

        # print(f'\n{l}: len(CC_l)={len(CC_l)}/{len(CC_all)}, CON_l_CC={self.get_CON_l_CC(l=l, CC=CC_l)}, '
        #       f'P_l={self.train_precisions[l.g][l]}\n')

        # if self.get_CON_l_CC(l=l, CC=CC_l) <= self.train_precisions[l.g][l]:
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
                    self.error_detection_rules[l] = EDCR.ErrorDetectionRule(l=l, DC_l=DC_l)

                for cond_l in DC_l:
                    if not (isinstance(cond_l, EDCR.PredCondition) and cond_l.l == l):
                        self.CC_all[g] = self.CC_all[g].union({(cond_l, l)})

                if utils.is_local():
                    progress_bar.update(1)

    def learn_correction_rules(self,
                               g: data_preprocessing.Granularity):

        granularity_labels = data_preprocessing.get_labels(g).values()
        other_g = data_preprocessing.granularities['fine' if str(g) == 'coarse' else 'coarse']

        print(f'\nLearning {g}-grain error correction rules...')
        processes_num = min(len(granularity_labels), mp.cpu_count())

        manager = mp.Manager()
        shared_index = manager.Value('i', 0)

        iterable = [(l, self.CC_all[g], shared_index) for l in granularity_labels]

        with mp.Pool(processes_num) as pool:
            CC_ls = pool.starmap(func=self._CorrRuleLearn,
                                 iterable=iterable)

        for l, CC_l in CC_ls:
            if len(CC_l):
                self.error_correction_rules[l] = EDCR.ErrorCorrectionRule(l=l, CC_l=CC_l)
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
        pred_fine_data, pred_coarse_data = self.get_predictions(test=test)
        altered_pred_granularity_data = self.get_predictions(test=test, g=g)

        for rule_g_l in {l: rule_l for l, rule_l in self.error_detection_rules.items() if l.g == g}.values():
            altered_pred_data_l = rule_g_l(pred_fine_data=pred_fine_data,
                                           pred_coarse_data=pred_coarse_data)
            altered_pred_granularity_data = np.where(altered_pred_data_l == -1, -1, altered_pred_granularity_data)

        self.pred_data['test' if test else 'train']['post_detection'][g] = altered_pred_granularity_data

        # error_mask = np.where(self.test_pred_data['post_detection'][g] == -1, -1, 0)

        self.post_detection_test_precisions[g], self.post_detection_test_recalls[g] = (
            self.get_g_precision_and_recall(g=g, test=test, stage='post_detection'))

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
        post_correction_l_precision = self.get_g_precision_and_recall(g=l.g, test=test, stage='post_correction')[0][l]

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

        for l, rule_g_l in {l: rule_l for l, rule_l in self.error_correction_rules.items() if l.g == g}.items():
            previous_l_precision = self.get_g_precision_and_recall(g=g, test=test, stage='post_correction')[0][l]

            correction_rule_theoretical_precision_increase = (
                self.get_l_correction_rule_theoretical_precision_increase(test=test, l=l))

            altered_pred_data_l = rule_g_l(
                fine_data=self.pred_data[test_or_train]['post_correction'][data_preprocessing.granularities['fine']],
                coarse_data=self.pred_data[test_or_train]['post_correction'][data_preprocessing.granularities['coarse']]
            )

            self.pred_data[test_or_train]['post_correction'][g] = np.where(
                # (collision_array != 1) &
                (altered_pred_data_l == l.index),
                l.index,
                self.pred_data[test_or_train]['post_correction'][g])

            self.evaluate_and_print_l_correction_rule_precision_increase(
                test=test,
                l=l,
                previous_l_precision=previous_l_precision,
                correction_rule_theoretical_precision_increase=correction_rule_theoretical_precision_increase)

        # collision_array = np.zeros_like(altered_pred_granularity_data)
        #
        # for l_1, altered_pred_data_l_1, in altered_pred_granularity_datas.items():
        #     for l_2, altered_pred_data_l_2 in altered_pred_granularity_datas.items():
        #         if l_1 != l_2:
        #             where_supposed_to_correct_to_l1 = np.where(altered_pred_data_l_1 == l_1.index, 1, 0)
        #             where_supposed_to_correct_to_l2 = np.where(altered_pred_data_l_2 == l_2.index, 1, 0)
        #             collision_array |= where_supposed_to_correct_to_l1 * where_supposed_to_correct_to_l2

        # for l, altered_pred_data_l in altered_pred_granularity_datas.items():

        self.post_correction_test_precisions[g], self.post_correction_test_recalls[g] = (
            self.get_g_precision_and_recall(g=g, test=True, stage='post_correction'))

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
                                                    self.pred_data[test_or_train]['original'][g],
                                                    pred_granularity_data)

    def get_l_detection_rule_support(self,
                                     test: bool,
                                     l: data_preprocessing.Label) -> float:
        if l not in self.error_detection_rules:
            return 0

        test_or_train = 'test' if test else 'train'

        N_l = np.sum(self.get_where_label_is_l(pred=True, test=test, l=l))
        r_l = self.error_detection_rules[l]
        where_l_detection_rule_body_is_satisfied = (
            r_l.get_where_body_is_satisfied(
                pred_fine_data=self.pred_data[test_or_train]['original'][data_preprocessing.granularities['fine']],
                pred_coarse_data=self.pred_data[test_or_train]['original'][data_preprocessing.granularities['coarse']]))
        num_predicted_l_and_any_conditions_satisfied = np.sum(where_l_detection_rule_body_is_satisfied)
        s_l = num_predicted_l_and_any_conditions_satisfied / N_l

        assert s_l <= 1

        return s_l

    def get_l_detection_rule_confidence(self,
                                        test: bool,
                                        l: data_preprocessing.Label) -> float:
        if l not in self.error_detection_rules:
            return 0

        test_or_train = 'test' if test else 'train'

        r_l = self.error_detection_rules[l]
        where_l_detection_rule_body_is_satisfied = (
            r_l.get_where_body_is_satisfied(
                pred_fine_data=self.pred_data[test_or_train]['original'][data_preprocessing.granularities['fine']],
                pred_coarse_data=self.pred_data[test_or_train]['original'][data_preprocessing.granularities['coarse']]))
        where_l_fp = self.get_where_fp_l(test=test, l=l)
        where_head_and_body_is_satisfied = where_l_detection_rule_body_is_satisfied * where_l_fp

        num_where_l_detection_rule_body_is_satisfied = np.sum(where_l_detection_rule_body_is_satisfied)

        if num_where_l_detection_rule_body_is_satisfied == 0:
            return 0

        c_l = np.sum(where_head_and_body_is_satisfied) / num_where_l_detection_rule_body_is_satisfied
        return c_l

    def get_l_detection_rule_theoretical_precision_increase(self,
                                                            test: bool,
                                                            l: data_preprocessing.Label) -> float:
        s_l = self.get_l_detection_rule_support(test=test, l=l)

        if s_l == 0:
            return 0

        c_l = self.get_l_detection_rule_confidence(test=test, l=l)
        p_l = self.get_g_precision_and_recall(g=l.g, test=test, stage='original')[0][l]

        return s_l / (1 - s_l) * (c_l + p_l - 1)

    def get_g_detection_rule_theoretical_precision_increase(self,
                                                            test: bool,
                                                            g: data_preprocessing.Granularity):
        precision_increases = [self.get_l_detection_rule_theoretical_precision_increase(test=test, l=l)
                               for l in data_preprocessing.get_labels(g).values()]
        return np.mean(precision_increases)

    def get_l_detection_rule_theoretical_recall_decrease(self,
                                                         test: bool,
                                                         l: data_preprocessing.Label) -> float:
        c_l = self.get_l_detection_rule_confidence(test=test, l=l)
        s_l = self.get_l_detection_rule_support(test=test, l=l)

        original_precision_and_recalls = self.get_g_precision_and_recall(g=l.g, test=test, stage='original')
        p_l = original_precision_and_recalls[0][l]
        r_l = original_precision_and_recalls[1][l]

        theoretical_recall_decrease = (1 - c_l) * s_l * r_l / p_l

        return theoretical_recall_decrease

    def get_g_detection_rule_theoretical_recall_decrease(self,
                                                         test: bool,
                                                         g: data_preprocessing.Granularity):
        recall_decreases = [self.get_l_detection_rule_theoretical_recall_decrease(test=test, l=l)
                            for l in data_preprocessing.get_labels(g).values()]
        return np.mean(recall_decreases)

    def get_l_correction_rule_confidence(self,
                                         test: bool,
                                         l: data_preprocessing.Label,
                                         pred_fine_data: np.array = None,
                                         pred_coarse_data: np.array = None
                                         ) -> float:
        if l not in self.error_correction_rules:
            return 0

        test_or_train = 'test' if test else 'train'

        r_l = self.error_correction_rules[l]
        where_l_correction_rule_body_is_satisfied = (
            r_l.get_where_body_is_satisfied(
                fine_data=self.pred_data[test_or_train]['post_correction'][data_preprocessing.granularities['fine']]
                if pred_fine_data is None else pred_fine_data,
                coarse_data=self.pred_data[test_or_train]['post_correction'][data_preprocessing.granularities['coarse']]
                if pred_coarse_data is None else pred_coarse_data))
        where_l_gt = self.get_where_label_is_l(pred=False, test=test, l=l)

        where_head_and_body_is_satisfied = where_l_correction_rule_body_is_satisfied * where_l_gt
        num_where_l_correction_rule_body_is_satisfied = np.sum(where_l_correction_rule_body_is_satisfied)

        if num_where_l_correction_rule_body_is_satisfied == 0:
            return 0

        c_l = np.sum(where_head_and_body_is_satisfied) / num_where_l_correction_rule_body_is_satisfied
        return c_l

    def check_g_correction_rule_precision_recall(self,
                                                 test: bool,
                                                 g: data_preprocessing.Granularity):

        for l in data_preprocessing.get_labels(g).values():
            c_l = self.get_l_correction_rule_confidence(test=test, l=l)
            p_l = self.post_detection_test_precisions[g][l]
            r_l = self.post_detection_test_recalls[g][l]
            (p_l_new, r_l_new) = self.post_correction_test_precisions[g][l], self.post_correction_test_recalls[g][l]

            try:
                assert r_l_new >= r_l
            except AssertionError:
                print(f'class {l}: new recall: {r_l_new}, old recall: {r_l}, '
                      f'diff: {r_l_new - r_l}')

            try:
                if c_l > p_l:
                    assert p_l_new > p_l

                if p_l_new > p_l:
                    assert c_l > p_l
            except AssertionError:
                print(f'class {l}: new precision: {p_l_new}, old precision: {p_l}, '
                      f'diff: {p_l_new - p_l}')
                print(f'class {l}: confidence: {c_l}')

    def get_l_correction_rule_support(self,
                                      test: bool,
                                      l: data_preprocessing.Label,
                                      pred_fine_data: np.array = None,
                                      pred_coarse_data: np.array = None
                                      ) -> float:
        if l not in self.error_correction_rules:
            return 0

        test_or_train = 'test' if test else 'train'

        N_l = np.sum(self.get_where_label_is_l(pred=True, test=test, l=l, stage='post_correction')
                     if (pred_fine_data is None and pred_coarse_data is None)
                     else self.get_where_label_is_l_in_data(l=l,
                                                            test_pred_fine_data=pred_fine_data,
                                                            test_pred_coarse_data=pred_coarse_data))
        r_l = self.error_correction_rules[l]
        where_rule_body_is_satisfied = (
            r_l.get_where_body_is_satisfied(
                fine_data=self.pred_data[test_or_train]['post_correction'][data_preprocessing.granularities['fine']]
                if pred_fine_data is None else pred_fine_data,
                coarse_data=self.pred_data[test_or_train]['post_correction'][
                    data_preprocessing.granularities['coarse']]
                if pred_coarse_data is None else pred_coarse_data))

        s_l = np.sum(where_rule_body_is_satisfied) / N_l

        return s_l

    def get_l_correction_rule_theoretical_precision_increase(self,
                                                             test: bool,
                                                             l: data_preprocessing.Label,
                                                             ) -> float:
        c_l = self.get_l_correction_rule_confidence(test=test, l=l)
        s_l = self.get_l_correction_rule_support(test=test, l=l)
        p_l_prior_correction = self.get_g_precision_and_recall(g=l.g,
                                                               test=test,
                                                               stage='post_correction')[0][l]

        return s_l * (c_l - p_l_prior_correction) / (1 + s_l)

    def get_g_correction_rule_theoretical_precision_increase(self,
                                                             test: bool,
                                                             g: data_preprocessing.Granularity):
        precision_increases = [self.get_l_correction_rule_theoretical_precision_increase(test=test, l=l)
                               for l in data_preprocessing.get_labels(g).values()]
        return np.mean(precision_increases)

    def evaluate_and_print_g_detection_rule_precision_increase(self,
                                                               test: bool,
                                                               g: data_preprocessing.Granularity,
                                                               threshold: float = 1e-5):
        original_precisions = self.get_g_precision_and_recall(g=g, test=test, stage='original')[0]
        post_detection_precisions = self.get_g_precision_and_recall(g=g, test=test, stage='post_detection')[0]

        original_mean_precision = np.mean(list(original_precisions.values()))
        post_detection_mean_precision = np.mean(list(post_detection_precisions.values()))

        precision_diff = post_detection_mean_precision - original_mean_precision
        detection_rule_theoretical_precision_increase = (
            self.get_g_detection_rule_theoretical_precision_increase(test=test, g=g))
        precision_theory_holds = abs(detection_rule_theoretical_precision_increase - precision_diff) < threshold
        precision_theory_holds_str = utils.green_text('The theory holds!') if precision_theory_holds else (
            utils.red_text('The theory does not hold!'))

        print('\n' + '#' * 20 + f'post detection {g}-grain precision results' + '#' * 20)

        print(f'{g}-grain new precision: {post_detection_mean_precision}, '
              f'{g}-grain old precision: {original_mean_precision}, '
              f'diff: {utils.blue_text(precision_diff)}\n'
              f'theoretical precision increase: {utils.blue_text(detection_rule_theoretical_precision_increase)}\n'
              f'{precision_theory_holds_str}'
              )

    def evaluate_and_print_g_detection_rule_recall_decrease(self,
                                                            test: bool,
                                                            g: data_preprocessing.Granularity,
                                                            threshold: float = 1e-5):
        original_recalls = self.get_g_precision_and_recall(g=g, test=test, stage='original')[1]
        post_detection_recalls = self.get_g_precision_and_recall(g=g, test=test, stage='post_detection')[1]

        original_recall = np.mean(list(original_recalls.values()))
        post_detection_avg_recall = np.mean(list(post_detection_recalls.values()))
        recall_diff = post_detection_avg_recall - original_recall

        detection_rule_theoretical_recall_decrease = (
            self.get_g_detection_rule_theoretical_recall_decrease(test=test, g=g))
        recall_theory_holds = abs(abs(detection_rule_theoretical_recall_decrease) - abs(recall_diff)) < threshold
        recall_theory_holds_str = utils.green_text('The theory holds!') if recall_theory_holds else (
            utils.red_text('The theory does not hold!'))

        print('\n' + '#' * 20 + f'post detection {g}-grain recall results' + '#' * 20)

        print(f'{g}-grain new recall: {post_detection_avg_recall}, '
              f'{g}-grain old recall: {original_recall}, '
              f'diff: {utils.blue_text(recall_diff)}\n'
              f'theoretical recall decrease: -{utils.blue_text(detection_rule_theoretical_recall_decrease)}\n'
              f'{recall_theory_holds_str}')


def plot_per_class(ps,
                   rs,
                   folder: str):
    for g in data_preprocessing.granularities:
        # plot all label per granularity:
        for label in data_preprocessing.get_labels(g).values():
            plt.plot(epsilons, [ps[g]['initial'][e][label] for e in epsilons],
                     label='initial average precision')
            plt.plot(epsilons, [ps[g]['pre_correction'][e][label] for e in epsilons],
                     label='pre correction average precision')
            plt.plot(epsilons, [ps[g]['post_correction'][e][label] for e in epsilons],
                     label='post correction average precision')

            plt.plot(epsilons, [rs[g]['initial'][e][label] for e in epsilons],
                     label='initial average recall')
            plt.plot(epsilons, [rs[g]['pre_correction'][e][label] for e in epsilons],
                     label='pre correction average recall')
            plt.plot(epsilons, [rs[g]['post_correction'][e][label] for e in epsilons],
                     label='post correction average recall')

            plt.legend()
            plt.tight_layout()
            plt.grid()
            plt.title(f'{label}')
            plt.savefig(f'figs/{folder}/{label}.png')
            plt.clf()
            plt.cla()


def plot_all(ps,
             rs,
             folder: str):
    for g in data_preprocessing.granularities:
        # plot average precision recall per granularity:

        plt.plot(epsilons, [np.mean(list(ps[g]['initial'][e].values())) for e in epsilons],
                 label='initial average precision')
        plt.plot(epsilons, [np.mean(list(ps[g]['pre_correction'][e].values())) for e in epsilons],
                 label='pre correction average precision')
        plt.plot(epsilons, [np.mean(list(ps[g]['post_correction'][e].values())) for e in epsilons],
                 label='post correction average precision')

        plt.plot(epsilons, [np.mean(list(rs[g]['initial'][e].values())) for e in epsilons],
                 label='initial average precision')
        plt.plot(epsilons, [np.mean(list(rs[g]['pre_correction'][e].values())) for e in epsilons],
                 label='pre correction average precision')
        plt.plot(epsilons, [np.mean(list(rs[g]['post_correction'][e].values())) for e in epsilons],
                 label='post correction average precision')

        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.title(f'average precision recall for {g}')
        plt.savefig(f'figs/{folder}/average_{g}.png')
        plt.clf()
        plt.cla()


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
                    num_epochs=20)
        edcr.print_metrics(test=test_bool, prior=True)

        for granularity in data_preprocessing.granularities.values():
            edcr.learn_detection_rules(g=granularity)
            edcr.learn_correction_rules(g=granularity)

        # print('\n' + '#' * 50 + 'post correction' + '#' * 50)

        for granularity in data_preprocessing.granularities.values():
            edcr.apply_detection_rules(test=test_bool, g=granularity)
            edcr.evaluate_and_print_g_detection_rule_precision_increase(test=test_bool, g=granularity)
            edcr.evaluate_and_print_g_detection_rule_recall_decrease(test=test_bool, g=granularity)

        edcr.print_metrics(test=test_bool, prior=False, stage='post_detection', print_inconsistencies=False)

        for gra in data_preprocessing.granularities.values():
            edcr.apply_correction_rules(test=test_bool, g=gra)
        # edcr.apply_reversion_rules(g=gra)

        # precision_dict[gra]['initial'][epsilon] = edcr.original_test_precisions[gra]
        # recall_dict[gra]['initial'][epsilon] = edcr.original_test_recalls[gra]
        # precision_dict[gra]['pre_correction'][epsilon] = edcr.post_detection_test_precisions[gra]
        # recall_dict[gra]['pre_correction'][epsilon] = edcr.post_detection_test_recalls[gra]
        # precision_dict[gra]['post_correction'][epsilon] = edcr.post_correction_test_precisions[gra]
        # recall_dict[gra]['post_correction'][epsilon] = edcr.post_correction_test_recalls[gra]

        # edcr.print_metrics(test=test_bool, prior=False, stage='post_correction', print_inconsistencies=False)

    # folder = "experiment_1"
    #
    # if not os.path.exists(f'figs/{folder}'):
    #     os.mkdir(f'figs/{folder}')
    #
    # plot_per_class(ps=precision_dict,
    #                rs=recall_dict,
    #                folder="experiment_1")
    # plot_all(precision_dict, recall_dict, "experiment_1")
