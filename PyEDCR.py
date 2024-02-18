from __future__ import annotations

import abc
import typing
import numpy as np
import multiprocessing as mp
import multiprocessing.managers
import warnings

warnings.filterwarnings('ignore')

import utils
import data_preprocessing
import vit_pipeline
import context_handlers


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
            self.__l = l

        def __call__(self,
                     fine_data: np.array,
                     coarse_data: np.array) -> np.array:
            granularity_data = fine_data if self.__l.g == data_preprocessing.granularities['fine'] else coarse_data
            return np.where(granularity_data == self.__l.index, 1, 0)

        def __str__(self) -> str:
            return f'pred_{self.__l}'

        @property
        def l(self):
            return self.__l

        def __hash__(self):
            return self.__l.__hash__()

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
                                  data: np.array) -> np.array:
            return np.where(data == self.l.index, 1, 0)

        @abc.abstractmethod
        def __call__(self,
                     test_pred_fine_data: np.array,
                     test_pred_coarse_data: np.array) -> np.array:
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
                                        test_pred_fine_data: np.array,
                                        test_pred_coarse_data: np.array) -> np.array:
            test_pred_granularity_data = test_pred_fine_data if self.l.g == data_preprocessing.granularities['fine'] \
                else test_pred_coarse_data
            where_predicted_l = self.get_where_predicted_l(data=test_pred_granularity_data)
            where_any_conditions_satisfied = EDCR.get_where_any_conditions_satisfied(C=self.C_l,
                                                                                     fine_data=test_pred_fine_data,
                                                                                     coarse_data=test_pred_coarse_data)
            where_body_is_satisfied = where_predicted_l * where_any_conditions_satisfied

            return where_body_is_satisfied

        def __call__(self,
                     test_pred_fine_data: np.array,
                     test_pred_coarse_data: np.array) -> np.array:
            """Infer the detection rule based on the provided prediction data.

            :param test_pred_fine_data: The fine-grained prediction data.
            :param test_pred_coarse_data: The coarse-grained prediction data.
            :return: modified prediction contains -1 at examples that have errors for a specific granularity as
            derived from Label l.
            """
            test_pred_granularity_data = test_pred_fine_data if self.l.g == data_preprocessing.granularities['fine'] \
                else test_pred_coarse_data
            where_predicted_l_and_any_conditions_satisfied = (
                self.get_where_body_is_satisfied(test_pred_fine_data=test_pred_fine_data,
                                                 test_pred_coarse_data=test_pred_coarse_data))
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
            super().__init__(l=l, C_l=CC_l)

        def get_where_any_pair_satisfied(self,
                                         test_pred_fine_data: np.array,
                                         test_pred_coarse_data: np.array) -> np.array:
            test_pred_granularity_data = test_pred_fine_data if self.l.g == data_preprocessing.granularities['fine'] \
                else test_pred_coarse_data

            where_any_pair_satisfied = np.zeros_like(test_pred_granularity_data)

            for cond, l_prime in self.C_l:
                where_condition_satisfied = (
                    EDCR.get_where_any_conditions_satisfied(C={cond},
                                                            fine_data=test_pred_fine_data,
                                                            coarse_data=test_pred_coarse_data))
                where_predicted_l_prime = np.where(test_pred_granularity_data == l_prime.index, 1, 0)
                where_pair_satisfied = where_condition_satisfied * where_predicted_l_prime

                where_any_pair_satisfied |= where_pair_satisfied

            return where_any_pair_satisfied

        def __call__(self,
                     test_pred_fine_data: np.array,
                     test_pred_coarse_data: np.array) -> np.array:
            """Infer the correction rule based on the provided prediction data.

            :param test_pred_fine_data: The fine-grained prediction data.
            :param test_pred_coarse_data: The coarse-grained prediction data.
            :return: new test prediction for a specific granularity as derived from Label l.
            """
            where_any_pair_satisfied = self.get_where_any_pair_satisfied(test_pred_fine_data=test_pred_fine_data,
                                                                         test_pred_coarse_data=test_pred_coarse_data)

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

        self.train_pred_data = {g: np.load(pred_paths['train'][str(g)])[self.K_train]
                                for g in data_preprocessing.granularities.values()}

        self.test_pred_data = {g: np.load(pred_paths['test'][str(g)])[self.K_test]
                               for g in data_preprocessing.granularities.values()}

        self.condition_datas = ({EDCR.PredCondition(l=l)
                                 for g in data_preprocessing.granularities.values()
                                 for l in data_preprocessing.get_labels(g).values()}.
                                union({EDCR.ConsistencyCondition()}))

        self.train_precisions = {}
        self.train_recalls = {}

        self.original_test_pred_data = self.test_pred_data.copy()

        for g in data_preprocessing.granularities.values():
            self.train_precisions[g], self.train_recalls[g] = self.get_g_precision_and_recall(g=g, test=False)

        self.original_test_precisions = {}
        self.original_test_recalls = {}

        for g in data_preprocessing.granularities.values():
            self.original_test_precisions[g], self.original_test_recalls[g] = (
                self.get_g_precision_and_recall(g=g, test=True))

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
                                   rules: typing.Dict[data_preprocessing.Label, {(_Condition, data_preprocessing.Label)}]):
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
                        original: bool = True) -> typing.Union[np.array, tuple[np.array]]:
        """Retrieves prediction data based on specified test/train mode.

        :param original:
        :param g:
        :param test: True for test data, False for training data.
        :return: Fine-grained and coarse-grained prediction data.
        """
        test_pred_data = (self.original_test_pred_data if original else self.test_pred_data)
        if g is not None:
            return (test_pred_data if test else self.train_pred_data)[g]

        pred_fine_data, pred_coarse_data = [(test_pred_data if test else self.train_pred_data)[g]
                                            for g in data_preprocessing.granularities.values()]

        return pred_fine_data, pred_coarse_data

    def get_where_label_is_l(self,
                             pred: bool,
                             test: bool,
                             l: data_preprocessing.Label,
                             original: bool = True) -> np.array:
        """ Retrieves indices of instances where the specified label is present.

        :param original:
        :param pred: True for prediction, False for ground truth
        :param test: Whether to use test data (True) or training data (False).
        :param l: The label to search for.
        :return: A boolean array indicating which instances have the given label.
        """
        data = self.get_predictions(test=test, g=l.g, original=original) if pred else (
            data_preprocessing.get_ground_truths(test=test, K=self.K_test if test else self.K_train, g=l.g))
        where_label_is_l = np.where(data == l.index, 1, 0)
        return where_label_is_l

    def print_metrics(self,
                      test: bool,
                      prior: bool,
                      print_inconsistencies: bool = True,
                      original: bool = True):

        """Prints performance metrics for given test/train data.

        Calculates and prints various metrics (accuracy, precision, recall, etc.)
        using appropriate true labels and prediction data based on the specified mode.

        :param original:
        :param print_inconsistencies:
        :param prior:
        :param test: True to use test data, False to use training data.
        """
        pred_fine_data, pred_coarse_data = self.get_predictions(test=test, original=original)
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
                                    original: bool = True) -> np.array:
        """Calculates true positive mask for given granularity and label.

        :param original:
        :param test: Whether to use test data (True) or training data (False).
        :param g: The granularity level.
        :return: A mask with 1s for true positive instances, 0s otherwise.
        """
        ground_truth = data_preprocessing.get_ground_truths(test=test, K=self.K_test if test else self.K_train, g=g)
        return np.where(self.get_predictions(test=test, g=g, original=original) == ground_truth, 1, 0)

    def get_where_predicted_incorrect(self,
                                      test: bool,
                                      g: data_preprocessing.Granularity,
                                      original: bool = True) -> np.array:
        """Calculates false positive mask for given granularity and label.

        :param original:
        :param test: whether to get prediction from train or test set
        :param g: The granularity level
        :return: A mask with 1s for false positive instances, 0s otherwise.
        """
        return 1 - self.get_where_predicted_correct(test=test, g=g, original=original)

    def get_where_tp_l(self,
                       test: bool,
                       l: data_preprocessing.Label,
                       original: bool = True) -> np.array:
        """ Retrieves indices of training instances where the true label is l and the model correctly predicted l.

        :param original:
        :param test:
        :param l: The label to query.
        :return: A boolean array indicating which training instances satisfy the criteria.
        """
        return (self.get_where_label_is_l(pred=True, test=test, l=l, original=original) *
                self.get_where_predicted_correct(test=test, g=l.g, original=original))

    def get_where_fp_l(self,
                       test: bool,
                       l: data_preprocessing.Label,
                       original: bool = True) -> np.array:
        """ Retrieves indices of instances where the predicted label is l and the ground truth is not l.

        :param original:
        :param test:
        :param l: The label to query.
        :return: A boolean array indicating which instances satisfy the criteria.
        """
        return (self.get_where_label_is_l(pred=True, test=test, l=l, original=original) *
                self.get_where_predicted_incorrect(test=test, g=l.g, original=original))

    def get_g_precision_and_recall(self,
                                   g: data_preprocessing.Granularity,
                                   test: bool,
                                   original: bool = True) -> (dict[data_preprocessing.Label, float],
                                                              dict[data_preprocessing.Label, float]):
        p_g = {}
        r_g = {}

        for l in data_preprocessing.get_labels(g).values():
            t_p_l = np.sum(self.get_where_tp_l(test=test, l=l, original=original))
            f_p_l = np.sum(self.get_where_fp_l(test=test, l=l, original=original))
            N_l_gt = np.sum(self.get_where_label_is_l(test=test, pred=False, l=l, original=original))

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

    def get_where_fn_l(self,
                       test: bool,
                       l: data_preprocessing.Label, ):
        N_l_gt = self.get_where_label_is_l(test=test, pred=False, l=l)
        t_p_l = self.get_where_tp_l(test=test, l=l)

        return N_l_gt - t_p_l

    def get_POS_l_C(self,
                    l: data_preprocessing.Label,
                    C: set[_Condition]) -> int:
        """Calculate the number of train samples that satisfy any conditions for some set of condition
        and are false positive.

        :param C: A set of `Condition` objects.
        :param l: The label of interest.
        :return: The number of instances that are false negative and satisfying some conditions.
        """
        where_was_wrong_with_respect_to_l = self.get_where_fn_l(test=False, l=l) + self.get_where_fp_l(test=False, l=l)
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
        POS_l_CC = self.get_POS_l_CC(l=l,
                                     where_any_pair_is_satisfied_in_train_pred=
                                     where_any_pair_is_satisfied_in_train_pred)
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

        if N_l:
            P_l = self.train_precisions[l.g][l]
            R_l = self.train_recalls[l.g][l]
            q_l = self.epsilon * N_l * P_l / R_l

            DC_star = {cond for cond in self.condition_datas if self.get_NEG_l_C(l=l, C={cond}) <= q_l}

            while DC_star != set():
                best_score = -1
                best_cond = None

                for cond in DC_star:
                    POS_l_c = self.get_POS_l_C(l=l, C=DC_l.union({cond}))
                    if POS_l_c >= best_score:
                        best_score = POS_l_c
                        best_cond = cond

                DC_l = DC_l.union({best_cond})
                DC_star = {cond for cond in self.condition_datas.difference(DC_l)
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
        CC_sorted = sorted(CC_all, key=lambda cond_and_l: self.get_CON_l_CC(l=l, CC={cond_and_l}), reverse=True)

        with context_handlers.WrapTQDM(total=len(CC_sorted)) as progress_bar:
            for cond_and_l in CC_sorted:
                a = self.get_CON_l_CC(l=l, CC=CC_l.union({cond_and_l})) - self.get_CON_l_CC(l=l, CC=CC_l)
                b = (self.get_CON_l_CC(l=l, CC=CC_l_prime.difference({cond_and_l})) -
                     self.get_CON_l_CC(l=l, CC=CC_l_prime))

                if a >= b:
                    CC_l = CC_l.union({cond_and_l})
                else:
                    CC_l_prime = CC_l_prime.difference({cond_and_l})

                if utils.is_local():
                    progress_bar.update(1)

        print(f'\n{l}: len(CC_l)={len(CC_l)}/{len(CC_all)}, CON_l_CC={self.get_CON_l_CC(l=l, CC=CC_l)}, '
              f'P_l={self.train_precisions[l.g][l]}\n')

        # if self.get_CON_l_CC(l=l, CC=CC_l) <= self.train_precisions[l.g][l]:
        #     CC_l = set()

        if not utils.is_local():
            shared_index.value += 1
            print(f'Completed {shared_index.value}/{len(data_preprocessing.get_labels(l.g).values())}')

        return l, CC_l

    def DetCorrRuleLearn(self,
                         g: data_preprocessing.Granularity,
                         learn_correction_rules: bool = True):
        """Learns error detection and correction rules for all labels at a given granularity.

        :param learn_correction_rules:
        :param g: The granularity level (e.g., 'fine', 'coarse').
        """
        CC_all = set()  # in this use case where the conditions are fine and coarse predictions
        granularity_labels = data_preprocessing.get_labels(g).values()

        # learning detection rules
        print(f'\nLearning {g}-grain error detection rules...')
        with context_handlers.WrapTQDM(total=len(granularity_labels)) as progress_bar:
            for l in granularity_labels:
                DC_l = self.DetRuleLearn(l=l)

                if len(DC_l):
                    self.error_detection_rules[l] = EDCR.ErrorDetectionRule(l=l, DC_l=DC_l)

                for cond_l in DC_l:
                    if not (isinstance(cond_l, EDCR.PredCondition) and cond_l.l == l):
                        CC_all = CC_all.union({(cond_l, l)})

                if utils.is_local():
                    progress_bar.update(1)

        # learning correction rules
        if learn_correction_rules:
            print(f'\nLearning {g}-grain error correction rules...')
            processes_num = min(len(granularity_labels), mp.cpu_count())

            manager = mp.Manager()
            shared_index = manager.Value('i', 0)

            iterable = [(l, CC_all, shared_index) for l in granularity_labels]

            with mp.Pool(processes_num) as pool:
                CC_ls = pool.starmap(func=self._CorrRuleLearn,
                                     iterable=iterable)

            for l, CC_l in CC_ls:
                if len(CC_l):
                    self.error_correction_rules[l] = EDCR.ErrorCorrectionRule(l=l, CC_l=CC_l)
                else:
                    print(utils.red_text('\n' + '#' * 10 + f' {l} does not have an error correction rule!\n'))

    def apply_detection_rules(self,
                              g: data_preprocessing.Granularity):
        """Applies error detection rules to test predictions for a given granularity. If a rule is satisfied for
        a particular label, the prediction data for that label is modified with a value of -1,
        indicating a potential error.

        :params g: The granularity of the predictions to be processed.
        """
        test_pred_fine_data, test_pred_coarse_data = self.get_predictions(test=True)
        altered_pred_granularity_data = self.get_predictions(test=True, g=g)

        for rule_g_l in {l: rule_l for l, rule_l in self.error_detection_rules.items() if l.g == g}.values():
            altered_pred_data_l = rule_g_l(test_pred_fine_data=test_pred_fine_data,
                                           test_pred_coarse_data=test_pred_coarse_data)
            altered_pred_granularity_data = np.where(altered_pred_data_l == -1, -1, altered_pred_granularity_data)

        self.test_pred_data[g] = altered_pred_granularity_data

        error_mask = np.where(self.test_pred_data[g] == -1, -1, 0)

        return error_mask

    def apply_correction_rules(self,
                               g: data_preprocessing.Granularity):
        """Applies error correction rules to test predictions for a given granularity. If a rule is satisfied for a
        particular label, the prediction data for that label is corrected using the rule's logic.

        :param g: The granularity of the predictions to be processed.
        """
        test_pred_fine_data, test_pred_coarse_data = self.get_predictions(test=True)
        altered_pred_granularity_data = self.get_predictions(test=True, g=g, original=False)

        altered_pred_granularity_datas = {}
        for l, rule_g_l in {l: rule_l for l, rule_l in self.error_correction_rules.items() if l.g == g}.items():
            altered_pred_granularity_datas[l] = rule_g_l(test_pred_fine_data=test_pred_fine_data,
                                                         test_pred_coarse_data=test_pred_coarse_data)

        collision_array = np.zeros_like(altered_pred_granularity_data)

        for l_1, altered_pred_data_l_1, in altered_pred_granularity_datas.items():
            for l_2, altered_pred_data_l_2 in altered_pred_granularity_datas.items():
                if l_1 != l_2:
                    where_supposed_to_correct_to_l1 = np.where(altered_pred_data_l_1 == l_1.index, 1, 0)
                    where_supposed_to_correct_to_l2 = np.where(altered_pred_data_l_2 == l_2.index, 1, 0)
                    collision_array |= where_supposed_to_correct_to_l1 * where_supposed_to_correct_to_l2

        for l, altered_pred_data_l in altered_pred_granularity_datas.items():
            altered_pred_granularity_data = np.where(
                (collision_array != 1) &
                (altered_pred_data_l == l.index),
                l.index,
                altered_pred_granularity_data)

        self.test_pred_data[g] = altered_pred_granularity_data

        return altered_pred_granularity_data

    def apply_reversion_rules(self,
                              g: data_preprocessing.Granularity):
        pred_granularity_data = self.get_predictions(test=True, g=g, original=False)

        self.test_pred_data[g] = np.where(pred_granularity_data == -1,
                                          self.original_test_pred_data[g], pred_granularity_data)

    def get_l_detection_rule_support_on_test(self,
                                             l: data_preprocessing.Label) -> float:
        if l not in self.error_detection_rules:
            return 0

        N_l = np.sum(self.get_where_label_is_l(pred=True, test=True, l=l))
        r_l = self.error_detection_rules[l]
        where_l_detection_rule_body_is_satisfied = (
            r_l.get_where_body_is_satisfied(
                test_pred_fine_data=self.original_test_pred_data[data_preprocessing.granularities['fine']],
                test_pred_coarse_data=self.original_test_pred_data[data_preprocessing.granularities['coarse']]))
        num_predicted_l_and_any_conditions_satisfied = np.sum(where_l_detection_rule_body_is_satisfied)
        s_l = num_predicted_l_and_any_conditions_satisfied / N_l

        assert s_l <= 1

        return s_l

    def get_l_detection_rule_confidence_on_test(self,
                                                l: data_preprocessing.Label) -> float:
        if l not in self.error_detection_rules:
            return 0

        r_l = self.error_detection_rules[l]
        where_l_detection_rule_body_is_satisfied = (
            r_l.get_where_body_is_satisfied(
                test_pred_fine_data=self.original_test_pred_data[data_preprocessing.granularities['fine']],
                test_pred_coarse_data=self.original_test_pred_data[data_preprocessing.granularities['coarse']]))
        where_l_fp = self.get_where_fp_l(test=True, l=l)
        where_head_and_body_is_satisfied = where_l_detection_rule_body_is_satisfied * where_l_fp

        num_where_l_detection_rule_body_is_satisfied = np.sum(where_l_detection_rule_body_is_satisfied)

        if num_where_l_detection_rule_body_is_satisfied == 0:
            return 0

        c_l = np.sum(where_head_and_body_is_satisfied) / num_where_l_detection_rule_body_is_satisfied
        return c_l

    def get_l_theoretical_precision_increase(self,
                                             l: data_preprocessing.Label) -> float:
        s_l = self.get_l_detection_rule_support_on_test(l=l)

        if s_l == 0:
            return 0

        c_l = self.get_l_detection_rule_confidence_on_test(l=l)
        p_l = self.original_test_precisions[l.g][l]

        return s_l / (1 - s_l) * (c_l + p_l - 1)

    def get_g_theoretical_precision_increase(self,
                                             g: data_preprocessing.Granularity):
        precision_increases = [self.get_l_theoretical_precision_increase(l=l)
                               for l in data_preprocessing.get_labels(g).values()]
        return np.mean(precision_increases)

    def get_l_theoretical_recall_decrease(self,
                                          l: data_preprocessing.Label) -> float:
        c_l = self.get_l_detection_rule_confidence_on_test(l=l)
        s_l = self.get_l_detection_rule_support_on_test(l=l)
        p_l = self.original_test_precisions[l.g][l]
        r_l = self.original_test_recalls[l.g][l]
        theoretical_recall_decrease = (1 - c_l) * s_l * r_l / p_l

        return theoretical_recall_decrease

    def get_g_theoretical_recall_decrease(self,
                                          g: data_preprocessing.Granularity):
        recall_decreases = [self.get_l_theoretical_recall_decrease(l=l)
                            for l in data_preprocessing.get_labels(g).values()]
        return np.mean(recall_decreases)

    def get_l_theorem_1_condition(self,
                                  l: data_preprocessing.Label):
        return self.get_l_detection_rule_support_on_test(l=l) + self.original_test_precisions[l] <= 1

    def get_g_theorem_1_condition(self,
                                  g: data_preprocessing.Granularity):
        return np.array([self.get_l_theorem_1_condition(l=l) for l in data_preprocessing.get_labels(g).values()])


if __name__ == '__main__':
    for e in [0.1 * i for i in range(1, 5)]:
        print('#' * 25 + f'eps = {e}' + '#' * 50)
        edcr = EDCR(epsilon=e,
                    main_model_name='vit_b_16',
                    combined=True,
                    loss='BCE',
                    lr=0.0001,
                    num_epochs=20)
        edcr.print_metrics(test=True, prior=False)

        for g in data_preprocessing.granularities.values():
            edcr.DetCorrRuleLearn(g=g, learn_correction_rules=False)

        # # print([edcr.get_l_correction_rule_support_on_test(l=l) for l in
        # #        list(data_preprocessing.fine_grain_labels.values()) +
        # #        list(data_preprocessing.coarse_grain_labels.values())])

        for g in data_preprocessing.granularities:
            edcr.apply_detection_rules(g=g)
            p, r = edcr.get_g_precision_and_recall(g=g, test=True, original=False)
            new_avg_precision = np.mean(list(p.values()))
            new_avg_recall = np.mean(list(r.values()))
            old_precision = np.mean(list(edcr.original_test_precisions[g].values()))
            old_recall = np.mean(list(edcr.original_test_recalls[g].values()))

            print(f'new precision: {new_avg_precision}, old precision: {old_precision}, '
                  f'diff: {new_avg_precision - old_precision}\n'
                  f'theoretical_precision_increase: {edcr.get_g_theoretical_precision_increase(g=g)}')
            print(f'new recall: {new_avg_recall}, old recall: {old_recall}, '
                  f'diff: {new_avg_recall - old_recall}\n'
                  f'theoretical_recall_decrease: {edcr.get_g_theoretical_recall_decrease(g=g)}')


            # for g in data_preprocessing.granularities:
            # edcr.apply_correction_rules(g=g)
            # edcr.apply_reversion_rules(g=g)

        edcr.print_metrics(test=True, prior=False, print_inconsistencies=False, original=False)

        #     edcr.apply_correction_rules(g=g)
        #     edcr.apply_reversion_rules(g=g)
        #
        # edcr.print_metrics(test=True, prior=False)
