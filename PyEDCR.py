from __future__ import annotations

import abc
import typing
import numpy as np
from sklearn.metrics import precision_score, recall_score
import multiprocessing as mp

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
        __main_model_name (str): Name of the primary model used for predictions.
        __combined (bool): Whether combined features (coarse and fine) were used during training.
        __loss (str): Loss function used during training.
        __lr: Learning rate used during training.
        __num_epochs (int): Number of training epochs.
        __epsilon: Value using for constraint in getting rules
        rules: ...
    """

    class _Condition(typing.Callable, abc.ABC):
        """Represents a condition that can be evaluated on examples.

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
            return np.where((fine_data if self.__l.g == data_preprocessing.granularities['fine'] else coarse_data)
                            == self.__l.index, 1, 0)

        def __str__(self) -> str:
            return f'pred_{self.__l}'

        @property
        def l(self):
            return self.__l

        def __hash__(self):
            return hash(self.__l)

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

    class _Rule(typing.Callable, abc.ABC):
        """Represents a rule for evaluating predictions based on conditions and labels.

        :param l: The label associated with the rule.
        :param C_l: The set of conditions that define the rule.
        """

        def __init__(self,
                     l: data_preprocessing.Label,
                     C_l: typing.Union[set[EDCR._Condition], set[(EDCR._Condition, data_preprocessing.Label)]]):
            self._l = l
            self._C_l = C_l

        def _get_datas(self,
                       test_pred_fine_data: np.array,
                       test_pred_coarse_data: np.array) -> (np.array, np.array):
            """Retrieves fine-grained or coarse-grained prediction data based on the label's granularity.

            :param test_pred_fine_data: The fine-grained prediction data.
            :param test_pred_coarse_data: The coarse-grained prediction data.
            :return: A tuple containing the relevant prediction data and a mask indicating where the label is predicted.
            """
            test_pred_granularity_data = test_pred_fine_data if isinstance(self._l, data_preprocessing.FineGrainLabel) \
                else test_pred_coarse_data
            where_predicted_l = np.where(test_pred_granularity_data == self._l.index, 1, 0)

            return test_pred_granularity_data, where_predicted_l

        @abc.abstractmethod
        def __call__(self,
                     test_pred_fine_data: np.array,
                     test_pred_coarse_data: np.array) -> typing.Union[bool, np.array]:
            pass

    class ErrorDetectionRule(_Rule):
        def __init__(self,
                     l: data_preprocessing.Label,
                     DC_l: set[EDCR._Condition]):
            """Construct a detection rule for evaluating predictions based on conditions and labels.

            :param l: The label associated with the rule.
            :param DC_l: The set of conditions that define the rule.
            """
            super().__init__(l=l, C_l=DC_l)
            assert all(cond.l != self._l for cond in {cond_prime for cond_prime in self._C_l
                                                      if isinstance(cond_prime, EDCR.PredCondition)})

        def __call__(self,
                     test_pred_fine_data: np.array,
                     test_pred_coarse_data: np.array) -> typing.Union[bool, np.array]:
            """Infer the detection rule based on the provided prediction data.

            :param test_pred_fine_data: The fine-grained prediction data.
            :param test_pred_coarse_data: The coarse-grained prediction data.
            :return: modified prediction contains -1 at examples that have errors for a specific granularity as 
            derived from Label l.
            """
            test_pred_granularity_data, where_predicted_l = self._get_datas(test_pred_fine_data=test_pred_fine_data,
                                                                            test_pred_coarse_data=test_pred_coarse_data)
            where_any_conditions_satisfied = EDCR._get_where_any_conditions_satisfied(C=self._C_l,
                                                                                      fine_data=test_pred_fine_data,
                                                                                      coarse_data=test_pred_coarse_data)
            where_predicted_l_and_any_conditions_satisfied = where_predicted_l * where_any_conditions_satisfied
            altered_pred_data = np.where(where_predicted_l_and_any_conditions_satisfied == 1, -1,
                                         test_pred_granularity_data)

            return altered_pred_data

        def __str__(self) -> str:
            return '\n'.join(f'error_{self._l}(x) <- pred_{self._l}(x) ^ {cond}(x)' for cond in self._C_l)

        def __len__(self):
            return len(self._C_l)

    class ErrorCorrectionRule(_Rule):
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
            test_pred_granularity_data = self._get_datas(test_pred_fine_data=test_pred_fine_data,
                                                         test_pred_coarse_data=test_pred_coarse_data)[0]

            where_any_pair_satisfied = np.zeros_like(test_pred_granularity_data)

            for cond, l_prime in self._C_l:
                where_condition_satisfied = (
                    EDCR._get_where_any_conditions_satisfied(C={cond},
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

            altered_pred_data = np.where(where_any_pair_satisfied == 1, self._l.index, -1)

            return altered_pred_data

        def __str__(self) -> str:
            return '\n'.join(f'corr_{self._l}(x) <- {cond}(x) ^ pred_{l_prime}(x)' for (cond, l_prime) in self._C_l)

    def __init__(self,
                 main_model_name: str,
                 combined: bool,
                 loss: str,
                 lr: typing.Union[str, float],
                 num_epochs: int,
                 epsilon: typing.Union[str, float],
                 K: int = None):
        self.__main_model_name = main_model_name
        self.__combined = combined
        self.__loss = loss
        self.__lr = lr
        self.__epsilon = epsilon

        test_pred_fine_path = vit_pipeline.get_filepath(model_name=main_model_name,
                                                        combined=combined,
                                                        test=True,
                                                        granularity='fine',
                                                        loss=loss,
                                                        lr=lr,
                                                        pred=True,
                                                        epoch=num_epochs)
        test_pred_coarse_path = vit_pipeline.get_filepath(model_name=main_model_name,
                                                          combined=combined,
                                                          test=True,
                                                          granularity='coarse',
                                                          loss=loss,
                                                          lr=lr,
                                                          pred=True,
                                                          epoch=num_epochs)

        train_pred_fine_path = vit_pipeline.get_filepath(model_name=main_model_name,
                                                         combined=combined,
                                                         test=False,
                                                         granularity='fine',
                                                         loss=loss,
                                                         lr=lr,
                                                         pred=True,
                                                         epoch=num_epochs)
        train_pred_coarse_path = vit_pipeline.get_filepath(model_name=main_model_name,
                                                           combined=combined,
                                                           test=False,
                                                           granularity='coarse',
                                                           loss=loss,
                                                           lr=lr,
                                                           pred=True,
                                                           epoch=num_epochs)

        self.__K = K if K is not None else np.load(train_pred_fine_path).shape[0]
        self.__T = np.load(train_pred_fine_path).shape[0]

        self.__train_pred_data = {g: np.load(train_pred_fine_path
                                             if str(g) == 'fine' else train_pred_coarse_path)[:self.__K]
                                  for g in data_preprocessing.granularities.values()}

        self.__test_pred_data = {g: np.load(test_pred_fine_path
                                            if str(g) == 'fine' else test_pred_coarse_path)[:self.__K]
                                 for g in data_preprocessing.granularities.values()}

        self.__condition_datas = ({EDCR.PredCondition(l=l)
                                   for g in data_preprocessing.granularities.values()
                                   for l in data_preprocessing.get_labels(g).values()}.
                                  union({EDCR.ConsistencyCondition()}))

        self.__train_precisions = {g: precision_score(y_true=data_preprocessing.get_ground_truths(test=False,
                                                                                                  K=self.__K,
                                                                                                  g=g),
                                                      y_pred=self.__train_pred_data[g],
                                                      average=None)
                                   for g in data_preprocessing.granularities.values()}

        self.__train_recalls = {g: recall_score(y_true=data_preprocessing.get_ground_truths(test=False,
                                                                                            K=self.__K,
                                                                                            g=g),
                                                y_pred=self.__train_pred_data[g],
                                                average=None)
                                for g in data_preprocessing.granularities.values()}

        self.error_detection_rules: dict[data_preprocessing.Label, EDCR.ErrorDetectionRule] = {}
        self.error_correction_rules: dict[data_preprocessing.Label, EDCR.ErrorCorrectionRule] = {}

        self.__post_detection_rules_test_predictions = {}
        self.__post_correction_rules_test_predictions = {}

    @classmethod
    def test(cls,
             epsilon: float,
             K: int,
             print_pred_and_true: bool = False):
        instance = cls(main_model_name='vit_b_16',
                       combined=True,
                       loss='BCE',
                       lr=0.0001,
                       num_epochs=20,
                       epsilon=epsilon,
                       K=K)

        print(f'Taking {instance.__K} / {instance.__T} train examples')

        if print_pred_and_true:
            fg = data_preprocessing.fine_grain_classes_str
            cg = data_preprocessing.coarse_grain_classes_str

            print('\n'.join([(
                f'pred: {(fg[fine_prediction_index], cg[coarse_prediction__index])}, '
                f'true: {(fg[fine_gt__index], cg[coarse_gt__index])}')
                for fine_prediction_index, coarse_prediction__index, fine_gt__index, coarse_gt__index
                in zip(*list(instance.__train_pred_data.values()),
                       *data_preprocessing.get_ground_truths(test=False, K=instance.__K))]))

        return instance

    def __get_predictions(self,
                          test: bool,
                          g: data_preprocessing.Granularity = None) -> typing.Union[np.array, tuple[np.array]]:
        """Retrieves prediction data based on specified test/train mode.

        :param test: True for test data, False for training data.
        :return: Fine-grained and coarse-grained prediction data.
        """
        if g is not None:
            return (self.__test_pred_data if test else self.__train_pred_data)[g]

        pred_fine_data, pred_coarse_data = [(self.__test_pred_data if test else self.__train_pred_data)[g]
                                            for g in data_preprocessing.granularities.values()]

        return pred_fine_data, pred_coarse_data

    def test_get_predictions(self):
        assert np.all(self.__get_predictions(test=True, g=data_preprocessing.granularities['fine']) ==
                      np.load('test/test_pred_fine.npy'))

    def __get_where_label_is_l(self,
                               pred: bool,
                               test: bool,
                               l: data_preprocessing.Label) -> np.array:
        """ Retrieves indices of instances where the specified label is present.

        :param test: Whether to use test data (True) or training data (False).
        :param l: The label to search for.
        :return: A boolean array indicating which instances have the given label.
        """
        granularity_data = self.__get_predictions(test=test, g=l.g) if pred else \
            (data_preprocessing.get_ground_truths(test=test, K=self.__K, g=l.g))
        return np.where(granularity_data == l.index, 1, 0)

    def __get_how_many_predicted_l(self,
                                   test: bool,
                                   l: data_preprocessing.Label) -> int:
        """ Retrieves number of instances where the specified label is present.

        :param test: Whether to use test data (True) or training data (False).
        :param l: The label to search for.
        :return: A boolean array indicating which instances have the given label.
        """
        return np.sum(self.__get_where_label_is_l(pred=True, test=test, l=l))

    def print_metrics(self,
                      test: bool,
                      prior: bool):

        """Prints performance metrics for given test/train data.

        Calculates and prints various metrics (accuracy, precision, recall, etc.)
        using appropriate true labels and prediction data based on the specified mode.

        :param prior:
        :param test: True to use test data, False to use training data.
        """
        pred_fine_data, pred_coarse_data = self.__get_predictions(test=test) if (not test) or prior else \
            [self.__post_correction_rules_test_predictions[g] for g in data_preprocessing.granularities.values()]
        true_fine_data, true_coarse_data = data_preprocessing.get_ground_truths(test=test, K=self.__K, )

        vit_pipeline.get_and_print_metrics(pred_fine_data=pred_fine_data,
                                           pred_coarse_data=pred_coarse_data,
                                           loss=self.__loss,
                                           true_fine_data=true_fine_data,
                                           true_coarse_data=true_coarse_data,
                                           test=test,
                                           prior=prior,
                                           combined=self.__combined,
                                           model_name=self.__main_model_name,
                                           lr=self.__lr)

    def __get_where_predicted_correct(self,
                                      test: bool,
                                      g: data_preprocessing.Granularity) -> np.array:
        """Calculates true positive mask for given granularity and label.

        :param g: The granularity level.
        :return: A mask with 1s for true positive instances, 0s otherwise.
        """
        return np.where(self.__get_predictions(test=test, g=g) ==
                        data_preprocessing.get_ground_truths(test=test, K=self.__K, g=g), 1, 0)

    def __get_where_predicted_incorrect(self,
                                        test: bool,
                                        g: data_preprocessing.Granularity) -> np.array:
        """Calculates false positive mask for given granularity and label.

        :param g: The granularity level
        :return: A mask with 1s for false positive instances, 0s otherwise.
        """
        return 1 - self.__get_where_predicted_correct(test=test, g=g)

    def __get_where_train_tp_l(self,
                               l: data_preprocessing.Label) -> np.array:
        """ Retrieves indices of training instances where the true label is l and the model correctly predicted l.
        
        :param l: The label to query.
        :return: A boolean array indicating which training instances satisfy the criteria.
        """
        return (self.__get_where_label_is_l(pred=True, test=False, l=l) *
                self.__get_where_predicted_correct(test=False, g=l.g))

    def __get_where_train_fp_l(self,
                               l: data_preprocessing.Label) -> np.array:
        """ Retrieves indices of training instances where the true label is l and the model incorrectly predicted l.
        
        :param l: The label to query.
        :return: A boolean array indicating which training instances satisfy the criteria.
        """
        return (self.__get_where_label_is_l(pred=True, test=False, l=l) *
                self.__get_where_predicted_incorrect(test=False, g=l.g))

    @staticmethod
    def _get_where_any_conditions_satisfied(C: set[_Condition],
                                            fine_data: typing.Union[np.array, typing.Iterable[np.array]],
                                            coarse_data: typing.Union[np.array, typing.Iterable[np.array]]) -> bool:
        """Checks if all given conditions are satisfied for each example.

        :param C: A set of `Condition` objects.
        :fine_data: Data that used for Condition having FineGrainLabel l 
        :coarse_data: Data that used for Condition having CoarseGrainLabel l 
        :return: A NumPy array with True values if the example satisfy all conditions and False otherwise.
        """
        any_condition_satisfied = np.zeros_like(fine_data)

        for cond in C:
            any_condition_satisfied |= cond(fine_data=fine_data, coarse_data=coarse_data)

        return any_condition_satisfied

    def __get_NEG_l(self,
                    l: data_preprocessing.Label,
                    C: set[_Condition]) -> int:
        """Calculate the number of samples that satisfy any of the conditions and are true positive.

        :param C: A set of `Condition` objects.
        :param l: The label of interest.
        :return: The number of instances that is true negative and satisfying all conditions.
        """
        where_train_tp_l = self.__get_where_train_tp_l(l=l)
        train_pred_fine_data, train_pred_coarse_data = self.__get_predictions(test=False)
        where_any_conditions_satisfied = self._get_where_any_conditions_satisfied(C=C,
                                                                                  fine_data=train_pred_fine_data,
                                                                                  coarse_data=train_pred_coarse_data)
        NEG_l = int(np.sum(where_train_tp_l * where_any_conditions_satisfied))

        return NEG_l

    def __get_POS_l(self,
                    l: data_preprocessing.Label,
                    C: set[_Condition]) -> int:
        """Calculate the number of samples that satisfy the conditions for some set of condition 
        and have false positive.

        :param C: A set of `Condition` objects.
        :param l: The label of interest.
        :return: The number of instances that is false negative and satisfying all conditions.
        """
        where_train_fp_l = self.__get_where_train_fp_l(l=l)
        train_pred_fine_data, train_pred_coarse_data = self.__get_predictions(test=False)
        where_any_conditions_satisfied = self._get_where_any_conditions_satisfied(C=C,
                                                                                  fine_data=train_pred_fine_data,
                                                                                  coarse_data=train_pred_coarse_data)
        POS_l = int(np.sum(where_train_fp_l * where_any_conditions_satisfied))

        return POS_l

    def __get_BOD_l(self,
                    l: data_preprocessing.Label,
                    CC: set[(_Condition, data_preprocessing.Label)]) -> (int, np.array):
        train_granularity_pred_data = self.__get_predictions(test=False, g=l.g)
        train_fine_pred_data, train_coarse_pred_data = self.__get_predictions(test=False)
        where_any_pair_is_satisfied_in_train_pred = np.zeros_like(train_granularity_pred_data)

        for cond, l_prime in CC:
            where_predicted_l_prime_in_train_pred = self.__get_where_label_is_l(pred=True, test=False, l=l_prime)
            where_condition_is_satisfied_in_train_pred = cond(train_fine_pred_data, train_coarse_pred_data)
            where_pair_is_satisfied = where_predicted_l_prime_in_train_pred * where_condition_is_satisfied_in_train_pred
            where_any_pair_is_satisfied_in_train_pred |= where_pair_is_satisfied

        BOD_l = np.sum(where_any_pair_is_satisfied_in_train_pred)

        return BOD_l, where_any_pair_is_satisfied_in_train_pred

    def test_BOD_l(self,
                   l: data_preprocessing.Label,
                   CC: set[(_Condition, data_preprocessing.Label)],
                   expected_result: float):
        res = self.__get_BOD_l(l=l, CC=CC)[0]
        print(res)
        assert res == expected_result

    def __get_CON_l(self,
                    l: data_preprocessing.Label,
                    CC: set[(_Condition, data_preprocessing.Label)]) -> float:
        """Calculate the ratio of number of samples that satisfy the rule body and head with the ones
        that only satisfy the body, given a set of condition class pairs.

        :param CC: A set of `Condition` - `Label` pairs.
        :param l: The label of interest.
        :return: ratio as defined above
        """

        where_train_ground_truths_is_l = self.__get_where_label_is_l(pred=False, test=False, l=l)
        BOD_l, where_any_pair_is_satisfied_in_train_pred = self.__get_BOD_l(l=l, CC=CC)
        POS_l = np.sum(where_any_pair_is_satisfied_in_train_pred * where_train_ground_truths_is_l)
        CON_l = POS_l / BOD_l if BOD_l else 0

        return CON_l

    def test_CON_l(self,
                   l: data_preprocessing.Label,
                   CC: set[(_Condition, data_preprocessing.Label)],
                   expected_result: float):
        print(self.__get_CON_l(l=l, CC=CC))
        assert self.__get_CON_l(l=l, CC=CC) == expected_result

    def __DetRuleLearn(self,
                       l: data_preprocessing.Label) -> set[_Condition]:
        """Learns error detection rules for a specific label and granularity. These rules capture conditions
        that, when satisfied, indicate a higher likelihood of prediction errors for a given label.

        :param l: The label of interest.
        :return: A set of `Condition` representing the learned error detection rules.
        """
        DC_l = set()

        N_l = self.__get_how_many_predicted_l(test=False, l=l)

        if N_l:
            P_l = self.__train_precisions[l.g][l.index]
            R_l = self.__train_recalls[l.g][l.index]
            q_l = self.__epsilon * N_l * P_l / R_l

            DC_star = {cond for cond in self.__condition_datas if self.__get_NEG_l(l=l, C={cond}) <= q_l}

            while len(DC_star) > 0:
                best_score = -1
                best_cond = None

                for cond in DC_star:
                    POS_l_c = self.__get_POS_l(l=l, C=DC_l.union({cond}))
                    if POS_l_c > best_score:
                        best_score = POS_l_c
                        best_cond = cond

                DC_l = DC_l.union({best_cond})

                DC_star = {cond for cond in self.__condition_datas.difference(DC_l)
                           if self.__get_NEG_l(l=l, C=DC_l.union({cond})) <= q_l}

        return DC_l

    def _CorrRuleLearn(self,
                       l: data_preprocessing.Label,
                       CC_all: set[(_Condition, data_preprocessing.Label)]) -> \
            set[tuple[_Condition, data_preprocessing.Label]]:
        """Learns error correction rules for a specific label and granularity. These rules associate conditions 
        with alternative labels that are more likely to be correct when those conditions are met.

        :param l: The label of interest.
        :param CC_all: A set of all condition-label pairs to consider for rule learning.
        :return: A set of condition-label pairs.
        """
        CC_l = set()
        CC_l_prime = CC_all

        CC_sorted = sorted(CC_all, key=lambda cc: self.__get_CON_l(l=cc[1], CC={cc}))

        with context_handlers.WrapTQDM(total=len(CC_sorted)) as progress_bar:
            for cond, l_prime in CC_sorted:
                a = self.__get_CON_l(l=l_prime, CC=CC_l.union({(cond, l_prime)})) - self.__get_CON_l(l=l_prime, CC=CC_l)
                b = (self.__get_CON_l(l=l_prime, CC=CC_l_prime.difference({(cond, l_prime)})) -
                     self.__get_CON_l(l=l_prime, CC=CC_l_prime))

                if a >= b:
                    CC_l = CC_l.union({(cond, l_prime)})
                else:
                    CC_l_prime = CC_l_prime.difference({(cond, l_prime)})

                if utils.is_local():
                    progress_bar.update(1)

        if self.__get_CON_l(l=l, CC=CC_l) <= self.__train_precisions[l.g][l.index]:
            print(f'\n{l}: len(CC_l)={len(CC_l)}, CON_l={self.__get_CON_l(l=l, CC=CC_l)}, '
                  f'P_l={self.__train_precisions[l.g][l.index]}\n')
            CC_l = set()

        return CC_l

    def DetCorrRuleLearn(self,
                         g: data_preprocessing.Granularity):
        """Learns error detection and correction rules for all labels at a given granularity.

        :param g: The granularity level (e.g., 'fine', 'coarse').
        """
        CC_all = set()  # in this use case where the conditions are fine and coarse predictions
        granularity_labels = data_preprocessing.get_labels(g)

        print(f'\nLearning {g}-grain error detection rules...')
        with context_handlers.WrapTQDM(total=len(granularity_labels)) as progress_bar:
            for l in granularity_labels.values():
                DC_l = self.__DetRuleLearn(l=l)
                if len(DC_l):
                    self.error_detection_rules[l] = EDCR.ErrorDetectionRule(l=l, DC_l=DC_l)

                for cond_l in DC_l:
                    CC_all = CC_all.union({(cond_l, l)})

                if utils.is_local():
                    progress_bar.update(1)

        print(f'\nLearning {g}-grain error correction rules...')
        with context_handlers.WrapTQDM(total=len(granularity_labels)) as progress_bar:
            processes_num = min(len(granularity_labels), mp.cpu_count())
            iterable = [(l, CC_all) for l in granularity_labels]

            with mp.Pool(processes_num) as pool:
                CC_ls = pool.starmap(func=self._CorrRuleLearn,
                                     iterable=iterable)

            for CC_l in CC_ls:
                if len(CC_l):
                    self.error_correction_rules[l] = EDCR.ErrorCorrectionRule(l=l, CC_l=CC_l)

        if utils.is_local():
            progress_bar.update(1)

    def apply_detection_rules(self,
                              g: data_preprocessing.Granularity):
        """Applies error detection rules to test predictions for a given granularity. If a rule is satisfied for
        a particular label, the prediction data for that label is modified with a value of -1,
        indicating a potential error.

        :params g: The granularity of the predictions to be processed.
        """
        test_pred_fine_data, test_pred_coarse_data = self.__get_predictions(test=True)

        altered_pred_granularity_datas = {}
        for l, rule_l in self.error_detection_rules.items():
            if l.g == g:
                altered_pred_granularity_datas[l] = rule_l(test_pred_fine_data=test_pred_fine_data,
                                                           test_pred_coarse_data=test_pred_coarse_data)

        altered_pred_granularity_data = self.__get_predictions(test=True, g=g)

        for altered_pred_data_l in altered_pred_granularity_datas.values():
            altered_pred_granularity_data = np.where(altered_pred_data_l == -1, -1, altered_pred_granularity_data)

        self.__post_detection_rules_test_predictions[g] = altered_pred_granularity_data

    def apply_correction_rules(self,
                               g: data_preprocessing.Granularity):
        """Applies error correction rules to test predictions for a given granularity. If a rule is satisfied for a 
        particular label, the prediction data for that label is corrected using the rule's logic. 

        :param g: The granularity of the predictions to be processed.
        """

        test_pred_fine_data, test_pred_coarse_data = self.__get_predictions(test=True)

        altered_pred_granularity_datas = {}
        for l, rule_l in self.error_correction_rules.items():
            if l.g == g:
                altered_pred_granularity_datas[l] = rule_l(test_pred_fine_data=test_pred_fine_data,
                                                           test_pred_coarse_data=test_pred_coarse_data)

        altered_pred_granularity_data = self.__get_predictions(test=True, g=g)

        collision_array = np.zeros_like(altered_pred_granularity_data)

        for (l_1, altered_pred_data_l_1), in altered_pred_granularity_datas.items():
            for (l_2, altered_pred_data_l_2) in altered_pred_granularity_datas.items():
                if l_1 != l_2:
                    where_supposed_to_correct_to_l1 = np.where(altered_pred_data_l_1 == l_1, 1, 0)
                    where_supposed_to_correct_to_l2 = np.where(altered_pred_data_l_2 == l_2, 1, 0)
                    collision_array |= where_supposed_to_correct_to_l1 * where_supposed_to_correct_to_l2

        for l, altered_pred_data_l in altered_pred_granularity_datas.items():
            altered_pred_granularity_data = np.where(collision_array != 1 & altered_pred_data_l == l.index, l.index,
                                                     altered_pred_granularity_data)

        self.__post_correction_rules_test_predictions[g] = altered_pred_granularity_data

    def get_l_correction_rule_support_on_test(self,
                                              l: data_preprocessing.Label) -> float:
        if l not in self.error_correction_rules:
            return 0

        r_l = self.error_correction_rules[l]
        N_l = self.__get_how_many_predicted_l(test=True, l=l)
        how_many_satisfies_any_pair = (
            np.sum(r_l.get_where_any_pair_satisfied(test_pred_fine_data=
                                                    self.__test_pred_data[data_preprocessing.granularities['fine']],
                                                    test_pred_coarse_data=
                                                    self.__test_pred_data[data_preprocessing.granularities['coarse']])))
        s_l = how_many_satisfies_any_pair / N_l
        return s_l


if __name__ == '__main__':
    edcr = EDCR(epsilon=0.1,
                main_model_name='vit_b_16',
                combined=True,
                loss='BCE',
                lr=0.0001,
                num_epochs=20)
    edcr.print_metrics(test=False, prior=True)
    edcr.print_metrics(test=True, prior=True)

    for g in data_preprocessing.granularities.values():
        edcr.DetCorrRuleLearn(g=g)

    print([edcr.get_l_correction_rule_support_on_test(l=l)
           for l in data_preprocessing.fine_grain_labels + data_preprocessing.coarse_grain_labels])

    # for g in data_preprocessing.granularities:
    #     edcr.apply_detection_rules(g=g)
    #     edcr.apply_correction_rules(g=g)

    # edcr.print_metrics(test=True, prior=False)
