import typing
import abc
import numpy as np

import data_preprocessing
import conditions


class Rule(typing.Callable, typing.Sized, abc.ABC):
    """Represents a rule for evaluating predictions based on conditions and labels.

    :param l: The label associated with the rule.
    :param C_l: The set of conditions that define the rule.
    """

    def __init__(self,
                 l: data_preprocessing.Label,
                 C_l: set[typing.Union[conditions.Condition, tuple[conditions.Condition, data_preprocessing.Label]]]):
        self.l = l
        self.C_l = C_l

    def get_where_predicted_l(self,
                              data: np.array,
                              l_prime: data_preprocessing.Label = None) -> np.array:
        return np.where(data == (self.l.index if l_prime is None else l_prime.index), 1, 0)

    @abc.abstractmethod
    def __call__(self,
                 pred_fine_data: np.array,
                 pred_coarse_data: np.array,
                 secondary_pred_fine_data: np.array,
                 secondary_pred_coarse_data: np.array) -> np.array:
        pass

    @abc.abstractmethod
    def get_where_body_is_satisfied(self,
                                    pred_fine_data: np.array,
                                    pred_coarse_data: np.array,
                                    secondary_pred_fine_data: np.array,
                                    secondary_pred_coarse_data: np.array):
        pass

    @staticmethod
    def get_where_any_conditions_satisfied(C: set[conditions.Condition],
                                           fine_data: np.array,
                                           coarse_data: np.array,
                                           secondary_fine_data: np.array,
                                           secondary_coarse_data: np.array
                                           ) -> np.array:
        """Checks where any given conditions are satisfied.

        :param secondary_fine_data:
        :param secondary_coarse_data:
        :param fine_data: Data that used for Condition having FineGrainLabel l
        :param coarse_data: Data that used for Condition having CoarseGrainLabel l
        :param C: A set of `Condition` objects.
        :return: A NumPy array with True values if the example satisfy any conditions and False otherwise.
        """
        any_condition_satisfied = np.zeros_like(fine_data)

        for cond in C:
            any_condition_satisfied |= cond(fine_data=fine_data,
                                            coarse_data=coarse_data,
                                            secondary_fine_data=secondary_fine_data,
                                            secondary_coarse_data=secondary_coarse_data)

        return any_condition_satisfied

    def __len__(self):
        return len(self.C_l)


class ErrorDetectionRule(Rule):
    def __init__(self,
                 l: data_preprocessing.Label,
                 DC_l: set[conditions.Condition]):
        """Construct a detection rule for evaluating predictions based on conditions and labels.

        :param l: The label associated with the rule.
        :param DC_l: The set of conditions that define the rule.
        """
        super().__init__(l=l, C_l=DC_l)
        assert all(cond.l != self.l for cond in {cond_prime for cond_prime in self.C_l
                                                 if isinstance(cond_prime, conditions.PredCondition)})

    def get_where_body_is_satisfied(self,
                                    pred_fine_data: np.array,
                                    pred_coarse_data: np.array,
                                    secondary_pred_fine_data: np.array,
                                    secondary_pred_coarse_data: np.array
                                    ) -> np.array:
        test_pred_granularity_data = pred_fine_data if self.l.g == data_preprocessing.granularities['fine'] \
            else pred_coarse_data
        where_predicted_l = self.get_where_predicted_l(data=test_pred_granularity_data)
        where_any_conditions_satisfied = (
            self.get_where_any_conditions_satisfied(C=self.C_l,
                                                    fine_data=pred_fine_data,
                                                    coarse_data=pred_coarse_data,
                                                    secondary_fine_data=secondary_pred_fine_data,
                                                    secondary_coarse_data=secondary_pred_coarse_data))
        where_body_is_satisfied = where_predicted_l * where_any_conditions_satisfied

        return where_body_is_satisfied

    def __call__(self,
                 pred_fine_data: np.array,
                 pred_coarse_data: np.array,
                 secondary_pred_fine_data: np.array,
                 secondary_pred_coarse_data: np.array) -> np.array:
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
                                             pred_coarse_data=pred_coarse_data,
                                             secondary_pred_fine_data=secondary_pred_fine_data,
                                             secondary_pred_coarse_data=secondary_pred_coarse_data))
        altered_pred_data = np.where(where_predicted_l_and_any_conditions_satisfied == 1, -1,
                                     test_pred_granularity_data)

        return altered_pred_data

    def __str__(self) -> str:
        return '\n'.join(f'error_{self.l}(x) <- pred_{self.l}(x) ^ {cond}(x)' for cond in self.C_l)


class ErrorCorrectionRule(Rule):
    def __init__(self,
                 l: data_preprocessing.Label,
                 CC_l: set[(conditions.Condition, data_preprocessing.Label)]):
        """Construct a detection rule for evaluating predictions based on conditions and labels.

        :param l: The label associated with the rule.
        :param CC_l: The set of condition-class pair that define the rule.
        """
        C_l = {(cond, l_prime) for cond, l_prime in CC_l if (isinstance(cond, conditions.InconsistencyCondition)
                                                             or cond.l.g != l_prime.g
                                                             or cond.secondary) and l_prime != l}

        super().__init__(l=l, C_l=C_l)

    def get_where_body_is_satisfied(self,
                                    pred_fine_data: np.array,
                                    pred_coarse_data: np.array,
                                    secondary_pred_fine_data: np.array,
                                    secondary_pred_coarse_data: np.array) -> np.array:
        test_pred_granularity_data = pred_fine_data if self.l.g == data_preprocessing.granularities['fine'] \
            else pred_coarse_data

        where_any_pair_satisfied = np.zeros_like(test_pred_granularity_data)

        for cond, l_prime in self.C_l:
            where_condition_satisfied = (
                self.get_where_any_conditions_satisfied(C={cond},
                                                        fine_data=pred_fine_data,
                                                        coarse_data=pred_coarse_data,
                                                        secondary_fine_data=secondary_pred_fine_data,
                                                        secondary_coarse_data=secondary_pred_coarse_data))
            where_predicted_l_prime = self.get_where_predicted_l(data=test_pred_granularity_data,
                                                                 l_prime=l_prime)
            where_pair_satisfied = where_condition_satisfied * where_predicted_l_prime
            where_any_pair_satisfied |= where_pair_satisfied

        return where_any_pair_satisfied

    def __call__(self,
                 pred_fine_data: np.array,
                 pred_coarse_data: np.array,
                 secondary_pred_fine_data: np.array,
                 secondary_pred_coarse_data: np.array) -> np.array:
        """Infer the correction rule based on the provided prediction data.

        :param pred_fine_data: The fine-grained prediction data.
        :param pred_coarse_data: The coarse-grained prediction data.
        :return: new test prediction for a specific granularity as derived from Label l.
        """
        where_any_pair_satisfied = (
            self.get_where_body_is_satisfied(pred_fine_data=pred_fine_data,
                                             pred_coarse_data=pred_coarse_data,
                                             secondary_pred_fine_data=secondary_pred_fine_data,
                                             secondary_pred_coarse_data=secondary_pred_coarse_data))

        altered_pred_data = np.where(where_any_pair_satisfied == 1, self.l.index, -1)

        return altered_pred_data

    def __str__(self) -> str:
        return '\n'.join(f'corr_{self.l}(x) <- {cond}(x) ^ pred_{l_prime}(x)' for (cond, l_prime) in self.C_l)
