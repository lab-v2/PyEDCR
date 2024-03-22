import typing
import abc
import numpy as np

import data_preprocessing


class Condition(typing.Hashable, typing.Callable, abc.ABC):
    """Represents a condition that can be evaluated on an example.

    When treated as a function, it takes an example (e.g., image, data) as input
    and returns a value between 0 and 1 indicating whether the condition is satisfied.
    A value of 0 means the condition is not met, while a value of 1 means it is fully met.
    """

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


class PredCondition(Condition):
    """Represents a condition based on a model's prediction of a specific class.

    It evaluates to 1 if the model predicts the specified class for a given example,
    and 0 otherwise.
    """

    def __init__(self,
                 l: data_preprocessing.Label,
                 secondary_model: bool = False,
                 second_predictions: bool = False):
        """Initializes a PredCondition instance.

        :param l: The target Label for which the condition is evaluated.
        """
        super().__init__()
        self.l = l
        self.secondary_model = secondary_model
        self.second_predictions = second_predictions

    def __call__(self,
                 fine_data: np.array,
                 coarse_data: np.array,
                 secondary_fine_data: np.array,
                 secondary_coarse_data: np.array,
                 second_predictions_fine_data: np.array,
                 second_predictions_coarse_data: np.array,
                 ) -> np.array:
        
        if self.secondary_model:
            if self.l.g == data_preprocessing.granularities['fine']:
                granularity_data = secondary_fine_data
            else:
                granularity_data = secondary_coarse_data
        elif self.second_predictions:
            if self.l.g == data_preprocessing.granularities['fine']:
                granularity_data = second_predictions_fine_data
            else:
                granularity_data = second_predictions_coarse_data
        else:
            if self.l.g == data_preprocessing.granularities['fine']:
                granularity_data = fine_data
            else:
                granularity_data = coarse_data

        return np.where(granularity_data == self.l.index, 1, 0)

    def __str__(self) -> str:
        secondary_str = 'secondary_' if self.secondary_model else ''
        second_prediction_str = '_second_prediction' if self.second_predictions else ''
        return f'{secondary_str}pred_{self.l}{second_prediction_str}'

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class InconsistencyCondition(Condition):
    def __call__(self,
                 fine_data: np.array,
                 coarse_data: np.array) -> np.array:
        values = []
        for fine_prediction_index, coarse_prediction_index in zip(fine_data, coarse_data):
            values += [int(data_preprocessing.fine_to_course_idx[fine_prediction_index]
                           != coarse_prediction_index)]

        return np.array(values)

    def __hash__(self):
        return hash('ConsistencyCondition')

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()
