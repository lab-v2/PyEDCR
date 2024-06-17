import abc
import typing
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
                 secondary_model_name: str = None,
                 lower_prediction_index: int = None,
                 binary: bool = False,
                 negated: bool = False):
        """Initializes a PredCondition instance.

        :param l: The target Label for which the condition is evaluated.
        """
        super().__init__()
        self.l = l
        self.secondary_model_name = secondary_model_name
        self.binary = binary
        self.lower_prediction_index = lower_prediction_index
        self.negated = negated

    def __call__(self,
                 fine_data: np.array,
                 coarse_data: np.array,
                 secondary_fine_data: np.array = None,
                 secondary_coarse_data: np.array = None,
                 lower_predictions_fine_data: dict = None,
                 lower_predictions_coarse_data: dict = None,
                 binary_data: typing.Dict[data_preprocessing.Label, np.array] = None) -> np.array:
        fine = self.l.g.g_str == 'fine'

        if self.secondary_model_name is not None:
            granularity_data = secondary_fine_data if fine else secondary_coarse_data
        elif self.lower_prediction_index is not None:
            granularity_data = lower_predictions_fine_data if fine else lower_predictions_coarse_data
        elif self.binary:
            granularity_data = None if binary_data is None else binary_data[self.l]
        else:
            granularity_data = fine_data if fine else coarse_data

        if granularity_data is None:
            raise ValueError(f'Condition with parameter: l={self.l}, '
                             f'secondary={self.secondary_model_name}, '
                             f'binary={self.binary},'
                             f'lower prediction index={self.lower_prediction_index}'
                             f'do not have associate data when do inference')

        positive_result = 0 if self.negated else 1

        return np.where(granularity_data == self.l.index, positive_result, 1 - positive_result)

    def __str__(self) -> str:
        secondary_str = f'_secondary_{self.secondary_model_name}' if self.secondary_model_name is not None else ''
        lower_prediction_index_str = f'_lower_{self.lower_prediction_index}' \
            if self.lower_prediction_index is not None else ''
        binary_str = '_binary' if self.binary else ''
        negated_str = '_negated' if self.negated else ''

        return f'pred_{self.l.g}_{self.l}{secondary_str}{lower_prediction_index_str}{binary_str}{negated_str}'

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class InconsistencyCondition(Condition):
    def __init__(self,
                 preprocessor: data_preprocessing.FineCoarseDataPreprocessor):
        super().__init__()
        self.preprocessor = preprocessor

    def __call__(self,
                 fine_data: np.array,
                 coarse_data: np.array) -> np.array:
        values = []
        for fine_prediction_index, coarse_prediction_index in zip(fine_data, coarse_data):
            values += [int(self.preprocessor.fine_to_course_idx[fine_prediction_index]
                           != coarse_prediction_index)]

        return np.array(values)

    def __hash__(self):
        return hash('ConsistencyCondition')

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()