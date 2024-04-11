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
                 lower_prediction_index: int = None,
                 binary: bool = False):
        """Initializes a PredCondition instance.

        :param l: The target Label for which the condition is evaluated.
        """
        super().__init__()
        self.l = l
        self.secondary_model = secondary_model
        self.binary = binary
        self.negatePredCondition = NegatePredCondition(l=l, secondary=secondary_model)
        self.lower_prediction_index = lower_prediction_index

    def __call__(self,
                 fine_data: np.array,
                 coarse_data: np.array,
                 secondary_fine_data: np.array,
                 secondary_coarse_data: np.array,
                 lower_predictions_fine_data: dict,
                 lower_predictions_coarse_data: dict,
                 binary_data: typing.Dict[data_preprocessing.Label, np.array]) -> np.array:

        if self.secondary_model:
            if self.l.g == data_preprocessing.DataPreprocessor.granularities['fine']:
                granularity_data = secondary_fine_data
            else:
                granularity_data = secondary_coarse_data
        elif self.lower_prediction_index is not None:
            if self.l.g == data_preprocessing.DataPreprocessor.granularities['fine']:
                granularity_data = lower_predictions_fine_data
            else:
                granularity_data = lower_predictions_coarse_data
        elif self.binary:
            granularity_data = binary_data[self.l]
        else:
            if self.l.g == data_preprocessing.DataPreprocessor.granularities['fine']:
                granularity_data = fine_data
            else:
                granularity_data = coarse_data

        return np.where(granularity_data == self.l.index, 1, 0)

    def __str__(self) -> str:
        if self.binary:
            return f'binary_pred_{self.l}'
        secondary_str = 'secondary_' if self.secondary_model else ''
        lower_prediction_index_str = f'_lower{self.lower_prediction_index}' \
            if self.lower_prediction_index is not None else ''
        return f'{secondary_str}pred_{self.l}{lower_prediction_index_str}'

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class InconsistencyCondition(Condition):
    def __init__(self,
                 preprocessor: data_preprocessing.DataPreprocessor):
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


class NegatePredCondition(Condition):
    """Represents a condition based on a model's prediction of a specific class.

    It evaluates to 0 if the model predicts the specified class for a given example,
    and 1 otherwise.
    """

    def __init__(self,
                 l: data_preprocessing.Label,
                 secondary: bool = False):
        """Initializes a PredCondition instance.

        :param l: The target Label for which the condition is evaluated.
        """
        super().__init__()
        self.l = l
        self.secondary = secondary

    def __call__(self,
                 fine_data: np.array,
                 coarse_data: np.array,
                 secondary_fine_data: np.array,
                 secondary_coarse_data: np.array,
                 ) -> np.array:
        granularity_data = (fine_data if not self.secondary else secondary_fine_data) \
            if self.l.g == data_preprocessing.DataPreprocessor.granularities['fine'] else \
            (coarse_data if not self.secondary else secondary_coarse_data)
        return np.where(granularity_data != self.l.index, 1, 0)

    def __str__(self) -> str:
        secondary_str = 'secondary_' if self.secondary else ''
        return f'not_{secondary_str}pred_{self.l}'

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()
