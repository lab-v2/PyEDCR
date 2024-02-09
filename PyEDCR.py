from sklearn.metrics import precision_score, recall_score
import numpy as np
import typing
import time

import utils
import data_preprocessing
import vit_pipeline
import context_handlers


class Condition:
    """Represents a condition that can be evaluated on examples.

    When treated as a function, it takes an example (e.g., image, data) as input
    and returns a value between 0 and 1 indicating whether the condition is satisfied.
    A value of 0 means the condition is not met, while a value of 1 means it is fully met.
    """

    def __init__(self,
                 data: np.array):
        """Initializes a Condition instance.

        Args:
            data: A NumPy array representing the condition data.
        """
        self.__data = data

    @property
    def data(self):
        """Gets the condition data.

        Returns:
            numpy.ndarray: The condition data.
        """
        return self.__data


class PredCondition(Condition):
    """Represents a condition based on a model's prediction of a specific class.

    It evaluates to 1 if the model predicts the specified class for a given example,
    and 0 otherwise.
    """

    def __init__(self,
                 pred_data: np.array,
                 l: data_preprocessing.Label):
        """Initializes a PredCondition instance.

        Args:
            pred_data: A NumPy array containing class predictions for examples.
            l: The target Label for which the condition is evaluated.
        """
        super().__init__(data=np.where(pred_data == l.index, 1, 0))
        self.__l = l


class Rule:
    pass


class EDCR:
    """Performs error detection and correction based on model predictions.

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
        __rules: ...

    """
    def __init__(self,
                 main_model_name: str,
                 combined: bool,
                 loss: str,
                 lr: typing.Union[str, float],
                 num_epochs: int,
                 epsilon: typing.Union[str, float]):
        self.__main_model_name = main_model_name
        self.__combined = combined
        self.__loss = loss
        self.__lr = lr
        self.__epsilon = epsilon

        test_pred_fine_path = vit_pipeline.get_filepath(model_name=main_model_name,
                                                        combined=True,
                                                        test=True,
                                                        granularity='fine',
                                                        loss=loss,
                                                        lr=lr,
                                                        pred=True,
                                                        epoch=num_epochs)
        test_pred_coarse_path = vit_pipeline.get_filepath(model_name=main_model_name,
                                                          combined=True,
                                                          test=True,
                                                          granularity='coarse',
                                                          loss=loss,
                                                          lr=lr,
                                                          pred=True,
                                                          epoch=num_epochs)

        train_pred_fine_path = vit_pipeline.get_filepath(model_name=main_model_name,
                                                         combined=True,
                                                         test=False,
                                                         granularity='fine',
                                                         loss=loss,
                                                         lr=lr,
                                                         pred=True,
                                                         epoch=num_epochs)
        train_pred_coarse_path = vit_pipeline.get_filepath(model_name=main_model_name,
                                                           combined=True,
                                                           test=False,
                                                           granularity='coarse',
                                                           loss=loss,
                                                           lr=lr,
                                                           pred=True,
                                                           epoch=num_epochs)

        self.__train_pred_data = {g: np.load(train_pred_fine_path if str(g) == 'fine' else train_pred_coarse_path)
                                  for g in data_preprocessing.granularities}

        self.__test_pred_data = {g: np.load(test_pred_fine_path if str(g) == 'fine' else test_pred_coarse_path)
                                 for g in data_preprocessing.granularities}

        self.__train_condition_datas = {PredCondition(pred_data=self.__train_pred_data[g], l=l)
                                        for g in data_preprocessing.granularities
                                        for l in data_preprocessing.get_labels(g)}

        self.__test_condition_datas = {g: [PredCondition(pred_data=self.__test_pred_data[g], l=l)
                                           for l in data_preprocessing.get_labels(g)]
                                       for g in data_preprocessing.granularities}

        self.__train_precisions = {g: precision_score(y_true=data_preprocessing.get_ground_truths(test=False, g=g),
                                                      y_pred=self.__train_pred_data[g],
                                                      average=None)
                                   for g in data_preprocessing.granularities}

        self.__train_recalls = {g: recall_score(y_true=data_preprocessing.get_ground_truths(test=False, g=g),
                                                y_pred=self.__train_pred_data[g],
                                                average=None)
                                for g in data_preprocessing.granularities}

        self.__rules: dict[str, dict[data_preprocessing.Label, set[Rule]]] = \
            {'error_detections': {l: set() for l in data_preprocessing.all_labels},
             'error_corrections': {l: set() for l in data_preprocessing.all_labels}}

    def __get_predictions(self,
                          test: bool) -> (np.array, np.array):
        """Retrieves prediction data based on specified test/train mode.

        :param test: True for test data, False for training data.
        :return: Fine-grained and coarse-grained prediction data.
        """
        
        pred_fine_data, pred_coarse_data = [self.__test_pred_data[g] for g in data_preprocessing.granularities] \
            if test else [self.__train_pred_data[g] for g in data_preprocessing.granularities]

        return pred_fine_data, pred_coarse_data

    def print_metrics(self,
                      test: bool):
        
        """Prints performance metrics for given test/train data.

        Calculates and prints various metrics (accuracy, precision, recall, etc.)
        using appropriate true labels and prediction data based on the specified mode.

        :param test: True to use test data, False to use training data.
        :return: None
        """
        pred_fine_data, pred_coarse_data = self.__get_predictions(test=test)
        true_fine_data, true_coarse_data = data_preprocessing.get_ground_truths(test=test)

        vit_pipeline.get_and_print_metrics(pred_fine_data=pred_fine_data,
                                           pred_coarse_data=pred_coarse_data,
                                           loss=self.__loss,
                                           true_fine_data=true_fine_data,
                                           true_coarse_data=true_coarse_data,
                                           test=test,
                                           combined=self.__combined,
                                           model_name=self.__main_model_name,
                                           lr=self.__lr)

    @staticmethod
    def __get_corr_l(g: data_preprocessing.Granularity,
                     l: data_preprocessing.Label):
        """Calculates correct label mask for given granularity and label.

        :param g: The granularity level 
        :param l: The label of interest.
        :return: A mask with 1s for correct labels, 0s otherwise.
        """
        return np.where(data_preprocessing.get_ground_truths(test=False, g=g) == l.index, 1, 0)

    def __get_tp_l(self,
                   g: data_preprocessing.Granularity,
                   l: data_preprocessing.Label):
        """Calculates true positive mask for given granularity and label.
            
        :param g: The granularity level.
        :param l: The label of interest.
        :return: A mask with 1s for true positive instances, 0s otherwise.
        """
        return np.where(self.__train_pred_data[g] == data_preprocessing.get_ground_truths(test=False, g=g) &
                        (self.__train_pred_data[g] == l.index), 1, 0)

    def __get_fp_l(self,
                   g: data_preprocessing.Granularity,
                   l: data_preprocessing.Label):
        """Calculates false positive mask for given granularity and label.

        :param g: The granularity level
        :param l: The label of interest.
        :return: A mask with 1s for false positive instances, 0s otherwise.
        """
        return np.where((self.__train_pred_data[g] == l.index) &
                        (data_preprocessing.get_ground_truths(test=False, g=g) != l.index), 1, 0)

    def __are_all_conditions_satisfied(self, C: set[Condition]):
        """Checks if all given conditions are satisfied for each example.
        
        :param C: A set of `Condition` objects.
        :return: A NumPy array with True values if the example satisfy all conditions and False otherwise.
        """
        all_conditions_satisfied = np.ones_like(self.__train_pred_data[data_preprocessing.granularities[0]])

        for cond in C:
            all_conditions_satisfied &= cond.data

        return all_conditions_satisfied

    def __get_NEG_l(self,
                    g: data_preprocessing.Granularity,
                    l: data_preprocessing.Label,
                    C: set[Condition]) -> int:
        """Calculate the number of samples that satisfy the conditions for some set of condition and have true positive.
        
        :param C: A set of `Condition` objects.
        :param g: The granularity level
        :param l: The label of interest.
        :return: The number of instances that is true negative and satisfying all conditions.
        """
        tp_indices = self.__get_tp_l(g=g, l=l)
        all_conditions_satisfied = self.__are_all_conditions_satisfied(C)
        NEG_l = int(np.sum(tp_indices * all_conditions_satisfied))

        return NEG_l

    def __get_POS_l(self,
                    g: data_preprocessing.Granularity,
                    l: data_preprocessing.Label,
                    C: set[Condition]) -> int:
        """Calculate the number of samples that satisfy the conditions for some set of condition and have false positive.
        
        :param C: A set of `Condition` objects.
        :param g: The granularity level
        :param l: The label of interest.
        :return: The number of instances that is false negative and satisfying all conditions.
        """
        fp_indices = self.__get_fp_l(g=g, l=l)
        all_conditions_satisfied = self.__are_all_conditions_satisfied(C)
        POS_l = int(np.sum(fp_indices * all_conditions_satisfied))

        return POS_l

    def __get_CON_l(self,
                    g: data_preprocessing.Granularity,
                    l: data_preprocessing.Label,
                    CC: set[(Condition, data_preprocessing.Label)]) -> float:
        """Calculate the ratio of number of samples that satisfy the rule body and head with the ones that only satisfy the body,
        given a condition class pair.
        
        :param C: A set of `Condition` objects.
        :param g: The granularity level
        :param l: The label of interest.
        :return: ratio as defined above
        """
        corr_l = self.__get_corr_l(g=g, l=l)

        any_pairs_satisfied = 0
        for cc in CC:
            cond, l_prime = cc
            pred_indices = np.where((self.__train_pred_data[g] == l_prime.index), 1, 0)
            cond_data = cond.data
            any_pairs_satisfied |= pred_indices * cond_data

        BOD = np.sum(any_pairs_satisfied)
        POS = np.sum(any_pairs_satisfied * corr_l)

        CON_l = POS / BOD if BOD else 0

        return CON_l

    def __DetRuleLearn(self,
                       g: data_preprocessing.Granularity,
                       l: data_preprocessing.Label) -> set[Condition]:
        """Learns error detection rules for a specific label and granularity. These rules capture conditions
        that, when satisfied, indicate a higher likelihood of prediction errors for a given label.

        :param g: The granularity level.
        :param l: The label of interest.
        :return: A set of `Condition` representing the learned error detection rules.
        """
        DC_l = set()

        N_l = np.sum(np.where(self.__train_pred_data[g] == l.index, 1, 0))
        q_l = self.__epsilon * N_l * self.__train_precisions[g][l.index] / self.__train_recalls[g][l.index]

        DC_star = {cond for cond in self.__train_condition_datas if self.__get_NEG_l(g=g, l=l, C={cond}) <= q_l}

        with context_handlers.WrapTQDM(total=len(DC_star)) as progress_bar:
            while len(DC_star) > 0:
                best_score = -1
                best_cond = None

                for cond in DC_star:
                    POS_l_c = self.__get_POS_l(g=g, l=l, C=DC_l.union({cond}))
                    if POS_l_c > best_score:
                        best_score = POS_l_c
                        best_cond = cond

                DC_l = DC_l.union({best_cond})

                DC_star = {cond for cond in self.__train_condition_datas.difference(DC_l)
                           if self.__get_NEG_l(g=g, l=l, C=DC_l.union({cond})) <= q_l}

                if utils.is_local():
                    time.sleep(0.1)
                    progress_bar.update(1)

        return DC_l

    def __CorrRuleLearn(self,
                        g: data_preprocessing.Granularity,
                        l: data_preprocessing.Label,
                        CC_all: set[(Condition, data_preprocessing.Label)]) -> \
            set[tuple[Condition, data_preprocessing.Label]]:
        """Learns error correction rules for a specific label and granularity. These rules associate conditions with alternative
        labels that are more likely to be correct when those conditions are met.

        :param g: The granularity level (e.g., 'fine', 'coarse').
        :param l: The label of interest.
        :param CC_all: A set of all condition-label pairs to consider for rule learning.
        :return: A set of condition-label pairs.
        """
        CC_l = set()
        CC_l_prime = CC_all

        CC_sorted = sorted(CC_all, key=lambda cc: self.__get_CON_l(g=g, l=cc[1], CC={(cc[0], cc[1])}))

        for (cond, l) in CC_sorted:
            a = self.__get_CON_l(g=g, l=l, CC=CC_l.union({(cond, l)})) - self.__get_CON_l(g=g, l=l, CC=CC_l)
            b = (self.__get_CON_l(g=g, l=l, CC=CC_l_prime.difference({(cond, l)})) -
                 self.__get_CON_l(g=g, l=l, CC=CC_l_prime))

            if a >= b:
                CC_l = CC_l.union({(cond, l)})
            else:
                CC_l_prime = CC_l_prime.difference({(cond, l)})

        # if self.__get_CON_l(g=g, l=l, CC=CC_l) <= self.__train_precisions[g][l]:
        #     CC_l = set()

        return CC_l

    def DetCorrRuleLearn(self,
                         g: data_preprocessing.Granularity):
        """Learns error detection and correction rules for all labels at a given granularity.

        :param g: The granularity level (e.g., 'fine', 'coarse').
        """
        CC_all = set()
        granularity_labels = data_preprocessing.get_labels(g)

        for l in granularity_labels:
            DC_l = self.__DetRuleLearn(g=g,
                                       l=l)
            if len(DC_l):
                self.__rules['error_detections'][l] = self.__rules['error_detections'][l].union(DC_l)

            for cond_l in DC_l:
                CC_all = CC_all.union({(cond_l, l)})

        for l in granularity_labels:
            CC_l = self.__CorrRuleLearn(g=g,
                                        l=l,
                                        CC_all=CC_all)
            if len(CC_l):
                self.__rules['error_corrections'][l] = self.__rules['error_corrections'][l].union(CC_l)


if __name__ == '__main__':
    edcr = EDCR(main_model_name='vit_b_16',
                combined=True,
                loss='BCE',
                lr=0.0001,
                num_epochs=20,
                epsilon=0.1)
    edcr.print_metrics(test=False)
    # edcr.print_metrics(test=True)

    # for g in data_preprocessing.granularities:
    #     edcr.DetCorrRuleLearn(g=g)
