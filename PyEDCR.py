from __future__ import annotations

import math
import typing
import numpy as np
import warnings
import multiprocessing as mp
from tqdm.contrib.concurrent import thread_map, process_map

warnings.filterwarnings('ignore')

import utils
import data_preprocessing
import neural_metrics
import symbolic_metrics
import conditions
import rules
import models
import google_sheets_api


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
                 epsilon: typing.Union[str, float] = None,
                 sheet_index: int = None,
                 K_train: typing.List[(int, int)] = None,
                 K_test: typing.List[(int, int)] = None,
                 include_inconsistency_constraint: bool = False,
                 secondary_model_name: str = None,
                 secondary_model_loss: str = None,
                 secondary_num_epochs: int = None,
                 secondary_lr: float = None,
                 lower_predictions_indices: typing.List[int] = [],
                 binary_l_strs: typing.List[str] = [],
                 binary_model_name: str = None,
                 binary_num_epochs: int = None,
                 binary_lr: typing.Union[str, float] = None,
                 num_train_images_per_class: int = None,
                 maximize_ratio: bool = True,
                 train_labels_noise_ratio: float = None,
                 indices_of_fine_labels_to_take_out: typing.List[int] = [],
                 use_google_api: bool = True,
                 negated_conditions: bool = False):
        self.data_str = data_str
        self.preprocessor = data_preprocessing.DataPreprocessor(data_str=data_str)
        self.main_model_name = main_model_name
        self.combined = combined
        self.loss = loss
        self.lr = lr
        self.original_num_epochs = original_num_epochs
        self.epsilon = epsilon
        self.sheet_index = 2 if self.epsilon is not None else (sheet_index + 3 if sheet_index is not None else None)
        self.secondary_model_name = secondary_model_name
        self.secondary_model_loss = secondary_model_loss
        self.secondary_num_epochs = secondary_num_epochs
        self.secondary_lr = secondary_lr if secondary_lr is not None else lr
        self.lower_predictions_indices = lower_predictions_indices
        self.binary_l_strs = binary_l_strs
        self.binary_model_name = binary_model_name
        self.binary_num_epochs = binary_num_epochs
        self.binary_lr = binary_lr
        self.num_train_images_per_class = num_train_images_per_class
        self.maximize_ratio = maximize_ratio
        self.train_labels_noise_ratio = train_labels_noise_ratio
        self.indices_of_fine_labels_to_take_out = indices_of_fine_labels_to_take_out
        self.indices_of_coarse_labels_to_take_out = []
        self.negated_conditions = negated_conditions

        # predictions data
        self.pred_paths: typing.Dict[str, dict] = {
            'test' if test else 'train': {g_str: models.get_filepath(data_str=data_str,
                                                                     model_name=main_model_name,
                                                                     combined=combined,
                                                                     test=test,
                                                                     granularity=g_str,
                                                                     loss=loss,
                                                                     lr=lr,
                                                                     pred=True,
                                                                     epoch=original_num_epochs)
                                          for g_str in data_preprocessing.DataPreprocessor.granularities_str}
            for test in [True, False]}

        if isinstance(K_train, typing.List):
            self.K_train = data_preprocessing.expand_ranges(K_train)
        elif isinstance(K_train, np.ndarray):
            self.K_train = K_train
        else:
            self.K_train = (
                data_preprocessing.expand_ranges([(0, np.load(self.pred_paths['train']['fine']).shape[0] - 1)]))

        if isinstance(K_test, typing.List):
            self.K_test = data_preprocessing.expand_ranges(K_test)
        elif isinstance(K_test, np.ndarray):
            self.K_test = K_test
        else:
            self.K_test = data_preprocessing.expand_ranges([(0, np.load(self.pred_paths['test']['fine']).shape[0] - 1)])

        self.T_train = np.load(self.pred_paths['train']['fine']).shape[0]
        self.T_test = np.load(self.pred_paths['test']['fine']).shape[0]

        print(f'Number of total train examples: {self.T_train}\n'
              f'Number of total test examples: {self.T_test}')

        self.pred_data = \
            {test_or_train: {'original': {g: np.load(self.pred_paths[test_or_train][g.g_str])[
                self.K_test if test_or_train == 'test' else self.K_train]
                                          for g in data_preprocessing.DataPreprocessor.granularities.values()},
                             'noisy': {g: np.load(self.pred_paths[test_or_train][g.g_str])[
                                 self.K_test if test_or_train == 'test' else self.K_train]
                                       for g in data_preprocessing.DataPreprocessor.granularities.values()},
                             'post_detection': {g: np.load(self.pred_paths[test_or_train][g.g_str])[
                                 self.K_test if test_or_train == 'test' else self.K_train]
                                                for g in data_preprocessing.DataPreprocessor.granularities.values()},
                             'post_correction': {
                                 g: np.load(self.pred_paths[test_or_train][g.g_str])[
                                     self.K_test if test_or_train == 'test' else self.K_train]
                                 for g in data_preprocessing.DataPreprocessor.granularities.values()}}
             for test_or_train in ['test', 'train']}

        self.noise_ratio = sum(self.preprocessor.get_num_of_train_fine_examples(fine_l_index=l_index)
                               for l_index in self.indices_of_fine_labels_to_take_out) / self.T_train

        self.set_coarse_labels_to_take_out()
        self.replace_labels_with_noise()

        # conditions data
        self.condition_datas = {}

        # if self.maximize_ratio:

        self.set_pred_conditions()
        self.set_binary_conditions()
        # else:
        #     if len(self.binary_l_strs) > 0:
        #         self.set_binary_conditions()
        #     else:
        #         self.set_pred_conditions()

        self.set_secondary_conditions()
        self.set_lower_prediction_conditions()

        if include_inconsistency_constraint:
            for g in data_preprocessing.DataPreprocessor.granularities.values():
                self.condition_datas[g] = self.condition_datas[g].union(
                    {conditions.InconsistencyCondition(preprocessor=self.preprocessor)})

        self.all_conditions = sorted(set().union(*[self.condition_datas[g]
                                                   for g in data_preprocessing.DataPreprocessor.granularities.values()])
                                     , key=lambda cond: str(cond))
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

        # actual_test_examples_with_errors = set()
        # for g in data_preprocessing.DataPreprocessor.granularities.values():
        #     actual_test_examples_with_errors = actual_test_examples_with_errors.union(set(
        #         np.where(self.get_where_predicted_incorrect(test=True, g=g) == 1)[0]))
        #
        # self.actual_examples_with_errors = np.array(list(actual_test_examples_with_errors))

        self.error_detection_rules: typing.Dict[data_preprocessing.Label, rules.ErrorDetectionRule] = {}
        self.error_correction_rules: typing.Dict[data_preprocessing.Label, rules.ErrorCorrectionRule] = {}
        self.predicted_test_errors = np.zeros_like(self.pred_data['test']['original'][
                                                       self.preprocessor.granularities['fine']])
        self.test_error_ground_truths = np.zeros_like(self.predicted_test_errors)
        self.inconsistency_error_ground_truths = np.zeros_like(self.predicted_test_errors)
        self.correction_model = None

        self.original_test_inconsistencies = (
            self.preprocessor.get_num_inconsistencies(
                fine_labels=self.get_predictions(test=True,
                                                 g=data_preprocessing.DataPreprocessor.granularities['fine']),
                coarse_labels=self.get_predictions(test=True,
                                                   g=data_preprocessing.DataPreprocessor.granularities['coarse'])))

        print(f'Number of fine classes: {self.preprocessor.num_fine_grain_classes}')
        print(f'Number of coarse classes: {self.preprocessor.num_coarse_grain_classes}')

        print(utils.blue_text(
            f"Number of fine conditions: "
            f"{len(self.condition_datas[data_preprocessing.DataPreprocessor.granularities['fine']])}\n"
            f"Number of coarse conditions: "
            f"{len(self.condition_datas[data_preprocessing.DataPreprocessor.granularities['coarse']])}\n"))

        if use_google_api:
            self.sheet_tab_name = google_sheets_api.get_sheet_tab_name(main_model_name=main_model_name,
                                                                       data_str=data_str,
                                                                       secondary_model_name=secondary_model_name,
                                                                       binary=len(binary_l_strs) > 0)
            print(f'\nsheet_tab_name: {self.sheet_tab_name}\n')

        self.recovered_constraints_recall = 0
        self.recovered_constraints_precision = 0

    def set_pred_conditions(self):
        for g in data_preprocessing.DataPreprocessor.granularities.values():
            fine = g.g_str == 'fine'
            self.condition_datas[g] = set()
            for l in self.preprocessor.get_labels(g).values():
                if not ((fine and l.index in self.indices_of_fine_labels_to_take_out)
                        or (not fine and l.index in self.indices_of_coarse_labels_to_take_out)):
                    conditions_to_add = {conditions.PredCondition(l=l)}
                    if self.negated_conditions:
                        conditions_to_add = conditions_to_add.union({conditions.PredCondition(l=l, negated=True)})
                    self.condition_datas[g] = self.condition_datas[g].union(conditions_to_add)

    def set_secondary_conditions(self):
        if self.secondary_model_name is not None:
            self.pred_paths['secondary_model'] = {
                'test' if test else 'train': {g_str: models.get_filepath(data_str=self.data_str,
                                                                         model_name=self.secondary_model_name,
                                                                         combined=self.combined,
                                                                         test=test,
                                                                         granularity=g_str,
                                                                         loss=self.secondary_model_loss,
                                                                         lr=self.secondary_lr,
                                                                         pred=True,
                                                                         epoch=self.secondary_num_epochs)
                                              for g_str in data_preprocessing.DataPreprocessor.granularities_str}
                for test in [True, False]}

            self.pred_data['secondary_model'] = \
                {test_or_train: {g: np.load(self.pred_paths['secondary_model'][test_or_train][g.g_str])
                                 for g in data_preprocessing.DataPreprocessor.granularities.values()}
                 for test_or_train in ['test', 'train']}

            for g in data_preprocessing.DataPreprocessor.granularities.values():
                for l in self.preprocessor.get_labels(g).values():
                    conditions_to_add = {conditions.PredCondition(l=l,
                                                                  secondary_model_name=self.secondary_model_name)}
                    if self.negated_conditions:
                        conditions_to_add = (
                            conditions_to_add.union({
                                conditions.PredCondition(l=l,
                                                         secondary_model_name=self.secondary_model_name,
                                                         negated=True)}))
                    self.condition_datas[g] = self.condition_datas[g].union(conditions_to_add)

    def set_lower_prediction_conditions(self):
        for lower_prediction_index in self.lower_predictions_indices:
            lower_prediction_key = f'lower_{lower_prediction_index}'

            self.pred_paths[lower_prediction_key] = {
                'test' if test else 'train':
                    {g_str: models.get_filepath(data_str=self.data_str,
                                                model_name=self.main_model_name,
                                                combined=self.combined,
                                                test=test,
                                                granularity=g_str,
                                                loss=self.loss,
                                                lr=self.lr,
                                                pred=True,
                                                epoch=self.original_num_epochs,
                                                lower_prediction_index=lower_prediction_index)
                     for g_str in data_preprocessing.DataPreprocessor.granularities_str}
                for test in [True, False]}

            self.pred_data[lower_prediction_key] = \
                {test_or_train: {g: np.load(self.pred_paths[lower_prediction_key][test_or_train][g.g_str])
                                 for g in data_preprocessing.DataPreprocessor.granularities.values()}
                 for test_or_train in ['test', 'train']}

            for g in data_preprocessing.DataPreprocessor.granularities.values():
                self.condition_datas[g] = self.condition_datas[g].union(
                    {conditions.PredCondition(l=l, lower_prediction_index=lower_prediction_index)
                     for l in self.preprocessor.get_labels(g).values()})

    def set_binary_conditions(self):
        self.pred_data['binary'] = {}

        for l_str in self.binary_l_strs:
            l = {**self.preprocessor.fine_grain_labels, **self.preprocessor.coarse_grain_labels}[l_str]
            self.pred_paths[l] = {
                'test' if test else 'train': models.get_filepath(data_str=self.data_str,
                                                                 model_name=self.binary_model_name,
                                                                 l=l,
                                                                 test=test,
                                                                 loss=self.loss,
                                                                 lr=self.binary_lr,
                                                                 pred=True,
                                                                 epoch=self.binary_num_epochs)
                for test in [True, False]}

            self.pred_data['binary'][l] = \
                {test_or_train: np.load(self.pred_paths[l][test_or_train])
                 for test_or_train in ['test', 'train']}

            binary_conditions = {conditions.PredCondition(l=l, binary=True)}

            if self.negated_conditions:
                binary_conditions = binary_conditions.union({conditions.PredCondition(l=l, binary=True, negated=False)})

            for g in self.preprocessor.granularities.values():
                if g in self.condition_datas:
                    self.condition_datas[g] = self.condition_datas[g].union(binary_conditions)
                else:
                    self.condition_datas[g] = {binary_conditions}

    def set_coarse_labels_to_take_out(self):
        g_coarse = self.preprocessor.granularities['coarse']

        for coarse_label in self.preprocessor.get_labels(g_coarse).values():
            fine_labels_of_curr_coarse_label = set(self.preprocessor.coarse_to_fine[coarse_label.l_str])
            if fine_labels_of_curr_coarse_label.issubset(set(self.indices_of_fine_labels_to_take_out)):
                self.indices_of_coarse_labels_to_take_out += [coarse_label.index]

    def replace_labels_with_noise(self):
        for g_str in ['fine', 'coarse']:
            g = self.preprocessor.granularities['fine']

            indices_of_labels_to_take_out = self.indices_of_fine_labels_to_take_out if g_str == 'fine' \
                else self.indices_of_coarse_labels_to_take_out

            if len(indices_of_labels_to_take_out):
                all_labels_indices = set(range(len(self.preprocessor.get_labels(g))))
                indices_of_labels_to_keep = list(all_labels_indices.difference(set(indices_of_labels_to_take_out)))
                max_value = len(indices_of_labels_to_keep)
                classes_str = self.preprocessor.fine_grain_classes_str if g_str == 'fine' \
                    else self.preprocessor.coarse_grain_classes_str
                i = 0

                for label_to_take_out_index in indices_of_labels_to_take_out:
                    next_true_label = indices_of_labels_to_keep[i % max_value]

                    train_true_indices_where_label_to_take_out = (
                        np.where(self.preprocessor.get_ground_truths(test=False, g=g) == label_to_take_out_index))[0]

                    if g_str == 'fine':
                        self.preprocessor.train_true_fine_data[train_true_indices_where_label_to_take_out] \
                            = next_true_label
                    else:
                        self.preprocessor.train_true_coarse_data[train_true_indices_where_label_to_take_out] \
                            = next_true_label

                    # indices_in_train_pred_where_label_to_take_out = (
                    #     np.where(self.get_predictions(test=False, g=g) == label_to_take_out_index))[0]
                    # self.pred_data['train']['noisy'][g_str][indices_in_train_pred_where_label_to_take_out] \
                    #     = next_pred_label
                    # indices_in_test_pred_where_label_to_take_out = (
                    #     np.where(self.get_predictions(test=True, g=g) == label_to_take_out_index))[0]
                    # self.pred_data['test']['noisy'][g_str][indices_in_test_pred_where_label_to_take_out] \
                    #     = next_pred_label

                    label_str = classes_str[label_to_take_out_index]
                    assert self.preprocessor.get_labels(g)[label_str].index == label_to_take_out_index

                    if label_str in self.binary_l_strs:
                        self.binary_l_strs.remove(label_str)

                    i += 1

                assert all(len(np.where(self.preprocessor.get_ground_truths(test=False, g=g) ==
                                        label_to_take_out_index)[0]) == 0
                           and classes_str[label_to_take_out_index] not in self.binary_l_strs
                           for label_to_take_out_index in indices_of_labels_to_take_out)

                print(f'Removed classes {indices_of_labels_to_take_out} from {g_str} grain')

    def set_error_detection_rules(self,
                                  input_rules: typing.Dict[data_preprocessing.Label, {conditions.Condition}]):
        """
        Manually sets the error detection rule dictionary.

        :params rules: A dictionary mapping label instances to error detection rule objects.
        """
        error_detection_rules = {}
        for label, DC_l in input_rules.items():
            error_detection_rules[label] = rules.ErrorDetectionRule(l=label,
                                                                    DC_l=DC_l,
                                                                    preprocessor=self.preprocessor)
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
            pred_data = {l: self.pred_data['binary'][l][test_str] for l in self.pred_data['binary']} \
                if 'binary' in self.pred_data else None
        else:
            pred_data = self.pred_data[test_str][stage]

        pred_data: dict

        if g is not None:
            return pred_data[g]
        elif binary:
            return pred_data

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
                      split: str,
                      prior: bool = True,
                      print_inconsistencies: bool = True,
                      stage: str = 'original',
                      print_actual_errors_num: bool = False):

        """Prints performance metrics for given test/train data.

        Calculates and prints various metrics (accuracy, precision, recall, etc.)
        using appropriate true labels and prediction data based on the specified mode.

        :param split:
        :param print_actual_errors_num:
        :param stage:
        :param print_inconsistencies: whether to print the inconsistencies metric or not
        :param prior:
        """

        original_pred_fine_data, original_pred_coarse_data = None, None
        test = split == 'test'

        # if stage != 'original':
        #     original_pred_fine_data, original_pred_coarse_data = self.get_predictions(test=test, stage='original')

        pred_fine_data, pred_coarse_data = self.get_predictions(test=test, stage=stage)
        true_fine_data, true_coarse_data = self.preprocessor.get_ground_truths(test=test,
                                                                               K=self.K_test if test else self.K_train)

        neural_metrics.get_and_print_metrics(preprocessor=self.preprocessor,
                                             pred_fine_data=pred_fine_data,
                                             pred_coarse_data=pred_coarse_data,
                                             loss=self.loss,
                                             true_fine_data=true_fine_data,
                                             true_coarse_data=true_coarse_data,
                                             split=split,
                                             prior=prior,
                                             combined=self.combined,
                                             model_name=self.main_model_name,
                                             lr=self.lr,
                                             print_inconsistencies=print_inconsistencies,
                                             current_num_test_inconsistencies=
                                             self.get_constraints_true_positives_and_total_positives()[0],
                                             original_test_inconsistencies=self.original_test_inconsistencies,
                                             original_pred_fine_data=original_pred_fine_data,
                                             original_pred_coarse_data=original_pred_coarse_data)

        # Calculate boolean masks for each condition
        # correct_coarse_incorrect_fine = np.logical_and(pred_coarse_data == true_coarse_data,
        #                                                pred_fine_data != true_fine_data)
        # incorrect_coarse_correct_fine = np.logical_and(pred_coarse_data != true_coarse_data,
        #                                                pred_fine_data == true_fine_data)
        # incorrect_both = np.logical_and(pred_coarse_data != true_coarse_data, pred_fine_data != true_fine_data)
        # correct_both = np.logical_and(pred_coarse_data == true_coarse_data,
        #                               pred_fine_data == true_fine_data)  # for completeness
        #
        # # Calculate total number of examples
        # total_examples = len(pred_coarse_data)  # Assuming shapes are compatible

        # fractions = {
        #     'correct_coarse_incorrect_fine': np.sum(correct_coarse_incorrect_fine) / total_examples,
        #     'incorrect_coarse_correct_fine': np.sum(incorrect_coarse_correct_fine) / total_examples,
        #     'incorrect_both': np.sum(incorrect_both) / total_examples,
        #     'correct_both': np.sum(correct_both) / total_examples
        # }
        #
        # print(f"fraction of error associate with each type: \n",
        #       f"correct_coarse_incorrect_fine: {round(fractions['correct_coarse_incorrect_fine'], 2)} \n",
        #       f"incorrect_coarse_correct_fine: {round(fractions['incorrect_coarse_correct_fine'], 2)} \n",
        #       f"incorrect_both: {round(fractions['incorrect_both'], 2)} \n",
        #       f"correct_both: {round(fractions['correct_both'], 2)} \n")

        # if print_actual_errors_num:
        #     print(utils.red_text(f'\nNumber of actual errors on test: {len(self.actual_examples_with_errors)} / '
        #                          f'{self.T_test}\n'))

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
        g_predictions = self.get_predictions(test=test, g=g, stage=stage)
        where_predicted_correct = np.where(g_predictions == ground_truth, 1, 0)

        return where_predicted_correct

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

    def get_where_predicted_inconsistently(self,
                                           test: bool,
                                           stage: str = 'original'
                                           ):
        predicted_derived_coarse = np.array([self.preprocessor.fine_to_course_idx[i]
                                             for i in self.get_predictions(test=test,
                                                                           g=self.preprocessor.granularities['fine'],
                                                                           stage=stage)])
        predicted_coarse = self.get_predictions(test=test,
                                                g=self.preprocessor.granularities['coarse'],
                                                stage=stage)
        return np.where(predicted_derived_coarse != predicted_coarse, 1, 0)

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
                                   test_pred_coarse_data: np.array = None) -> (
            typing.Dict[data_preprocessing.Label, float],
            typing.Dict[data_preprocessing.Label, float]):
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
            self.get_predictions(test=False, secondary=True)) if self.secondary_model_name is not None else \
            (None, None)
        lower_train_pred_fine_data, lower_train_pred_coarse_data = (
            self.get_predictions(test=False, lower_predictions=True)) \
            if len(self.lower_predictions_indices) else (None, None)
        binary_data = self.get_predictions(test=False, binary=True) if len(self.binary_l_strs) else None

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
                    l: data_preprocessing.Label,
                    C: set[conditions.Condition]) -> int:
        """Calculate the number of train samples that satisfy any conditions for some set of conditions and
        are positives

        :param l:
        :param C: A set of `Condition` objects.
        :return: The number of instances that are false negative and satisfying some conditions.
        """
        where_predicted_l = self.get_where_label_is_l(pred=True, test=False, l=l)
        train_pred_fine_data, train_pred_coarse_data = self.get_predictions(test=False)
        secondary_train_pred_fine_data, secondary_train_pred_coarse_data = (
            self.get_predictions(test=False, secondary=True)) if self.secondary_model_name is not None else (None, None)
        lower_train_pred_fine_data, lower_train_pred_coarse_data = (
            self.get_predictions(test=False, lower_predictions=True)) \
            if len(self.lower_predictions_indices) else (None, None)
        binary_data = self.get_predictions(test=False, binary=True) if len(self.binary_l_strs) else None

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
        BOD_l = np.sum(where_any_conditions_satisfied_on_train * where_predicted_l)

        return BOD_l

    def get_POS_l_C(self,
                    l: data_preprocessing.Label,
                    C: set[conditions.Condition],
                    stage: str = 'original') -> int:
        """Calculate the number of train samples that satisfy any conditions for some set of conditions
        and are false positive.

        :param stage:
        :param C: A set of `Condition` objects.
        :param l: The label of interest.
        :return: The number of instances that are false negative and satisfying some conditions.
        """
        where_fp_l = self.get_where_fp_l(test=False, l=l, stage=stage)
        train_pred_fine_data, train_pred_coarse_data = self.get_predictions(test=False, stage=stage)
        secondary_train_pred_fine_data, secondary_train_pred_coarse_data = (
            self.get_predictions(test=False, secondary=True)) if self.secondary_model_name is not None else (None, None)
        lower_train_pred_fine_data, lower_train_pred_coarse_data = (
            self.get_predictions(test=False, lower_predictions=True)) \
            if len(self.lower_predictions_indices) else (None, None)
        binary_data = self.get_predictions(test=False, binary=True) if len(self.binary_l_strs) else None

        where_any_conditions_satisfied_on_train = (
            rules.Rule.get_where_any_conditions_satisfied(C=C,
                                                          fine_data=train_pred_fine_data,
                                                          coarse_data=train_pred_coarse_data,
                                                          secondary_fine_data=secondary_train_pred_fine_data,
                                                          secondary_coarse_data=secondary_train_pred_coarse_data,
                                                          lower_predictions_fine_data=lower_train_pred_fine_data,
                                                          lower_predictions_coarse_data=lower_train_pred_coarse_data,
                                                          binary_data=binary_data))
        POS_l = np.sum(where_fp_l * where_any_conditions_satisfied_on_train)

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
            P_l, R_l = self.get_l_precision_and_recall(test=False, l=l, stage=stage)
            q_l = self.epsilon * N_l * P_l / R_l

            DC_star = [cond for cond in self.all_conditions if self.get_NEG_l_C(l=l,
                                                                                C={cond},
                                                                                stage=stage) <= q_l
                       and not (isinstance(cond, conditions.PredCondition) and cond.l == l
                                and cond.secondary_model_name is None)]

            while DC_star:
                best_cond = sorted(DC_star,
                                   key=lambda cond: self.get_POS_l_C(l=l, C=DC_l.union({cond}), stage=stage))[-1]

                DC_l = DC_l.union({best_cond})
                DC_star = sorted([cond for cond in
                                  set(self.all_conditions).difference(DC_l)
                                  if self.get_NEG_l_C(l=l, C=DC_l.union({cond}), stage=stage) <= q_l
                                  and not (isinstance(cond, conditions.PredCondition) and cond.l == l
                                           and cond.secondary_model_name is None)
                                  ],
                                 key=lambda cond: str(cond))

        return DC_l

    def get_minimization_denominator(self,
                                     l: data_preprocessing.Label,
                                     C: set[conditions.Condition],
                                     stage: str = 'original'):
        POS = 2 * self.get_POS_l_C(l=l, C=C, stage=stage)
        return POS

    @staticmethod
    def get_f_margin(f: typing.Callable,
                     l: data_preprocessing.Label,
                     DC_l_i: set[conditions.Condition],
                     cond: conditions.Condition):
        return f(l=l, C=DC_l_i.union({cond})) - f(l=l, C=DC_l_i)

    def get_minimization_numerator(self,
                                   l: data_preprocessing.Label,
                                   C: set[conditions.Condition]):
        return self.get_BOD_l_C(l=l, C=C) + np.sum(self.get_where_fp_l(l=l, test=False))

    def get_ratio_of_margins(self,
                             l: data_preprocessing.Label,
                             DC_l_i: set[conditions.Condition],
                             cond: conditions.Condition,
                             init_value: float) -> float:
        minimization_numerator_margin = self.get_f_margin(f=self.get_minimization_numerator,
                                                          l=l,
                                                          DC_l_i=DC_l_i,
                                                          cond=cond)
        minimization_denominator_margin = self.get_f_margin(f=self.get_minimization_denominator,
                                                            l=l,
                                                            DC_l_i=DC_l_i,
                                                            cond=cond)

        return (minimization_numerator_margin / minimization_denominator_margin) \
            if (minimization_denominator_margin > 0) else init_value

    def get_minimization_ratio(self,
                               l: data_preprocessing.Label,
                               DC_l_i: set[conditions.Condition],
                               init_value: float) -> float:
        minimization_numerator = self.get_minimization_numerator(l=l, C=DC_l_i)
        minimization_denominator = self.get_minimization_denominator(l=l, C=DC_l_i)

        return (minimization_numerator / minimization_denominator) if (minimization_denominator > 0) else init_value

    def RatioDetRuleLearn(self,
                          l: data_preprocessing.Label) -> set[conditions.Condition]:
        """Learns error detection rules for a specific label and granularity. These rules capture conditions
        that, when satisfied, indicate a higher likelihood of prediction errors for a given label.

        :param l: The label of interest.
        :return: A set of `Condition` representing the learned error detection rules.
        """
        stage = 'original' if self.correction_model is None else 'post_detection'
        N_l = np.sum(self.get_where_label_is_l(pred=True, test=False, l=l, stage=stage))
        init_value = float('inf')
        DC_ls = {0: set()}
        DC_l_scores = {0: init_value}

        if N_l:
            # print(f'Curvature for {l} is: {self.get_numerator_curvature(l=l)}')
            i = 0
            # 1. sorting the conditions by their 1/f1 value from least to greatest
            DC_star = sorted([cond for cond in self.all_conditions
                              if not (isinstance(cond, conditions.PredCondition) and cond.l == l
                                      and cond.secondary_model_name is None)],
                             key=lambda cond: self.get_minimization_ratio(l=l,
                                                                          DC_l_i={cond},
                                                                          init_value=init_value))

            while DC_star:
                DC_l_i = DC_ls[i]

                # print({str(cond): ((self.get_BOD_l_C(l=l, C=DC_l_i.union({cond})) - self.get_BOD_l_C(l=l, C=DC_l_i)) /
                #         (self.get_POS_l_C(l=l, C=DC_l_i.union({cond}), stage=stage) -
                #          self.get_POS_l_C(l=l, C=DC_l_i, stage=stage)))
                #        if ((self.get_POS_l_C(l=l, C=DC_l_i.union({cond}), stage=stage) -
                #             self.get_POS_l_C(l=l, C=DC_l_i, stage=stage)) > 0)
                #        else init_value for cond in DC_star})

                # 2. minimizing the margin of 1/f1 based on the algorithm
                best_cond = sorted(DC_star,
                                   key=lambda cond: self.get_ratio_of_margins(l=l,
                                                                              DC_l_i=DC_l_i,
                                                                              cond=cond,
                                                                              init_value=init_value,
                                                                              ))[0]

                DC_l_i_1 = DC_l_i.union({best_cond})
                DC_ls[i + 1] = DC_l_i_1
                # 3. updating the new set 1/f1 score
                DC_l_scores[i + 1] = self.get_minimization_ratio(l=l,
                                                                 DC_l_i=DC_l_i_1,
                                                                 init_value=init_value)

                # 4. updating the set of conditions based on the algorithm and also again sorting like in step 1
                # from least to greatest
                DC_star = sorted([cond for cond in DC_star
                                  if init_value > self.get_f_margin(f=self.get_minimization_denominator,
                                                                    l=l,
                                                                    DC_l_i=DC_l_i,
                                                                    cond=cond) > 0],
                                 key=lambda cond: self.get_minimization_ratio(l=l,
                                                                              DC_l_i={cond},
                                                                              init_value=init_value))

                i += 1

        # best_set_index = sorted(DC_l_scores.keys(), key=lambda i: DC_l_scores[i])[0]

        # 5. picking the set with the most conditions from all the sets that have the same best score
        best_set_score = sorted(DC_l_scores.values())[0]
        best_score_DC_ls = [DC_ls[i] for i, score in DC_l_scores.items() if score == best_set_score]
        best_set = sorted(best_score_DC_ls, key=lambda DC_l_i: len(DC_l_i))[0]
        # print(f'\nBest set for {l}: {[str(cond) for cond in best_set]}\n')

        return best_set

    def get_curvature_term(self,
                           l: data_preprocessing.Label,
                           cond: conditions.Condition,
                           ):
        return self.get_f_margin(f=self.get_minimization_numerator,
                                 l=l,
                                 DC_l_i=set(self.all_conditions).difference({cond}),
                                 cond=cond) / self.get_minimization_numerator(l=l, C={cond})

    def get_numerator_curvature(self,
                                l: data_preprocessing.Label,
                                ):
        min_value = sorted([self.get_curvature_term(l=l, cond=cond) for cond in self.all_conditions])[0]
        return 1 - min_value

    def learn_detection_rules(self,
                              g: data_preprocessing.Granularity,
                              multi_processing: bool = True):
        # self.CC_all[g] = set()  # in this use case where the conditions are fine and coarse predictions
        granularity_labels = list(self.preprocessor.get_labels(g).values())
        processes_num = min(len(granularity_labels), mp.cpu_count())

        print(f'\nLearning {g}-grain error detection rules...')

        maximizer: typing.Callable = self.RatioDetRuleLearn if self.maximize_ratio else self.DetRuleLearn
        mapper = process_map if multi_processing else thread_map

        DC_ls = mapper(maximizer,
                       granularity_labels,
                       max_workers=processes_num) if (not utils.is_debug_mode()) else \
            [maximizer(l=l) for l in granularity_labels]

        for l, DC_l in zip(granularity_labels, DC_ls):
            if len(DC_l):
                self.error_detection_rules[l] = rules.ErrorDetectionRule(l=l,
                                                                         DC_l=DC_l,
                                                                         preprocessor=self.preprocessor)

            # for cond_l in DC_l:
            #     if not (isinstance(cond_l, conditions.PredCondition) and (not cond_l.secondary_model)
            #             and (cond_l.lower_prediction_index is None) and (cond_l.l == l)):
            #         self.CC_all[g] = self.CC_all[g].union({(cond_l, l)})

    def get_num_all_possible_constraint_can_recover_in_train(self):
        train_fine_prediction, train_coarse_prediction = self.get_predictions(test=False)

        unique_set_of_fine_coarse_predictions = set([(train_fine_prediction[i], train_coarse_prediction[i])
                                                     for i in range(len(train_fine_prediction))])

        consistent_prediction = set([(fine_label_idx, coarse_label_idx)
                                     for fine_label_idx, coarse_label_idx in
                                     self.preprocessor.fine_to_course_idx.items()])

        return len(unique_set_of_fine_coarse_predictions.difference(consistent_prediction))

    def apply_detection_rules(self,
                              test: bool,
                              g: data_preprocessing.Granularity,
                              stage: str):
        """Applies error detection rules to test predictions for a given granularity. If a rule is satisfied for
        a particular label, the prediction data for that label is modified with a value of -1,
        indicating a potential error.

        :params g: The granularity of the predictions to be processed.
        """
        test_str = 'test' if test else 'train'
        pred_fine_data, pred_coarse_data = self.get_predictions(test=test, stage=stage)

        secondary_pred_fine_data, secondary_pred_coarse_data = (
            self.get_predictions(test=test, secondary=True) if self.secondary_model_name is not None else (None, None))
        lower_train_pred_fine_data, lower_train_pred_coarse_data = (
            self.get_predictions(test=test, lower_predictions=True)) \
            if len(self.lower_predictions_indices) else (None, None)
        binary_data = self.get_predictions(test=test, binary=True) if len(self.binary_l_strs) else None

        altered_pred_granularity_data = self.get_predictions(test=test, g=g, stage=stage)

        # self.pred_data['test' if test else 'train']['mid_learning'][g] = altered_pred_granularity_data

        inconsistency_error_ground_truths = self.get_where_predicted_inconsistently(test=test, stage=stage)
        granularity_error_predictions = np.zeros_like(inconsistency_error_ground_truths)

        for rule_g_l in {rule_l for l, rule_l in self.error_detection_rules.items() if l.g == g}:
            altered_pred_l_data = rule_g_l(pred_fine_data=pred_fine_data,
                                           pred_coarse_data=pred_coarse_data,
                                           secondary_pred_fine_data=secondary_pred_fine_data,
                                           secondary_pred_coarse_data=secondary_pred_coarse_data,
                                           lower_predictions_fine_data=lower_train_pred_fine_data,
                                           lower_predictions_coarse_data=lower_train_pred_coarse_data,
                                           binary_data=binary_data)
            altered_pred_granularity_data = np.where(altered_pred_l_data == -1, -1, altered_pred_granularity_data)

            l_error_predictions = np.where(altered_pred_l_data == -1, 1, 0)
            granularity_error_predictions |= l_error_predictions

        if test:
            self.pred_data[test_str]['post_detection'][g] = altered_pred_granularity_data
            self.predicted_test_errors |= granularity_error_predictions

            test_granularity_error_ground_truths = (
                self.get_where_predicted_incorrect(test=test, g=g, stage=stage))
            self.test_error_ground_truths |= test_granularity_error_ground_truths
            self.inconsistency_error_ground_truths |= inconsistency_error_ground_truths

            if g.g_str == 'fine':
                recovered_constraints_true_positives, recovered_constraints_positives = (
                    self.get_constraints_true_positives_and_total_positives())
                # inconsistencies_from_original_test_data = self.original_test_inconsistencies[1]
                all_possible_consistency_constraints = self.get_num_all_possible_constraint_can_recover_in_train()

                # print(f'Total unique recoverable constraints from the {test_str} predictions: '
                #       f'{utils.red_text(inconsistencies_from_original_test_data)}\n'
                #       f'Recovered constraints: {recovered_constraints_str}')

                self.recovered_constraints_recall = min(recovered_constraints_true_positives /
                                                        all_possible_consistency_constraints, 1)

                self.recovered_constraints_precision = min(recovered_constraints_true_positives /
                                                           recovered_constraints_positives, 1) \
                    if recovered_constraints_positives else 0

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
                                    stage: str,
                                    ):
        test_or_train = 'test' if test else 'train'
        print(f'\nNum not assigned in {test_or_train} {stage} {g}-grain predictions: ' +
              utils.red_text(f"{np.sum(np.where(self.get_predictions(test=test, g=g, stage=stage) == -1, 1, 0))}\n"))

    def run_error_detection_application_pipeline(self,
                                                 test: bool,
                                                 print_results: bool = True,
                                                 save_to_google_sheets: bool = True):
        for g in data_preprocessing.DataPreprocessor.granularities.values():
            self.apply_detection_rules(test=test,
                                       g=g,
                                       stage='noisy'
                                       if len(self.indices_of_fine_labels_to_take_out) > 0 else 'original')

            if print_results:
                symbolic_metrics.evaluate_and_print_g_detection_rule_precision_increase(edcr=self,
                                                                                        test=test,
                                                                                        g=g)
                symbolic_metrics.evaluate_and_print_g_detection_rule_recall_decrease(edcr=self,
                                                                                     test=test,
                                                                                     g=g)
                self.print_how_many_not_assigned(test=test,
                                                 g=g,
                                                 stage='post_detection')

        if test and save_to_google_sheets:
            error_accuracy, error_balanced_accuracy, error_f1, error_precision, error_recall, = \
                [f'{round(metric_result * 100, 2)}%' for metric_result in neural_metrics.get_individual_metrics(
                    pred_data=self.predicted_test_errors,
                    true_data=self.test_error_ground_truths,
                    labels=[1],
                    binary=True)]

            # inconsistency_error_accuracy, inconsistency_error_f1, _, _, _ = \
            #     [f'{round(metric_result * 100, 2)}%' for metric_result in neural_metrics.get_individual_metrics(
            #         pred_data=self.predicted_test_errors,
            #         true_data=self.inconsistency_error_ground_truths,
            #         labels=[0])]

            # set values
            input_values = [round(self.epsilon, 3) if self.epsilon is not None else '',
                            self.noise_ratio,
                            error_accuracy,
                            error_balanced_accuracy,
                            error_precision,
                            error_recall,
                            error_f1,
                            self.recovered_constraints_precision,
                            self.recovered_constraints_recall,
                            2 / ((1 / self.recovered_constraints_precision) + (1 / self.recovered_constraints_recall))
                            if self.recovered_constraints_precision and self.recovered_constraints_recall else 0
                            ]

            print(input_values)

            google_sheets_api.update_sheet(range_=f'{self.sheet_tab_name}!A{self.sheet_index}:'
                                                  f'{chr(len(input_values) + 64)}{self.sheet_index}',
                                           body={'values': [input_values]})

        if print_results:
            self.print_metrics(split='test' if test else 'train',
                               prior=False,
                               stage='post_detection',
                               print_inconsistencies=False)

    def get_constraints_true_positives_and_total_positives(self):
        true_recovered_constraints: dict[str, set[str]] = {}
        recovered_constraints: dict[str, set[str]] = {}

        for l, error_detection_rule in self.error_detection_rules.items():
            error_detection_rule: rules.ErrorDetectionRule

            for cond in error_detection_rule.C_l:
                if ((isinstance(cond, conditions.PredCondition)) and (cond.secondary_model_name is None)
                    and (not cond.binary)) and (cond.lower_prediction_index is None):
                    if cond.l.g != l.g:
                        if cond.l.g.g_str == 'fine':
                            fine_index = cond.l.index
                            coarse_index = l.index
                        else:
                            fine_index = l.index
                            coarse_index = cond.l.index

                        fine_label_str = self.preprocessor.fine_grain_classes_str[fine_index]
                        coarse_label_str = self.preprocessor.coarse_grain_classes_str[coarse_index]

                        if self.preprocessor.fine_to_course_idx[fine_index] != coarse_index:
                            if fine_label_str not in true_recovered_constraints:
                                true_recovered_constraints[fine_label_str] = {coarse_label_str}
                            else:
                                true_recovered_constraints[fine_label_str] = (
                                    true_recovered_constraints[fine_label_str].union({coarse_label_str}))

                        if fine_label_str not in recovered_constraints:
                            recovered_constraints[fine_label_str] = {coarse_label_str}
                        else:
                            recovered_constraints[fine_label_str] = (
                                recovered_constraints[fine_label_str].union({coarse_label_str}))

        assert all(self.preprocessor.fine_to_coarse[fine_label_str] not in coarse_dict
                   for fine_label_str, coarse_dict in true_recovered_constraints.items())

        num_true_recovered_constraints = sum(len(coarse_dict) for coarse_dict in true_recovered_constraints.values())
        num_recovered_constraints = sum(len(coarse_dict) for coarse_dict in recovered_constraints.values())

        return num_true_recovered_constraints, num_recovered_constraints


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

        # precision_typing.Dict[gra]['initial'][epsilon] = edcr.original_test_precisions[gra]
        # recall_typing.Dict[gra]['initial'][epsilon] = edcr.original_test_recalls[gra]
        # precision_typing.Dict[gra]['pre_correction'][epsilon] = edcr.post_detection_test_precisions[gra]
        # recall_typing.Dict[gra]['pre_correction'][epsilon] = edcr.post_detection_test_recalls[gra]
        # precision_typing.Dict[gra]['post_correction'][epsilon] = edcr.post_correction_test_precisions[gra]
        # recall_typing.Dict[gra]['post_correction'][epsilon] = edcr.post_correction_test_recalls[gra]

    # folder = "experiment_1"
    #
    # if not os.path.exists(f'figs/{folder}'):
    #     os.mkdir(f'figs/{folder}')
    #
    # plot_per_class(ps=precision_dict,
    #                rs=recall_dict,
    #                folder="experiment_1")
    # plot_all(precision_dict, recall_dict, "experiment_1")
