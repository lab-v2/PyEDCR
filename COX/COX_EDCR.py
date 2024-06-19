import typing
import numpy as np
from sklearn.metrics import accuracy_score
import multiprocessing as mp
from tqdm.contrib.concurrent import thread_map, process_map

import label
import rule
import condition
import utils
import data_preprocessing
import NeuralPyEDCR


class COX_EDCR(NeuralPyEDCR.NeuralPyEDCR):
    def __init__(self,
                 data_str: str,
                 secondary_model_name: str = None,
                 secondary_model_loss: str = None,
                 secondary_num_epochs: int = None,
                 secondary_lr: float = None):
        self.data_str = 'COX'
        self.main_model_name = 'main_model'
        self.preprocessor = data_preprocessing.OneLevelDataPreprocessor(data_str=self.data_str)
        self.secondary_model_name = secondary_model_name
        self.secondary_model_loss = secondary_model_loss
        self.secondary_num_epochs = secondary_num_epochs
        self.secondary_lr = secondary_lr

        # predictions data
        self.pred_paths = {
            'test' if test else 'train': data_preprocessing.get_filepath(data_str=self.data_str,
                                                                         model_name=self.main_model_name,
                                                                         test=test,
                                                                         pred=True) for test in [True, False]}

        print(f'Number of total train examples: {self.preprocessor.get_ground_truths(test=False).shape[0]}\n'
              f'Number of total test examples: {self.preprocessor.get_ground_truths(test=True).shape[0]}')

        self.pred_data = \
            {test_or_train: {'original': np.load(self.pred_paths[test_or_train]),
                             'post_detection': np.load(self.pred_paths[test_or_train]),
                             'post_correction': np.load(self.pred_paths[test_or_train])}
             for test_or_train in ['test', 'train']}

        self.pred_paths['secondary_model'] = {
            'test' if test else 'train': data_preprocessing.get_filepath(data_str=self.data_str,
                                                                         model_name=self.secondary_model_name,
                                                                         test=test,
                                                                         loss=self.secondary_model_loss,
                                                                         lr=self.secondary_lr,
                                                                         pred=True,
                                                                         epoch=self.secondary_num_epochs)
            for test in [True, False]}

        self.pred_data['secondary_model'] = \
            {test_or_train: np.load(self.pred_paths['secondary_model'][test_or_train])
             for test_or_train in ['test', 'train']}
        # conditions data
        self.condition_datas = {condition.PredCondition(l=l) for l in self.preprocessor.get_labels().values()}.union({
            condition.PredCondition(l=l, secondary_model_name=self.secondary_model_name)
            for l in self.preprocessor.get_labels().values()})

        self.all_conditions = sorted(self.condition_datas, key=lambda cond: str(cond))

        self.error_detection_rules: typing.Dict[label.Label, rule.ErrorDetectionRule] = {}
        self.error_correction_rules: typing.Dict[label.Label, rule.ErrorCorrectionRule] = {}
        self.correction_model = None

        print(utils.blue_text(f"Number of fine conditions: {len(self.condition_datas)}\n"))

    @typing.no_type_check
    def get_predictions(self,
                        test: bool,
                        stage: str = 'original',
                        secondary: bool = False) -> typing.Union[np.array, tuple[np.array]]:
        test_str = 'test' if test else 'train'

        if secondary:
            pred_data = self.pred_data['secondary_model'][test_str]
        else:
            pred_data = self.pred_data[test_str][stage]

        return pred_data

    @typing.no_type_check
    def get_where_label_is_l(self,
                             pred: bool,
                             test: bool,
                             l: label.Label,
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
        data = self.get_predictions(test=test, stage=stage, secondary=secondary) if pred else (
            self.preprocessor.get_ground_truths(test=test))
        return np.where(data == l.index, 1, 0)

    @typing.no_type_check
    def get_BOD_l_C(self,
                    l: label.Label,
                    C: set[condition.Condition]) -> int:
        """Calculate the number of train samples that satisfy any conditions for some set of conditions and
        are positives

        :param l:
        :param C: A set of `Condition` objects.
        :return: The number of instances that are false negative and satisfying some condition.
        """
        where_predicted_l = self.get_where_label_is_l(pred=True, test=False, l=l)
        train_pred_data = self.get_predictions(test=False)
        secondary_train_pred_data = self.get_predictions(test=False, secondary=True) \
            if self.secondary_model_name is not None else None

        where_any_conditions_satisfied_on_train = (
            rule.Rule.get_where_any_conditions_satisfied(C=C,
                                                         data=train_pred_data,
                                                         secondary_data=secondary_train_pred_data,
                                                         ))
        BOD_l = np.sum(where_any_conditions_satisfied_on_train * where_predicted_l)

        return BOD_l

    @typing.no_type_check
    def print_metrics(self,
                      test: bool,
                      stage: str = 'original'):
        print(accuracy_score(y_true=np.array([self.preprocessor.gt_labels[Y_gt_i]
                                              for Y_gt_i in self.preprocessor.get_ground_truths(test=test)]),
                             y_pred=np.array([self.preprocessor.pred_labels[pred_i]
                                              for pred_i in self.pred_data['test' if test else 'train'][stage]])))

    @typing.no_type_check
    def learn_detection_rules(self,
                              multi_processing: bool = True):
        processes_num = mp.cpu_count()

        print(f'\nLearning error detection rules...')

        mapper = process_map if multi_processing else thread_map

        DC_ls = mapper(self.RatioDetRuleLearn,
                       self.preprocessor.get_labels().values(),
                       max_workers=processes_num) if (not utils.is_debug_mode()) else \
            [self.RatioDetRuleLearn(l=l) for l in self.preprocessor.get_labels().values()]

        for l, DC_l in zip(self.preprocessor.get_labels(), DC_ls):
            if len(DC_l):
                self.error_detection_rules[l] = rule.ErrorDetectionRule(l=l,
                                                                        DC_l=DC_l,
                                                                        preprocessor=self.preprocessor)

            # for cond_l in DC_l:
            #     if not (isinstance(cond_l, condition.PredCondition) and (not cond_l.secondary_model)
            #             and (cond_l.lower_prediction_index is None) and (cond_l.l == l)):
            #         self.CC_all[g] = self.CC_all[g].union({(cond_l, l)})

    def run_learning_pipeline(self,
                              multi_processing: bool = True):
        print('Started learning pipeline...\n')
        # self.print_metrics(test=False, prior=True)

        self.learn_detection_rules(multi_processing=multi_processing)

        # self.learn_correction_rules(g=g)
        # self.learn_correction_rules_alt(g=g)

        print('\nRule learning completed\n')


if __name__ == '__main__':
    data_str = 'COX'
    secondary_model_name = 'MLP'
    cox_edcr = COX_EDCR(data_str=data_str,
                        secondary_model_name=secondary_model_name,
                        secondary_model_loss='CE',
                        secondary_num_epochs=500,
                        secondary_lr=0.5)
    cox_edcr.print_metrics(test=True)
    cox_edcr.run_learning_pipeline(multi_processing=False)

    # cox_edcr.run_error_detection_application_pipeline(test=True,
    #                                                   print_results=False,
    #                                                   save_to_google_sheets=True)
