import typing
import numpy as np

import models
import rules
import conditions
import utils
import data_preprocessing


class COX_EDCR:
    def set_pred_conditions(self):
        self.condition_datas = {conditions.PredCondition(l=l) for l in self.preprocessor.get_labels()}

    def __init__(self,
                 secondary_model_name: str = None,
                 secondary_model_loss: str = None,
                 secondary_num_epochs: int = None,
                 secondary_lr: float = None):
        self.data_str = 'COX'
        self.main_model_name = 'model_name'
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

        # conditions data
        self.condition_datas = {}

        self.set_pred_conditions()
        self.set_secondary_conditions()

        self.all_conditions = sorted(set().union(*[self.condition_datas[g]
                                                   for g in
                                                   data_preprocessing.FineCoarseDataPreprocessor.granularities.values()])
                                     , key=lambda cond: str(cond))
        self.CC_all = {g: set() for g in data_preprocessing.FineCoarseDataPreprocessor.granularities.values()}

        self.num_predicted_l = {'original': {g: {}
                                             for g in
                                             data_preprocessing.FineCoarseDataPreprocessor.granularities.values()},
                                'post_detection': {g: {}
                                                   for g in
                                                   data_preprocessing.FineCoarseDataPreprocessor.granularities.values()},
                                'post_correction':
                                    {g: {} for g in
                                     data_preprocessing.FineCoarseDataPreprocessor.granularities.values()}}

        for g in data_preprocessing.FineCoarseDataPreprocessor.granularities.values():
            for l in self.preprocessor.get_labels(g).values():
                self.num_predicted_l['original'][g][l] = np.sum(self.get_where_label_is_l(pred=True,
                                                                                          test=True,
                                                                                          l=l,
                                                                                          stage='original'))

        self.error_detection_rules: typing.Dict[data_preprocessing.Label, rules.ErrorDetectionRule] = {}
        self.error_correction_rules: typing.Dict[data_preprocessing.Label, rules.ErrorCorrectionRule] = {}
        self.predicted_test_errors = np.zeros_like(self.pred_data['test']['original'][
                                                       self.preprocessor.granularities['fine']])
        self.test_error_ground_truths = np.zeros_like(self.predicted_test_errors)
        self.inconsistency_error_ground_truths = np.zeros_like(self.predicted_test_errors)
        self.correction_model = None

        print(utils.blue_text(
            f"Number of fine conditions: "
            f"{len(self.condition_datas[data_preprocessing.FineCoarseDataPreprocessor.granularities['fine']])}\n"
            f"Number of coarse conditions: "
            f"{len(self.condition_datas[data_preprocessing.FineCoarseDataPreprocessor.granularities['coarse']])}\n"))
