from PyEDCR import EDCR
import data_preprocessing
import utils
import conditions
import rules
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from itertools import product
import typing
import metrics


class EDCR_experiment(EDCR):
    def __init__(self,
                 main_model_name: str,
                 combined: bool,
                 loss: str,
                 lr: typing.Union[str, float],
                 num_epochs: int,
                 epsilon: typing.Union[str, float],
                 K_train: list[(int, int)] = None,
                 K_test: list[(int, int)] = None,
                 include_inconsistency_constraint: bool = False,
                 secondary_model_name: str = None):
        super().__init__(main_model_name=main_model_name,
                         combined=combined,
                         loss=loss,
                         lr=lr,
                         num_epochs=num_epochs,
                         epsilon=epsilon,
                         K_train=K_train,
                         K_test=K_test,
                         include_inconsistency_constraint=include_inconsistency_constraint,
                         secondary_model_name=secondary_model_name)

        self.error_correction_rules: dict[data_preprocessing.Label, rules.ErrorCorrectionRule2] = {}

    def get_accuracy(self,
                     prediction: np.array,
                     g: data_preprocessing.Granularity):
        return accuracy_score(y_true=data_preprocessing.get_ground_truths(test=False,
                                                                          K=self.K_train,
                                                                          g=g),
                              y_pred=prediction)

    def new_objective_function(self,
                               l: data_preprocessing.Label,
                               CC: set[(set[conditions.Condition], data_preprocessing.Label)],
                               g: data_preprocessing.Granularity):
        Rule_CC_l = rules.ErrorCorrectionRule2(l=l, CC_l=CC)
        prediction_after_apply_rule_mask = Rule_CC_l(
            pred_fine_data=self.pred_data['train']['original']['fine'],
            pred_coarse_data=self.pred_data['train']['original']['coarse'],
            secondary_pred_fine_data=self.pred_data['secondary_model']['train']['fine'],
            secondary_pred_coarse_data=self.pred_data['secondary_model']['train']['coarse']
        )
        prediction_after_apply_rule = np.where(prediction_after_apply_rule_mask == -1,
                                               self.pred_data['train']['original'][g],
                                               prediction_after_apply_rule_mask)
        return self.get_accuracy(prediction=prediction_after_apply_rule, g=g)

    def learn_correction_rules(self,
                               g: data_preprocessing.Granularity):

        granularity_labels = data_preprocessing.get_labels(g).values()
        other_g = data_preprocessing.granularities['fine' if str(g) == 'coarse' else 'coarse']

        print(f'\nLearning {g}-grain error correction rules...')

        CC_ls = {l: set() for l in data_preprocessing.get_labels(g).values()}
        condition_pair = product([cond for cond in self.condition_datas[other_g] if not cond.secondary],
                                 [cond for cond in self.condition_datas[other_g] if cond.secondary])
        CC_all = product(condition_pair,
                         data_preprocessing.get_labels(g).values())
        for CC in tqdm(CC_all):
            max_score = 0
            assign_l = None
            for l in data_preprocessing.get_labels(g).values():
                score = self.new_objective_function(l=l, CC={CC}, g=g)
                if score > max_score:
                    assign_l = l
                    max_score = score

            if assign_l is not None:
                CC_ls[assign_l] = CC_ls[assign_l].union({CC})

        for l, CC_l in CC_ls.items():
            if len(CC_l):
                self.error_correction_rules[l] = rules.ErrorCorrectionRule2(l=l, CC_l=CC_l)
            else:
                print(utils.red_text('\n' + '#' * 10 + f' {l} does not have an error correction rule!\n'))

    def run_learning_pipeline(self,
                              EDCR_epoch_num=0):
        print('Started learning pipeline...\n')
        self.print_metrics(test=False, prior=True)

        for g in data_preprocessing.granularities.values():
            self.learn_correction_rules(g=g)

        print('\nRule learning completed\n')

    def apply_correction_rules(self,
                               test: bool,
                               g: data_preprocessing.Granularity):
        """Applies error correction rules to test predictions for a given granularity. If a rule is satisfied for a
        particular label, the prediction data for that label is corrected using the rule's logic.

        :param test:
        :param g: The granularity of the predictions to be processed.
        """

        test_or_train = 'test' if test else 'train'
        g_l_rules = {l: rule_l for l, rule_l in self.error_correction_rules.items() if l.g == g}
        secondary_fine_data, secondary_coarse_data = self.get_predictions(test=test, secondary=True)

        for l, rule_g_l in g_l_rules.items():
            fine_data, coarse_data = self.get_predictions(test=test, stage='post_correction')
            altered_pred_data_l = rule_g_l(pred_fine_data=fine_data,
                                           pred_coarse_data=coarse_data,
                                           secondary_pred_fine_data=secondary_fine_data,
                                           secondary_pred_coarse_data=secondary_coarse_data, )

            self.pred_data[test_or_train]['post_correction'][g] = np.where(
                # (collision_array != 1) &
                (altered_pred_data_l == l.index),
                l.index,
                self.get_predictions(test=test, g=g, stage='post_correction'))

            self.print_metrics(test=test, prior=False, stage='post_correction', print_inconsistencies=True)

        for l in data_preprocessing.get_labels(g).values():
            self.num_predicted_l['post_correction'][g][l] = np.sum(self.get_where_label_is_l(pred=True,
                                                                                             test=True,
                                                                                             l=l,
                                                                                             stage='post_correction'))


if __name__ == '__main__':
    epsilons = [0.1 * i for i in range(2, 3)]
    test_bool = False

    for eps in epsilons:
        print('#' * 25 + f'eps = {eps}' + '#' * 50)
        edcr = EDCR_experiment(epsilon=eps,
                               main_model_name='vit_b_16',
                               combined=True,
                               loss='BCE',
                               lr=0.0001,
                               num_epochs=20,
                               include_inconsistency_constraint=False,
                               secondary_model_name='vit_b_16_soft_marginal')
        edcr.print_metrics(test=test_bool, prior=True)

        edcr.run_learning_pipeline()
        # edcr.run_error_detection_application_pipeline(test=test_bool)
        for gra in data_preprocessing.granularities.values():
            edcr.apply_correction_rules(test=test_bool, g=gra)
