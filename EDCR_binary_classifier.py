import numpy as np

from PyEDCR import EDCR
import data_preprocessing
import vit_pipeline
import typing
import conditions
import rules
import metrics


class EDCR_binary_classifier(EDCR):
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


        # for test_or_train in ['test', 'train']:
        #     self.pred_data[test_or_train]['binary'] = {}
        #     for g in data_preprocessing.granularities.values():
        #         for l in data_preprocessing.get_labels(g).values():
        #             self.pred_data[test_or_train]['binary'][l] = None

        for test_or_train in ['test', 'train']:
            self.pred_data[test_or_train]['binary'] = {}
            for g in data_preprocessing.granularities.values():
                for l in data_preprocessing.get_labels(g).values():
                    pred_path = vit_pipeline.get_filepath(model_name=main_model_name,
                                                          combined=combined,
                                                          test=test_or_train == 'test',
                                                          granularity=g.g_str,
                                                          loss=loss,
                                                          lr=lr,
                                                          pred=True,
                                                          epoch=num_epochs,
                                                          binary_classifier=True,
                                                          l=l)
                    # self.pred_data[test_or_train]['binary'][l] = np.load(pred_path)
                    self.pred_data[test_or_train]['binary'][l] = (
                        np.zeros_like(self.pred_data[test_or_train]['original'][g]))
                    self.condition_datas[g] = self.condition_datas[g].union(
                        {conditions.PredCondition(l=l, binary=True)})

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
            self.get_predictions(test=False, secondary=True)) if self.secondary_model_name is not None else (None, None)

        where_any_conditions_satisfied_on_train = (
            rules.Rule.get_where_any_conditions_satisfied(C=C,
                                                          fine_data=train_pred_fine_data,
                                                          coarse_data=train_pred_coarse_data,
                                                          secondary_fine_data=secondary_train_pred_fine_data,
                                                          secondary_coarse_data=secondary_train_pred_coarse_data,
                                                          binary_data=self.pred_data['train']['binary']))
        where_train_tp_l = self.get_where_tp_l(test=False, l=l, stage=stage)
        NEG_l = np.sum(where_train_tp_l * where_any_conditions_satisfied_on_train)

        return NEG_l

    def get_BOD_l_C(self,
                    C: set[conditions.Condition]) -> int:
        """Calculate the number of train samples that satisfy any conditions for some set of condition.

        :param C: A set of `Condition` objects.
        :return: The number of instances that are false negative and satisfying some conditions.
        """
        train_pred_fine_data, train_pred_coarse_data = self.get_predictions(test=False)
        secondary_train_pred_fine_data, secondary_train_pred_coarse_data = (
            self.get_predictions(test=False, secondary=True))
        where_any_conditions_satisfied_on_train = (
            rules.Rule.get_where_any_conditions_satisfied(C=C,
                                                          fine_data=train_pred_fine_data,
                                                          coarse_data=train_pred_coarse_data,
                                                          secondary_fine_data=secondary_train_pred_fine_data,
                                                          secondary_coarse_data=secondary_train_pred_coarse_data,
                                                          binary_data=self.pred_data['train']['binary']))
        BOD_l = np.sum(where_any_conditions_satisfied_on_train)

        return BOD_l

    def get_POS_l_C(self,
                    l: data_preprocessing.Label,
                    C: set[conditions.Condition],
                    stage: str = 'original') -> int:
        """Calculate the number of train samples that satisfy any conditions for some set of condition
        and are false positive.

        :param stage:
        :param C: A set of `Condition` objects.
        :param l: The label of interest.
        :return: The number of instances that are false negative and satisfying some conditions.
        """
        where_was_wrong_with_respect_to_l = self.get_where_fp_l(test=False, l=l, stage=stage)
        train_pred_fine_data, train_pred_coarse_data = self.get_predictions(test=False, stage=stage)
        secondary_train_pred_fine_data, secondary_train_pred_coarse_data = (
            self.get_predictions(test=False, secondary=True)) if self.secondary_model_name is not None else (None, None)

        where_any_conditions_satisfied_on_train = (
            rules.Rule.get_where_any_conditions_satisfied(C=C,
                                                          fine_data=train_pred_fine_data,
                                                          coarse_data=train_pred_coarse_data,
                                                          secondary_fine_data=secondary_train_pred_fine_data,
                                                          secondary_coarse_data=secondary_train_pred_coarse_data,
                                                          binary_data=self.pred_data['train']['binary']))
        POS_l = np.sum(where_was_wrong_with_respect_to_l * where_any_conditions_satisfied_on_train)

        return POS_l

    def apply_detection_rules(self,
                              test: bool,
                              g: data_preprocessing.Granularity):
        """Applies error detection rules to test predictions for a given granularity. If a rule is satisfied for
        a particular label, the prediction data for that label is modified with a value of -1,
        indicating a potential error.

        :params g: The granularity of the predictions to be processed.
        """
        stage = 'original' if self.correction_model is None else 'post_detection'
        pred_fine_data, pred_coarse_data = self.get_predictions(test=test, stage=stage)
        secondary_pred_fine_data, secondary_pred_coarse_data = (
            self.get_predictions(test=test, secondary=True) if self.secondary_model_name is not None else None, None)
        altered_pred_granularity_data = self.get_predictions(test=test, g=g, stage=stage)

        self.pred_data['test' if test else 'train']['mid_learning'][g] = altered_pred_granularity_data

        for rule_g_l in {l: rule_l for l, rule_l in self.error_detection_rules.items() if l.g == g}.values():
            altered_pred_data_l = rule_g_l(pred_fine_data=pred_fine_data,
                                           pred_coarse_data=pred_coarse_data,
                                           secondary_pred_fine_data=secondary_pred_fine_data,
                                           secondary_pred_coarse_data=secondary_pred_coarse_data,
                                           binary_data=self.pred_data['test' if test else 'train']['binary'])
            altered_pred_granularity_data = np.where(altered_pred_data_l == -1, -1, altered_pred_granularity_data)

        self.pred_data['test' if test else 'train']['post_detection'][g] = altered_pred_granularity_data

    def apply_correction_rules(self,
                               test: bool,
                               g: data_preprocessing.Granularity):
        """Applies error correction rules to test predictions for a given granularity. If a rule is satisfied for a
        particular label, the prediction data for that label is corrected using the rule's logic.

        :param test:
        :param g: The granularity of the predictions to be processed.
        """
        # test_pred_fine_data, test_pred_coarse_data = self.get_predictions(test=True)
        # self.test_pred_data['post_correction'][g] = self.get_predictions(test=True, g=g)

        test_or_train = 'test' if test else 'train'
        g_l_rules = {l: rule_l for l, rule_l in self.error_correction_rules.items() if l.g == g}

        secondary_fine_data, secondary_coarse_data = self.get_predictions(test=test, secondary=True)

        for l, rule_g_l in g_l_rules.items():
            previous_l_precision, previous_l_recall = self.get_l_precision_and_recall(l=l, test=test,
                                                                                      stage='post_correction')

            correction_rule_theoretical_precision_increase = (
                metrics.get_l_correction_rule_theoretical_precision_increase(edcr=self, test=test, l=l))
            correction_rule_theoretical_recall_increase = (
                metrics.get_l_correction_rule_theoretical_recall_increase(edcr=self, test=test, l=l,
                                                                          CC_l=self.error_correction_rules[l].C_l))

            fine_data, coarse_data = self.get_predictions(test=test, stage='post_correction')

            altered_pred_data_l = rule_g_l(pred_fine_data=fine_data,
                                           pred_coarse_data=coarse_data,
                                           secondary_pred_fine_data=secondary_fine_data,
                                           secondary_pred_coarse_data=secondary_coarse_data,
                                           binary_data=self.pred_data['test' if test else 'train']['binary'])

            self.pred_data[test_or_train]['post_correction'][g] = np.where(
                # (collision_array != 1) &
                (altered_pred_data_l == l.index),
                l.index,
                self.get_predictions(test=test, g=g, stage='post_correction'))

            # self.print_how_many_not_assigned(test=test, g=g, stage='post_correction')

            self.evaluate_and_print_l_correction_rule_precision_increase(
                test=test,
                l=l,
                previous_l_precision=previous_l_precision,
                correction_rule_theoretical_precision_increase=correction_rule_theoretical_precision_increase)

            self.evaluate_and_print_l_correction_rule_recall_increase(
                test=test,
                l=l,
                previous_l_recall=previous_l_recall,
                correction_rule_theoretical_recall_increase=correction_rule_theoretical_recall_increase)

            self.print_metrics(test=test, prior=False, stage='post_correction', print_inconsistencies=False)

    def run_learning_pipeline(self,
                              EDCR_epoch_num=0):
        print('Started learning pipeline...\n')
        self.print_metrics(test=False, prior=True)

        for g in data_preprocessing.granularities.values():
            self.learn_detection_rules(g=g)
            self.apply_detection_rules(test=False, g=g)

        print('\nRule learning completed\n')

    def run_evaluating_pipeline(self):
        pass


if __name__ == '__main__':
    epsilons = [0.1 * i for i in range(2, 3)]
    test_bool = False

    for eps in epsilons:
        print('#' * 25 + f'eps = {eps}' + '#' * 50)
        edcr = EDCR_binary_classifier(
            epsilon=eps,
            main_model_name='vit_b_16',
            combined=True,
            loss='BCE',
            lr=0.0001,
            num_epochs=20,
            include_inconsistency_constraint=False,
            secondary_model_name='vit_b_16_soft_marginal')
        edcr.print_metrics(test=test_bool, prior=True)

        edcr.run_learning_pipeline()
