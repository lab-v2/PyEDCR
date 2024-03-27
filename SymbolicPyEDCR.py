import numpy as np
import multiprocessing as mp
import multiprocessing.managers
import typing
import random

import data_preprocessing
import PyEDCR
import rules
import conditions
import utils
import context_handlers
import symbolic_metrics

randomized: bool = False


class SymbolicPyEDCR(PyEDCR.EDCR):
    @staticmethod
    def get_CC_str(CC: set[(conditions.Condition, data_preprocessing.Label)]) -> str:
        return ('{' + ', '.join(['(' + ', '.join(item_repr) + ')' for item_repr in
                                 [[str(obj) for obj in item] for item in CC]]) + '}')

    def set_error_correction_rules(self,
                                   input_rules: typing.Dict[
                                       data_preprocessing.Label, {(conditions.Condition, data_preprocessing.Label)}]):
        """
        Manually sets the error correction rule dictionary.

        :params rules: A dictionary mapping label instances to error detection rule objects.
        """
        error_correction_rules = {}
        for label, CC_l in input_rules.items():
            error_correction_rules[label] = rules.ErrorCorrectionRule(label, CC_l=CC_l)
        self.error_correction_rules = error_correction_rules

    def apply_reversion_rules(self,
                              test: bool,
                              g: data_preprocessing.Granularity):
        """Applies error reversion rules to recover a prediction for a given granularity. If the inference of detection
        and correction rules do not change the label, the prediction label for that example is set to be the original
        one.

        :param test:
        :param g: The granularity of the predictions to be processed.
        """
        test_or_train = 'test' if test else 'train'

        pred_granularity_data = self.get_predictions(test=test, g=g, stage='post_correction')

        self.pred_data[test_or_train][g] = np.where(pred_granularity_data == -1,
                                                    self.get_predictions(test=test, g=g, stage='original'),
                                                    pred_granularity_data)

    def get_BOD_CC(self,
                   CC: set[(conditions.Condition, data_preprocessing.Label)]) -> (int, np.array):
        """Calculate the number of train samples that satisfy the body of the 2nd rule for some set of condition
        class pair.

        :param CC: A set of `Condition`-`Class` pair.
        :return: The number of instances that satisfy the body of the 2nd rule and the boolean array it.
        """
        train_fine_pred_data, train_coarse_pred_data = self.get_predictions(test=False)
        where_any_pair_is_satisfied_in_train_pred = np.zeros_like(train_fine_pred_data)

        for cond, l_prime in CC:
            where_predicted_l_prime_in_train = self.get_where_label_is_l(pred=True, test=False, l=l_prime)
            where_condition_is_satisfied_in_train_pred = cond(train_fine_pred_data, train_coarse_pred_data)
            where_pair_is_satisfied = where_predicted_l_prime_in_train * where_condition_is_satisfied_in_train_pred
            where_any_pair_is_satisfied_in_train_pred |= where_pair_is_satisfied

        BOD_l = np.sum(where_any_pair_is_satisfied_in_train_pred)

        return BOD_l, where_any_pair_is_satisfied_in_train_pred

    def get_POS_l_CC(self,
                     test: bool,
                     l: data_preprocessing.Label,
                     CC: set[(conditions.Condition, data_preprocessing.Label)]) -> int:
        """Calculate the number of samples that satisfy the body of the 2nd rule and head
        (ground truth is l) for a label l and some set of condition class pair.

        :param test:
        :param CC:
        :param l: The label of interest.
        :return: The number of instances that satisfy the body of the 2nd rule and the boolean array it.
        """
        where_ground_truths_is_l = self.get_where_label_is_l(pred=False, test=test, l=l)

        fine_pred_data, coarse_pred_data = self.get_predictions(test=test)
        where_any_pair_is_satisfied = np.zeros_like(fine_pred_data)

        for cond, l_prime in CC:
            where_predicted_l_prime = self.get_where_label_is_l(pred=True, test=test, l=l_prime)
            where_condition_is_satisfied = cond(fine_pred_data, coarse_pred_data)
            where_pair_is_satisfied = where_predicted_l_prime * where_condition_is_satisfied
            where_any_pair_is_satisfied |= where_pair_is_satisfied

        POS_l_CC = np.sum(where_any_pair_is_satisfied * where_ground_truths_is_l)

        return POS_l_CC

    def get_CON_l_CC(self,
                     test: bool,
                     l: data_preprocessing.Label,
                     CC: set[(conditions.Condition, data_preprocessing.Label)]) -> float:
        """Calculate the ratio of number of samples that satisfy the rule body and head with the ones
        that only satisfy the body, given a set of condition class pairs.

        :param test:
        :param CC: A set of `Condition` - `Label` pairs.
        :param l: The label of interest.
        :return: ratio as defined above
        """

        BOD_CC, where_any_pair_is_satisfied_in_train_pred = self.get_BOD_CC(CC=CC)
        POS_l_CC = self.get_POS_l_CC(test=test, l=l, CC=CC)
        CON_l_CC = POS_l_CC / BOD_CC if BOD_CC else 0

        return CON_l_CC

    def _CorrRuleLearn(self,
                       l: data_preprocessing.Label,
                       CC_all: set[(conditions.Condition, data_preprocessing.Label)],
                       shared_index: mp.managers.ValueProxy) -> \
            (data_preprocessing.Label, [tuple[conditions.Condition, data_preprocessing.Label]]):
        """Learns error correction rules for a specific label and granularity. These rules associate conditions
        with alternative labels that are more likely to be correct when those conditions are met.

        :param l: The label of interest.
        :param CC_all: A set of all condition-label pairs to consider for rule learning.
        :return: A set of condition-label pairs.
        """
        CC_l = set()

        CC_l_prime = CC_all
        CC_sorted = sorted(CC_l_prime, key=lambda c_l: self.get_CON_l_CC(test=False, l=l, CC={c_l}))

        with context_handlers.WrapTQDM(total=len(CC_sorted)) as progress_bar:
            for cond_and_l in CC_sorted:
                a = self.get_CON_l_CC(test=False, l=l, CC=CC_l.union({cond_and_l})) - self.get_CON_l_CC(test=False,
                                                                                                        l=l, CC=CC_l)
                b = (self.get_CON_l_CC(test=False, l=l, CC=CC_l_prime.difference({cond_and_l})) -
                     self.get_CON_l_CC(test=False, l=l, CC=CC_l_prime))

                # randomized algorithm
                a_prime = max(a, 0)
                b_prime = max(b, 0)
                P = a_prime / (a_prime + b_prime) if not (a_prime == 0 and b_prime == 0) else 1

                # if a >= b:
                if ((not randomized) and a >= b) or (randomized and (random.random() < P)):
                    CC_l = CC_l.union({cond_and_l})
                else:
                    CC_l_prime = CC_l_prime.difference({cond_and_l})

                if utils.is_local():
                    progress_bar.update(1)

        assert CC_l_prime == CC_l

        p_l = self.get_l_precision_and_recall(test=False, l=l)[0]
        CON_CC_l = self.get_CON_l_CC(test=False, l=l, CC=CC_l)

        print(f'\n{l}: len(CC_l)={len(CC_l)}/{len(CC_all)}, CON_l_CC={CON_CC_l}, '
              f'p_l={p_l}\n')

        # if CON_CC_l <= p_l:
        #     CC_l = set()

        if not utils.is_local():
            shared_index.value += 1
            print(f'Completed {shared_index.value}/{len(data_preprocessing.get_labels(l.g).values())}')

        return l, CC_l

    def learn_correction_rules(self,
                               g: data_preprocessing.Granularity):

        granularity_labels = data_preprocessing.get_labels(g).values()

        print(f'\nLearning {g}-grain error correction rules...')
        processes_num = min(len(granularity_labels), mp.cpu_count())

        manager = mp.Manager()
        shared_index = manager.Value('i', 0)

        iterable = [(l, self.CC_all[g],
                     shared_index
                     ) for l in granularity_labels]

        with mp.Pool(processes_num) as pool:
            CC_ls = pool.starmap(func=self._CorrRuleLearn,
                                 iterable=iterable)

        for l, CC_l in CC_ls:
            if len(CC_l):
                self.error_correction_rules[l] = rules.ErrorCorrectionRule(l=l, CC_l=CC_l)
            else:
                print(utils.red_text('\n' + '#' * 10 + f' {l} does not have an error correction rule!\n'))


    def evaluate_and_print_l_correction_rule_precision_increase(self,
                                                                test: bool,
                                                                l: data_preprocessing.Label,
                                                                previous_l_precision: float,
                                                                correction_rule_theoretical_precision_increase: float,
                                                                threshold: float = 1e-5
                                                                ):
        post_correction_l_precision = self.get_l_precision_and_recall(l=l, test=test, stage='post_correction')[0]

        precision_diff = post_correction_l_precision - previous_l_precision

        precision_theory_holds = abs(correction_rule_theoretical_precision_increase - precision_diff) < threshold
        precision_theory_holds_str = utils.green_text('The theory holds!') if precision_theory_holds else (
            utils.red_text('The theory does not hold!'))

        print(f'class {l} new precision: {post_correction_l_precision}, '
              f'class {l} old precision: {previous_l_precision}, '
              f'diff: {utils.blue_text(precision_diff)}\n'
              f'theoretical precision increase: {utils.blue_text(correction_rule_theoretical_precision_increase)}\n'
              f'{precision_theory_holds_str}'
              )

    def evaluate_and_print_l_correction_rule_recall_increase(self,
                                                             test: bool,
                                                             l: data_preprocessing.Label,
                                                             previous_l_recall: float,
                                                             correction_rule_theoretical_recall_increase: float,
                                                             threshold: float = 1e-5
                                                             ):
        post_correction_l_recall = self.get_l_precision_and_recall(l=l, test=test, stage='post_correction')[1]

        precision_diff = post_correction_l_recall - previous_l_recall

        precision_theory_holds = abs(correction_rule_theoretical_recall_increase - precision_diff) < threshold
        precision_theory_holds_str = utils.green_text('The theory holds!') if precision_theory_holds else (
            utils.red_text('The theory does not hold!'))

        print(f'class {l} new recall: {post_correction_l_recall}, '
              f'class {l} old recall: {previous_l_recall}, '
              f'diff: {utils.blue_text(precision_diff)}\n'
              f'theoretical recall increase: {utils.blue_text(correction_rule_theoretical_recall_increase)}\n'
              f'{precision_theory_holds_str}'
              )

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

        secondary_fine_data, secondary_coarse_data = self.get_predictions(test=test, secondary=True) \
            if self.secondary_model_name is not None else (None, None)
        lower_train_pred_fine_data, lower_train_pred_coarse_data = (
            self.get_predictions(test=False, lower_predictions=True)) \
            if self.lower_predictions_indices is not None else (None, None)

        for l, rule_g_l in g_l_rules.items():
            previous_l_precision, previous_l_recall = self.get_l_precision_and_recall(l=l, test=test,
                                                                                      stage='post_correction')

            correction_rule_theoretical_precision_increase = (
                metrics.get_l_correction_rule_theoretical_precision_increase(edcr=self, test=test, l=l))
            correction_rule_theoretical_recall_increase = (
                metrics.get_l_correction_rule_theoretical_recall_increase(edcr=self, test=test, l=l,
                                                                          CC_l=self.error_correction_rules[l].C_l))

            fine_data, coarse_data = self.get_predictions(test=test, stage='post_correction')

            altered_pred_data_l = rule_g_l(pred_fine_data=fine_data, pred_coarse_data=coarse_data,
                                           secondary_pred_fine_data=secondary_fine_data,
                                           secondary_pred_coarse_data=secondary_coarse_data,
                                           lower_predictions_fine_data=lower_train_pred_fine_data,
                                           lower_predictions_coarse_data=lower_train_pred_coarse_data)

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

        # collision_array = np.zeros_like(altered_pred_granularity_data)
        #
        # for l_1, altered_pred_data_l_1, in altered_pred_granularity_datas.items():
        #     for l_2, altered_pred_data_l_2 in altered_pred_granularity_datas.items():
        #         if l_1 != l_2:
        #             where_supposed_to_correct_to_l1 = np.where(altered_pred_data_l_1 == l_1.index, 1, 0)
        #             where_supposed_to_correct_to_l2 = np.where(altered_pred_data_l_2 == l_2.index, 1, 0)
        #             collision_array |= where_supposed_to_correct_to_l1 * where_supposed_to_correct_to_l2

        # for l, altered_pred_data_l in altered_pred_granularity_datas.items():

        for l in data_preprocessing.get_labels(g).values():
            self.num_predicted_l['post_correction'][g][l] = np.sum(self.get_where_label_is_l(pred=True,
                                                                                             test=True,
                                                                                             l=l,
                                                                                             stage='post_correction'))

        # return altered_pred_granularity_data

    def run_error_correction_application_pipeline(self,
                                                  test: bool):
        print('\n' + '#' * 50 + 'post correction' + '#' * 50)

        for g in data_preprocessing.granularities.values():
            self.apply_correction_rules(test=test, g=g)

        self.print_metrics(test=test, prior=False, stage='post_correction', print_inconsistencies=False)
