import numpy as np
from sklearn.metrics import accuracy_score
import typing
import heapq
import copy
import multiprocessing as mp
import multiprocessing.managers
import tqdm

from PyEDCR import EDCR
import data_preprocessing
import conditions
import rules
import utils


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

    def objective_function(self,
                           l: data_preprocessing.Label,
                           CC: set[(set[conditions.Condition], data_preprocessing.Label)]):
        Rule_CC_l = rules.ErrorCorrectionRule2(l=l, CC_l=CC)
        prediction_after_apply_rule_mask = Rule_CC_l(
            pred_fine_data=self.pred_data['train']['original']['fine'],
            pred_coarse_data=self.pred_data['train']['original']['coarse'],
            secondary_pred_fine_data=self.pred_data['secondary_model']['train']['fine'],
            secondary_pred_coarse_data=self.pred_data['secondary_model']['train']['coarse']
        )
        prediction_after_apply_rule = np.where(prediction_after_apply_rule_mask == -1,
                                               self.pred_data['train']['original'][l.g],
                                               prediction_after_apply_rule_mask)
        return self.get_accuracy(prediction=prediction_after_apply_rule, g=l.g)

    def CorrRuleLearn(self,
                      l: data_preprocessing.Label,
                      CC_all: set[(conditions.Condition, data_preprocessing.Label)],
                      shared_index: mp.managers.ValueProxy,
                      beam_width: int = 30,
                      relaxation: int = 0.3) -> \
            (data_preprocessing.Label, [tuple[conditions.Condition, data_preprocessing.Label]]):
        """Learns error correction rules for a specific label and granularity. These rules associate conditions
        with alternative labels that are more likely to be correct when those conditions are met.

        :param l: The label of interest.
        :param CC_all: A set of all condition-label pairs to consider for rule learning.
        :return: A set of condition-label pairs.
        """
        beam = []
        counter = 0

        print(f"Start corr rule learn for class {l}")

        # Initialise the beam with the top beam_width candidates
        for cond_and_l in heapq.nlargest(
                n=beam_width,
                iterable=self.CC_all[l.g],
                key=lambda c_l: self.objective_function(
                    l=l,
                    CC={c_l} # if isinstance(c_l, conditions.Condition) else c_l
                )):
            CC_candidate = {cond_and_l}
            score = self.objective_function(l=l, CC=CC_candidate)
            heapq.heappush(beam, (score, CC_candidate))

        while counter <= 10:
            temp_beam = copy.copy(beam)

            for _, CC_candidate in beam:
                expanded_candidates = [
                    CC_candidate.union({cond_and_l})
                    for cond_and_l in CC_all.difference(CC_candidate)
                ]

                # Evaluate and maintain the top-k candidates in the beam
                for expanded_candidate in expanded_candidates:
                    score = self.objective_function(l=l, CC=expanded_candidate)
                    heapq.heappush(temp_beam, (score, expanded_candidate))
                    if len(temp_beam) > beam_width:
                        heapq.heappop(temp_beam)  # Keep only the top-k candidates

            beam = temp_beam
            counter += 1

        # Choose the best candidate from the final beam
        best_CC_l = sorted(beam, key=lambda x: x[0], reverse=True)[0][1]
        CC_l = best_CC_l

        # p_l = self.get_l_precision_and_recall(test=False, l=l)[0]
        # CON_CC_l = self.get_CON_l_CC(l=l, CC=CC_l, test=False)
        # POS_CC_l = self.get_BOD_CC(CC=CC_l)[0] * self.get_CON_l_CC(l=l, CC=CC_l, test=False)

        print(f'\n{l}: len(CC_l)={len(CC_l)}/{len(CC_all)}')

        # if CON_CC_l <= p_l:
        #     CC_l = set()

        if not utils.is_local():
            shared_index.value += 1
            print(f'Completed {shared_index.value}/{len(data_preprocessing.get_labels(l.g).values())}')

        return l, CC_l

    def learn_correction_rules(self,
                               g: data_preprocessing.Granularity,
                               beam_width: int = 30):
        cand = set()
        dummy_l = list(data_preprocessing.get_labels(g).values())[0]
        frac_correct_example_satisfy_cond_and_l = {}
        for i, (cond, l_prime) in enumerate(self.CC_all[g]):
            dummy_rules = rules.ErrorCorrectionRule(l=dummy_l, CC_l={(cond, l_prime)})
            example_satisfy_cond_and_l = dummy_rules.get_where_body_is_satisfied(
                pred_fine_data=self.pred_data['train']['original']['fine'],
                pred_coarse_data=self.pred_data['train']['original']['coarse'],
                secondary_pred_fine_data=self.pred_data['secondary_model']['train']['fine'],
                secondary_pred_coarse_data=self.pred_data['secondary_model']['train']['coarse']
            )
            used_cond_and_l = set()
            for l_i in data_preprocessing.get_labels(g).values():
                frac_correct_example_satisfy_cond_and_l[l_i] = np.sum(
                    example_satisfy_cond_and_l
                    & self.get_where_label_is_l(pred=False, test=False, l=l_i)
                ) / np.sum(example_satisfy_cond_and_l) if np.sum(example_satisfy_cond_and_l) != 0 else 0

                for secondary_cond in (cond for cond in self.condition_datas[g] if cond.secondary):
                    for l_j in data_preprocessing.get_labels(g).values():
                        dummy_rules = rules.ErrorCorrectionRule2(l=dummy_l, CC_l={((secondary_cond, cond), l_prime)})
                        example_satisfy_two_cond_and_l = dummy_rules.get_where_body_is_satisfied(
                            pred_fine_data=self.pred_data['train']['original']['fine'],
                            pred_coarse_data=self.pred_data['train']['original']['coarse'],
                            secondary_pred_fine_data=self.pred_data['secondary_model']['train']['fine'],
                            secondary_pred_coarse_data=self.pred_data['secondary_model']['train']['coarse']
                        )
                        frac_correct_example_satisfy_two_cond_and_l = np.sum(
                            example_satisfy_two_cond_and_l
                            & self.get_where_label_is_l(pred=False, test=False, l=l_j)
                        ) / np.sum(example_satisfy_two_cond_and_l) if np.sum(example_satisfy_two_cond_and_l) != 0 else 0
                        if frac_correct_example_satisfy_two_cond_and_l > frac_correct_example_satisfy_cond_and_l[l_i]:
                            cand.add(((cond, secondary_cond), l_prime))
                            used_cond_and_l.add(secondary_cond)
            cand.add(((*(dummy_cond.negatePredCondition for dummy_cond in used_cond_and_l), cond), l_prime))
            print(f"finish {i}/{len(self.CC_all[g])}")

        self.CC_all[g] = cand

        print(f"Retrieve {len(cand)} condition class pair")

        # Beam search part:
        granularity_labels = data_preprocessing.get_labels(g).values()
        processes_num = min(len(granularity_labels), mp.cpu_count())

        manager = mp.Manager()
        shared_index = manager.Value('i', 0)

        iterable = [(l, self.CC_all[g], shared_index) for l in granularity_labels]

        with mp.Pool(processes_num) as pool:
            CC_ls = pool.starmap(func=self.CorrRuleLearn,
                                 iterable=iterable)

        for l, CC_l in CC_ls:
            if len(CC_l):
                self.error_correction_rules[l] = rules.ErrorCorrectionRule2(l=l, CC_l=CC_l)
            else:
                print(utils.red_text('\n' + '#' * 10 + f' {l} does not have an error correction rule!\n'))

    def run_learning_pipeline(self,
                              EDCR_epoch_num=0):
        print('Started learning pipeline...\n')
        self.print_metrics(test=False, prior=True)

        for g in data_preprocessing.granularities.values():
            self.learn_detection_rules(g=g)

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
        for gra in data_preprocessing.granularities.values():
            edcr.apply_correction_rules(test=test_bool, g=gra)
