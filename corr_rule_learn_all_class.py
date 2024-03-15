from PyEDCR import EDCR
import data_preprocessing
import utils
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class EDCR_experiment(EDCR):
    def get_accuracy(self,
                     prediction: np.array,
                     g: data_preprocessing.Granularity):
        return accuracy_score(y_true=data_preprocessing.get_ground_truths(test=False,
                                                                          g=g),
                              y_pred=prediction)

    def new_objective_function(self,
                               l: data_preprocessing.Label,
                               CC: set[(EDCR._Condition, data_preprocessing.Label)],
                               g: data_preprocessing.Granularity):
        Rule_CC_l = EDCR.ErrorCorrectionRule(l=l, CC_l=CC)
        prediction_after_apply_rule = Rule_CC_l(fine_data=self.pred_data['train']['original']['fine'],
                                                coarse_data=self.pred_data['train']['original']['coarse'])
        return self.get_accuracy(prediction=prediction_after_apply_rule,
                                 g=g)

    def learn_correction_rules(self,
                               g: data_preprocessing.Granularity):

        granularity_labels = data_preprocessing.get_labels(g).values()
        other_g = data_preprocessing.granularities['fine' if str(g) == 'coarse' else 'coarse']

        print(f'\nLearning {g}-grain error correction rules...')

        CC_ls = {l: set() for l in data_preprocessing.get_labels(g).values()}

        for CC in tqdm(self.CC_all[g]):
            max_score = 100
            assign_l = None
            for l in data_preprocessing.get_labels(g).values():
                score = self.new_objective_function(l=l, CC=CC, g=g)
                if score > max_score and self.get_BOD_CC(CC) != 0:
                    assign_l = l
                    max_score = score

            CC_ls[assign_l] = CC_ls[assign_l].union({CC})

        for l, CC_l in CC_ls.items():
            if len(CC_l):
                self.error_correction_rules[l] = EDCR.ErrorCorrectionRule(l=l, CC_l=CC_l)
            else:
                print(utils.red_text('\n' + '#' * 10 + f' {l} does not have an error correction rule!\n'))


if __name__ == '__main__':
    epsilons = [0.1 * i for i in range(2, 3)]
    test_bool = True

    for eps in epsilons:
        print('#' * 25 + f'eps = {eps}' + '#' * 50)
        edcr = EDCR(epsilon=eps,
                    main_model_name='vit_b_16',
                    combined=True,
                    loss='BCE',
                    lr=0.0001,
                    num_epochs=20,
                    include_inconsistency_constraint=False)
        edcr.print_metrics(test=test_bool, prior=True)

        edcr.run_learning_pipeline()
        # edcr.run_error_detection_application_pipeline(test=test_bool)
        edcr.run_error_correction_application_pipeline(test=test_bool)
