from sklearn.metrics import precision_score, recall_score
import numpy as np
import typing
import time

import utils
import data_preprocessing
import vit_pipeline
import context_handlers


class Condition:
    def __init__(self,
                 data: np.array):
        self.__data = data

    def get_value(self,
                  x: data_preprocessing.Example) -> int:
        return self.__data[x.index]

    @property
    def data(self):
        return self.__data


class PredCondition(Condition):
    def __init__(self,
                 pred_data: np.array,
                 l: data_preprocessing.Label):
        super().__init__(data=np.where(pred_data == l.index, 1, 0))

        self.__l = l

    def get_label(self) -> data_preprocessing.Label:
        return self.__l


class Rule:
    pass


class EDCR:
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

        combined_str = 'combined' if combined else 'individual'
        test_pred_fine_path = (f'{combined_str}_results/{main_model_name}_test_fine_pred_{loss}_lr{lr}'
                               f'_e{num_epochs - 1}.npy')
        test_pred_coarse_path = (f'{combined_str}_results/{main_model_name}_test_coarse_pred_{loss}_lr{lr}'
                                 f'_e{num_epochs - 1}.npy')

        train_pred_fine_path = f'{combined_str}_results/{main_model_name}_train_fine_pred_{loss}_lr{lr}.npy'
        train_pred_coarse_path = f'{combined_str}_results/{main_model_name}_train_coarse_pred_{loss}_lr{lr}.npy'

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
        pred_fine_data, pred_coarse_data = [self.__test_pred_data[g] for g in data_preprocessing.granularities] \
            if test else [self.__train_pred_data[g] for g in data_preprocessing.granularities]

        return pred_fine_data, pred_coarse_data

    def print_metrics(self,
                      test: bool):

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
        return np.where(data_preprocessing.get_ground_truths(test=False, g=g) == l.index, 1, 0)

    def __get_tp_l(self,
                   g: data_preprocessing.Granularity,
                   l: data_preprocessing.Label):
        return np.where(self.__train_pred_data[g] == data_preprocessing.get_ground_truths(test=False, g=g) &
                        (self.__train_pred_data[g] == l.index), 1, 0)

    def __get_fp_l(self,
                   g: data_preprocessing.Granularity,
                   l: data_preprocessing.Label):
        return np.where((self.__train_pred_data[g] == l.index) &
                        (data_preprocessing.get_ground_truths(test=False, g=g) != l.index), 1, 0)

    def __are_all_conditions_satisfied(self, C: set[Condition]):
        all_conditions_satisfied = np.ones_like(self.__train_pred_data[data_preprocessing.granularities[0]])

        for cond in C:
            all_conditions_satisfied &= cond.data

        return all_conditions_satisfied

    def __get_NEG_l(self,
                    g: data_preprocessing.Granularity,
                    l: data_preprocessing.Label,
                    C: set[Condition]) -> int:
        tp_indices = self.__get_tp_l(g=g, l=l)
        all_conditions_satisfied = self.__are_all_conditions_satisfied(C)
        NEG_l = int(np.sum(tp_indices * all_conditions_satisfied))

        return NEG_l

    def __get_POS_l(self,
                    g: data_preprocessing.Granularity,
                    l: data_preprocessing.Label,
                    C: set[Condition]) -> int:
        fp_indices = self.__get_fp_l(g=g, l=l)
        all_conditions_satisfied = self.__are_all_conditions_satisfied(C)
        POS_l = int(np.sum(fp_indices * all_conditions_satisfied))

        return POS_l

    def __get_CON_l(self,
                    g: data_preprocessing.Granularity,
                    l: data_preprocessing.Label,
                    CC: set[(Condition, data_preprocessing.Label)]) -> float:
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
    edcr.print_metrics(test=True)

    for g in data_preprocessing.granularities:
        edcr.DetCorrRuleLearn(g=g)





