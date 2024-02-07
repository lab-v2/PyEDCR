import numpy as np
import typing

import utils
import data_preprocessing
import vit_pipeline


def run_EDCR_pipeline(test_pred_fine_path: str,
                      test_pred_coarse_path: str,
                      test_true_fine_path: str,
                      test_true_coarse_path: str,
                      train_pred_fine_path: str,
                      train_pred_coarse_path: str,
                      train_true_fine_path: str,
                      train_true_coarse_path: str,
                      main_lr: typing.Union[str, float],
                      combined: bool,
                      loss: str,
                      epsilon: float,
                      consistency_constraints: bool,
                      multiprocessing: bool = True):
    (test_pred_fine_data,
     test_pred_coarse_data,
     test_true_fine_data,
     test_true_coarse_data,
     train_pred_fine_data,
     train_pred_coarse_data,
     train_true_fine_data,
     train_true_coarse_data,
     possible_test_consistency_constraints) = load_priors(test_pred_coarse_path=test_pred_coarse_path,
                                                          test_pred_fine_path=test_pred_fine_path,
                                                          test_true_fine_path=test_true_fine_path,
                                                          test_true_coarse_path=test_true_coarse_path,
                                                          train_pred_fine_path=train_pred_fine_path,
                                                          train_pred_coarse_path=train_pred_coarse_path,
                                                          train_true_fine_path=train_true_fine_path,
                                                          train_true_coarse_path=train_true_coarse_path,
                                                          main_lr=main_lr,
                                                          loss=loss,
                                                          combined=combined)
    train_condition_datas = get_conditions(pred_fine_data=train_pred_fine_data,
                                           pred_coarse_data=train_pred_coarse_data,
                                           # secondary_fine_data=secondary_fine_data
                                           )
    test_condition_datas = get_conditions(pred_fine_data=test_pred_fine_data,
                                          pred_coarse_data=test_pred_coarse_data,
                                          )
    pipeline_results = {}
    error_detections = []

    for main_granularity in data_preprocessing.granularities_str:
        if main_granularity == 'fine':
            test_pred_granularity = test_pred_fine_data
            test_true_granularity = test_true_fine_data

            train_pred_granularity = train_pred_fine_data
            train_true_granularity = train_true_fine_data
        else:
            test_pred_granularity = test_pred_coarse_data
            test_true_granularity = test_true_coarse_data

            train_pred_granularity = train_pred_coarse_data
            train_true_granularity = train_true_coarse_data

        res = (
            run_EDCR_for_granularity(combined=combined,
                                     main_lr=main_lr,
                                     main_granularity=main_granularity,
                                     test_pred_granularity=test_pred_granularity,
                                     test_true_granularity=test_true_granularity,
                                     train_pred_granularity=train_pred_granularity,
                                     train_true_granularity=train_true_granularity,
                                     train_condition_datas=train_condition_datas,
                                     test_condition_datas=test_condition_datas,
                                     multiprocessing=multiprocessing,
                                     possible_test_consistency_constraints=possible_test_consistency_constraints,
                                     epsilon=epsilon,
                                     consistency_constraints=consistency_constraints))
        pipeline_results[main_granularity] = res[0]
        if multiprocessing:
            error_detections += [res[1]]

    if multiprocessing:
        error_detections = np.mean(np.array(error_detections))
        print(utils.green_text(f'Mean error detections found {np.mean(error_detections)}'))

    vit_pipeline.get_and_print_metrics(pred_fine_data=pipeline_results['fine'],
                                       pred_coarse_data=pipeline_results['coarse'],
                                       loss=loss,
                                       true_fine_data=test_true_fine_data,
                                       true_coarse_data=test_true_coarse_data,
                                       prior=False,
                                       combined=combined,
                                       model_name=main_model_name,
                                       lr=main_lr)


class Condition:
    def __init__(self,
                 data: np.array):
        self.__data = data_preprocessing.get_one_hot_encoding(arr=data)

    def get_condition_value(self,
                            x: data_preprocessing.Example) -> int:
        return self.__data[x.index]


class PredCondition:
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

        self.__train_pred_fine_data = np.load(train_pred_fine_path)
        self.__train_pred_coarse_data = np.load(train_pred_coarse_path)

        self.__test_pred_fine_data = np.load(test_pred_fine_path)
        self.__test_pred_coarse_data = np.load(test_pred_coarse_path)

        self.__train_fine_condition_datas = [PredCondition(pred_data=self.__train_pred_fine_data, l=l)
                                             for l in data_preprocessing.fine_grain_labels]
        self.__train_coarse_condition_datas = [PredCondition(pred_data=self.__train_pred_coarse_data, l=l)
                                               for l in data_preprocessing.coarse_grain_labels]

        self.__test_fine_condition_datas = [PredCondition(pred_data=self.__test_pred_fine_data, l=l)
                                            for l in data_preprocessing.fine_grain_labels]
        self.__test_coarse_condition_datas = [PredCondition(pred_data=self.__test_pred_coarse_data, l=l)
                                              for l in data_preprocessing.coarse_grain_labels]

        self.__rules: dict[str, dict[data_preprocessing.Label, set[Rule]]] = \
            {'error_detections': {l: set() for l in data_preprocessing.all_labels},
             'error_corrections': {l: set() for l in data_preprocessing.all_labels}}

    def __get_predictions(self,
                          test: bool):
        if test:
            pred_fine_data = self.__test_pred_fine_data
            pred_coarse_data = self.__test_pred_coarse_data
        else:
            pred_fine_data = self.__train_pred_fine_data
            pred_coarse_data = self.__train_pred_coarse_data

        return pred_fine_data, pred_coarse_data

    def get_and_print_metrics(self,
                              test: bool):

        pred_fine_data, pred_coarse_data = self.__get_predictions(test=test)
        true_fine_data, true_coarse_data = data_preprocessing.get_ground_truths(test=test)

        vit_pipeline.get_and_print_metrics(pred_fine_data=pred_fine_data,
                                           pred_coarse_data=pred_coarse_data,
                                           loss=self.__loss,
                                           true_fine_data=true_fine_data,
                                           true_coarse_data=true_coarse_data,
                                           combined=self.__combined,
                                           model_name=self.__main_model_name,
                                           lr=self.__lr)

    def __DetRuleLearn(self,
                       l: data_preprocessing.Label) -> set[Condition]:
        pass

    def __CorrRuleLearn(self,
                        l: data_preprocessing.Label,
                        CC_all: set[tuple[Condition, data_preprocessing.Label]]) -> \
            set[tuple[Condition, data_preprocessing.Label]]:
        pass

    def DetCorrRuleLearn(self,
                         g: data_preprocessing.Granularity):
        CC_all = {}

        granularity_labels = data_preprocessing.get_labels(g)

        for l in granularity_labels:
            DC_l = self.__DetRuleLearn(l=l)
            if len(DC_l):
                self.__rules['error_detections'][l] = self.__rules['error_detections'][l].union(DC_l)

            for cond_l in DC_l:
                CC_all = CC_all.union({(cond_l, l)})

        for l in granularity_labels:
            CC_l = self.__CorrRuleLearn(l=l,
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
    edcr.get_and_print_metrics(test=False)
    edcr.get_and_print_metrics(test=True)
