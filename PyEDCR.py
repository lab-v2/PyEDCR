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

    for main_granularity in data_preprocessing.granularities:
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

        test_true_fine_path = f'{combined_str}_results/test_true_fine.npy'
        test_true_coarse_path = f'{combined_str}_results/test_true_coarse.npy'

        train_pred_fine_path = f'{combined_str}_results/{main_model_name}_train_fine_pred_{loss}_lr{lr}.npy'
        train_pred_coarse_path = f'{combined_str}_results/{main_model_name}_train_coarse_pred_{loss}_lr{lr}.npy'

        train_true_fine_path = f'{combined_str}_results/train_true_fine.npy'
        train_true_coarse_path = f'{combined_str}_results/train_true_coarse.npy'

        self.__test_pred_fine_data = np.load(test_pred_fine_path)
        self.__test_pred_coarse_data = np.load(test_pred_coarse_path)

        self.__test_true_fine_data = np.load(test_true_fine_path)
        self.__test_true_coarse_data = np.load(test_true_coarse_path)

        self.__train_pred_fine_data = np.load(train_pred_fine_path)
        self.__train_pred_coarse_data = np.load(train_pred_coarse_path)

        self.__train_true_fine_data = np.load(train_true_fine_path)
        self.__train_true_coarse_data = np.load(train_true_coarse_path)

        self.__train_condition_datas = self.__get_conditions(test=True)
        self.__test_condition_datas = self.__get_conditions(test=False)

    def __get_predictions(self,
                          test: bool):
        if test:
            pred_fine_data = self.__test_pred_fine_data
            pred_coarse_data = self.__test_pred_coarse_data
        else:
            pred_fine_data = self.__train_pred_fine_data
            pred_coarse_data = self.__train_pred_coarse_data

        return pred_fine_data, pred_coarse_data

    def __get_ground_truths(self,
                            test: bool):
        if test:
            true_fine_data = self.__test_true_fine_data
            true_coarse_data = self.__test_true_coarse_data
        else:
            true_fine_data = self.__train_true_fine_data
            true_coarse_data = self.__train_true_coarse_data

        return true_fine_data, true_coarse_data

    def get_and_print_metrics(self,
                              test: bool):

        pred_fine_data, pred_coarse_data = self.__get_predictions(test=test)
        true_fine_data, true_coarse_data = self.__get_ground_truths(test=test)

        vit_pipeline.get_and_print_metrics(pred_fine_data=pred_fine_data,
                                           pred_coarse_data=pred_coarse_data,
                                           loss=self.__loss,
                                           true_fine_data=true_fine_data,
                                           true_coarse_data=true_coarse_data,
                                           combined=self.__combined,
                                           model_name=self.__main_model_name,
                                           lr=self.__lr)

    def __get_conditions(self,
                         test: bool) -> dict[str, dict[str, np.array]]:
        condition_datas = {}
        pred_fine_data, pred_coarse_data = self.__get_predictions(test=test)

        for granularity in data_preprocessing.granularities:
            cla_data = self.__train_pred_fine_data if granularity == 'fine' else pred_coarse_data
            condition_datas[granularity] = data_preprocessing.get_one_hot_encoding(cla_data)

        return condition_datas

    def DetRuleLearn(self):
        pass

    def CorrRuleLearn(self):
        pass

    def DetCorrRuleLearn(self):
        pass


if __name__ == '__main__':
    edcr = EDCR(main_model_name='vit_b_16',
                combined=True,
                loss='BCE',
                lr=0.0001,
                num_epochs=20,
                epsilon=0.1)
    edcr.get_and_print_metrics(test=False)
    edcr.get_and_print_metrics(test=True)
