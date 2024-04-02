import os
import torch.utils.data
import context_handlers
import models
import ltn
import ltn_support
import numpy as np

from PyEDCR import EDCR
import data_preprocessing
import vit_pipeline
import typing
import config


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
                 secondary_model_name: str = None,
                 config=None):
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

        self.batch_size = config.batch_size
        self.scheduler_gamma = config.scheduler_gamma
        self.num_epochs = config.num_ltn_epochs
        self.scheduler_step_size = num_epochs
        self.pretrain_path = config.main_pretrained_path


    def run_learning_pipeline(self,
                              EDCR_epoch_num=0):
        print('Started learning pipeline...\n')
        self.print_metrics(test=False, prior=True)

        for g in data_preprocessing.granularities.values():
            self.learn_detection_rules(g=g)

        print('\nRule learning completed\n')

    def run_evaluating_pipeline(self):
        pass

if __name__ == '__main__':
    epsilons = [0.1 * i for i in range(2, 3)]
    test_bool = False
    main_pretrained_path = config

    for eps in epsilons:
        print('#' * 25 + f'eps = {eps}' + '#' * 50)
        edcr = EDCR_binary_classifier(
            epsilon=eps,
            main_model_name=config.vit_model_name,
            combined=config.combined,
            loss=config.loss,
            lr=config.lr,
            num_epochs=config.num_epochs,
            include_inconsistency_constraint=config.include_inconsistency_constraint,
            secondary_model_name=config.secondary_model_name,
            config=config)
        edcr.print_metrics(test=test_bool, prior=True)

        edcr.run_learning_pipeline()
        edcr.run_evaluating_pipeline()
