import os
import torch.utils.data
from tqdm import tqdm
import typing

import ltn
import ltn_support
import numpy as np
import copy

from src.PyEDCR.data_processing import data_preprocessor
from src.PyEDCR.utils import utils, context_handlers
from src.PyEDCR.PyEDCR import EDCR
from src.PyEDCR.neural_fine_tuning import neural_fine_tuning, backbone_pipeline, models
from src.PyEDCR.neural_fine_tuning.models import models

from src.PyEDCR.evaluation import neural_evaluation, metrics
from src.PyEDCR import config

import torch
import torch.nn as nn


def evaluate_on_test(data_str: str,
                     model_name: str,
                     lr: typing.Union[str, float],
                     fine_tuner: models.FineTuner,
                     loss: str,
                     num_epochs: int,
                     save_files: bool = False):
    neural_evaluation.run_combined_evaluating_pipeline(data_str=data_str,
                                                       model_name=model_name,
                                                       split='test',
                                                       lr=lr,
                                                       loss=loss,
                                                       pretrained_fine_tuner=fine_tuner,
                                                       num_epochs=num_epochs,
                                                       print_results=True,
                                                       save_files=save_files)

    print('#' * 100)


class EDCR_LTN_experiment(EDCR):
    def __init__(self,
                 data_str: str,
                 main_model_name: str,
                 combined: bool,
                 loss: str,
                 lr: typing.Union[str, float],
                 original_num_epochs: int,
                 epsilon: typing.Union[str, float],
                 secondary_model_name: str = None,
                 secondary_model_loss: str = None,
                 secondary_num_epochs: int = None,
                 binary_l_strs: typing.List[str] = [],
                 binary_num_epochs: int = None,
                 binary_lr: typing.Union[str, float] = None,
                 binary_model_name: str = None):

        super().__init__(data_str=data_str,
                         main_model_name=main_model_name,
                         combined=combined,
                         loss=loss,
                         lr=lr,
                         original_num_epochs=original_num_epochs,
                         epsilon=epsilon,
                         secondary_model_name=secondary_model_name if config.use_secondary_model else None,
                         secondary_model_loss=secondary_model_loss if config.use_secondary_model else None,
                         secondary_num_epochs=secondary_num_epochs if config.use_secondary_model else None,
                         binary_l_strs=binary_l_strs if config.use_binary_model else [],
                         binary_num_epochs=binary_num_epochs if config.use_binary_model else None,
                         binary_lr=binary_lr if config.use_binary_model else None,
                         binary_model_name=binary_model_name if config.use_binary_model else None,
                         use_google_api=False)

        print(utils.blue_text(f'EDCR has {config.use_binary_model} flag for using binary model '
                              f'and {config.use_secondary_model} flag for using secondary model'))

        self.batch_size = config.batch_size
        self.scheduler_gamma = config.scheduler_gamma
        self.num_ltn_epochs = config.ltn_num_epochs
        self.scheduler_step_size = config.scheduler_step_size
        self.beta = config.beta
        self.ltn_loss = config.ltn_loss
        self.beta_learn = config.beta_learn
        self.noise_learn_log = config.noise_learn_log
        self.dynamic_weight_averaging = config.dynamic_weight

        self.preprocessor, self.fine_tuners, self.loaders, self.devices = backbone_pipeline.initiate(
            data_str=self.data_str,
            preprocessor=self.preprocessor,
            lr=self.lr,
            model_name=main_model_name,
            train_eval_split=0.8
        )

    def run_learning_pipeline(self,
                              multi_processing: bool = True):
        print('Started learning pipeline...\n')

        for g in data_preprocessor.FineCoarseDataPreprocessor.granularities.values():
            self.learn_detection_rules(g=g,
                                       multi_processing=multi_processing)

        print('\nRule learning completed\n')

    def fine_tune_and_evaluate_combined_model(self,
                                              print_rules: bool = False,
                                              additional_info: str = None):
        fine_tuner = self.fine_tuners[0]
        device = self.devices[0]
        preprocessor = self.preprocessor
        loss = self.ltn_loss
        beta = self.beta
        data_str = self.preprocessor.data_str
        model_name = self.main_model_name
        lr = self.lr
        evaluate_on_test_between_epochs = True
        loaders = self.loaders
        early_stopping = False

        fine_tuner.to(device)
        fine_tuner.train()
        train_loader = loaders['train']
        num_batches = len(train_loader)

        # initialise learnable parameters
        # initialise beta,sigma1,sigma2

        # initialise using torch.normal_ and limit the numbers using torch.clamp

        # Learnable Beta is an added beta parameter between 1e2 and 1e4 to weigh LTN loss only
        # Noise optimisation is based on Multi Task learning paper which optimizes sigma parameters

        temperature = None
        beta_param = None
        sigma_1 = None
        sigma_2 = None

        if self.beta_learn is True:
            shape = (1,)
            mean = 100
            std = 10

            beta_param = nn.Parameter(torch.empty(*shape, dtype=torch.float).normal_(mean, std).to(device))

            model_parameters = list(fine_tuner.parameters()) + [beta_param]

            optimizer = torch.optim.Adam(params=model_parameters, lr=self.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                        step_size=config.scheduler_step_size,
                                                        gamma=self.scheduler_gamma)

        elif self.noise_learn_log is True:
            shape = (1,)
            sigma_1 = nn.Parameter(torch.empty(*shape, dtype=torch.float).normal_().to(device))
            sigma_2 = nn.Parameter(torch.empty(*shape, dtype=torch.float).normal_().to(device))
            model_parameters = list(fine_tuner.parameters()) + [sigma_1, sigma_2]
            optimizer = torch.optim.Adam(params=model_parameters, lr=self.lr)
        elif self.dynamic_weight_averaging is True:
            shape = (1,)
            temperature = nn.Parameter(torch.empty(*shape, dtype=torch.float).normal_().to(device))
            model_parameters = list(fine_tuner.parameters()) + [temperature]
            optimizer = torch.optim.Adam(params=model_parameters, lr=self.lr)
        else:
            optimizer = torch.optim.Adam(params=fine_tuner.parameters())

        logits_to_predicate = ltn.Predicate(ltn_support.LogitsToPredicate()).to(ltn.device)

        early_stopping_value_list = []
        best_fine_tuner = copy.deepcopy(fine_tuner)

        if print_rules:
            print(utils.blue_text(f'Rules that are using: '))
            for label, rules in self.error_detection_rules.items():
                print(f'rule for {label}: {rules}')

        neural_fine_tuning.print_fine_tuning_initialization(fine_tuner=fine_tuner,
                                                            num_epochs=self.num_ltn_epochs,
                                                            lr=self.lr,
                                                            device=device,
                                                            early_stopping=True)
        print(utils.blue_text(f'additional parameter: beta = {beta}, loss = {loss}'))
        print('#' * 100 + '\n')
        print('#' * 100 + '\n')

        consecutive_epochs_with_no_train_eval_loss_decrease_from_the_minimum = 0

        LTN_losses = []
        NN_losses = []
        ratios_LTN = []
        ratios_NN = []

        for epoch in range(self.num_ltn_epochs):

            with ((context_handlers.TimeWrapper())):
                total_running_loss = torch.Tensor([0.0]).to(device)

                total_train_fine_predictions = []
                total_train_coarse_predictions = []

                total_train_fine_ground_truths = []
                total_train_coarse_ground_truths = []

                batches = tqdm(enumerate(train_loader),
                               total=num_batches)

                for batch_num, batch in batches:
                    with context_handlers.ClearCache(device=device):
                        optimizer.zero_grad()
                        X, Y_true_fine, Y_true_coarse, indices = [batch[i].to(device) for i in [0, 1, 3, 4]]

                        indices = indices.detach().to('cpu')
                        # slice the condition from the indices you get above
                        original_pred_fine_batch = torch.tensor(
                            self.pred_data['train']['original']['fine'][indices]).to(device)
                        original_pred_coarse_batch = torch.tensor(
                            self.pred_data['train']['original']['coarse'][indices]).to(device)

                        secondary_pred_fine_batch = None if not config.use_secondary_model else torch.tensor(
                            self.pred_data['secondary_model']['train']['fine'][indices]).to(device)
                        secondary_pred_coarse_batch = None if not config.use_secondary_model else torch.tensor(
                            self.pred_data['secondary_model']['train']['coarse'][indices]).to(device)

                        # binary_pred = {label: torch.tensor(binary_data).to(device)
                        #                for label, binary_data in self.pred_data['binary'].items()}

                        Y_true_fine_one_hot = torch.nn.functional.one_hot(Y_true_fine, num_classes=len(
                            preprocessor.fine_grain_classes_str))
                        Y_true_coarse_one_hot = torch.nn.functional.one_hot(Y_true_coarse, num_classes=len(
                            preprocessor.coarse_grain_classes_str))

                        Y_true_combine = torch.cat(tensors=[Y_true_fine_one_hot, Y_true_coarse_one_hot], dim=1).float()

                        # currently we have many option to get prediction, depend on whether fine_tuner predict
                        # fine / coarse or both
                        Y_pred = fine_tuner(X)

                        Y_pred_fine_grain = Y_pred[:, :len(preprocessor.fine_grain_classes_str)]
                        Y_pred_coarse_grain = Y_pred[:, len(preprocessor.fine_grain_classes_str):]

                        # You can modify how the loss is defined here
                        if 'LTN_BCE' in loss:
                            criterion = torch.nn.BCEWithLogitsLoss()
                        if "LTN_soft_marginal" in loss:
                            criterion = torch.nn.MultiLabelSoftMarginLoss()

                        sat_agg = ltn_support.compute_sat_normally(
                            preprocessor=preprocessor,
                            logits_to_predicate=logits_to_predicate,
                            train_pred_fine_batch=Y_pred_fine_grain,
                            train_pred_coarse_batch=Y_pred_coarse_grain,
                            train_true_fine_batch=Y_true_fine,
                            train_true_coarse_batch=Y_true_coarse,
                            original_train_pred_fine_batch=original_pred_fine_batch,
                            original_train_pred_coarse_batch=original_pred_coarse_batch,
                            secondary_train_pred_fine_batch=secondary_pred_fine_batch,
                            secondary_train_pred_coarse_batch=secondary_pred_coarse_batch,
                            binary_pred=self.pred_data['binary'],
                            error_detection_rules=self.error_detection_rules,
                            device=device
                        )

                        # modify the batch loss according to the optimisation used
                        if self.dynamic_weight_averaging is True:
                            print("Using dynamic weight allocation")
                            LTN_losses.append((1. - sat_agg).item())
                            NN_losses.append((criterion(Y_pred, Y_true_combine)).item())

                            if len(LTN_losses) < 2 and len(NN_losses) < 2:
                                ratio_LTN = torch.randn(1)
                                ratio_NN = torch.randn(1)
                                ratios_LTN.append(ratio_LTN)
                                ratios_NN.append(ratio_NN)

                            else:
                                ratio_LTN = (LTN_losses[-1] / LTN_losses[-2])
                                ratio_NN = (NN_losses[-1] / NN_losses[-2])
                                ratios_LTN.append(ratio_LTN)
                                ratios_NN.append(ratio_NN)

                            if len(ratios_LTN) < 1 and len(ratios_NN) < 1:
                                alpha_1 = torch.randn(1)
                                alpha_2 = torch.randn(1)

                            else:
                                ratio_tensor = torch.Tensor([ratio_LTN, ratio_NN])

                                softmax = torch.nn.Softmax(dim=1)
                                alphas = softmax(ratio_tensor / temperature)
                                alpha_1 = 2 * alphas[0]
                                alpha_2 = 2 * alphas[1]

                            batch_total_loss = alpha_1 * (1. - sat_agg) + alpha_2 * criterion(Y_pred, Y_true_combine)

                        if self.beta_learn is True:
                            print('Using learnable beta')
                            min_value = 1e1
                            max_value = 1e4
                            # Apply torch.clamp to ensure the values fall within the specified range
                            beta_param.data = torch.clamp(beta_param.data, min_value, max_value)

                            print(beta_param.item())

                            # modified loss to make LTN loss comparable to NN loss
                            batch_total_loss = beta_param * (1. - sat_agg) + criterion(Y_pred, Y_true_combine)

                        elif self.noise_learn_log is True:
                            print('Using learnable noise')
                            # keeping the magnitude more than 1e-6 order
                            if sigma_1.data < 0:
                                sigma_1.data = torch.clamp(sigma_1.data, -float('inf'), 1e-3)
                            else:
                                sigma_1.data = torch.clamp(sigma_1.data, 1e-3, float('inf'))

                            if sigma_2.data < 0:
                                sigma_2.data = torch.clamp(sigma_1.data, -float('inf'), 1e-3)
                            else:
                                sigma_2.data = torch.clamp(sigma_1.data, 1e-3, float('inf'))

                            # modified loss for multitask learning
                            # adding 1e-8 to the log term to avoid log(0)
                            batch_total_loss = (0.5 / sigma_1 ** 2) * (1. - sat_agg) + \
                                               (1.0 / sigma_2 ** 2) * (criterion(Y_pred, Y_true_combine)) + \
                                               torch.log(sigma_1 * sigma_2 + 1e-8)
                        else:
                            print('using constant beta')
                            batch_total_loss = beta * (1. - sat_agg) + (1 - beta) * criterion(Y_pred, Y_true_combine)

                        current_train_fine_predictions = torch.max(Y_pred_fine_grain, 1)[1]
                        current_train_coarse_predictions = torch.max(Y_pred_coarse_grain, 1)[1]

                        total_train_fine_predictions += current_train_fine_predictions.tolist()
                        total_train_coarse_predictions += current_train_coarse_predictions.tolist()

                        total_train_fine_ground_truths += Y_true_fine.tolist()
                        total_train_coarse_ground_truths += Y_true_coarse.tolist()

                        del X, Y_true_fine, Y_true_coarse, Y_pred, Y_pred_fine_grain, Y_pred_coarse_grain

                        total_running_loss += batch_total_loss.item()

                        if batch_num > 4 and batch_num % 5 == 0:
                            metrics.get_and_print_post_metrics(preprocessor=preprocessor,
                                                               curr_batch_num=batch_num,
                                                               total_batch_num=len(batches),
                                                               train_fine_ground_truth=np.array(
                                                                   total_train_fine_ground_truths),
                                                               train_fine_prediction=np.array(
                                                                   total_train_fine_predictions),
                                                               train_coarse_ground_truth=np.array(
                                                                   total_train_coarse_ground_truths),
                                                               train_coarse_prediction=np.array(
                                                                   total_train_coarse_predictions))
                        batch_total_loss.backward()
                        optimizer.step()

                if epoch == 0:
                    print(utils.blue_text(
                        f'coarse grain label use and count: '
                        f'{np.unique(np.array(total_train_coarse_ground_truths), return_counts=True)}'))
                    print(utils.blue_text(
                        f'fine grain label use and count: '
                        f'{np.unique(np.array(total_train_fine_ground_truths), return_counts=True)}'))

                metrics.get_and_print_post_metrics(preprocessor=preprocessor,
                                                   curr_epoch=epoch,
                                                   total_num_epochs=self.num_ltn_epochs,
                                                   train_fine_ground_truth=np.array(
                                                       total_train_fine_ground_truths),
                                                   train_fine_prediction=np.array(
                                                       total_train_fine_predictions),
                                                   train_coarse_ground_truth=np.array(
                                                       total_train_coarse_ground_truths),
                                                   train_coarse_prediction=np.array(
                                                       total_train_coarse_predictions))

                if evaluate_on_test_between_epochs:
                    evaluate_on_test(data_str=preprocessor.data_str,
                                     model_name=self.main_model_name,
                                     lr=lr,
                                     fine_tuner=best_fine_tuner,
                                     loss=loss,
                                     num_epochs=self.num_ltn_epochs,
                                     save_files=False)

                if early_stopping:
                    # You can modify how to perform early stopping by modify the early stopping value. The example here
                    # showing how harmonic mean is maximized and use for early stopping
                    train_eval_fine_accuracy, train_eval_coarse_accuracy, train_eval_fine_f1, train_eval_coarse_f1 = \
                        neural_evaluation.evaluate_combined_model(
                            preprocessor=preprocessor,
                            fine_tuner=fine_tuner,
                            loaders=loaders,
                            loss=loss,
                            device=device,
                            split='train_eval')[-5:-1]
                    train_eval_mean_accuracy = (train_eval_fine_accuracy + train_eval_coarse_accuracy) / 2
                    train_eval_mean_f1 = (train_eval_fine_f1 + train_eval_coarse_f1) / 2

                    # Negative early stopping value means we are maximizing the value, and positive value
                    # have an opposite meaning
                    early_stopping_value = - 2 / (1 / train_eval_mean_accuracy + 1 / train_eval_mean_f1)

                    print(f'The current value use for early stopping is {utils.red_text(early_stopping_value)}')

                    if not len(early_stopping_value_list) or \
                            (len(early_stopping_value_list) and early_stopping_value < min(early_stopping_value_list)):
                        print(utils.green_text(
                            f'The last value is lower than previous ones. Updating the best fine tuner'))
                        best_fine_tuner = copy.deepcopy(fine_tuner)

                    if len(early_stopping_value_list) and early_stopping_value >= min(early_stopping_value_list):
                        consecutive_epochs_with_no_train_eval_loss_decrease_from_the_minimum += 1
                    else:
                        consecutive_epochs_with_no_train_eval_loss_decrease_from_the_minimum = 0

                    if consecutive_epochs_with_no_train_eval_loss_decrease_from_the_minimum == 6:
                        print(utils.red_text(f'finish training, stop criteria met!!!'))
                        break

                    early_stopping_value_list += [early_stopping_value]

        if not evaluate_on_test_between_epochs:
            evaluate_on_test(data_str=data_str,
                             model_name=model_name,
                             lr=lr,
                             fine_tuner=best_fine_tuner,
                             loss=loss,
                             num_epochs=self.num_ltn_epochs,
                             save_files=False)

        torch.save(best_fine_tuner.state_dict(),
                   f"models/{data_str}_{best_fine_tuner}_lr{lr}_{loss}_e{self.num_ltn_epochs - 1}_beta{beta}.pth")

        # save prediction file
        neural_evaluation.run_combined_evaluating_pipeline(data_str=data_str,
                                                           model_name=model_name,
                                                           split='train',
                                                           lr=lr,
                                                           loss=loss,
                                                           pretrained_fine_tuner=best_fine_tuner,
                                                           num_epochs=self.num_ltn_epochs,
                                                           print_results=True,
                                                           save_files=True,
                                                           additional_info=additional_info)
        neural_evaluation.run_combined_evaluating_pipeline(data_str=data_str,
                                                           model_name=model_name,
                                                           split='test',
                                                           lr=lr,
                                                           loss=loss,
                                                           pretrained_fine_tuner=best_fine_tuner,
                                                           num_epochs=self.num_ltn_epochs,
                                                           print_results=True,
                                                           save_files=True,
                                                           additional_info=additional_info)
        print('#' * 100)


if __name__ == '__main__':
    data_str = 'military_vehicles'
    epsilon = 0.1

    main_model_name = 'vit_b_16'
    main_lr = 0.0001
    original_num_epochs = 20

    secondary_model_name = 'vit_l_16'
    secondary_model_loss = 'BCE'
    secondary_num_epochs = 20

    binary_num_epochs = 10
    binary_lr = 0.0001
    binary_model_name = 'vit_b_16'

    # data_str = 'imagenet'
    # epsilon = 0.1
    #
    # main_model_name = 'dinov2_vits14'
    # main_lr = 0.000001
    # original_num_epochs = 8
    #
    # secondary_model_name = 'dinov2_vitl14'
    # secondary_model_loss = 'BCE'
    # secondary_num_epochs = 2
    #
    # binary_num_epochs = 5
    # binary_lr = 0.000001
    # binary_model_name = 'dinov2_vits14'

    binary_l_strs = list({f.split(f'e{binary_num_epochs - 1}_')[-1].replace('.npy', '')
                          for f in os.listdir('../../../../binary_results')
                          if f.startswith(f'{data_str}_{main_model_name}')
                          })

    edcr = EDCR_LTN_experiment(data_str=data_str,
                               epsilon=epsilon,
                               main_model_name=main_model_name,
                               combined=True,
                               loss='BCE',
                               lr=main_lr,
                               original_num_epochs=original_num_epochs,
                               secondary_model_name=secondary_model_name,
                               secondary_model_loss=secondary_model_loss,
                               secondary_num_epochs=secondary_num_epochs,
                               binary_l_strs=binary_l_strs,
                               binary_lr=binary_lr,
                               binary_num_epochs=binary_num_epochs,
                               binary_model_name=binary_model_name)
    edcr.run_learning_pipeline()
    edcr.fine_tune_and_evaluate_combined_model(
        additional_info=f"LTN{'_binary' if config.use_binary_model else ''}"
                        f"{'_secondary' if config.use_secondary_model else ''}")
