import copy
import os

import utils

if utils.is_local():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import numpy as np
import torch
import torch.utils.data
import typing
from tqdm import tqdm

import data_preprocessing
import models
import context_handlers
import neural_evaluation
import neural_metrics
import backbone_pipeline
import neural_fine_tuning


def evaluate_on_test(data_str: str,
                     model_name: str,
                     preprocessor: data_preprocessing.DataPreprocessor,
                     lr: typing.Union[str, float],
                     fine_tuner: models.FineTuner,
                     device: torch.device,
                     loaders: typing.Dict[str, torch.utils.data.DataLoader],
                     loss: str,
                     num_epochs: int,
                     save_files: bool = False):
    if loss == "error_BCE":
        _, _, test_accuracy, test_f1 = neural_evaluation.evaluate_binary_model(fine_tuner=fine_tuner,
                                                                               loaders=loaders,
                                                                               loss=loss,
                                                                               device=device,
                                                                               split='test',
                                                                               preprocessor=preprocessor)
        # test_harmonic_mean = 2 / (1 / test_accuracy + 1 / test_f1)
        print(utils.blue_text(f'test f1: {test_f1}, test accuracy: {test_accuracy}'))
        # print(utils.blue_text(f'harmonic mean of test: {test_harmonic_mean}'))
    else:
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


def fine_tune_combined_model(data_str: str,
                             model_name: str,
                             preprocessor: data_preprocessing.DataPreprocessor,
                             lr: typing.Union[str, float],
                             fine_tuner: models.FineTuner,
                             device: torch.device,
                             loaders: typing.Dict[str, torch.utils.data.DataLoader],
                             loss: str,
                             num_epochs: int,
                             beta: float = 0.1,
                             save_files: bool = True,
                             evaluate_on_test_between_epochs: bool = True,
                             early_stopping: bool = False,
                             additional_model: bool = False,
                             save_ground_truth: bool = False):
    fine_tuner.to(device)
    fine_tuner.train()
    train_loader = loaders['train']
    num_batches = len(train_loader)

    total_train_fine_predictions = None
    total_train_coarse_predictions = None

    optimizer = torch.optim.Adam(params=fine_tuner.parameters(),
                                 lr=lr)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
    #                                             step_size=scheduler_step_size,
    #                                             gamma=scheduler_gamma)

    alpha = preprocessor.num_fine_grain_classes / (preprocessor.num_fine_grain_classes +
                                                   preprocessor.num_coarse_grain_classes)

    test_fine_ground_truths = []
    test_coarse_ground_truths = []

    train_eval_losses = [0]
    best_fine_tuner = copy.deepcopy(fine_tuner)

    if loss.split('_')[0] == 'LTN':
        import ltn
        import ltn_support

        num_epochs = backbone_pipeline.ltn_num_epochs
        logits_to_predicate = ltn.Predicate(ltn_support.LogitsToPredicate()).to(ltn.device)

    neural_fine_tuning.print_fine_tuning_initialization(fine_tuner=fine_tuner,
                                                        num_epochs=num_epochs,
                                                        lr=lr,
                                                        device=device,
                                                        early_stopping=early_stopping)
    print('#' * 100 + '\n')

    consecutive_epochs_with_no_train_eval_loss_decrease = 0

    for epoch in range(num_epochs):
        # print(f"Current lr={optimizer.param_groups[0]['lr']}")

        with context_handlers.TimeWrapper():
            total_running_loss = torch.Tensor([0.0]).to(device)
            running_fine_loss = torch.Tensor([0.0]).to(device)
            running_coarse_loss = torch.Tensor([0.0]).to(device)

            total_train_fine_predictions = []
            total_train_coarse_predictions = []

            total_train_fine_ground_truths = []
            total_train_coarse_ground_truths = []

            error_predictions = []
            error_ground_truths = []

            batches = tqdm(enumerate(train_loader, 0),
                           total=num_batches)

            for batch_num, batch in batches:
                with context_handlers.ClearCache(device=device):
                    if loss == "error_BCE":
                        X, Y_pred_fine, Y_pred_coarse, E_true = [b.to(device) for b in batch]

                        Y_pred_fine_one_hot = torch.nn.functional.one_hot(Y_pred_fine, num_classes=len(
                            preprocessor.fine_grain_classes_str))
                        Y_pred_coarse_one_hot = torch.nn.functional.one_hot(Y_pred_coarse, num_classes=len(
                            preprocessor.coarse_grain_classes_str))

                        Y_pred = torch.cat(tensors=[Y_pred_fine_one_hot, Y_pred_coarse_one_hot], dim=1).float()

                        E_pred = fine_tuner(X, Y_pred)
                        criterion = torch.nn.BCEWithLogitsLoss()
                        batch_total_loss = criterion(E_pred, E_true.float())

                        error_predictions += torch.where(E_pred > 0.5, 1, 0).tolist()
                        error_ground_truths += E_true.tolist()

                        del X, Y_pred_fine, Y_pred_coarse, E_true
                    else:
                        X, Y_true_fine, Y_true_coarse = [batch[i].to(device) for i in [0, 1, 3]]
                        Y_true_fine_one_hot = torch.nn.functional.one_hot(Y_true_fine, num_classes=len(
                            preprocessor.fine_grain_classes_str))
                        Y_true_coarse_one_hot = torch.nn.functional.one_hot(Y_true_coarse, num_classes=len(
                            preprocessor.coarse_grain_classes_str))

                        Y_true = torch.cat(tensors=[Y_true_fine_one_hot, Y_true_coarse_one_hot], dim=1).float()
                        optimizer.zero_grad()

                        Y_pred = fine_tuner(X)
                        Y_pred_fine = Y_pred[:, :preprocessor.num_fine_grain_classes]
                        Y_pred_coarse = Y_pred[:, preprocessor.num_fine_grain_classes:]

                        if loss == "weighted":
                            criterion = torch.nn.CrossEntropyLoss()

                            batch_fine_grain_loss = criterion(Y_pred_fine, Y_true_fine)
                            batch_coarse_grain_loss = criterion(Y_pred_coarse, Y_true_coarse)

                            running_fine_loss += batch_fine_grain_loss
                            running_coarse_loss += batch_coarse_grain_loss

                            batch_total_loss = alpha * batch_fine_grain_loss + (1 - alpha) * batch_coarse_grain_loss

                        elif loss == "BCE":
                            criterion = torch.nn.BCEWithLogitsLoss()
                            batch_total_loss = criterion(Y_pred, Y_true)

                        elif loss == "CE":
                            criterion = torch.nn.CrossEntropyLoss()
                            batch_total_loss = criterion(Y_pred, Y_true)

                        elif loss == "soft_marginal":
                            criterion = torch.nn.MultiLabelSoftMarginLoss()

                            batch_total_loss = criterion(Y_pred, Y_true)

                        elif loss.split('_')[0] == 'LTN':
                            if loss == 'LTN_BCE':
                                criterion = torch.nn.BCEWithLogitsLoss()
                                sat_agg = ltn_support.compute_sat_normally(logits_to_predicate,
                                                                           Y_pred, Y_true_coarse, Y_true_fine)
                                batch_total_loss = beta * (1. - sat_agg) + (1 - beta) * criterion(Y_pred, Y_true)

                            if loss == "LTN_soft_marginal":
                                criterion = torch.nn.MultiLabelSoftMarginLoss()

                                sat_agg = ltn_support.compute_sat_normally(logits_to_predicate,
                                                                           Y_pred, Y_true_coarse, Y_true_fine)
                                batch_total_loss = beta * (1. - sat_agg) + (1 - beta) * (criterion(Y_pred, Y_true))

                        current_train_fine_predictions = torch.max(Y_pred_fine, 1)[1]
                        current_train_coarse_predictions = torch.max(Y_pred_coarse, 1)[1]

                        total_train_fine_predictions += current_train_fine_predictions.tolist()
                        total_train_coarse_predictions += current_train_coarse_predictions.tolist()

                        total_train_fine_ground_truths += Y_true_fine.tolist()
                        total_train_coarse_ground_truths += Y_true_coarse.tolist()

                        del X, Y_true_fine, Y_true_coarse, Y_pred, Y_pred_fine, Y_pred_coarse

                    total_running_loss += batch_total_loss.item()

                    if batch_num > 4 and batch_num % 5 == 0:
                        if loss == "error_BCE":
                            neural_metrics.print_post_batch_binary_metrics(batch_num=batch_num,
                                                                           num_batches=num_batches,
                                                                           train_predictions=error_predictions,
                                                                           train_ground_truths=error_ground_truths,
                                                                           batch_total_loss=batch_total_loss.item(), )
                        else:
                            neural_metrics.get_and_print_post_metrics(preprocessor=preprocessor,
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

            # if epoch == 0:
            #     print(utils.blue_text(
            #         f'label use and count: {np.unique(np.array(error_ground_truths), return_counts=True)}'))

            if loss == "error_BCE":
                neural_metrics.get_and_print_post_epoch_binary_metrics(
                    epoch=epoch,
                    num_epochs=num_epochs,
                    train_predictions=error_predictions,
                    train_ground_truths=error_ground_truths,
                    total_running_loss=total_running_loss.item()
                )
            else:
                neural_metrics.get_and_print_post_metrics(preprocessor=preprocessor,
                                                          curr_epoch=epoch,
                                                          total_num_epochs=num_epochs,
                                                          train_fine_ground_truth=np.array(
                                                              total_train_fine_ground_truths),
                                                          train_fine_prediction=np.array(
                                                              total_train_fine_predictions),
                                                          train_coarse_ground_truth=np.array(
                                                              total_train_coarse_ground_truths),
                                                          train_coarse_prediction=np.array(
                                                              total_train_coarse_predictions))

            if evaluate_on_test_between_epochs:
                evaluate_on_test(data_str=data_str,
                                 model_name=model_name,
                                 preprocessor=preprocessor,
                                 lr=lr,
                                 fine_tuner=best_fine_tuner,
                                 device=device,
                                 loaders=loaders,
                                 loss=loss,
                                 num_epochs=num_epochs,
                                 save_files=save_files)

            if early_stopping:
                if loss == "error_BCE":
                    _, _, train_eval_accuracy, train_eval_f1 = neural_evaluation.evaluate_binary_model(
                        fine_tuner=fine_tuner,
                        loaders=loaders,
                        loss=loss,
                        device=device,
                        split='train_eval',
                        preprocessor=preprocessor)
                    train_eval_harmonic_mean = 2 / (1 / train_eval_accuracy + 1 / train_eval_f1)
                    print(utils.blue_text(f'harmonic mean of train eval: {train_eval_harmonic_mean}'))
                    current_stopping_criterion_value = train_eval_harmonic_mean

                else:
                    curr_train_eval_loss = neural_evaluation.evaluate_combined_model(preprocessor=preprocessor,
                                                                                     fine_tuner=fine_tuner,
                                                                                     loaders=loaders,
                                                                                     loss=loss,
                                                                                     device=device,
                                                                                     split='train_eval')[-1]

                print(f'The current train eval loss is {utils.red_text(curr_train_eval_loss)}')
                if curr_train_eval_loss < min(train_eval_losses):
                    print(utils.green_text(f'The last loss is lower than previous ones. Updating the best fine tuner'))
                    best_fine_tuner = copy.deepcopy(fine_tuner)

                if curr_train_eval_loss >= train_eval_losses[-1]:
                    consecutive_epochs_with_no_train_eval_loss_decrease += 1
                else:
                    consecutive_epochs_with_no_train_eval_loss_decrease = 0

                if consecutive_epochs_with_no_train_eval_loss_decrease == 4:
                    print(utils.red_text(f'finish training, stop criteria met!!!'))
                    break

                train_eval_losses += [curr_train_eval_loss]

    if not evaluate_on_test_between_epochs:
        evaluate_on_test(data_str=data_str,
                         model_name=model_name,
                         preprocessor=preprocessor,
                         lr=lr,
                         fine_tuner=best_fine_tuner,
                         device=device,
                         loaders=loaders,
                         loss=loss,
                         num_epochs=num_epochs,
                         save_files=save_files)

    additional_str = 'additional' if additional_model else ''

    if loss == "error_BCE":
        torch.save(best_fine_tuner.state_dict(),
                   f"models/binary_models/binary_error_{best_fine_tuner}_"
                   f"lr{lr}_loss_{loss}_e{num_epochs}_{additional_str}.pth")
    elif loss.split('_')[0] == 'LTN':
        torch.save(best_fine_tuner.state_dict(), f"models/{best_fine_tuner}_lr{lr}_{loss}_beta{beta}.pth")
    else:
        if not os.path.isdir(f'models'):
            os.mkdir(f'models')

        torch.save(best_fine_tuner.state_dict(),
                   f"models/{data_str}_{best_fine_tuner}_lr{lr}_{loss}_e{num_epochs}_{additional_str}.pth")

    print('#' * 100)

    if save_ground_truth:
        if not os.path.exists(f"{backbone_pipeline.combined_results_path}test_fine_true.npy"):
            np.save(f"{backbone_pipeline.combined_results_path}test_fine_true.npy", test_fine_ground_truths)
        if not os.path.exists(f"{backbone_pipeline.combined_results_path}test_coarse_true.npy"):
            np.save(f"{backbone_pipeline.combined_results_path}test_coarse_true.npy", test_coarse_ground_truths)

    return total_train_fine_predictions, total_train_coarse_predictions


def run_combined_fine_tuning_pipeline(data_str: str,
                                      model_name: str,
                                      lr: typing.Union[str, float],
                                      num_epochs: int,
                                      loss: str = 'BCE',
                                      pretrained_path: str = None,
                                      save_files: bool = True,
                                      debug: bool = utils.is_debug_mode(),
                                      additional_model: bool = False,
                                      evaluate_on_test_between_epochs: bool = True,
                                      evaluate_train_eval: bool = True):
    preprocessor, fine_tuners, loaders, devices = (
        backbone_pipeline.initiate(data_str=data_str,
                                   model_name=model_name,
                                   lr=lr,
                                   combined=True,
                                   pretrained_path=pretrained_path,
                                   train_eval_split=0.8 if evaluate_train_eval else None,
                                   debug=debug)
    )
    for fine_tuner in fine_tuners:
        fine_tune_combined_model(
            data_str=data_str,
            model_name=model_name,
            preprocessor=preprocessor,
            lr=lr,
            fine_tuner=fine_tuner,
            device=devices[0],
            loaders=loaders,
            loss=loss,
            num_epochs=num_epochs,
            save_files=save_files,
            additional_model=additional_model,
            early_stopping=evaluate_train_eval,
            evaluate_on_test_between_epochs=evaluate_on_test_between_epochs
        )
        print('#' * 100)


if __name__ == '__main__':
    run_combined_fine_tuning_pipeline(data_str='military_vehicles',
                                      model_name='vit_b_16',
                                      lr=0.0001,
                                      num_epochs=50,
                                      loss='BCE',
                                      additional_model=True,
                                      evaluate_on_test_between_epochs=False)
