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

save_ground_truth = False


def fine_tune_combined_model(
        data_str: str,
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
        evaluate_on_test: bool = True,
        evaluate_on_train_eval: bool = False,
        additional_model: bool = False):

    fine_tuner.to(device)
    fine_tuner.train()
    train_loader = loaders['train']
    num_batches = len(train_loader)

    train_fine_predictions = None
    train_coarse_predictions = None

    optimizer = torch.optim.Adam(params=fine_tuner.parameters(),
                                 lr=lr)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
    #                                             step_size=scheduler_step_size,
    #                                             gamma=scheduler_gamma)

    alpha = preprocessor.num_fine_grain_classes / (preprocessor.num_fine_grain_classes +
                                                   preprocessor.num_coarse_grain_classes)

    test_fine_ground_truths = []
    test_coarse_ground_truths = []

    slicing_window = [0, 0]
    best_fine_tuner = copy.deepcopy(fine_tuner)
    stopping_citeria = 0

    test_fine_accuracies = []
    test_coarse_accuracies = []

    train_eval_fine_accuracy, train_eval_coarse_accuracy = None, None

    if loss.split('_')[0] == 'LTN':
        import ltn
        import ltn_support

        epochs = backbone_pipeline.ltn_num_epochs
        logits_to_predicate = ltn.Predicate(ltn_support.LogitsToPredicate()).to(ltn.device)
    else:
        epochs = num_epochs

    neural_fine_tuning.print_fine_tuning_initialization(fine_tuner=fine_tuner,
                                                        num_epochs=num_epochs,
                                                        lr=lr,
                                                        device=device)
    print('#' * 100 + '\n')

    for epoch in range(epochs):
        print(f"Current lr={optimizer.param_groups[0]['lr']}")

        with (context_handlers.TimeWrapper()):
            total_running_loss = torch.Tensor([0.0]).to(device)
            running_fine_loss = torch.Tensor([0.0]).to(device)
            running_coarse_loss = torch.Tensor([0.0]).to(device)

            train_fine_predictions = []
            train_coarse_predictions = []

            train_fine_ground_truths = []
            train_coarse_ground_truths = []

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
                        X, Y_true_fine, Y_true_coarse = (batch[0].to(device), batch[1].to(device), batch[3].to(device))
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

                        predicted_fine = torch.max(Y_pred_fine, 1)[1]
                        predicted_coarse = torch.max(Y_pred_coarse, 1)[1]

                        train_fine_predictions += predicted_fine.tolist()
                        train_coarse_predictions += predicted_coarse.tolist()

                        train_fine_ground_truths += Y_true_fine.tolist()
                        train_coarse_ground_truths += Y_true_coarse.tolist()

                        del X, Y_true_fine, Y_true_coarse, Y_pred, Y_pred_fine, Y_pred_coarse

                    total_running_loss += batch_total_loss.item()
                    if loss == "error_BCE":
                        neural_metrics.print_post_batch_binary_metrics(batch_num=batch_num,
                                                                       num_batches=num_batches,
                                                                       train_predictions=error_predictions,
                                                                       train_ground_truths=error_ground_truths,
                                                                       batch_total_loss=batch_total_loss.item(), )
                    else:
                        neural_metrics.print_post_batch_metrics(batch_num=batch_num,
                                                                num_batches=num_batches,
                                                                batch_total_loss=batch_total_loss.item())
                    batch_total_loss.backward()
                    optimizer.step()

            if epochs == 0:
                print(utils.blue_text(
                    f'label use and count: {np.unique(np.array(error_ground_truths), return_counts=True)}'))

            if loss == "error_BCE":
                error_accuracy, error_f1 = neural_metrics.get_and_print_post_epoch_binary_metrics(
                    epoch=epoch,
                    num_epochs=num_epochs,
                    train_predictions=error_predictions,
                    train_ground_truths=error_ground_truths,
                    total_running_loss=total_running_loss.item()
                )
            else:
                training_fine_accuracy, training_coarse_accuracy = (
                    neural_metrics.get_and_print_post_epoch_metrics(preprocessor=preprocessor,
                                                                    epoch=epoch,
                                                                    num_epochs=num_epochs,
                                                                    # running_fine_loss=running_fine_loss.item(),
                                                                    # running_coarse_loss=running_coarse_loss.item(),
                                                                    # num_batches=num_batches,
                                                                    train_fine_ground_truth=np.array(
                                                                        train_fine_ground_truths),
                                                                    train_fine_prediction=np.array(
                                                                        train_fine_predictions),
                                                                    train_coarse_ground_truth=np.array(
                                                                        train_coarse_ground_truths),
                                                                    train_coarse_prediction=np.array(
                                                                        train_coarse_predictions)))

            if evaluate_on_train_eval:
                if loss == "error_BCE":
                    _, _, test_accuracy, test_f1 = neural_evaluation.evaluate_binary_model(fine_tuner=fine_tuner,
                                                                                           loaders=loaders,
                                                                                           loss=loss,
                                                                                           device=device,
                                                                                           split='train_eval',
                                                                                           preprocessor=preprocessor)
                    test_harmonic_mean = 2 / (1 / test_accuracy + 1 / test_f1)
                    print(utils.blue_text(f'harmonic mean of train eval: {test_harmonic_mean}'))
                    stopping_citeria = test_harmonic_mean

                else:
                    curr_train_eval_fine_accuracy, curr_train_eval_coarse_accuracy = \
                        neural_evaluation.evaluate_combined_model(
                            preprocessor=preprocessor,
                            fine_tuner=fine_tuner,
                            loaders=loaders,
                            loss=loss,
                            device=device,
                            split='train_eval')[-2:]
                    # if train_eval_fine_accuracy is not None and train_eval_coarse_accuracy is not None and \
                    #         curr_train_eval_fine_accuracy < train_eval_fine_accuracy and \
                    #         curr_train_eval_coarse_accuracy < train_eval_coarse_accuracy:
                    #     print(utils.red_text('Early stopping!!!'))
                    #     break
                    #
                    # train_eval_fine_accuracy = curr_train_eval_fine_accuracy
                    # train_eval_coarse_accuracy = curr_train_eval_coarse_accuracy
                    stopping_citeria = (curr_train_eval_fine_accuracy + curr_train_eval_coarse_accuracy) / 2

                # Update slicing window, and break if the sum of current sliding window is smaller than previous one:
                if stopping_citeria > slicing_window[1]:
                    print(utils.green_text(f'harmonic mean of current fine_tuner is better. Update fine_tuner'))
                    best_fine_tuner = copy.deepcopy(fine_tuner)
                current_sliding_window = [slicing_window[1], test_harmonic_mean]
                print(f'current sliding window is {current_sliding_window} and previous one is {slicing_window}')
                if sum(slicing_window) > sum(current_sliding_window):
                    print(utils.red_text(f'finish training, stop criteria met!!!'))
                    break
                slicing_window = current_sliding_window

    if evaluate_on_test and loss == "error_BCE":
        _, _, test_accuracy, test_f1 = neural_evaluation.evaluate_binary_model(fine_tuner=fine_tuner,
                                                                               loaders=loaders,
                                                                               loss=loss,
                                                                               device=device,
                                                                               split='test',
                                                                               preprocessor=preprocessor)
        test_harmonic_mean = 2 / (1 / test_accuracy + 1 / test_f1)
        print(utils.blue_text(f'harmonic mean of test: {test_harmonic_mean}'))

    elif evaluate_on_test:
        neural_evaluation.run_combined_evaluating_pipeline(data_str=data_str,
                                                           model_name=model_name,
                                                           split='train',
                                                           lr=lr,
                                                           loss=loss,
                                                           num_epochs=num_epochs,
                                                           print_results=True,
                                                           save_files=save_files)
        neural_evaluation.run_combined_evaluating_pipeline(data_str=data_str,
                                                           model_name=model_name,
                                                           split='test',
                                                           lr=lr,
                                                           loss=loss,
                                                           num_epochs=num_epochs,
                                                           print_results=True,
                                                           save_files=save_files)

        print('#' * 100)

    if loss == "error_BCE":
        torch.save(best_fine_tuner.state_dict(),
                   f"models/binary_models/binary_error_{best_fine_tuner}_"
                   f"lr{lr}_loss_{loss}_e{num_epochs}_{'additional' if additional_model else ''}.pth")
    else:
        if loss.split('_')[0] == 'LTN':
            torch.save(fine_tuner.state_dict(), f"models/{fine_tuner}_lr{lr}_{loss}_beta{beta}.pth")
        else:
            torch.save(fine_tuner.state_dict(), f"models/{fine_tuner}_lr{lr}_{loss}.pth")

    print('#' * 100)

    if save_ground_truth:
        if not os.path.exists(f"{backbone_pipeline.combined_results_path}test_fine_true.npy"):
            np.save(f"{backbone_pipeline.combined_results_path}test_fine_true.npy", test_fine_ground_truths)
        if not os.path.exists(f"{backbone_pipeline.combined_results_path}test_coarse_true.npy"):
            np.save(f"{backbone_pipeline.combined_results_path}test_coarse_true.npy", test_coarse_ground_truths)

    return train_fine_predictions, train_coarse_predictions


def run_combined_fine_tuning_pipeline(data_str: str,
                                      model_name: str,
                                      lr: typing.Union[str, float],
                                      num_epochs: int,
                                      loss: str = 'BCE',
                                      pretrained_path: str = None,
                                      save_files: bool = True,
                                      debug: bool = utils.is_debug_mode(),
                                      additional_model: bool = False,
                                      evaluate_train_eval=True):
    preprocessor, fine_tuners, loaders, devices = (
        backbone_pipeline.initiate(data_str=data_str,
                                   model_name=model_name,
                                   lr=lr,
                                   combined=True,
                                   pretrained_path=pretrained_path,
                                   train_eval_split=0.8 if evaluate_train_eval else None,
                                   debug=debug))
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
            evaluate_on_train_eval=evaluate_train_eval
        )
        print('#' * 100)


if __name__ == '__main__':
    run_combined_fine_tuning_pipeline(data_str='military_vehicles',
                                      model_name='vit_b_16',
                                      lr=0.0001,
                                      num_epochs=10,
                                      loss='BCE',
                                      additional_model=True,
                                      )
