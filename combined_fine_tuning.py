import os
import numpy as np
import torch
import torch.utils.data
import typing

import data_preprocessing
import models
import utils
import context_handlers
import neural_evaluation
import neural_metrics
import vit_pipeline
import neural_fine_tuning


def fine_tune_combined_model(preprocessor: data_preprocessing.DataPreprocessor,
                             lrs: list[typing.Union[str, float]],
                             fine_tuner: models.FineTuner,
                             device: torch.device,
                             loaders: dict[str, torch.utils.data.DataLoader],
                             loss: str,
                             num_epochs: int,
                             beta: float = 0.1,
                             save_files: bool = True,
                             debug: bool = False,
                             evaluate_on_test: bool = True,
                             evaluate_on_train_eval: bool = False):
    fine_tuner.to(device)
    fine_tuner.train()
    train_loader = loaders['train']
    num_batches = len(train_loader)

    train_fine_predictions = None
    train_coarse_predictions = None

    for lr in lrs:
        optimizer = torch.optim.Adam(params=fine_tuner.parameters(),
                                     lr=lr)

        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
        #                                             step_size=scheduler_step_size,
        #                                             gamma=scheduler_gamma)

        alpha = preprocessor.num_fine_grain_classes / (preprocessor.num_fine_grain_classes +
                                                       preprocessor.num_coarse_grain_classes)

        train_total_losses = []
        train_fine_losses = []
        train_coarse_losses = []

        train_fine_accuracies = []
        train_coarse_accuracies = []

        test_fine_ground_truths = []
        test_coarse_ground_truths = []

        test_fine_accuracies = []
        test_coarse_accuracies = []

        train_eval_fine_accuracy, train_eval_coarse_accuracy = None, None

        if loss.split('_')[0] == 'LTN':
            import ltn
            import ltn_support

            epochs = vit_pipeline.ltn_num_epochs
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

            with context_handlers.TimeWrapper():
                total_running_loss = torch.Tensor([0.0]).to(device)
                running_fine_loss = torch.Tensor([0.0]).to(device)
                running_coarse_loss = torch.Tensor([0.0]).to(device)

                train_fine_predictions = []
                train_coarse_predictions = []

                train_fine_ground_truths = []
                train_coarse_ground_truths = []

                batches = neural_fine_tuning.get_fine_tuning_batches(train_loader=train_loader,
                                                                     num_batches=num_batches,
                                                                     debug=debug)

                for batch_num, batch in batches:
                    with context_handlers.ClearCache(device=device):
                        X, Y_fine_grain, Y_coarse_grain = batch[0].to(device), batch[1].to(device), batch[3].to(device)
                        Y_fine_grain_one_hot = torch.nn.functional.one_hot(Y_fine_grain, num_classes=len(
                            preprocessor.fine_grain_classes_str))
                        Y_coarse_grain_one_hot = torch.nn.functional.one_hot(Y_coarse_grain, num_classes=len(
                            preprocessor.coarse_grain_classes_str))

                        Y_combine = torch.cat(tensors=[Y_fine_grain_one_hot, Y_coarse_grain_one_hot], dim=1).float()
                        optimizer.zero_grad()

                        Y_pred = fine_tuner(X)
                        Y_pred_fine_grain = Y_pred[:, :preprocessor.num_fine_grain_classes]
                        Y_pred_coarse_grain = Y_pred[:, preprocessor.num_fine_grain_classes:]

                        if loss == "weighted":
                            criterion = torch.nn.CrossEntropyLoss()

                            batch_fine_grain_loss = criterion(Y_pred_fine_grain, Y_fine_grain)
                            batch_coarse_grain_loss = criterion(Y_pred_coarse_grain, Y_coarse_grain)

                            running_fine_loss += batch_fine_grain_loss
                            running_coarse_loss += batch_coarse_grain_loss

                            batch_total_loss = alpha * batch_fine_grain_loss + (1 - alpha) * batch_coarse_grain_loss

                        elif loss == "BCE":
                            criterion = torch.nn.BCEWithLogitsLoss()
                            batch_total_loss = criterion(Y_pred, Y_combine)

                        elif loss == "CE":
                            criterion = torch.nn.CrossEntropyLoss()
                            batch_total_loss = criterion(Y_pred, Y_combine)

                        elif loss == "soft_marginal":
                            criterion = torch.nn.MultiLabelSoftMarginLoss()

                            batch_total_loss = criterion(Y_pred, Y_combine)

                        elif loss.split('_')[0] == 'LTN':
                            if loss == 'LTN_BCE':
                                criterion = torch.nn.BCEWithLogitsLoss()

                                sat_agg = ltn_support.compute_sat_normally(logits_to_predicate,
                                                                           Y_pred, Y_coarse_grain, Y_fine_grain)
                                batch_total_loss = beta * (1. - sat_agg) + (1 - beta) * criterion(Y_pred, Y_combine)

                            if loss == "LTN_soft_marginal":
                                criterion = torch.nn.MultiLabelSoftMarginLoss()

                                sat_agg = ltn_support.compute_sat_normally(logits_to_predicate,
                                                                           Y_pred, Y_coarse_grain, Y_fine_grain)
                                batch_total_loss = beta * (1. - sat_agg) + (1 - beta) * (criterion(Y_pred, Y_combine))

                        neural_metrics.print_post_batch_metrics(batch_num=batch_num,
                                                                num_batches=num_batches,
                                                                batch_total_loss=batch_total_loss.item())

                        batch_total_loss.backward()
                        optimizer.step()

                        total_running_loss += batch_total_loss.item()

                        predicted_fine = torch.max(Y_pred_fine_grain, 1)[1]
                        predicted_coarse = torch.max(Y_pred_coarse_grain, 1)[1]

                        train_fine_predictions += predicted_fine.tolist()
                        train_coarse_predictions += predicted_coarse.tolist()

                        train_fine_ground_truths += Y_fine_grain.tolist()
                        train_coarse_ground_truths += Y_coarse_grain.tolist()

                        del X, Y_fine_grain, Y_coarse_grain, Y_pred, Y_pred_fine_grain, Y_pred_coarse_grain

                training_fine_accuracy, training_coarse_accuracy = (
                    neural_metrics.get_and_print_post_epoch_metrics(epoch=epoch,
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

                train_fine_accuracies += [training_fine_accuracy]
                train_coarse_accuracies += [training_coarse_accuracy]

                train_total_losses += [total_running_loss.item() / num_batches]
                train_fine_losses += [running_fine_loss.item() / num_batches]
                train_coarse_losses += [running_coarse_loss.item() / num_batches]

                # scheduler.step()

                if evaluate_on_test:
                    (test_fine_ground_truths, test_coarse_ground_truths, test_fine_predictions, test_coarse_predictions,
                     test_fine_lower_predictions, test_coarse_lower_predictions, test_fine_accuracy,
                     test_coarse_accuracy) = (
                        neural_evaluation.evaluate_combined_model(fine_tuner=fine_tuner,
                                                                  loaders=loaders,
                                                                  loss=loss,
                                                                  device=device,
                                                                  split='test'))

                    test_fine_accuracies += [test_fine_accuracy]
                    test_coarse_accuracies += [test_coarse_accuracy]

                print('#' * 100)

                if (epoch == num_epochs) and save_files:
                    vit_pipeline.save_prediction_files(test=False,
                                                       fine_tuners=fine_tuner,
                                                       combined=True,
                                                       lrs=lr,
                                                       epoch=epoch,
                                                       fine_prediction=test_fine_predictions,
                                                       coarse_prediction=test_coarse_predictions,
                                                       loss=loss)

                if evaluate_on_train_eval:
                    curr_train_eval_fine_accuracy, curr_train_eval_coarse_accuracy = \
                        neural_evaluation.evaluate_combined_model(
                            fine_tuner=fine_tuner,
                            loaders=loaders,
                            loss=loss,
                            device=device,
                            split='train_eval')[-2:]
                    if train_eval_fine_accuracy is not None and train_eval_coarse_accuracy is not None and \
                            curr_train_eval_fine_accuracy < train_eval_fine_accuracy and \
                            curr_train_eval_coarse_accuracy < train_eval_coarse_accuracy:
                        print(utils.red_text('Early stopping!!!'))
                        break

                    train_eval_fine_accuracy = curr_train_eval_fine_accuracy
                    train_eval_coarse_accuracy = curr_train_eval_coarse_accuracy

        if save_files:
            if not os.path.exists(f"{vit_pipeline.combined_results_path}test_fine_true.npy"):
                np.save(f"{vit_pipeline.combined_results_path}test_fine_true.npy", test_fine_ground_truths)
            if not os.path.exists(f"{vit_pipeline.combined_results_path}test_coarse_true.npy"):
                np.save(f"{vit_pipeline.combined_results_path}test_coarse_true.npy", test_coarse_ground_truths)

            if loss.split('_')[0] == 'LTN':
                torch.save(fine_tuner.state_dict(), f"models/{fine_tuner}_lr{lr}_{loss}_beta{beta}.pth")
            else:
                torch.save(fine_tuner.state_dict(), f"models/{fine_tuner}_lr{lr}_{loss}.pth")

        return train_fine_predictions, train_coarse_predictions


def run_combined_fine_tuning_pipeline(data: str,
                                      model_names: list[str],
                                      lrs: list[typing.Union[str, float]],
                                      num_epochs: int,
                                      loss: str = 'BCE',
                                      save_files: bool = True,
                                      debug: bool = utils.is_debug_mode()):
    preprocessor, fine_tuners, loaders, devices, num_fine_grain_classes, num_coarse_grain_classes = (
        vit_pipeline.initiate(data=data,
                              model_names=model_names,
                              lrs=lrs,
                              combined=True,
                              debug=debug))
    for fine_tuner in fine_tuners:
        with context_handlers.ClearSession():
            fine_tune_combined_model(preprocessor=preprocessor,
                                     lrs=lrs,
                                     fine_tuner=fine_tuner,
                                     device=devices[0],
                                     loaders=loaders,
                                     loss=loss,
                                     num_epochs=num_epochs,
                                     save_files=save_files,
                                     debug=debug)
            print('#' * 100)


if __name__ == '__main__':
    run_combined_fine_tuning_pipeline(data='imagenet',
                                      model_names=['dinov2_vits14'],
                                      lrs=[0.000001],
                                      num_epochs=15,
                                      loss='BCE')
