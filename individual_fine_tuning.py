import os
import numpy as np
import torch
import torch.utils.data
import typing

import models
import utils
import context_handlers
import neural_evaluation
import neural_metrics
import vit_pipeline
import neural_fine_tuning


def fine_tune_individual_models(fine_tuners: list[models.FineTuner],
                                devices: list[torch.device],
                                loaders: dict[str, torch.utils.data.DataLoader],
                                num_epochs: int,
                                fine_lr: float = 1e-4,
                                coarse_lr: float = 1e-4,
                                save_files: bool = True,
                                debug: bool = False):
    fine_fine_tuner, coarse_fine_tuner = fine_tuners
    device_1, device_2 = devices
    fine_fine_tuner.to(device_1)
    fine_fine_tuner.train()

    coarse_fine_tuner.to(device_2)
    coarse_fine_tuner.train()

    train_loader = loaders['train']
    num_batches = len(train_loader)
    criterion = torch.nn.CrossEntropyLoss()

    fine_optimizer = torch.optim.Adam(params=fine_fine_tuner.parameters(),
                                      lr=fine_lr)
    coarse_optimizer = torch.optim.Adam(params=coarse_fine_tuner.parameters(),
                                        lr=coarse_lr)

    # fine_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=fine_optimizer,
    #                                                  step_size=scheduler_step_size,
    #                                                  gamma=scheduler_gamma)
    # coarse_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=coarse_optimizer,
    #                                                    step_size=scheduler_step_size,
    #                                                    gamma=scheduler_gamma)

    train_fine_losses = []
    train_fine_accuracies = []

    test_true_fine_data = []
    test_fine_accuracies = []

    train_coarse_losses = []
    train_coarse_accuracies = []

    test_true_coarse_data = []
    test_coarse_accuracies = []

    print(f'Started fine-tuning individual models with fine_lr={fine_lr} and coarse_lr={coarse_lr}'
          f'for {num_epochs} epochs on {device_1} and {device_2}...')

    for epoch in range(num_epochs):
        print(f"Current fine lr={fine_optimizer.param_groups[0]['lr']}")
        print(f"Current coarse lr={coarse_optimizer.param_groups[0]['lr']}")

        with context_handlers.TimeWrapper():
            running_fine_loss = 0.0
            running_coarse_loss = 0.0

            train_fine_predictions = []
            train_coarse_predictions = []

            train_fine_ground_truths = []
            train_coarse_ground_truths = []

            batches = neural_fine_tuning.get_fine_tuning_batches(train_loader=train_loader,
                                                                 num_batches=num_batches,
                                                                 debug=debug)

            for batch_num, batch in batches:
                with context_handlers.ClearCache(device=device_1):
                    with context_handlers.ClearCache(device=device_2):
                        X, Y_true_fine, Y_true_coarse = batch[0], batch[1].to(device_1), batch[3].to(device_2)

                        fine_optimizer.zero_grad()
                        coarse_optimizer.zero_grad()

                        Y_pred_fine = fine_fine_tuner(X.to(device_1))
                        Y_pred_coarse = coarse_fine_tuner(X.to(device_2))

                        batch_fine_grain_loss = criterion(Y_pred_fine, Y_true_fine)
                        batch_coarse_grain_loss = criterion(Y_pred_coarse, Y_true_coarse)

                        batch_fine_grain_loss.backward()
                        batch_coarse_grain_loss.backward()

                        fine_optimizer.step()
                        coarse_optimizer.step()

                        running_fine_loss += batch_fine_grain_loss.item()
                        running_coarse_loss += batch_coarse_grain_loss.item()

                        predicted_fine = torch.max(Y_pred_fine, 1)[1]
                        predicted_coarse = torch.max(Y_pred_coarse, 1)[1]

                        train_fine_ground_truths += Y_true_fine.tolist()
                        train_coarse_ground_truths += Y_true_coarse.tolist()

                        train_fine_predictions += predicted_fine.tolist()
                        train_coarse_predictions += predicted_coarse.tolist()

                        del X, Y_true_fine, Y_true_coarse

                        neural_metrics.print_post_batch_metrics(batch_num=batch_num,
                                                                num_batches=num_batches,
                                                                batch_fine_grain_loss=batch_fine_grain_loss.item(),
                                                                batch_coarse_grain_loss=batch_coarse_grain_loss.item())

            true_fine_labels = np.array(train_fine_ground_truths)
            true_coarse_labels = np.array(train_coarse_ground_truths)

            predicted_fine_labels = np.array(train_fine_predictions)
            predicted_coarse_labels = np.array(train_coarse_predictions)

            training_fine_accuracy, training_coarse_accuracy = (
                neural_metrics.get_and_print_post_epoch_metrics(epoch=epoch,
                                                                num_epochs=num_epochs,
                                                                # running_fine_loss=running_fine_loss,
                                                                # running_coarse_loss=running_coarse_loss,
                                                                # num_batches=num_batches,
                                                                train_fine_ground_truth=true_fine_labels,
                                                                train_fine_prediction=predicted_fine_labels,
                                                                train_coarse_ground_truth=true_coarse_labels,
                                                                train_coarse_prediction=predicted_coarse_labels))

            train_fine_accuracies += [training_fine_accuracy]
            train_coarse_accuracies += [training_coarse_accuracy]

            train_fine_losses += [running_fine_loss / num_batches]
            train_coarse_losses += [running_coarse_loss / num_batches]

            # fine_scheduler.step()
            # coarse_scheduler.step()

            (test_true_fine_data, test_true_coarse_data, test_pred_fine_data, test_pred_coarse_data,
             test_fine_accuracy, test_coarse_accuracy) = (
                neural_evaluation.evaluate_individual_models(fine_tuners=fine_tuners,
                                                             loaders=loaders,
                                                             devices=devices,
                                                             test=True))

            test_fine_accuracies += [test_fine_accuracy]
            test_coarse_accuracies += [test_coarse_accuracy]
            print('#' * 100)

            np.save(f"{vit_pipeline.individual_results_path}{fine_fine_tuner}"
                    f"_test_pred_lr{fine_lr}_e{epoch}_fine_individual.npy",
                    test_pred_fine_data)
            np.save(f"{vit_pipeline.individual_results_path}{coarse_fine_tuner}"
                    f"_test_pred_lr{coarse_lr}_e{epoch}_coarse_individual.npy",
                    test_pred_coarse_data)

            if save_files:
                vit_pipeline.save_prediction_files(test=True,
                                                   fine_tuners={'fine': fine_fine_tuner,
                                                                'coarse': coarse_fine_tuner},
                                                   combined=False,
                                                   lrs={'fine': fine_lr,
                                                        'coarse': coarse_lr},
                                                   epoch=epoch,
                                                   fine_prediction=test_pred_fine_data,
                                                   coarse_prediction=test_pred_coarse_data)

    torch.save(fine_fine_tuner.state_dict(), f"{fine_fine_tuner}_lr{fine_lr}_fine_individual.pth")
    torch.save(coarse_fine_tuner.state_dict(), f"{coarse_fine_tuner}_lr{coarse_lr}_coarse_individual.pth")

    if not os.path.exists(f"{vit_pipeline.individual_results_path}test_true_fine_individual.npy"):
        np.save(f"{vit_pipeline.individual_results_path}test_true_fine_individual.npy", test_true_fine_data)
    if not os.path.exists(f"{vit_pipeline.individual_results_path}test_true_coarse_individual.npy"):
        np.save(f"{vit_pipeline.individual_results_path}test_true_coarse_individual.npy", test_true_coarse_data)


def run_individual_fine_tuning_pipeline(vit_model_names: list[str],
                                        lrs: list[typing.Union[str, float]],
                                        num_epochs: int,
                                        save_files: bool = True,
                                        debug: bool = utils.is_debug_mode()):
    fine_tuners, loaders, devices = (
        vit_pipeline.initiate(model_names=vit_model_names,
                              lrs=lrs,
                              combined=False,
                              debug=debug))

    for fine_tuner in fine_tuners:
        print(f'Initiating {fine_tuner}')

        with context_handlers.ClearSession():
            fine_tune_individual_models(fine_tuners=fine_tuners,
                                        devices=devices,
                                        loaders=loaders,
                                        num_epochs=num_epochs,
                                        save_files=save_files)
            print('#' * 100)
