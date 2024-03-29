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


def get_fine_tuning_batches(train_loader: torch.utils.data.DataLoader,
                            num_batches: int,
                            debug: bool = False):
    if utils.is_local():
        from tqdm import tqdm
        batches = tqdm(enumerate([list(train_loader)[0]] if debug else train_loader, 0),
                       total=num_batches)
    else:
        batches = enumerate(train_loader, 0)

    return batches


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

            batches = get_fine_tuning_batches(train_loader=train_loader,
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


def print_fine_tuning_initialization(fine_tuner: models.FineTuner,
                                     num_epochs: int,
                                     lr: float,
                                     device: torch.device):
    print(f'\nFine-tuning {fine_tuner} with {utils.format_integer(len(fine_tuner))} '
          f'parameters for {num_epochs} epochs using lr={lr} on {device}...')


def fine_tune_binary_model(l: data_preprocessing.Label,
                           lrs: list[typing.Union[str, float]],
                           fine_tuner: models.FineTuner,
                           device: torch.device,
                           loaders: dict[str, torch.utils.data.DataLoader],
                           num_epochs: int,
                           weight: list[float],
                           save_files: bool = True,
                           evaluate_on_test: bool = True):
    fine_tuner.to(device)
    fine_tuner.train()
    train_loader = loaders['train']
    num_batches = len(train_loader)
    weight = torch.tensor(weight).float().to(device)
    loss = 'BCE'

    for lr in lrs:
        optimizer = torch.optim.Adam(params=fine_tuner.parameters(),
                                     lr=lr)

        print_fine_tuning_initialization(fine_tuner=fine_tuner,
                                         num_epochs=num_epochs,
                                         lr=lr,
                                         device=device)

        print('#' * 100 + '\n')

        for epoch in range(num_epochs):
            with context_handlers.TimeWrapper():
                total_running_loss = torch.Tensor([0.0]).to(device)

                train_predictions = []
                train_ground_truths = []

                batches = get_fine_tuning_batches(train_loader=train_loader,
                                                  num_batches=num_batches)

                for batch_num, batch in batches:
                    with context_handlers.ClearCache(device=device):
                        X, Y = batch[0].to(device), batch[1].to(device)
                        Y_one_hot = torch.nn.functional.one_hot(Y, num_classes=2).float()
                        optimizer.zero_grad()
                        Y_pred = fine_tuner(X)

                        criterion = torch.nn.BCEWithLogitsLoss(weight=weight)

                        batch_total_loss = criterion(Y_pred, Y_one_hot)

                        neural_metrics.print_post_batch_metrics(batch_num=batch_num,
                                                                num_batches=num_batches,
                                                                batch_total_loss=batch_total_loss.item())

                        batch_total_loss.backward()
                        optimizer.step()

                        total_running_loss += batch_total_loss.item()
                        predicted = torch.max(Y_pred, 1)[1]
                        train_predictions += predicted.tolist()
                        train_ground_truths += Y.tolist()

                        del X, Y, Y_pred

                training_accuracy, training_f1 = neural_metrics.get_and_print_post_epoch_binary_metrics(
                    epoch=epoch,
                    num_epochs=num_epochs,
                    train_predictions=train_predictions,
                    train_ground_truths=train_ground_truths,
                    total_running_loss=total_running_loss.item()
                )

                if evaluate_on_test:
                    test_ground_truths, test_predictions, test_accuracy = (
                        neural_evaluation.evaluate_binary_model(l=l,
                                                                fine_tuner=fine_tuner,
                                                                loaders=loaders,
                                                                loss=loss,
                                                                device=device,
                                                                split='test'))
                print('#' * 100)

        if save_files:
            vit_pipeline.save_binary_prediction_files(test=False,
                                                      fine_tuner=fine_tuner,
                                                      lr=lr,
                                                      epoch=num_epochs,
                                                      l=l,
                                                      predictions=train_predictions,
                                                      ground_truths=train_ground_truths)

        return train_predictions


def fine_tune_combined_model(lrs: list[typing.Union[str, float]],
                             fine_tuner: models.FineTuner,
                             device: torch.device,
                             loaders: dict[str, torch.utils.data.DataLoader],
                             loss: str,
                             num_epochs: int,
                             beta: float = 0.1,
                             save_files: bool = True,
                             debug: bool = False,
                             evaluate_on_test: bool = True,
                             evaluate_on_train_eval: bool = False,
                             Y_original_fine: np.array = None,
                             Y_original_coarse: np.array = None):
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

        alpha = data_preprocessing.num_fine_grain_classes / (data_preprocessing.num_fine_grain_classes +
                                                             data_preprocessing.num_coarse_grain_classes)

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

        print_fine_tuning_initialization(fine_tuner=fine_tuner,
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

                batches = get_fine_tuning_batches(train_loader=train_loader,
                                                  num_batches=num_batches,
                                                  debug=debug)

                for batch_num, batch in batches:
                    with context_handlers.ClearCache(device=device):
                        X, Y_fine_grain, Y_coarse_grain = batch[0].to(device), batch[1].to(device), batch[3].to(device)
                        Y_fine_grain_one_hot = torch.nn.functional.one_hot(Y_fine_grain, num_classes=len(
                            data_preprocessing.fine_grain_classes_str))
                        Y_coarse_grain_one_hot = torch.nn.functional.one_hot(Y_coarse_grain, num_classes=len(
                            data_preprocessing.coarse_grain_classes_str))

                        Y_combine = torch.cat(tensors=[Y_fine_grain_one_hot, Y_coarse_grain_one_hot], dim=1).float()
                        optimizer.zero_grad()

                        Y_pred = fine_tuner(X)
                        Y_pred_fine_grain = Y_pred[:, :data_preprocessing.num_fine_grain_classes]
                        Y_pred_coarse_grain = Y_pred[:, data_preprocessing.num_fine_grain_classes:]

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

                        if batch_total_loss is not None and Y_original_fine is not None:
                            end_index = (batch_num + 1) * vit_pipeline.batch_size if batch_num + 1 < num_batches else \
                                len(Y_original_fine)
                            Y_original_fine_one_hot = torch.nn.functional.one_hot(
                                torch.tensor(Y_original_fine[batch_num *
                                                             vit_pipeline.batch_size:end_index]).to(device),
                                num_classes=len(data_preprocessing.fine_grain_classes_str))
                            Y_original_coarse_one_hot = torch.nn.functional.one_hot(
                                torch.tensor(Y_original_coarse[batch_num *
                                                               vit_pipeline.batch_size:end_index]).to(device),
                                num_classes=len(data_preprocessing.coarse_grain_classes_str))

                            Y_original_combine = torch.cat(tensors=[Y_original_fine_one_hot,
                                                                    Y_original_coarse_one_hot],
                                                           dim=1).float()
                            batch_total_loss -= vit_pipeline.original_prediction_weight * criterion(Y_pred,
                                                                                                    Y_original_combine)

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

                if (epoch == num_epochs - 1) and save_files:
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
                torch.save(fine_tuner.state_dict(), f"{fine_tuner}_lr{lr}_{loss}_beta{beta}.pth")
            else:
                torch.save(fine_tuner.state_dict(), f"{fine_tuner}_lr{lr}_{loss}.pth")

        return train_fine_predictions, train_coarse_predictions


def run_g_binary_fine_tuning_pipeline(vit_model_names: list[str],
                                      g: data_preprocessing.Granularity,
                                      lr: float,
                                      num_epochs: int,
                                      save_files: bool = True):
    for l in data_preprocessing.get_labels(g=g).values():
        if not os.path.exists(f"{os.getcwd()}/models/binary_models/binary_{l}_vit_b_16_lr{lr}_"
                              f"loss_BCE_e{num_epochs}.pth"):
            fine_tuners, loaders, devices, weight = vit_pipeline.initiate(vit_model_names=vit_model_names,
                                                                          lrs=[lr],
                                                                          l=l)
            for fine_tuner in fine_tuners:
                with context_handlers.ClearSession():
                    fine_tune_binary_model(l=l,
                                           lrs=[lr],
                                           fine_tuner=fine_tuner,
                                           device=devices[0],
                                           loaders=loaders,
                                           num_epochs=num_epochs,
                                           save_files=save_files,
                                           weight=weight)
                    print('#' * 100)
        else:
            print(f'Skipping {l}')


def run_combined_fine_tuning_pipeline(vit_model_names: list[str],
                                      lrs: list[typing.Union[str, float]],
                                      num_epochs: int,
                                      loss: str = 'BCE',
                                      save_files: bool = True,
                                      debug: bool = utils.is_debug_mode()):
    fine_tuners, loaders, devices, num_fine_grain_classes, num_coarse_grain_classes = (
        vit_pipeline.initiate(vit_model_names=vit_model_names,
                              lrs=lrs,
                              combined=True,
                              debug=debug))
    for fine_tuner in fine_tuners:
        with context_handlers.ClearSession():
            fine_tune_combined_model(lrs=lrs,
                                     fine_tuner=fine_tuner,
                                     device=devices[0],
                                     loaders=loaders,
                                     loss=loss,
                                     num_epochs=num_epochs,
                                     save_files=save_files,
                                     debug=debug)
            print('#' * 100)


def run_individual_fine_tuning_pipeline(vit_model_names: list[str],
                                        lrs: list[typing.Union[str, float]],
                                        num_epochs: int,
                                        save_files: bool = True,
                                        debug: bool = utils.is_debug_mode()):
    fine_tuners, loaders, devices = (
        vit_pipeline.initiate(vit_model_names=vit_model_names,
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


if __name__ == '__main__':
    run_combined_fine_tuning_pipeline(vit_model_names=['vit_l_16'],
                                      lrs=[0.0001],
                                      num_epochs=20,
                                      loss='BCE')

    # for g in data_preprocessing.granularities.values():
    #     run_g_binary_fine_tuning_pipeline(g=g,
    #                                       lr=0.0001,
    #                                       num_epochs=10,
    #                                       save_files=True)
