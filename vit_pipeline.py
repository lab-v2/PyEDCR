import os
import torch
import torch.utils.data
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from time import time
import pathlib

import context_handlers
import models
import utils
import data_preprocessing

batch_size = 32
lrs = [1e-4]
scheduler_gamma = 0.1
num_epochs = 10
vit_model_names = [f'vit_{vit_model_name}' for vit_model_name in ['b_16']]

files_path = '/content/drive/My Drive/' if utils.is_running_in_colab() else ''
combined_results_path = fr'{files_path}combined_results/'
individual_results_path = fr'{files_path}individual_results/'
cwd = pathlib.Path(__file__).parent.resolve()
scheduler_step_size = num_epochs


def print_num_inconsistencies(fine_predictions: np.array, coarse_predictions: np.array):
    inconsistencies = 0
    for fine_prediction, coarse_prediction in zip(fine_predictions, coarse_predictions):
        if data_preprocessing.fine_to_course_idx[fine_prediction] != coarse_prediction:
            inconsistencies += 1

    print(
        f'Total prior inconsistencies {inconsistencies}/{len(fine_predictions)} '
        f'which is {round(inconsistencies / len(fine_predictions) * 100, 2)}%')


def get_and_print_post_test_metrics(test_fine_ground_truth: np.array,
                                    test_fine_prediction: np.array,
                                    test_coarse_ground_truth: np.array,
                                    test_coarse_prediction: np.array,
                                    num_fine_grain_classes: int,
                                    num_coarse_grain_classes: int):
    test_fine_accuracy = accuracy_score(y_true=test_fine_ground_truth,
                                        y_pred=test_fine_prediction)
    test_coarse_accuracy = accuracy_score(y_true=test_coarse_ground_truth,
                                          y_pred=test_coarse_prediction)
    test_fine_f1 = f1_score(y_true=test_fine_ground_truth,
                            y_pred=test_fine_prediction,
                            labels=range(num_fine_grain_classes),
                            average='macro')
    test_coarse_f1 = f1_score(y_true=test_fine_ground_truth,
                              y_pred=test_fine_prediction,
                              labels=range(num_coarse_grain_classes),
                              average='macro')

    print(f'\nTest fine accuracy: {round(test_fine_accuracy * 100, 2)}%'
          f', fine f1: {round(test_fine_f1 * 100, 2)}%'
          f'\nTest coarse accuracy: {round(test_coarse_accuracy * 100, 2)}%'
          f', coarse f1: {round(test_coarse_f1 * 100, 2)}%\n')

    print_num_inconsistencies(fine_predictions=test_fine_prediction, coarse_predictions=test_coarse_prediction)

    return test_fine_accuracy, test_coarse_accuracy


def test_individual_models(fine_tuners: list[models.FineTuner],
                           loaders: dict[str, torch.utils.data.DataLoader],
                           devices: list[torch.device],
                           num_fine_grain_classes: int,
                           num_coarse_grain_classes: int
                           ) -> (list[int], list[int], float):
    test_loader = loaders[f'test']
    fine_fine_tuner, coarse_fine_tuner = fine_tuners

    device_1, device_2 = devices
    fine_fine_tuner.to(device_1)
    coarse_fine_tuner.to(device_2)

    fine_fine_tuner.eval()
    coarse_fine_tuner.eval()

    test_fine_prediction = []
    test_coarse_prediction = []

    test_fine_ground_truth = []
    test_coarse_ground_truth = []

    name_list = []

    print(f'Started testing...')

    with torch.no_grad():
        if utils.is_local():
            from tqdm import tqdm
            gen = tqdm(enumerate(test_loader), total=len(test_loader))
        else:
            gen = enumerate(test_loader)

        for i, data in gen:
            X, Y_fine_grain, names, Y_coarse_grain = data[0], data[1].to(device_1), data[2], data[3].to(device_2)

            Y_pred_fine_grain = fine_fine_tuner(X.to(device_1))
            Y_pred_coarse_grain = coarse_fine_tuner(X.to(device_2))

            predicted_fine = torch.max(Y_pred_fine_grain, 1)[1]
            predicted_coarse = torch.max(Y_pred_coarse_grain, 1)[1]

            test_fine_ground_truth += Y_fine_grain.tolist()
            test_coarse_ground_truth += Y_coarse_grain.tolist()

            test_fine_prediction += predicted_fine.tolist()
            test_coarse_prediction += predicted_coarse.tolist()

            name_list += names

    test_fine_accuracy, test_coarse_accuracy = (
        get_and_print_post_test_metrics(test_fine_ground_truth=test_fine_ground_truth,
                                        test_fine_prediction=test_fine_prediction,
                                        test_coarse_ground_truth=test_coarse_ground_truth,
                                        test_coarse_prediction=test_coarse_prediction,
                                        num_fine_grain_classes=num_fine_grain_classes,
                                        num_coarse_grain_classes=num_coarse_grain_classes))

    return (test_fine_ground_truth, test_coarse_ground_truth, test_fine_prediction, test_coarse_prediction,
            test_fine_accuracy, test_coarse_accuracy)


def test_combined_model(fine_tuner: models.FineTuner,
                        loaders: dict[str, torch.utils.data.DataLoader],
                        device: torch.device,
                        num_fine_grain_classes: int,
                        num_coarse_grain_classes: int) -> (list[int], list[int], list[int], list[int], float, float):
    test_loader = loaders['test']
    fine_tuner.to(device)
    fine_tuner.eval()

    test_fine_prediction = []
    test_coarse_prediction = []

    test_fine_ground_truth = []
    test_coarse_ground_truth = []

    name_list = []

    print(f'Testing {fine_tuner} on {device}...')

    with torch.no_grad():
        if utils.is_local():
            from tqdm import tqdm
            gen = tqdm(enumerate(test_loader), total=len(test_loader))
        else:
            gen = enumerate(test_loader)

        for i, data in gen:
            X, Y_fine_grain, names, Y_coarse_grain = data[0].to(device), data[1].to(device), data[2], data[3].to(device)
            Y_pred = fine_tuner(X)
            Y_pred_fine_grain = Y_pred[:, :num_fine_grain_classes]
            Y_pred_coarse_grain = Y_pred[:, num_fine_grain_classes:]

            predicted_fine = torch.max(Y_pred_fine_grain, 1)[1]
            predicted_coarse = torch.max(Y_pred_coarse_grain, 1)[1]

            test_fine_ground_truth += Y_fine_grain.tolist()
            test_coarse_ground_truth += Y_coarse_grain.tolist()

            test_fine_prediction += predicted_fine.tolist()
            test_coarse_prediction += predicted_coarse.tolist()

            name_list += names

    test_fine_accuracy, test_coarse_accuracy = (
        get_and_print_post_test_metrics(test_fine_ground_truth=test_fine_ground_truth,
                                        test_fine_prediction=test_fine_prediction,
                                        test_coarse_ground_truth=test_coarse_ground_truth,
                                        test_coarse_prediction=test_coarse_prediction,
                                        num_fine_grain_classes=num_fine_grain_classes,
                                        num_coarse_grain_classes=num_coarse_grain_classes))

    return (test_fine_ground_truth, test_coarse_ground_truth, test_fine_prediction, test_coarse_prediction,
            test_fine_accuracy, test_coarse_accuracy)


def get_and_print_post_epoch_metrics(epoch: int,
                                     epoch_start_time: float,
                                     running_fine_loss: float,
                                     running_coarse_loss: float,
                                     num_batches: int,
                                     train_fine_ground_truth,
                                     train_fine_prediction,
                                     train_coarse_ground_truth,
                                     train_coarse_prediction,
                                     num_fine_grain_classes,
                                     num_coarse_grain_classes):
    training_fine_accuracy = accuracy_score(y_true=train_fine_ground_truth, y_pred=train_fine_prediction)
    training_coarse_accuracy = accuracy_score(y_true=train_coarse_ground_truth, y_pred=train_coarse_prediction)
    training_fine_f1 = f1_score(y_true=train_fine_ground_truth, y_pred=train_fine_prediction,
                                labels=range(num_fine_grain_classes), average='macro')
    training_coarse_f1 = f1_score(y_true=train_coarse_ground_truth, y_pred=train_coarse_prediction,
                                  labels=range(num_coarse_grain_classes), average='macro')

    print(f'\nEpoch {epoch + 1}/{num_epochs} done in {utils.format_seconds(int(time() - epoch_start_time))}, '
          f'\nTraining fine loss: {round(running_fine_loss / num_batches, 2)}'
          f'\ntraining coarse loss: {round(running_coarse_loss / num_batches, 2)}'
          f'\ntraining fine accuracy: {round(training_fine_accuracy * 100, 2)}'
          f', fine f1: {round(training_fine_f1 * 100, 2)}%'
          f'\ntraining coarse accuracy: {round(training_coarse_accuracy * 100, 2)}%'
          f', coarse f1: {round(training_coarse_f1 * 100, 2)}%\n')

    return training_fine_accuracy, training_coarse_accuracy


def print_post_batch_metrics(batch_num: int,
                             num_batches: int,
                             batch_start_time: float,
                             batch_fine_grain_loss: float,
                             batch_coarse_grain_loss: float,
                             alpha_value: float = None):
    if not utils.is_local() and batch_num > 0 and batch_num % 10 == 0:
        print(f'Completed batch num {batch_num}/{num_batches} in {int(time() - batch_start_time)} '
              f'seconds. Batch fine-grain loss: {round(batch_fine_grain_loss, 2)}, '
              f'batch coarse-grain loss: {round(batch_coarse_grain_loss, 2)}'
              + (f', alpha value: {round(alpha_value, 2)}' if alpha_value is not None else ''))


def fine_tune_individual_models(fine_tuners: list[models.FineTuner],
                                devices: list[torch.device],
                                loaders: dict[str, torch.utils.data.DataLoader],
                                num_fine_grain_classes: int,
                                num_coarse_grain_classes: int,
                                fine_lr: float = 1e-4,
                                coarse_lr: float = 1e-4):
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

    fine_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=fine_optimizer,
                                                     step_size=scheduler_step_size,
                                                     gamma=scheduler_gamma)
    coarse_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=coarse_optimizer,
                                                       step_size=scheduler_step_size,
                                                       gamma=scheduler_gamma)

    train_fine_losses = []
    train_fine_accuracies = []

    test_fine_ground_truths = []
    test_fine_accuracies = []

    train_coarse_losses = []
    train_coarse_accuracies = []

    test_coarse_ground_truths = []
    test_coarse_accuracies = []

    print(f'Started fine-tuning individual models with fine_lr={fine_lr} and coarse_lr={coarse_lr}'
          f'for {num_epochs} epochs on {device_1} and {device_2}...')

    for epoch in range(num_epochs):
        epoch_start_time = time()
        running_fine_loss = 0.0
        running_coarse_loss = 0.0

        train_fine_predictions = []
        train_coarse_predictions = []

        train_fine_ground_truths = []
        train_coarse_ground_truths = []

        if utils.is_local():
            from tqdm import tqdm
            batches = tqdm(enumerate(train_loader, 0), total=num_batches)
        else:
            batches = enumerate(train_loader, 0)

        for batch_num, batch in batches:
            with context_handlers.ClearCache(device=device_1):
                with context_handlers.ClearCache(device=device_2):
                    batch_start_time = time()

                    X, Y_fine_grain, Y_coarse_grain = batch[0], batch[1].to(device_1), batch[3].to(device_2)

                    fine_optimizer.zero_grad()
                    coarse_optimizer.zero_grad()

                    Y_pred_fine = fine_fine_tuner(X.to(device_1))
                    Y_pred_coarse = coarse_fine_tuner(X.to(device_2))

                    batch_fine_grain_loss = criterion(Y_pred_fine, Y_fine_grain)
                    batch_coarse_grain_loss = criterion(Y_pred_coarse, Y_coarse_grain)

                    batch_fine_grain_loss.backward()
                    batch_coarse_grain_loss.backward()

                    fine_optimizer.step()
                    coarse_optimizer.step()

                    running_fine_loss += batch_fine_grain_loss.item()
                    running_coarse_loss += batch_coarse_grain_loss.item()

                    predicted_fine = torch.max(Y_pred_fine, 1)[1]
                    predicted_coarse = torch.max(Y_pred_coarse, 1)[1]

                    train_fine_ground_truths += Y_fine_grain.tolist()
                    train_coarse_ground_truths += Y_coarse_grain.tolist()

                    train_fine_predictions += predicted_fine.tolist()
                    train_coarse_predictions += predicted_coarse.tolist()

                    del X, Y_fine_grain, Y_coarse_grain

                    print_post_batch_metrics(batch_num=batch_num,
                                             num_batches=num_batches,
                                             batch_start_time=batch_start_time,
                                             batch_fine_grain_loss=batch_fine_grain_loss.item(),
                                             batch_coarse_grain_loss=batch_coarse_grain_loss.item())

        true_fine_labels = np.array(train_fine_ground_truths)
        true_coarse_labels = np.array(train_coarse_ground_truths)

        predicted_fine_labels = np.array(train_fine_predictions)
        predicted_coarse_labels = np.array(train_coarse_predictions)

        training_fine_accuracy, training_coarse_accuracy = (
            get_and_print_post_epoch_metrics(epoch=epoch,
                                             epoch_start_time=epoch_start_time,
                                             running_fine_loss=running_fine_loss,
                                             running_coarse_loss=running_coarse_loss,
                                             num_batches=num_batches,
                                             train_fine_ground_truth=true_fine_labels,
                                             train_fine_prediction=predicted_fine_labels,
                                             train_coarse_ground_truth=true_coarse_labels,
                                             train_coarse_prediction=predicted_coarse_labels,
                                             num_fine_grain_classes=num_fine_grain_classes,
                                             num_coarse_grain_classes=num_coarse_grain_classes))

        train_fine_accuracies += [training_fine_accuracy]
        train_coarse_accuracies += [training_coarse_accuracy]

        train_fine_losses += [running_fine_loss / num_batches]
        train_coarse_losses += [running_coarse_loss / num_batches]

        fine_scheduler.step()
        coarse_scheduler.step()

        (test_fine_ground_truth, test_coarse_ground_truth, test_fine_prediction, test_coarse_prediction,
         test_fine_accuracy, test_coarse_accuracy) = (
            test_individual_models(fine_tuners=fine_tuners,
                                   loaders=loaders,
                                   devices=devices,
                                   num_fine_grain_classes=num_fine_grain_classes,
                                   num_coarse_grain_classes=num_coarse_grain_classes
                                   ))
        test_fine_accuracies += [test_fine_accuracy]
        test_coarse_accuracies += [test_coarse_accuracy]
        print('#' * 100)

        np.save(f"{individual_results_path}{fine_fine_tuner}"
                f"_test_pred_lr{fine_lr}_e{epoch}_fine_individual.npy",
                test_fine_prediction)
        np.save(f"{individual_results_path}{coarse_fine_tuner}"
                f"_test_pred_lr{coarse_lr}_e{epoch}_coarse_individual.npy",
                test_coarse_prediction)

        torch.save(fine_fine_tuner.state_dict(), f"{fine_fine_tuner}_lr{fine_lr}_fine_individual.pth")
        torch.save(coarse_fine_tuner.state_dict(), f"{coarse_fine_tuner}_lr{coarse_lr}_coarse_individual.pth")

        if not os.path.exists(f"{individual_results_path}test_true_fine_individual.npy"):
            np.save(f"{individual_results_path}test_true_fine_individual.npy", test_fine_ground_truths)
        if not os.path.exists(f"{individual_results_path}test_true_coarse_individual.npy"):
            np.save(f"{individual_results_path}test_true_coarse_individual.npy", test_coarse_ground_truths)


def fine_tune_combined_model(fine_tuner: models.FineTuner,
                             device: torch.device,
                             loaders: dict[str, torch.utils.data.DataLoader],
                             num_fine_grain_classes: int,
                             num_coarse_grain_classes: int):
    fine_tuner.to(device)
    fine_tuner.train()

    train_loader = loaders['train']
    num_batches = len(train_loader)
    criterion = torch.nn.CrossEntropyLoss()

    for lr in lrs:
        optimizer = torch.optim.Adam(params=fine_tuner.parameters(),
                                     lr=lr)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                    step_size=scheduler_step_size,
                                                    gamma=scheduler_gamma)

        learned_weighted_loss = models.LearnedHierarchicalWeightedLoss(
            num_fine_grain_classes=num_fine_grain_classes,
            num_coarse_grain_classes=num_coarse_grain_classes).to(device)

        train_total_losses = []
        train_fine_losses = []
        train_coarse_losses = []

        train_fine_accuracies = []
        train_coarse_accuracies = []

        test_fine_ground_truths = []
        test_coarse_ground_truths = []

        test_fine_accuracies = []
        test_coarse_accuracies = []

        print(f'Fine-tuning {fine_tuner} with {len(fine_tuner)} parameters for {num_epochs} epochs '
              f'using lr={lr} on {device}...')
        print('#' * 100 + '\n')

        for epoch in range(num_epochs):
            epoch_start_time = time()

            total_running_loss = torch.Tensor([0.0]).to(device)
            running_fine_loss = torch.Tensor([0.0]).to(device)
            running_coarse_loss = torch.Tensor([0.0]).to(device)

            train_fine_predictions = []
            train_coarse_predictions = []

            train_fine_ground_truths = []
            train_coarse_ground_truths = []

            if utils.is_local():
                from tqdm import tqdm
                batches = tqdm(enumerate(train_loader, 0), total=num_batches)
            else:
                batches = enumerate(train_loader, 0)

            for batch_num, batch in batches:
                with context_handlers.ClearCache(device=device):
                    batch_start_time = time()

                    X, Y_fine_grain, Y_coarse_grain = batch[0].to(device), batch[1].to(device), batch[3].to(device)
                    optimizer.zero_grad()

                    Y_pred = fine_tuner(X)
                    Y_pred_fine_grain = Y_pred[:, :num_fine_grain_classes]
                    Y_pred_coarse_grain = Y_pred[:, num_fine_grain_classes:]

                    batch_fine_grain_loss = criterion(Y_pred_fine_grain, Y_fine_grain)
                    batch_coarse_grain_loss = criterion(Y_pred_coarse_grain, Y_coarse_grain)

                    running_fine_loss += batch_fine_grain_loss
                    running_coarse_loss += batch_coarse_grain_loss

                    batch_total_loss = learned_weighted_loss(batch_fine_loss=batch_fine_grain_loss,
                                                             batch_coarse_loss=batch_coarse_grain_loss,
                                                             total_fine_loss=running_fine_loss,
                                                             total_coarse_loss=running_coarse_loss)
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

                    print_post_batch_metrics(batch_num=batch_num,
                                             num_batches=num_batches,
                                             batch_start_time=batch_start_time,
                                             batch_fine_grain_loss=batch_fine_grain_loss.item(),
                                             batch_coarse_grain_loss=batch_coarse_grain_loss.item(),
                                             alpha_value=learned_weighted_loss.alpha.item())

            training_fine_accuracy, training_coarse_accuracy = (
                get_and_print_post_epoch_metrics(epoch=epoch,
                                                 epoch_start_time=epoch_start_time,
                                                 running_fine_loss=running_fine_loss.item(),
                                                 running_coarse_loss=running_coarse_loss.item(),
                                                 num_batches=num_batches,
                                                 train_fine_ground_truth=np.array(train_fine_ground_truths),
                                                 train_fine_prediction=np.array(train_fine_predictions),
                                                 train_coarse_ground_truth=np.array(train_coarse_ground_truths),
                                                 train_coarse_prediction=np.array(train_coarse_predictions),
                                                 num_fine_grain_classes=num_fine_grain_classes,
                                                 num_coarse_grain_classes=num_coarse_grain_classes))

            train_fine_accuracies += [training_fine_accuracy]
            train_coarse_accuracies += [training_coarse_accuracy]

            train_total_losses += [total_running_loss.item() / num_batches]
            train_fine_losses += [running_fine_loss.item() / num_batches]
            train_coarse_losses += [running_coarse_loss.item() / num_batches]

            scheduler.step()
            (test_fine_ground_truth, test_coarse_ground_truth, test_fine_prediction, test_coarse_prediction,
             test_fine_accuracy, test_coarse_accuracy) = (
                test_combined_model(fine_tuner=fine_tuner,
                                    loaders=loaders,
                                    device=device,
                                    num_fine_grain_classes=num_fine_grain_classes,
                                    num_coarse_grain_classes=num_coarse_grain_classes))
            test_fine_accuracies += [test_fine_accuracy]
            test_coarse_accuracies += [test_coarse_accuracy]
            print('#' * 100)

            np.save(f"{combined_results_path}{fine_tuner}_test_fine_pred_lr{lr}_e{epoch}.npy",
                    test_fine_prediction)
            np.save(f"{combined_results_path}{fine_tuner}_test_coarse_pred_lr{lr}_e{epoch}.npy",
                    test_coarse_prediction)

        np.save(f"{combined_results_path}{fine_tuner}_test_fine_acc_lr{lr}.npy", test_fine_accuracies)
        np.save(f"{combined_results_path}{fine_tuner}_test_coarse_acc_lr{lr}.npy", test_coarse_accuracies)

        if not os.path.exists(f"{combined_results_path}test_fine_true.npy"):
            np.save(f"{combined_results_path}test_fine_true.npy", test_fine_ground_truths)
        if not os.path.exists(f"{combined_results_path}test_coarse_true.npy"):
            np.save(f"{combined_results_path}test_coarse_true.npy", test_coarse_ground_truths)

        torch.save(fine_tuner.state_dict(), f"{fine_tuner}_lr{lr}.pth")


def initiate(combined: bool,
             train: bool,
             debug: bool = False):
    print(f'Models: {vit_model_names}\nEpochs num: {num_epochs}\nLearning rates: {lrs}')
    datasets, num_fine_grain_classes, num_coarse_grain_classes = data_preprocessing.get_datasets(cwd=cwd)

    if combined:
        device = torch.device('cpu') if debug and utils.is_local() and not train else (
            torch.device('mps' if torch.backends.mps.is_available() else
                         ("cuda" if torch.cuda.is_available() else 'cpu')))
        devices = [device]
        print(f'Using {device}')

        fine_tuners = [models.VITFineTuner(vit_model_name=vit_model_name,
                                           num_classes=num_fine_grain_classes + num_coarse_grain_classes)
                       for vit_model_name in vit_model_names]
        results_path = combined_results_path
    else:
        num_gpus = torch.cuda.device_count()

        if num_gpus < 2:
            raise ValueError("This setup requires at least 2 GPUs.")

        # Assign models to different GPUs
        devices = [torch.device("cuda:0"), torch.device("cuda:1")]

        fine_tuners = ([models.VITFineTuner(vit_model_name=vit_model_name,
                                            num_classes=num_fine_grain_classes)
                        for vit_model_name in vit_model_names] +
                       [models.VITFineTuner(vit_model_name=vit_model_name,
                                            num_classes=num_coarse_grain_classes)
                        for vit_model_name in vit_model_names])
        results_path = individual_results_path

    utils.create_directory(results_path)
    loaders = data_preprocessing.get_loaders(datasets=datasets,
                                             batch_size=batch_size)

    return fine_tuners, loaders, devices, num_fine_grain_classes, num_coarse_grain_classes


def run_combined_fine_tuning_pipeline(debug: bool = False):
    fine_tuners, loaders, devices, num_fine_grain_classes, num_coarse_grain_classes = initiate(combined=True,
                                                                                               train=True,
                                                                                               debug=debug)
    for fine_tuner in fine_tuners:
        with context_handlers.ClearSession():
            fine_tune_combined_model(fine_tuner=fine_tuner,
                                     device=devices[0],
                                     loaders=loaders,
                                     num_fine_grain_classes=num_fine_grain_classes,
                                     num_coarse_grain_classes=num_coarse_grain_classes)
            print('#' * 100)


def run_combined_testing_pipeline():
    fine_tuners, loaders, devices, num_fine_grain_classes, num_coarse_grain_classes = initiate(combined=True,
                                                                                               train=False)

    test_combined_model(fine_tuner=fine_tuners[0],
                        loaders=loaders,
                        device=devices[0],
                        num_fine_grain_classes=num_fine_grain_classes,
                        num_coarse_grain_classes=num_coarse_grain_classes)


def run_individual_fine_tuning_pipeline(debug: bool = False):
    fine_tuners, loaders, devices, num_fine_grain_classes, num_coarse_grain_classes = initiate(combined=False,
                                                                                               train=True,
                                                                                               debug=debug)

    for fine_tuner in fine_tuners:
        print(f'Initiating {fine_tuner}')

        with context_handlers.ClearSession():
            fine_tune_individual_models(fine_tuners=fine_tuners,
                                        devices=devices,
                                        loaders=loaders,
                                        num_fine_grain_classes=num_fine_grain_classes,
                                        num_coarse_grain_classes=num_coarse_grain_classes)
            print('#' * 100)


if __name__ == '__main__':
    run_individual_fine_tuning_pipeline()
