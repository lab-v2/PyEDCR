import os
import torch
import torch.utils.data
import numpy as np
from sklearn.metrics import accuracy_score
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
vit_model_names = [f'vit_{vit_model_name}' for vit_model_name in ['l_16']]

files_path = '/content/drive/My Drive/' if utils.is_running_in_colab() else ''
combined_results_path = fr'{files_path}combined_results/'
individual_results_path = fr'{files_path}individual_results/'
cwd = pathlib.Path(__file__).parent.resolve()
scheduler_step_size = num_epochs


def test_individual_models(fine_tuners: list[models.FineTuner],
                           loaders: dict[str, torch.utils.data.DataLoader],
                           devices: list[torch.device]) -> (list[int], list[int], float):
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

    test_fine_accuracy = round(accuracy_score(y_true=test_fine_ground_truth, y_pred=test_fine_prediction), 3)
    test_coarse_accuracy = round(accuracy_score(y_true=test_coarse_ground_truth, y_pred=test_coarse_prediction), 3)

    print(f'\nTest fine accuracy: {test_fine_accuracy}'
          f'\nTest coarse accuracy: {test_coarse_accuracy}')

    return (test_fine_ground_truth, test_coarse_ground_truth, test_fine_prediction, test_coarse_prediction,
            test_fine_accuracy, test_coarse_accuracy)


def test_combined_model(fine_tuner: models.FineTuner,
                        loaders: dict[str, torch.utils.data.DataLoader],
                        device: torch.device,
                        num_fine_grain_classes: int) -> (list[int], list[int], list[int], list[int], float, float):
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

    test_fine_accuracy = round(accuracy_score(y_true=test_fine_ground_truth,
                                              y_pred=test_fine_prediction), 3)
    test_coarse_accuracy = round(accuracy_score(y_true=test_coarse_ground_truth,
                                                y_pred=test_coarse_prediction), 3)
    print(f'\nTest fine accuracy: {test_fine_accuracy}'
          f'\nTest coarse accuracy: {test_coarse_accuracy}')

    return (test_fine_ground_truth, test_coarse_ground_truth, test_fine_prediction, test_coarse_prediction,
            test_fine_accuracy, test_coarse_accuracy)


def fine_tune_individual_models(fine_tuners: list[models.FineTuner],
                                devices: list[torch.device],
                                loaders: dict[str, torch.utils.data.DataLoader]):
    fine_fine_tuner, coarse_fine_tuner = fine_tuners
    device_1, device_2 = devices
    fine_fine_tuner.to(device_1)
    fine_fine_tuner.train()

    coarse_fine_tuner.to(device_2)
    coarse_fine_tuner.train()

    train_loader = loaders['train']
    num_batches = len(train_loader)
    criterion = torch.nn.CrossEntropyLoss()

    for lr in lrs:
        fine_optimizer = torch.optim.Adam(params=fine_fine_tuner.parameters(),
                                          lr=lr)
        coarse_optimizer = torch.optim.Adam(params=coarse_fine_tuner.parameters(),
                                            lr=lr)

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

        print(f'Started fine-tuning individual models with lr={lr} on {device_1} and {device_2}...')

        for epoch in range(num_epochs):
            t1 = time()
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
                        X, Y_fine_grain, Y_coarse_grain = batch[0], batch[1].to(device_1), batch[3].to(device_2)

                        fine_optimizer.zero_grad()
                        coarse_optimizer.zero_grad()

                        Y_pred_fine = fine_fine_tuner(X.to(device_1))
                        Y_pred_coarse = coarse_fine_tuner(X.to(device_2))

                        fine_loss = criterion(Y_pred_fine, Y_fine_grain)
                        coarse_loss = criterion(Y_pred_coarse, Y_coarse_grain)

                        fine_loss.backward()
                        coarse_loss.backward()

                        fine_optimizer.step()
                        coarse_optimizer.step()

                        running_fine_loss += fine_loss.item()
                        running_coarse_loss += coarse_loss.item()

                        predicted_fine = torch.max(Y_pred_fine, 1)[1]
                        predicted_coarse = torch.max(Y_pred_coarse, 1)[1]

                        train_fine_ground_truths += Y_fine_grain.tolist()
                        train_coarse_ground_truths += Y_coarse_grain.tolist()

                        train_fine_predictions += predicted_fine.tolist()
                        train_coarse_predictions += predicted_coarse.tolist()

                        del X, Y_fine_grain, Y_coarse_grain

                        if not utils.is_local() and batch_num > 0 and batch_num % 10 == 0:
                            print(f'\nCompleted batch num {batch_num}/{num_batches}. '
                              f'Batch fine-grain loss: {round(fine_loss.item(), 3)}, '
                              f' coarse-grain loss: {round(coarse_loss.item(), 3)}')

            true_fine_labels = np.array(train_fine_ground_truths)
            true_coarse_labels = np.array(train_coarse_ground_truths)

            predicted_fine_labels = np.array(train_fine_predictions)
            predicted_coarse_labels = np.array(train_coarse_predictions)

            fine_acc = accuracy_score(true_fine_labels, predicted_fine_labels)
            coarse_acc = accuracy_score(true_coarse_labels, predicted_coarse_labels)

            print(f'\nEpoch {epoch + 1}/{num_epochs} done in {utils.format_seconds(int(time() - t1))}, '
                  f'\nTraining fine loss: {round(running_fine_loss / num_batches, 3)}'
                  f'\ntraining coarse loss: {round(running_fine_loss / num_batches, 3)}'
                  f'\ntraining fine accuracy: {round(fine_acc, 3)}\n'
                  f'\ntraining coarse accuracy: {round(coarse_acc, 3)}\n')

            train_fine_accuracies += [fine_acc]
            train_coarse_accuracies += [coarse_acc]

            train_fine_losses += [running_fine_loss / num_batches]
            train_coarse_losses += [running_coarse_loss / num_batches]

            fine_scheduler.step()
            coarse_scheduler.step()

            (test_fine_ground_truth, test_coarse_ground_truth, test_fine_prediction, test_coarse_prediction,
             test_fine_accuracy, test_coarse_accuracy) = test_individual_models(fine_tuners=fine_tuners,
                                                                                loaders=loaders,
                                                                                devices=devices)
            test_fine_accuracies += [test_fine_accuracy]
            test_coarse_accuracies += [test_coarse_accuracy]
            print('#' * 100)

            np.save(f"{individual_results_path}{fine_fine_tuner}"
                    f"_test_pred_lr{lr}_e{epoch}_fine_individual.npy",
                    test_fine_prediction)
            np.save(f"{individual_results_path}{fine_fine_tuner}"
                    f"_test_pred_lr{lr}_e{epoch}_coarse_individual.npy",
                    test_coarse_prediction)

        torch.save(fine_fine_tuner.state_dict(), f"{fine_fine_tuner}_lr{lr}_fine_individual.pth")
        torch.save(coarse_fine_tuner.state_dict(), f"{coarse_fine_tuner}_lr{lr}_coarse_individual.pth")

        if not os.path.exists(f"{individual_results_path}test_true_fine_individual.npy"):
            np.save(f"{individual_results_path}test_true_fine_individual.npy", test_fine_ground_truths)
        if not os.path.exists(f"{individual_results_path}test_true_coarse_individual.npy"):
            np.save(f"{individual_results_path}test_true_coarse_individual.npy", test_coarse_ground_truths)


class LearnedWeightedLoss(torch.nn.Module):
    def __init__(self):
        super(LearnedWeightedLoss, self).__init__()
        self.a1 = torch.nn.Parameter(torch.Tensor([1.0]),
                                     requires_grad=True)
        self.a2 = torch.nn.Parameter(torch.Tensor([1.0]),
                                     requires_grad=True)

    def forward(self, L1: torch.Tensor, L2: torch.Tensor) -> torch.Tensor:
        return self.a1 * L1 + self.a2 * L2


def fine_tune_combined_model(fine_tuner: models.FineTuner,
                             device: torch.device,
                             loaders: dict[str, torch.utils.data.DataLoader],
                             num_fine_grain_classes: int):
    fine_tuner.to(device)
    fine_tuner.train()

    train_loader = loaders['train']
    num_batches = len(train_loader)
    criterion = torch.nn.CrossEntropyLoss()

    # alpha = num_fine_grain_classes / (num_fine_grain_classes + num_coarse_grain_classes)

    for lr in lrs:
        optimizer = torch.optim.Adam(params=fine_tuner.parameters(),
                                     lr=lr)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                    step_size=scheduler_step_size,
                                                    gamma=scheduler_gamma)

        learned_weighted_loss = LearnedWeightedLoss()

        train_total_losses = []
        train_fine_losses = []
        train_coarse_losses = []

        train_fine_accuracies = []
        train_coarse_accuracies = []

        test_fine_ground_truths = []
        test_coarse_ground_truths = []

        test_fine_accuracies = []
        test_coarse_accuracies = []

        print(f'Fine-tuning {fine_tuner} with {len(fine_tuner)} parameters using lr={lr} on {device}...')
        print('#' * 100 + '\n')

        for epoch in range(num_epochs):

            epoch_start_time = time()

            total_running_loss = 0.0
            fine_running_loss = 0.0
            coarse_running_loss = 0.0

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

                    batch_total_loss = learned_weighted_loss(batch_fine_grain_loss, batch_coarse_grain_loss)
                    # batch_total_loss = alpha * batch_fine_grain_loss + (1 - alpha) * batch_coarse_grain_loss
                    batch_total_loss.backward()
                    optimizer.step()

                    total_running_loss += batch_total_loss.item()
                    fine_running_loss += batch_fine_grain_loss.item()
                    coarse_running_loss += batch_coarse_grain_loss.item()

                    predicted_fine = torch.max(Y_pred_fine_grain, 1)[1]
                    predicted_coarse = torch.max(Y_pred_coarse_grain, 1)[1]

                    train_fine_predictions += predicted_fine.tolist()
                    train_coarse_predictions += predicted_coarse.tolist()

                    train_fine_ground_truths += Y_fine_grain.tolist()
                    train_coarse_ground_truths += Y_coarse_grain.tolist()

                    del X, Y_fine_grain, Y_coarse_grain, Y_pred, Y_pred_fine_grain, Y_pred_coarse_grain

                    if not utils.is_local() and batch_num > 0 and batch_num % 10 == 0:
                        print(f'Completed batch num {batch_num}/{num_batches} in {round(time() - batch_start_time, 1)} '
                              f'seconds. Batch fine-grain loss: {round(batch_fine_grain_loss.item(), 3)}, '
                              f'batch coarse-grain loss: {round(batch_coarse_grain_loss.item(), 3)}')

            training_fine_accuracy = accuracy_score(y_true=np.array(train_fine_ground_truths),
                                                    y_pred=np.array(train_fine_predictions))
            training_coarse_accuracy = accuracy_score(y_true=np.array(train_coarse_ground_truths),
                                                      y_pred=np.array(train_coarse_predictions))

            print(f'\nModel: {fine_tuner}\n'
                  f'Epoch {epoch + 1}/{num_epochs} done in {utils.format_seconds(int(time() - epoch_start_time))}\n'
                  f'Training total loss: {round(total_running_loss / num_batches, 3)}\n'
                  f'Training fine loss: {round(fine_running_loss / num_batches, 3)}\n'
                  f'Training coarse loss: {round(coarse_running_loss / num_batches, 3)}\n'
                  f'Training fine accuracy: {round(training_fine_accuracy, 3)}\n'
                  f'Training coarse accuracy: {round(training_coarse_accuracy, 3)}\n')

            train_fine_accuracies += [training_fine_accuracy]
            train_coarse_accuracies += [training_coarse_accuracy]

            train_total_losses += [total_running_loss / num_batches]
            train_fine_losses += [fine_running_loss / num_batches]
            train_coarse_losses += [coarse_running_loss / num_batches]

            scheduler.step()
            (test_fine_ground_truth, test_coarse_ground_truth, test_fine_prediction, test_coarse_prediction,
             test_fine_accuracy, test_coarse_accuracy) = test_combined_model(fine_tuner=fine_tuner,
                                                                             loaders=loaders,
                                                                             device=device,
                                                                             num_fine_grain_classes=
                                                                             num_fine_grain_classes)
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
    print(f'Models: {vit_model_names}\nLearning rates: {lrs}\n')
    datasets, num_fine_grain_classes, num_coarse_grain_classes = data_preprocessing.get_datasets(cwd=cwd)

    if combined:
        device = torch.device('cpu') if debug and utils.is_local() and not train else (
            torch.device('mps' if torch.backends.mps.is_available() else
                         ("cuda" if torch.cuda.is_available() else 'cpu')))
        devices = [device]
        print(f'Using {device}')
    else:
        # Check the number of available GPUs
        num_gpus = torch.cuda.device_count()

        if num_gpus < 2:
            raise ValueError("This setup requires at least 2 GPUs.")

        # Assign models to different GPUs
        device_1 = torch.device("cuda:0")  # Choose GPU 0
        device_2 = torch.device("cuda:1")  # Choose GPU 1
        devices = [device_1, device_2]

    if combined:
        fine_tuners = [models.VITFineTuner(vit_model_name=vit_model_name,
                                           num_classes=num_fine_grain_classes + num_coarse_grain_classes)
                       for vit_model_name in vit_model_names]
        results_path = combined_results_path
    else:
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
                                     num_fine_grain_classes=num_fine_grain_classes)
            print('#' * 100)


def run_combined_testing_pipeline():
    fine_tuners, loaders, devices, num_fine_grain_classes, num_coarse_grain_classes = initiate(combined=True,
                                                                                               train=False)

    test_combined_model(fine_tuner=fine_tuners[0],
                        loaders=loaders,
                        device=devices[0],
                        num_fine_grain_classes=num_fine_grain_classes)


def run_individual_fine_tuning_pipeline(debug: bool = False):
    fine_tuners, loaders, devices, num_fine_grain_classes, num_coarse_grain_classes = initiate(combined=False,
                                                                                               train=True,
                                                                                               debug=debug)

    for fine_tuner in fine_tuners:
        print(f'Initiating {fine_tuner}')

        with context_handlers.ClearSession():
            fine_tune_individual_models(fine_tuners=fine_tuners,
                                        devices=devices,
                                        loaders=loaders)
            print('#' * 100)


if __name__ == '__main__':
    run_individual_fine_tuning_pipeline()
