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
lrs = [1e-6]
scheduler_gamma = 0.1
num_epochs = 10
vit_model_names = [f'vit_{vit_model_name}' for vit_model_name in ['b_16']]

cwd = pathlib.Path(__file__).parent.resolve()
scheduler_step_size = num_epochs


def test(fine_tuner: models.FineTuner,
         loaders: dict[str, torch.utils.data.DataLoader],
         device: torch.device,
         num_fine_grain_classes: int) -> (list[int], list[int], list[int], list[int], float, float):
    test_loader = loaders['test']
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


def fine_tune(fine_tuner: models.FineTuner,
              device: torch.device,
              loaders: dict[str, torch.utils.data.DataLoader],
              results_path: str,
              num_fine_grain_classes: int,
              num_coarse_grain_classes: int
              ):
    fine_tuner.to(device)
    fine_tuner.train()

    train_loader = loaders['train']
    num_batches = len(train_loader)
    criterion = torch.nn.CrossEntropyLoss()

    alpha = num_fine_grain_classes / (num_fine_grain_classes + num_coarse_grain_classes)

    for lr in lrs:
        optimizer = torch.optim.Adam(params=fine_tuner.parameters(),
                                     lr=lr)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                    step_size=scheduler_step_size,
                                                    gamma=scheduler_gamma)

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

                    batch_total_loss = alpha * batch_fine_grain_loss + (1 - alpha) * batch_coarse_grain_loss
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

                    if not utils.is_local() and batch_num % 10 == 0:
                        print(f'Completed batch {batch_num}/{num_batches} in {round(time() - batch_start_time, 1)} '
                              f'seconds. Fine loss: {round(batch_fine_grain_loss.item(), 3)}, '
                              f'coarse loss: {round(batch_coarse_grain_loss.item(), 3)}')

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
             test_fine_accuracy, test_coarse_accuracy) = test(fine_tuner=fine_tuner,
                                                              loaders=loaders,
                                                              device=device,
                                                              num_fine_grain_classes=num_fine_grain_classes)
            test_fine_accuracies += [test_fine_accuracy]
            test_coarse_accuracies += [test_coarse_accuracy]
            print('#' * 100)

            np.save(f"{results_path}{fine_tuner}_test_fine_acc_lr{lr}_e{epoch}.npy", test_fine_accuracies)
            np.save(f"{results_path}{fine_tuner}_test_coarse_acc_lr{lr}_e{epoch}.npy", test_coarse_accuracies)

            np.save(f"{results_path}{fine_tuner}_test_fine_pred_lr{lr}_e{epoch}.npy", test_fine_prediction)
            np.save(f"{results_path}{fine_tuner}_test_coarse_pred_lr{lr}_e{epoch}.npy", test_coarse_prediction)

        torch.save(fine_tuner.state_dict(), f"{fine_tuner}_lr{lr}.pth")

        if not os.path.exists(f"{results_path}test_fine_true.npy"):
            np.save(f"{results_path}test_true.npy", test_fine_ground_truths)
        if not os.path.exists(f"{results_path}test_coarse_true.npy"):
            np.save(f"{results_path}test_true.npy", test_coarse_ground_truths)


def run_pipeline(debug: bool = False):
    files_path = '/content/drive/My Drive/' if utils.is_running_in_colab() else ''
    results_path = fr'{files_path}results/'
    utils.create_directory(results_path)

    datasets, num_fine_grain_classes, num_coarse_grain_classes = data_preprocessing.get_datasets(cwd=cwd)

    device = torch.device('cpu') if debug and utils.is_local() else (
        torch.device('mps' if torch.backends.mps.is_available() else
                     ("cuda" if torch.cuda.is_available() else 'cpu')))
    print(f'Using {device}')

    fine_tuners = [models.VITFineTuner(vit_model_name=vit_model_name,
                                       num_classes=num_fine_grain_classes + num_coarse_grain_classes)
                   for vit_model_name in vit_model_names]

    loaders = data_preprocessing.get_loaders(datasets=datasets,
                                             batch_size=batch_size)

    for fine_tuner in fine_tuners:
        with context_handlers.ClearSession():
            fine_tune(fine_tuner=fine_tuner,
                      device=device,
                      loaders=loaders,
                      results_path=results_path,
                      num_fine_grain_classes=num_fine_grain_classes,
                      num_coarse_grain_classes=num_coarse_grain_classes)
            print('#' * 100)


def main():
    print(f'Models: {vit_model_names}\nLearning rates: {lrs}\n')
    run_pipeline(debug=False)


if __name__ == '__main__':
    main()
