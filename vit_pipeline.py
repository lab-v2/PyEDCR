import os
import torch
import torch.utils.data
import numpy as np
from sklearn.metrics import accuracy_score
from time import time
from typing import Tuple
import pathlib

import context_handlers
import models
import utils
import data_preprocessing

batch_size = 32
lrs = [1e-6]
scheduler_gamma = 0.1
num_epochs = 10
vit_model_names = [f'vit_{vit_model_name}' for vit_model_name in ['l_16']]

cwd = pathlib.Path(__file__).parent.resolve()
scheduler_step_size = num_epochs


def test(fine_tuner: models.FineTuner,
         loaders: dict[str, torch.utils.data.DataLoader],
         device: torch.device) -> Tuple[list[int], list[int], float]:
    test_loader = loaders['test']
    fine_tuner.eval()
    correct = 0
    total = 0
    test_prediction = []
    test_ground_truth = []
    name_list = []

    print(f'Testing {fine_tuner} on {device}...')

    with torch.no_grad():
        if utils.is_local():
            from tqdm import tqdm
            gen = tqdm(enumerate(test_loader), total=len(test_loader))
        else:
            gen = enumerate(test_loader)

        for i, data in gen:
            pred_temp = []
            truth_temp = []
            name_temp = []
            images, labels, names = data[0].to(device), data[1].to(device), data[2]
            outputs = fine_tuner(images)
            predicted = torch.max(outputs.data, 1)[1]
            test_ground_truth += labels.tolist()
            test_prediction += predicted.tolist()
            name_list += names  # Collect the name values
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pred_temp += predicted.tolist()
            truth_temp += labels.tolist()
            name_temp += names

    test_accuracy = round(accuracy_score(y_true=test_ground_truth, y_pred=test_prediction), 3)
    print(f'\nTest accuracy: {test_accuracy}')

    return test_ground_truth, test_prediction, test_accuracy


def fine_tune(fine_tuner: models.FineTuner,
              device: torch.device,
              loaders: dict[str, torch.utils.data.DataLoader],
              results_path: str,
              num_fine_grain_classes: int,
              ):
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

        train_losses = []
        train_accuracies = []
        test_ground_truths = []

        test_accuracies = []

        print(f'Fine-tuning {fine_tuner} with {len(fine_tuner)} parameters using lr={lr} on {device}...')

        for epoch in range(num_epochs):

            epoch_start_time = time()
            running_loss = 0.0
            train_predictions = []
            train_ground_truths = []

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

                    fine_grain_loss = criterion(Y_pred_fine_grain, Y_fine_grain)
                    coarse_grain_loss = criterion(Y_pred_coarse_grain, Y_coarse_grain)
                    loss = fine_grain_loss + coarse_grain_loss
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                    predicted_fine = torch.max(Y_pred_fine_grain, 1)[1]
                    predicted_coarse = torch.max(Y_pred_coarse_grain, 1)[1]
                    train_ground_truths += Y_fine_grain.tolist() + Y_coarse_grain.tolist()
                    train_predictions += predicted_fine.tolist() + predicted_coarse.tolist()

                    del X, Y_fine_grain, Y_coarse_grain, Y_pred, Y_pred_fine_grain, Y_pred_coarse_grain

                    if not utils.is_local() and batch_num % 10 == 0:
                        print(f'Completed batch {batch_num}/{num_batches} in {time() - batch_start_time} seconds')

            true_labels = np.array(train_ground_truths)
            predicted_labels = np.array(train_predictions)
            acc = accuracy_score(true_labels, predicted_labels)

            print(f'\nModel: {fine_tuner}\n'
                  f'epoch {epoch + 1}/{num_epochs} done in {utils.format_seconds(int(time() - epoch_start_time))}, '
                  f'\nTraining loss: {round(running_loss / num_batches, 3)}'
                  f'\nTraining accuracy: {round(acc, 3)}\n')

            train_accuracies += [acc]
            train_losses += [running_loss / num_batches]
            scheduler.step()
            test_ground_truths, test_predictions, test_accuracy = test(fine_tuner=fine_tuner,
                                                                       loaders=loaders,
                                                                       device=device)
            test_accuracies += [test_accuracy]
            print('#' * 100)

            np.save(f"{results_path}{fine_tuner}_train_acc_lr{lr}_e{epoch}.npy", train_accuracies)
            np.save(f"{results_path}{fine_tuner}_train_loss_lr{lr}_e{epoch}.npy", train_losses)

            np.save(f"{results_path}{fine_tuner}_test_acc_lr{lr}_e{epoch}.npy", test_accuracies)
            np.save(f"{results_path}{fine_tuner}_test_pred_lr{lr}_e{epoch}.npy", test_predictions)

        torch.save(fine_tuner.state_dict(), f"{fine_tuner}_lr{lr}.pth")

        if not os.path.exists(f"{results_path}test_true.npy"):
            np.save(f"{results_path}test_true.npy", test_ground_truths)


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
                      num_fine_grain_classes=num_fine_grain_classes)
            print('#' * 100)


def main():
    print(f'Models: {vit_model_names}\nLearning rates: {lrs}\n')
    run_pipeline(debug=True)

    # with mp.Pool(processes=len(data_preprocessing.granularities)) as pool:
    #     pool.starmap(func=run_pipeline,
    #                  iterable=[[granularity] for granularity in data_preprocessing.granularities.values()])


if __name__ == '__main__':
    main()
