import os
import torch
import torchvision
import torch.utils.data
import numpy as np
from sklearn.metrics import accuracy_score
from time import time
from typing import Tuple
import pathlib

from context import ClearCache, ClearSession
from models import FineTuner, VITFineTuner
from utils import is_running_in_colab, create_directory, format_seconds, is_local
from data_preprocessing import granularities, get_datasets, get_loaders

batch_size = 32
lrs = [1e-6, 1e-5, 5e-5]
scheduler_gamma = 0.1
num_epochs = 4
vit_model_names = [f'vit_{vit_model_name}' for vit_model_name in ['b_16', 'b_32', 'l_16', 'l_32', 'h_14']]

cwd = pathlib.Path(__file__).parent.resolve()
scheduler_step_size = num_epochs



def test(fine_tuner: FineTuner,
         loaders: dict[str, torch.utils.data.DataLoader],
         device: torch.device,
         test_folder_name: str) -> Tuple[list[int], list[int], float]:
    test_loader = loaders[f'{fine_tuner}_{test_folder_name}']
    fine_tuner.eval()
    correct = 0
    total = 0
    test_prediction = []
    test_ground_truth = []
    name_list = []

    print(f'Started testing {fine_tuner} on {device}...')

    with torch.no_grad():
        if is_local():
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


def fine_tune(fine_tuner: FineTuner,
              device: torch.device,
              loaders: dict[str, torch.utils.data.DataLoader],
              granularity: str,
              train_folder_name: str,
              test_folder_name: str,
              results_path: str):
    fine_tuner.to(device)
    fine_tuner.train()

    train_loader = loaders[f'{fine_tuner}_{train_folder_name}']
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

        print(f'Started fine-tuning {fine_tuner} with lr={lr} on {device}...')

        for epoch in range(num_epochs):

            # print(f'Started epoch {epoch + 1}/{num_epochs}...')
            t1 = time()
            running_loss = 0.0
            train_predictions = []
            train_ground_truths = []

            if is_local():
                from tqdm import tqdm
                batches = tqdm(enumerate(train_loader, 0), total=num_batches)
            else:
                batches = enumerate(train_loader, 0)

            for batch_num, batch in batches:
                with ClearCache(device=device):
                    # if batch_num % 10 == 0:
                    #     print(f'Started batch {batch_num + 1}/{num_batches}')

                    X, Y = batch[0].to(device), batch[1].to(device)
                    optimizer.zero_grad()
                    Y_pred = fine_tuner(X)

                    if isinstance(Y_pred, torchvision.models.InceptionOutputs):
                        Y_pred = Y_pred[0]

                    loss = criterion(Y_pred, Y)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                    predicted = torch.max(Y_pred, 1)[1]
                    train_ground_truths += Y.tolist()
                    train_predictions += predicted.tolist()

                    del X
                    del Y

            true_labels = np.array(train_ground_truths)
            predicted_labels = np.array(train_predictions)
            acc = accuracy_score(true_labels, predicted_labels)
            # train.report({"mean_accuracy": acc})

            print(f'\nModel: {fine_tuner} with {len(fine_tuner)} parameters\n'
                  f'epoch {epoch + 1}/{num_epochs} done in {format_seconds(int(time() - t1))}, '
                  f'\nTraining loss: {round(running_loss / num_batches, 3)}'
                  f'\ntraining accuracy: {round(acc, 3)}\n')

            train_accuracies += [acc]
            train_losses += [running_loss / num_batches]
            scheduler.step()
            test_ground_truths, test_predictions, test_accuracy = test(fine_tuner=fine_tuner,
                                                                       loaders=loaders,
                                                                       device=device,
                                                                       test_folder_name=test_folder_name)
            test_accuracies += [test_accuracy]
            print('#' * 100)

            np.save(f"{results_path}{fine_tuner}_train_acc_lr{lr}_e{epoch}_{granularity}.npy", train_accuracies)
            np.save(f"{results_path}{fine_tuner}_train_loss_lr{lr}_e{epoch}_{granularity}.npy", train_losses)

            np.save(f"{results_path}{fine_tuner}_test_acc_lr{lr}_e{epoch}_{granularity}.npy", test_accuracies)
            np.save(f"{results_path}{fine_tuner}_test_pred_lr{lr}_e{epoch}_{granularity}.npy", test_predictions)

        torch.save(fine_tuner.state_dict(), f"{fine_tuner}_lr{lr}_{granularity}.pth")

        if not os.path.exists(f"{results_path}test_true_{granularity}.npy"):
            np.save(f"{results_path}test_true_{granularity}.npy", test_ground_truths)


def run_pipeline(granularity_index: int):
    granularity = granularities[granularity_index]

    train_folder_name = f'train_{granularity}'
    test_folder_name = f'test_{granularity}'
    files_path = '/content/drive/My Drive/' if is_running_in_colab() else ''
    results_path = fr'{files_path}results/'
    create_directory(results_path)

    print(f'Running {granularity}-grain pipeline...')
    print(F'Learning rates: {lrs}')


    print(f'Models: {vit_model_names}')

    datasets, n = get_datasets(model_names=vit_model_names,
                               cwd=cwd,
                               train_folder_name=train_folder_name,
                               test_folder_name=test_folder_name)

    device = torch.device('mps' if torch.backends.mps.is_available() else
                          ("cuda" if torch.cuda.is_available() else 'cpu'))
    print(f'Using {device}')

    all_fine_tuners = {f'vit_{vit_model_name}': VITFineTuner for vit_model_name in vit_model_names}

    fine_tuners = []
    for model_name in vit_model_names:
        fine_tuners_constructor = all_fine_tuners[model_name]
        fine_tuner = fine_tuners_constructor(*tuple([model_name, vit_model_names, n] if 'vit' in model_name else [n]))
        fine_tuners += [fine_tuner]
    print(f'Fine tuners: {[str(ft) for ft in fine_tuners]}')

    loaders = get_loaders(datasets=datasets,
                          batch_size=batch_size,
                          model_names=vit_model_names,
                          train_folder_name=train_folder_name,
                          test_folder_name=test_folder_name)

    for fine_tuner in fine_tuners:
        # try:
        print(f'Initiating {fine_tuner}')

        with ClearSession():
            fine_tune(fine_tuner=fine_tuner,
                      device=device,
                      loaders=loaders,
                      granularity=granularity,
                      train_folder_name=train_folder_name,
                      test_folder_name=test_folder_name,
                      results_path=results_path)
            print('#' * 100)


if __name__ == '__main__':
    granularity_index = 0
    run_pipeline(granularity_index=granularity_index)
