import copy
import os.path

import torch
import torch.utils.data
from tqdm import tqdm
import numpy as np

import data_preprocessing
import models
import context_handlers
import neural_evaluation
import neural_metrics
import backbone_pipeline
import neural_fine_tuning
import utils

# All the label having sufficient result is saved in the file":
best_f1_label_file_name = 'best_f1_label_file_name.txt'


def save_label_with_good_f1_score(l: data_preprocessing.Label):
    # Check if file exists (optional)
    if os.path.exists(best_f1_label_file_name):
        # Open the file in read mode ("r")
        with open(best_f1_label_file_name, "r") as file:
            # Read all lines into a set (efficient for checking membership)
            existing_labels = set(line.strip() for line in file)

        # Check if l.l_str is not present in the set
        if l.l_str not in existing_labels:
            # If not present, open in write mode ("a") to append
            with open(best_f1_label_file_name, "a") as file:
                file.write(f'{l.l_str}\n')
    else:
        # If file doesn't exist, create it and write (same as before)
        with open(best_f1_label_file_name, "w") as file:
            file.write(f'{l.l_str}\n')


def fine_tune_binary_model(data_str: str,
                           l: data_preprocessing.Label,
                           lr: float,
                           fine_tuner: models.FineTuner,
                           device: torch.device,
                           loaders: dict[str, torch.utils.data.DataLoader],
                           num_epochs: int,
                           positive_class_weight: list[float] = None,
                           save_files: bool = True,
                           evaluate_on_test: bool = True,
                           train_eval_split: float = None,
                           previous_f1_score: float = None):
    fine_tuner.to(device)
    fine_tuner.train()
    train_loader = loaders['train']
    num_batches = len(train_loader)
    if positive_class_weight is not None:
        positive_class_weight = torch.tensor(positive_class_weight).float().to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_class_weight)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(params=fine_tuner.parameters(),
                                 lr=lr)

    neural_fine_tuning.print_fine_tuning_initialization(
        fine_tuner=fine_tuner,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        experiment_name=f'{l.l_str} with {"train_eval" if train_eval_split is not None else ""}')

    slicing_window = [0, 0]
    best_fine_tuner = copy.deepcopy(fine_tuner)

    print('#' * 100 + '\n')

    for epoch in range(num_epochs):
        with context_handlers.TimeWrapper():
            total_running_loss = torch.Tensor([0.0]).to(device)

            train_predictions = []
            train_ground_truths = []

            batches = tqdm(enumerate(train_loader, 0),
                           total=num_batches)

            for batch_num, batch in batches:
                with context_handlers.ClearCache(device=device):
                    X, Y = batch[0].to(device), batch[1].to(device)
                    Y_one_hot = torch.nn.functional.one_hot(Y, num_classes=2).float()
                    optimizer.zero_grad()
                    Y_pred = fine_tuner(X)

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

            print(utils.blue_text(f'class is used for training and number of them '))
            print(np.unique(train_ground_truths, return_counts=True))

            neural_metrics.get_and_print_post_epoch_binary_metrics(
                epoch=epoch,
                num_epochs=num_epochs,
                train_predictions=train_predictions,
                train_ground_truths=train_ground_truths,
                total_running_loss=total_running_loss.item()
            )

            if train_eval_split is not None:
                _, _, f1 = neural_evaluation.run_binary_evaluating_pipeline(model_name=model_name_in_main,
                                                                            l=l,
                                                                            split='train_eval',
                                                                            lr=lr,
                                                                            loss='BCE',
                                                                            num_epochs=num_epochs,
                                                                            pretrained_fine_tuner=fine_tuner,
                                                                            data_str=data_str,
                                                                            train_eval_split=train_eval_split)
                # Update slicing window, and break if the sum of current sliding window is smaller than previous one:
                if f1 > slicing_window[1]:
                    print(utils.green_text(f'f1 of current fine_tuner is better. Update fine_tuner'))
                    best_fine_tuner = copy.deepcopy(fine_tuner)
                current_sliding_window = [slicing_window[1], f1]
                print(f'current sliding window is {current_sliding_window} and previous one is {slicing_window}')
                if sum(slicing_window) > sum(current_sliding_window):
                    print(utils.green_text(f'finish training, stop criteria met'))
                    break
                slicing_window = current_sliding_window

    if evaluate_on_test:
        _, _, test_f1 = neural_evaluation.run_binary_evaluating_pipeline(model_name=model_name_in_main,
                                                                         l=l,
                                                                         split='test',
                                                                         lr=lr,
                                                                         loss='BCE',
                                                                         num_epochs=num_epochs,
                                                                         pretrained_fine_tuner=best_fine_tuner,
                                                                         data_str=data_str,
                                                                         save_files=False)
        if previous_f1_score is not None and test_f1 < previous_f1_score:
            print(utils.red_text(f'previous f1 score is {previous_f1_score}, greater than current f1 score {test_f1}'
                                 f'do not save model'))
            save_files = False
        elif test_f1 > 0.7:
            print(utils.green_text(f'f1 score for {l} is sufficient on test: {test_f1}'))
            save_label_with_good_f1_score(l=l)
        else:
            print(utils.red_text(f'f1 score for {l} is not sufficient on test: {test_f1}'))
            save_files = False
    print('#' * 100)

    if save_files:
        neural_evaluation.run_binary_evaluating_pipeline(model_name=model_name_in_main,
                                                         l=l,
                                                         split='test',
                                                         lr=lr,
                                                         loss='BCE',
                                                         num_epochs=num_epochs,
                                                         pretrained_fine_tuner=best_fine_tuner,
                                                         data_str=data_str,
                                                         save_files=True)
        torch.save(best_fine_tuner.state_dict(),
                   f"models/binary_models/binary_{l}_{best_fine_tuner}_lr{lr}_loss_{loss}_e{num_epochs}.pth")

        run_l_binary_evaluating_pipeline_from_train(data_str=data_str,
                                                    lr=lr,
                                                    num_epochs=num_epochs,
                                                    model_name=model_name_in_main,
                                                    l=l)


def run_l_binary_fine_tuning_pipeline(data_str: str,
                                      model_name: str,
                                      l: data_preprocessing.Label,
                                      lr: float,
                                      num_epochs: int,
                                      train_eval_split: float = None,
                                      save_files: bool = True,
                                      previous_f1_score: float = None):
    preprocessor, fine_tuners, loaders, devices = backbone_pipeline.initiate(
        data_str=data_str,
        lr=lr,
        model_name=model_name,
        train_eval_split=train_eval_split,
        l=l)

    for fine_tuner in fine_tuners:
        fine_tune_binary_model(data_str=data_str,
                               l=l,
                               lr=lr,
                               fine_tuner=fine_tuner,
                               device=devices[0],
                               loaders=loaders,
                               num_epochs=num_epochs,
                               save_files=save_files,
                               train_eval_split=train_eval_split,
                               previous_f1_score=previous_f1_score
                               )
        print('#' * 100)


def run_l_binary_evaluating_pipeline_from_train(data_str: str,
                                                model_name: str,
                                                l: data_preprocessing.Label,
                                                lr: float,
                                                num_epochs: int):
    pretrained_path = f"models/binary_models/binary_{l}_{model_name}_lr{lr}_loss_{loss}_e{num_epochs}.pth"
    try:
        neural_evaluation.run_binary_evaluating_pipeline(model_name=model_name_in_main,
                                                         l=l,
                                                         split='train',
                                                         lr=lr,
                                                         loss='BCE',
                                                         num_epochs=num_epochs,
                                                         pretrained_path=pretrained_path,
                                                         data_str=data_str)
    except FileNotFoundError:
        print(f'There is no pretrained {model_name} model for {l}')

    print('#' * 100)


if __name__ == '__main__':
    # data_str_in_main = 'imagenet'
    # num_epochs_in_main = 5
    # lr_in_main = 0.000001
    # model_name_in_main = 'dinov2_vits14'
    # loss = 'BCE'
    # train_eval_split_in_main = 0.8

    # data_str_in_main = 'military_vehicles'
    # num_epochs_in_main = 5
    # lr_in_main = 0.0001
    # model_name_in_main = 'vit_b_16'
    # loss = 'BCE'
    # train_eval_split_in_main = 0.8

    data_str_in_main = 'openimage'
    num_epochs_in_main = 5
    lr_in_main = 0.000001
    model_name_in_main = 'dinov2_vits14'
    loss = 'BCE'
    train_eval_split_in_main = 0.8

    preprocessor_in_main = data_preprocessing.DataPreprocessor(data_str_in_main)

    download_path = []
    g_str = None

    # for label_idx in range(len(preprocessor_in_main.coarse_grain_classes_str)):
    #     l_str = preprocessor_in_main.coarse_grain_classes_str[label_idx]
    #     l_in_main = preprocessor_in_main.coarse_grain_labels[l_str]
    #     g_str = 'coarse'

    for label_idx in range(len(preprocessor_in_main.fine_grain_classes_str)):
        l_str = preprocessor_in_main.fine_grain_classes_str[label_idx]
        l_in_main = preprocessor_in_main.fine_grain_labels[l_str]
        g_str = 'fine'
        save_metric = neural_evaluation.evaluate_binary_models_from_files(data_str=data_str_in_main,
                                                                          g_str=g_str,  # change this also!!!
                                                                          test=True,
                                                                          lr=lr_in_main,
                                                                          num_epochs=num_epochs_in_main,
                                                                          model_name=model_name_in_main,
                                                                          l=l_in_main)
        if save_metric is not None:
            if save_metric[1] > 0.7:
                print(f'binary model of class {l_in_main} is finished with sufficient f1 score {save_metric[1]}')
                print(f'get prediction from train set)')
                save_label_with_good_f1_score(l=l_in_main)

                test_save_path = models.get_filepath(model_name=model_name_in_main,
                                                     l=l_in_main,
                                                     test=True,
                                                     loss=loss,
                                                     lr=lr_in_main,
                                                     pred=True,
                                                     epoch=num_epochs_in_main,
                                                     data_str=data_str_in_main)

                train_save_path = models.get_filepath(model_name=model_name_in_main,
                                                      l=l_in_main,
                                                      test=False,
                                                      loss=loss,
                                                      lr=lr_in_main,
                                                      pred=True,
                                                      epoch=num_epochs_in_main,
                                                      data_str=data_str_in_main)
                if os.path.exists(train_save_path):
                    print(utils.green_text(f'file {train_save_path} already exist, train prediction is checkout'))
                    download_path.append(train_save_path)
                    download_path.append(test_save_path)
                else:
                    print(utils.red_text(f'file {train_save_path} do not exist, train prediction is created'))
                    run_l_binary_evaluating_pipeline_from_train(data_str=data_str_in_main,
                                                                lr=lr_in_main,
                                                                num_epochs=num_epochs_in_main,
                                                                model_name=model_name_in_main,
                                                                l=l_in_main)
                continue

        print(utils.red_text(f'Train model for {l_in_main} to get sufficient f1 score'))

        run_l_binary_fine_tuning_pipeline(data_str=data_str_in_main,
                                          model_name=model_name_in_main,
                                          l=l_in_main,
                                          lr=lr_in_main,
                                          num_epochs=num_epochs_in_main,
                                          save_files=True,
                                          train_eval_split=train_eval_split_in_main,
                                          previous_f1_score=save_metric[1] if save_metric is not None else None)

    print(utils.green_text('#' * 50))
    print(utils.green_text(f'Use these file for binary condition. This is for {data_str_in_main} dataset,'
                           f'model name {model_name_in_main} and {g_str} grain class'))
    print(utils.green_text('#' * 50))

    for file_path in download_path:
        print(file_path)
