import torch
import torch.utils.data
import typing

import data_preprocessing
import models
import context_handlers
import neural_evaluation
import neural_metrics
import backbone_pipeline
import neural_fine_tuning


def fine_tune_binary_model(l: data_preprocessing.Label,
                           lrs: list[typing.Union[str, float]],
                           fine_tuner: models.FineTuner,
                           device: torch.device,
                           loaders: dict[str, torch.utils.data.DataLoader],
                           num_epochs: int,
                           positive_class_weight: list[float],
                           save_files: bool = True,
                           evaluate_on_test: bool = True):
    fine_tuner.to(device)
    fine_tuner.train()
    train_loader = loaders['train']
    num_batches = len(train_loader)
    positive_class_weight = torch.tensor(positive_class_weight).float().to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_class_weight)

    for lr in lrs:
        optimizer = torch.optim.Adam(params=fine_tuner.parameters(),
                                     lr=lr)

        neural_fine_tuning.print_fine_tuning_initialization(fine_tuner=fine_tuner,
                                                            num_epochs=num_epochs,
                                                            lr=lr,
                                                            device=device)

        print('#' * 100 + '\n')

        for epoch in range(num_epochs):
            with context_handlers.TimeWrapper():
                total_running_loss = torch.Tensor([0.0]).to(device)

                train_predictions = []
                train_ground_truths = []

                batches = neural_fine_tuning.get_fine_tuning_batches(train_loader=train_loader,
                                                                     num_batches=num_batches)

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

                training_accuracy, training_f1 = neural_metrics.get_and_print_post_epoch_binary_metrics(
                    epoch=epoch,
                    num_epochs=num_epochs,
                    train_predictions=train_predictions,
                    train_ground_truths=train_ground_truths,
                    total_running_loss=total_running_loss.item()
                )

                if evaluate_on_test:
                    # test_ground_truths, test_predictions, test_accuracy = (
                    #     neural_evaluation.evaluate_binary_model(l=l,
                    #                                             fine_tuner=fine_tuner,
                    #                                             loaders=loaders,
                    #                                             loss=loss,
                    #                                             device=device,
                    #                                             split='test'))
                    if epoch == num_epochs - 1:
                        neural_evaluation.run_binary_evaluating_pipeline(model_name='vit_b_16',
                                                                         l=l,
                                                                         split='train',
                                                                         lrs=[lr],
                                                                         loss='BCE',
                                                                         num_epochs=num_epochs,
                                                                         pretrained_fine_tuner=fine_tuner)
                        neural_evaluation.run_binary_evaluating_pipeline(model_name='vit_b_16',
                                                                         l=l,
                                                                         split='test',
                                                                         lrs=[lr],
                                                                         loss='BCE',
                                                                         num_epochs=num_epochs,
                                                                         pretrained_fine_tuner=fine_tuner)
                print('#' * 100)

        if save_files:
            torch.save(fine_tuner.state_dict(),
                       f"models/binary_models/binary_{l}_{fine_tuner}_lr{lr}_loss_{loss}_e{num_epochs}.pth")

        return train_predictions


def run_l_binary_fine_tuning_pipeline(data: str,
                                      vit_model_names: list[str],
                                      l: data_preprocessing.Label,
                                      lr: float,
                                      num_epochs: int,
                                      save_files: bool = True):
    # if not os.path.exists(f"{os.getcwd()}/models/binary_models/binary_{l}_vit_b_16_lr{lr}_"
    #                       f"loss_BCE_e{num_epochs}.pth"):
    fine_tuners, loaders, devices, positive_class_weight = backbone_pipeline.initiate(data=data,
                                                                                      model_names=vit_model_names,
                                                                                      lrs=[lr],
                                                                                      l=l)
    for fine_tuner in fine_tuners:
        fine_tune_binary_model(l=l,
                               lrs=[lr],
                               fine_tuner=fine_tuner,
                               device=devices[0],
                               loaders=loaders,
                               num_epochs=num_epochs,
                               save_files=save_files,
                               positive_class_weight=positive_class_weight)
        print('#' * 100)


# else:
#     print(f'Skipping {l}')


def run_g_binary_fine_tuning_pipeline(data: str,
                                      vit_model_names: list[str],
                                      g: data_preprocessing.Granularity,
                                      lr: float,
                                      num_epochs: int,
                                      save_files: bool = True):
    for l in data_preprocessing.get_labels(g=g).values():
        run_l_binary_fine_tuning_pipeline(data=data,
                                          vit_model_names=vit_model_names,
                                          l=l,
                                          lr=lr,
                                          num_epochs=num_epochs,
                                          save_files=save_files)


if __name__ == '__main__':
    data = 'imagenet'
    num_epochs = 10
    lr = 0.0001
    model_name = 'vit_b_16'
    loss = 'BCE'

    # for g in data_preprocessing.granularities.values():
    #     run_g_binary_fine_tuning_pipeline(vit_model_names=[model_name],
    #                                       g=g,
    #                                       lr=lr,
    #                                       num_epochs=num_epochs,
    #                                       save_files=True)

    l_str = data_preprocessing.fine_grain_classes_str[1]

    # for l_str in data_preprocessing.fine_grain_classes_str:
    l = data_preprocessing.fine_grain_labels[l_str]
    run_l_binary_fine_tuning_pipeline(data=data,
                                      vit_model_names=[model_name],
                                      l=data_preprocessing.fine_grain_labels[l_str],
                                      lr=lr,
                                      num_epochs=num_epochs,
                                      save_files=True)

    neural_evaluation.evaluate_binary_models_from_files(g_str='fine',
                                                        test=False,
                                                        lrs=lr,
                                                        num_epochs=num_epochs)

    # neural_evaluation.run_binary_evaluating_pipeline(
    #     model_name=model_name,
    #     l=l,
    #     split='train',
    #     lrs=[lr],
    #     loss=loss,
    #     num_epochs=num_epochs,
    #     pretrained_path=f'models/binary_models/binary_{l}_{model_name}_lr{lr}_loss_{loss}_e{num_epochs}.pth')
