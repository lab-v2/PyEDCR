import os
import torch.utils.data
import context_handlers
import models
import ltn
import ltn_support
import numpy as np

from PyEDCR import EDCR
import data_preprocessing
import vit_pipeline
import typing

num_fine_grain_classes = len(data_preprocessing.fine_grain_classes_str)
num_coarse_grain_classes = len(data_preprocessing.coarse_grain_classes_str)

class EDCR_LTN_experiment(EDCR):
    def __init__(self,
                 main_model_name: str,
                 combined: bool,
                 loss: str,
                 lr: typing.Union[str, float],
                 num_epochs: int,
                 epsilon: typing.Union[str, float],
                 K_train: list[(int, int)] = None,
                 K_test: list[(int, int)] = None,
                 include_inconsistency_constraint: bool = False,
                 secondary_model_name: str = None):
        super().__init__(main_model_name=main_model_name,
                         combined=combined,
                         loss=loss,
                         lr=lr,
                         num_epochs=num_epochs,
                         epsilon=epsilon,
                         K_train=K_train,
                         K_test=K_test,
                         include_inconsistency_constraint=include_inconsistency_constraint,
                         secondary_model_name=secondary_model_name)

        self.batch_size = 32
        self.scheduler_gamma = 0.9
        self.num_epochs = 5
        self.ltn_num_epochs = 5
        self.vit_model_names = [f'vit_{vit_model_name}' for vit_model_name in ['b_16']]

        self.combined_results_path = fr'combined_results'
        self.individual_results_path = fr'individual_results'

        self.scheduler_step_size = num_epochs
        self.original_prediction_weight = 1 / (len(data_preprocessing.fine_grain_classes_str) +
                                               len(data_preprocessing.coarse_grain_classes_str))

    def fine_tune_combined_model(self,
                                 fine_tuner: models.FineTuner,
                                 device: torch.device,
                                 loaders: dict[str, torch.utils.data.DataLoader],
                                 loss: str,
                                 ltn_num_epochs: int = None,
                                 beta: float = 0.1,
                                 save_files: bool = True,
                                 debug: bool = False,
                                 evaluate_on_test: bool = True,
                                 Y_original_fine: np.array = None,
                                 Y_original_coarse: np.array = None):
        fine_tuner.to(device)
        fine_tuner.train()
        train_loader = loaders['train']
        num_batches = len(train_loader)

        train_fine_predictions = None
        train_coarse_predictions = None

        optimizer = torch.optim.Adam(params=fine_tuner.parameters(),
                                     lr=self.lr)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                    step_size=self.scheduler_step_size,
                                                    gamma=self.scheduler_gamma)

        train_total_losses = []
        train_fine_losses = []
        train_coarse_losses = []

        train_fine_accuracies = []
        train_coarse_accuracies = []

        test_fine_ground_truths = []
        test_coarse_ground_truths = []

        test_fine_accuracies = []
        test_coarse_accuracies = []

        epochs = ltn_num_epochs
        logits_to_predicate = ltn.Predicate(ltn_support.LogitsToPredicate()).to(ltn.device)

        print(f'\nFine-tuning {fine_tuner} with {len(fine_tuner)} parameters for {epochs} epochs '
              f'using lr={self.lr} on {device}...')
        print('#' * 100 + '\n')

        for epoch in range(epochs):
            with ((context_handlers.TimeWrapper())):
                total_running_loss = torch.Tensor([0.0]).to(device)
                running_fine_loss = torch.Tensor([0.0]).to(device)
                running_coarse_loss = torch.Tensor([0.0]).to(device)

                train_fine_predictions = []
                train_coarse_predictions = []

                train_fine_ground_truths = []
                train_coarse_ground_truths = []

                # TODO: modify vit_pipeline.get_fine_tuning_batches to get indices of the example X

                batches = vit_pipeline.get_fine_tuning_batches(train_loader=train_loader,
                                                               num_batches=num_batches,
                                                               debug=debug)

                for batch_num, batch in batches:
                    with context_handlers.ClearCache(device=device):
                        X, Y_fine_grain, Y_coarse_grain = batch[0].to(device), batch[1].to(device), batch[3].to(
                            device)

                        # TODO: slice the condition from the indices you get above
                        # Hint: condition can get from self.pred_data[original][train][...]
                        train_pred_fine_batch = self.pred_data['original']['train'][...]
                        train_pred_coarse_batch = ...
                        secondary_train_pred_fine_batch = ...
                        ...
                        Y_fine_grain_one_hot = torch.nn.functional.one_hot(Y_fine_grain, num_classes=len(
                            data_preprocessing.fine_grain_classes_str))
                        Y_coarse_grain_one_hot = torch.nn.functional.one_hot(Y_coarse_grain, num_classes=len(
                            data_preprocessing.coarse_grain_classes_str))

                        Y_combine = torch.cat(tensors=[Y_fine_grain_one_hot, Y_coarse_grain_one_hot], dim=1).float()
                        optimizer.zero_grad()

                        # TODO: get Y_pred
                        # currently we have many option to get prediction, depend on whether fine_tuner predict
                        # fine / coarse or both
                        Y_pred = fine_tuner(X)

                        if loss == 'LTN_BCE':
                            criterion = torch.nn.BCEWithLogitsLoss()

                            # TODO: fill in the rest of the argument
                            sat_agg = ltn_support.compute_sat_normally(
                                logits_to_predicate=logits_to_predicate,
                                train_pred_fine_batch=train_pred_fine_batch,
                                original_secondary_train_pred_fine_batch=secondary_train_pred_fine_batch,
                                error_detection_rules=self.error_detection_rules
                            )
                            batch_total_loss = beta * (1. - sat_agg) + (1 - beta) * criterion(Y_pred, Y_combine)

                        if loss == "LTN_soft_marginal":
                            criterion = torch.nn.MultiLabelSoftMarginLoss()

                            sat_agg = ltn_support.compute_sat_normally(logits_to_predicate,
                                                                       Y_pred, Y_coarse_grain, Y_fine_grain)
                            batch_total_loss = beta * (1. - sat_agg) + (1 - beta) * (
                                criterion(Y_pred, Y_combine))

                        vit_pipeline.print_post_batch_metrics(batch_num=batch_num,
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
                        vit_pipeline.get_and_print_post_epoch_metrics(
                                                         epoch=epoch,
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

                    if evaluate_on_test:
                        (test_fine_ground_truths, test_coarse_ground_truths, test_fine_predictions,
                         test_coarse_predictions,
                         test_fine_accuracy, test_coarse_accuracy) = (
                            vit_pipeline.evaluate_combined_model(fine_tuner=fine_tuner,
                                                                 loaders=loaders,
                                                                 loss=loss,
                                                                 device=device))

                        test_fine_accuracies += [test_fine_accuracy]
                        test_coarse_accuracies += [test_coarse_accuracy]
                    print('#' * 100)

                    if (epoch == vit_pipeline.num_epochs - 1) and save_files:
                        vit_pipeline.save_prediction_files(
                                              test=False,
                                              fine_tuners=fine_tuner,
                                              combined=True,
                                              lrs=lr,
                                              epoch=epoch,
                                              test_fine_prediction=test_fine_predictions,
                                              test_coarse_prediction=test_coarse_predictions,
                                              loss=loss)

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

    def run_learning_pipeline(self,
                              EDCR_epoch_num=0):
        print('Started learning pipeline...\n')
        self.print_metrics(test=False, prior=True)

        for g in data_preprocessing.granularities.values():
            self.learn_correction_rules(g=g)

        print('\nRule learning completed\n')


if __name__ == '__main__':
    epsilons = [0.1 * i for i in range(2, 3)]
    test_bool = False

    for eps in epsilons:
        print('#' * 25 + f'eps = {eps}' + '#' * 50)
        edcr = EDCR_LTN_experiment(epsilon=eps,
                                   main_model_name='vit_b_16',
                                   combined=True,
                                   loss='BCE',
                                   lr=0.0001,
                                   num_epochs=20,
                                   include_inconsistency_constraint=False,
                                   secondary_model_name='vit_b_16_soft_marginal')
        edcr.print_metrics(test=test_bool, prior=True)

        edcr.run_learning_pipeline()
        for gra in data_preprocessing.granularities.values():
            edcr.apply_correction_rules(test=test_bool, g=gra)
