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
import config
import utils


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
                 secondary_model_name: str = None,
                 config=None):
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

        self.batch_size = config.batch_size
        self.scheduler_gamma = config.scheduler_gamma
        self.num_epochs = config.ltn_num_epochs
        self.scheduler_step_size = num_epochs
        self.pretrain_path = config.main_pretrained_path
        self.beta = config.beta
        self.correction_model = {}
        self.num_models = 5

    def fine_tune_and_evaluate_combined_model(self,
                                              fine_tuner: models.FineTuner,
                                              device: torch.device,
                                              loaders: dict[str, torch.utils.data.DataLoader],
                                              loss: str,
                                              mode: 'str',
                                              epoch: int = 0,
                                              beta: float = 0.1,
                                              optimizer=None,
                                              scheduler=None):
        fine_tuner.to(device)
        fine_tuner.train()
        loader = loaders[mode]
        num_batches = len(loader)
        train_or_test = 'train' if mode != 'test' else 'test'

        total_losses = []
        logits_to_predicate = ltn.Predicate(ltn_support.LogitsToPredicate()).to(ltn.device)

        print(f'\n{mode} {fine_tuner} with {len(fine_tuner)} parameters for {self.num_epochs} epochs '
              f'using lr={self.lr} on {device}...')
        print('#' * 100 + '\n')

        with context_handlers.TimeWrapper():
            total_running_loss = 0.0

            fine_predictions = []
            coarse_predictions = []

            fine_ground_truths = []
            coarse_ground_truths = []

            original_pred_fines = []
            original_pred_coarses = []

            original_secondary_pred_fines = []
            original_secondary_pred_coarses = []

            batches = vit_pipeline.get_fine_tuning_batches(train_loader=loader,
                                                           num_batches=num_batches,
                                                           debug=False)

            for batch_num, batch in batches:
                with context_handlers.ClearCache(device=device):
                    X, Y_fine_grain, Y_coarse_grain, indices = (
                        batch[0].to(device), batch[1].to(device), batch[3].to(device), batch[4])

                    # slice the condition from the indices you get above
                    original_pred_fine_batch = torch.tensor(
                        self.pred_data[train_or_test]['original']['fine'][indices]).to(device)
                    original_pred_coarse_batch = torch.tensor(
                        self.pred_data[train_or_test]['original']['fine'][indices]).to(device)
                    original_secondary_pred_fine_batch = torch.tensor(
                        self.pred_data['secondary_model'][train_or_test]['fine'][indices]).to(device)
                    original_secondary_pred_coarse_batch = torch.tensor(
                        self.pred_data['secondary_model'][train_or_test]['coarse'][indices]).to(device)

                    Y_fine_grain_one_hot = torch.nn.functional.one_hot(Y_fine_grain, num_classes=len(
                        data_preprocessing.fine_grain_classes_str))
                    Y_coarse_grain_one_hot = torch.nn.functional.one_hot(Y_coarse_grain, num_classes=len(
                        data_preprocessing.coarse_grain_classes_str))

                    Y_combine = torch.cat(tensors=[Y_fine_grain_one_hot, Y_coarse_grain_one_hot], dim=1).float()

                    # currently we have many option to get prediction, depend on whether fine_tuner predict
                    # fine / coarse or both
                    Y_pred = fine_tuner(X)

                    Y_pred_fine_grain = Y_pred[:, :len(data_preprocessing.fine_grain_classes_str)]
                    Y_pred_coarse_grain = Y_pred[:, len(data_preprocessing.fine_grain_classes_str):]

                    if mode == 'train' and optimizer is not None and scheduler is not None:
                        optimizer.zero_grad()

                        if loss == 'BCE':
                            criterion = torch.nn.BCEWithLogitsLoss()

                            sat_agg = ltn_support.compute_sat_normally(
                                logits_to_predicate=logits_to_predicate,
                                train_pred_fine_batch=Y_pred_fine_grain,
                                train_pred_coarse_batch=Y_pred_coarse_grain,
                                train_true_fine_batch=Y_fine_grain,
                                train_true_coarse_batch=Y_coarse_grain,
                                original_train_pred_fine_batch=original_pred_fine_batch,
                                original_train_pred_coarse_batch=original_pred_coarse_batch,
                                original_secondary_train_pred_fine_batch=original_secondary_pred_fine_batch,
                                original_secondary_train_pred_coarse_batch=original_secondary_pred_coarse_batch,
                                error_detection_rules=self.error_detection_rules,
                                device=device
                            )
                            batch_total_loss = beta * (1. - sat_agg) + (1 - beta) * criterion(Y_pred, Y_combine)

                        if loss == "soft_marginal":
                            criterion = torch.nn.MultiLabelSoftMarginLoss()

                            sat_agg = ltn_support.compute_sat_normally(
                                logits_to_predicate=logits_to_predicate,
                                train_pred_fine_batch=Y_pred_fine_grain,
                                train_pred_coarse_batch=Y_pred_coarse_grain,
                                train_true_fine_batch=Y_fine_grain,
                                train_true_coarse_batch=Y_coarse_grain,
                                original_train_pred_fine_batch=original_pred_fine_batch,
                                original_train_pred_coarse_batch=original_pred_coarse_batch,
                                original_secondary_train_pred_fine_batch=original_secondary_pred_fine_batch,
                                original_secondary_train_pred_coarse_batch=original_secondary_pred_coarse_batch,
                                error_detection_rules=self.error_detection_rules,
                                device=device
                            )
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

                    fine_predictions += predicted_fine.tolist()
                    coarse_predictions += predicted_coarse.tolist()

                    fine_ground_truths += Y_fine_grain.tolist()
                    coarse_ground_truths += Y_coarse_grain.tolist()

                    original_pred_fines += original_pred_fine_batch.tolist()
                    original_pred_coarses += original_pred_coarse_batch.tolist()

                    original_secondary_pred_fines += original_secondary_pred_fine_batch.tolist()
                    original_secondary_pred_coarses += original_secondary_pred_coarse_batch.tolist()

                    del X, Y_fine_grain, Y_coarse_grain, indices, Y_pred_fine_grain, Y_pred_coarse_grain

        fine_accuracy, coarse_accuracy = vit_pipeline.get_and_print_post_epoch_metrics(
            epoch=epoch,
            num_batches=num_batches,
            train_fine_ground_truth=np.array(fine_ground_truths),
            train_fine_prediction=np.array(fine_predictions),
            train_coarse_ground_truth=np.array(coarse_ground_truths),
            train_coarse_prediction=np.array(coarse_predictions),
            num_fine_grain_classes=len(data_preprocessing.fine_grain_classes_str),
            num_coarse_grain_classes=len(data_preprocessing.coarse_grain_classes_str))

        ltn_support.compute_sat_testing_value(
            logits_to_predicate=logits_to_predicate,
            pred_fine_batch=torch.tensor(fine_predictions).to(device),
            pred_coarse_batch=torch.tensor(fine_predictions).to(device),
            true_fine_batch=torch.tensor(fine_ground_truths).to(device),
            true_coarse_batch=torch.tensor(coarse_ground_truths).to(device),
            original_pred_fine_batch=torch.tensor(original_pred_fines).to(device),
            original_pred_coarse_batch=torch.tensor(original_pred_coarses).to(device),
            original_secondary_pred_fine_batch=torch.tensor(original_secondary_pred_fines).to(device),
            original_secondary_pred_coarse_batch=torch.tensor(original_secondary_pred_coarses).to(device),
            error_detection_rules=self.error_detection_rules,
            device=device
        )

        if mode == 'train':
            scheduler.step()

        print('#' * 100)

        return fine_accuracy, coarse_accuracy, fine_predictions, coarse_predictions

    def run_learning_pipeline(self,
                              model_index: int,
                              EDCR_epoch_num=0):
        print('Started learning pipeline...\n')
        self.print_metrics(test=False, prior=True)

        for g in data_preprocessing.granularities.values():
            self.learn_detection_rules(g=g)

        print('\nRule learning completed\n')

        print(f'\nStarted train and eval LTN model {model_index}...\n')

        fine_tuners, loaders, devices, num_fine_grain_classes, num_coarse_grain_classes = (
            vit_pipeline.initiate(combined=self.combined, pretrained_path=self.pretrain_path, debug=False,
                                  get_indices=True, train_eval_split=0.8))

        self.correction_model[model_index] = fine_tuners[0]

        train_fine_accuracies = []
        train_coarse_accuracies = []

        optimizer = torch.optim.Adam(params=self.correction_model[model_index].parameters(),
                                     lr=self.lr)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                    step_size=self.scheduler_step_size,
                                                    gamma=self.scheduler_gamma)
        for epoch in range(self.num_epochs):
            with context_handlers.ClearSession():

                self.fine_tune_and_evaluate_combined_model(
                    fine_tuner=self.correction_model[model_index],
                    device=devices[0],
                    loaders=loaders,
                    loss=self.loss,
                    epoch=epoch,
                    mode='train',
                    optimizer=optimizer,
                    scheduler=scheduler
                )

                training_fine_accuracy, training_coarse_accuracy, _, _ = self.fine_tune_and_evaluate_combined_model(
                    fine_tuner=self.correction_model[model_index],
                    device=devices[0],
                    loaders=loaders,
                    loss=self.loss,
                    epoch=epoch,
                    mode='train_eval'
                )

                train_fine_accuracies += [training_fine_accuracy]
                train_coarse_accuracies += [training_coarse_accuracy]

                slicing_window_last = (sum(train_fine_accuracies[-3:]) + sum(train_coarse_accuracies[-3:])) / 6
                slicing_window_before_last = (sum(train_fine_accuracies[-4:-2]) + sum(train_coarse_accuracies[-4:-2]))

                if epoch >= 6 and slicing_window_last <= slicing_window_before_last:
                    break

        print(f'\nfinish train and eval model {model_index}!\n')

        print('#' * 100)

    def run_evaluating_pipeline(self,
                                model_index: int):

        print(f'\nStarted testing LTN model {model_index}...\n')

        _, loaders, devices, num_fine_grain_classes, num_coarse_grain_classes = (
            vit_pipeline.initiate(combined=self.combined, debug=False, evaluation=True, lrs=[self.lr],
                                  get_indices=True))

        _, _, fine_predictions, coarse_prediction = self.fine_tune_and_evaluate_combined_model(
            fine_tuner=self.correction_model[model_index],
            device=devices[0],
            loaders=loaders,
            loss=self.loss,
            mode='test'
        )
        return fine_predictions, coarse_prediction

    def get_majority_vote(self,
                          predictions: dict[int, list],
                          g: data_preprocessing.Granularity):
        """
        Performs majority vote on a list of 1D numpy arrays representing predictions.

        Args:
            predictions: A list of 1D numpy arrays, where each array represents the
                         predictions from a single model.

        Returns:
            A 1D numpy array representing the majority vote prediction for each element.
        """
        # Count the occurrences of each class for each example (axis=0)
        all_prediction = torch.zeros_like(torch.nn.functional.one_hot(torch.tensor(predictions[0]),
                                                                      num_classes=len(
                                                                          data_preprocessing.get_labels(g))))
        for i in range(self.num_models):
            all_prediction += torch.nn.functional.one_hot(torch.tensor(predictions[i]),
                                                          num_classes=len(data_preprocessing.get_labels(g)))

        # Get the index of the majority class
        majority_votes = torch.argmax(all_prediction, dim=1)

        return majority_votes

    def run_evaluating_pipeline_all_models(self):
        fine_prediction, coarse_prediction = {}, {}
        for i in range(self.num_models):
            self.run_learning_pipeline(model_index=i)
            fine_prediction[i], coarse_prediction[i] = self.run_evaluating_pipeline(model_index=i)

        print("\nGot all the prediction from test model!\n")

        final_fine_prediction = self.get_majority_vote(fine_prediction,
                                                       g=data_preprocessing.granularities['fine'])
        final_coarse_prediction = self.get_majority_vote(coarse_prediction,
                                                         g=data_preprocessing.granularities['coarse'])

        np.save(f"combined_results/LTN_EDCR_result/vit_b_16_test_fine_{self.loss}_"
                f"lr_{self.lr}_batch_size_{self.batch_size}_"
                f"ltn_epoch_{self.num_epochs}_"
                f"beta_{self.beta}_num_model_{self.num_models}.npy",
                np.array(final_fine_prediction))

        np.save(f"combined_results/LTN_EDCR_result/vit_b_16_test_coarse_{self.loss}_"
                f"lr_{self.lr}_batch_size_{self.batch_size}_"
                f"ltn_epoch_{self.num_epochs}_"
                f"beta_{self.beta}_num_model_{self.num_models}.npy",
                np.array(final_fine_prediction))

        vit_pipeline.get_and_print_metrics(pred_fine_data=np.array(final_fine_prediction),
                                           pred_coarse_data=np.array(final_coarse_prediction),
                                           loss=self.loss,
                                           true_fine_data=data_preprocessing.get_ground_truths(
                                               test=True,
                                               g=data_preprocessing.granularities['fine']),
                                           true_coarse_data=data_preprocessing.get_ground_truths(
                                               test=True,
                                               g=data_preprocessing.granularities['fine']),
                                           test=True)


if __name__ == '__main__':
    epsilons = [0.1 * i for i in range(2, 3)]
    test_bool = False
    main_pretrained_path = config

    for eps in epsilons:
        print('#' * 25 + f'eps = {eps}' + '#' * 50)
        edcr = EDCR_LTN_experiment(
            epsilon=eps,
            main_model_name=config.vit_model_names[0],
            combined=config.combined,
            loss=config.loss,
            lr=config.lr,
            num_epochs=config.num_epochs,
            include_inconsistency_constraint=config.include_inconsistency_constraint,
            secondary_model_name=config.secondary_model_name,
            config=config)
        edcr.print_metrics(test=test_bool, prior=True)
        edcr.run_evaluating_pipeline_all_models()
