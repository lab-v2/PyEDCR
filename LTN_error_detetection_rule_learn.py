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

    def fine_tune_combined_model(self,
                                 fine_tuner: models.FineTuner,
                                 device: torch.device,
                                 loaders: dict[str, torch.utils.data.DataLoader],
                                 loss: str,
                                 beta: float = 0.1,
                                 debug: bool = False):
        fine_tuner.to(device)
        fine_tuner.train()
        train_loader = loaders['train']
        num_batches = len(train_loader)

        optimizer = torch.optim.Adam(params=fine_tuner.parameters(),
                                     lr=self.lr)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                    step_size=self.scheduler_step_size,
                                                    gamma=self.scheduler_gamma)

        train_total_losses = []

        train_fine_accuracies = []
        train_coarse_accuracies = []
        logits_to_predicate = ltn.Predicate(ltn_support.LogitsToPredicate()).to(ltn.device)

        print(f'\nFine-tuning {fine_tuner} with {len(fine_tuner)} parameters for {self.num_epochs} epochs '
              f'using lr={self.lr} on {device}...')
        print('#' * 100 + '\n')

        for epoch in range(self.num_epochs):
            with context_handlers.TimeWrapper():
                total_running_loss = 0.0

                train_fine_predictions = []
                train_coarse_predictions = []

                train_fine_ground_truths = []
                train_coarse_ground_truths = []

                all_Y_pred_fine = []
                all_Y_pred_coarse = []

                batches = vit_pipeline.get_fine_tuning_batches(train_loader=train_loader,
                                                               num_batches=num_batches,
                                                               debug=debug)

                for batch_num, batch in batches:
                    with context_handlers.ClearCache(device=device):
                        X, Y_fine_grain, Y_coarse_grain, indices = (
                            batch[0].to(device), batch[1].to(device), batch[3].to(device), batch[4])

                        # slice the condition from the indices you get above
                        original_train_pred_fine_batch = torch.tensor(
                            self.pred_data['train']['original']['fine'][indices]).to(device)
                        original_train_pred_coarse_batch = torch.tensor(
                            self.pred_data['train']['original']['fine'][indices]).to(device)
                        original_secondary_train_pred_fine_batch = torch.tensor(
                            self.pred_data['secondary_model']['train']['fine'][indices]).to(device)
                        original_secondary_train_pred_coarse_batch = torch.tensor(
                            self.pred_data['secondary_model']['train']['coarse'][indices]).to(device)

                        Y_fine_grain_one_hot = torch.nn.functional.one_hot(Y_fine_grain, num_classes=len(
                            data_preprocessing.fine_grain_classes_str))
                        Y_coarse_grain_one_hot = torch.nn.functional.one_hot(Y_coarse_grain, num_classes=len(
                            data_preprocessing.coarse_grain_classes_str))

                        Y_combine = torch.cat(tensors=[Y_fine_grain_one_hot, Y_coarse_grain_one_hot], dim=1).float()
                        optimizer.zero_grad()

                        # currently we have many option to get prediction, depend on whether fine_tuner predict
                        # fine / coarse or both
                        Y_pred = fine_tuner(X)

                        Y_pred_fine_grain = Y_pred[:, :len(data_preprocessing.fine_grain_classes_str)]
                        Y_pred_coarse_grain = Y_pred[:, len(data_preprocessing.fine_grain_classes_str):]

                        all_Y_pred_fine.append(Y_pred_fine_grain)
                        all_Y_pred_coarse.append(Y_pred_coarse_grain)

                        if loss == 'BCE':
                            criterion = torch.nn.BCEWithLogitsLoss()

                            sat_agg = ltn_support.compute_sat_normally(
                                logits_to_predicate=logits_to_predicate,
                                train_pred_fine_batch=Y_pred_fine_grain,
                                train_pred_coarse_batch=Y_pred_coarse_grain,
                                train_true_fine_batch=Y_fine_grain,
                                train_true_coarse_batch=Y_coarse_grain,
                                original_train_pred_fine_batch=original_train_pred_fine_batch,
                                original_train_pred_coarse_batch=original_train_pred_coarse_batch,
                                original_secondary_train_pred_fine_batch=original_secondary_train_pred_fine_batch,
                                original_secondary_train_pred_coarse_batch=original_secondary_train_pred_coarse_batch,
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
                                original_train_pred_fine_batch=original_train_pred_fine_batch,
                                original_train_pred_coarse_batch=original_train_pred_coarse_batch,
                                original_secondary_train_pred_fine_batch=original_secondary_train_pred_fine_batch,
                                original_secondary_train_pred_coarse_batch=original_secondary_train_pred_coarse_batch,
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

                        train_fine_predictions += predicted_fine.tolist()
                        train_coarse_predictions += predicted_coarse.tolist()

                        train_fine_ground_truths += Y_fine_grain.tolist()
                        train_coarse_ground_truths += Y_coarse_grain.tolist()

                        del X, Y_fine_grain, Y_coarse_grain, indices, Y_pred_fine_grain, Y_pred_coarse_grain

                training_fine_accuracy, training_coarse_accuracy = (
                    vit_pipeline.get_and_print_post_epoch_metrics(
                        epoch=epoch,
                        num_batches=num_batches,
                        train_fine_ground_truth=np.array(train_fine_ground_truths),
                        train_fine_prediction=np.array(train_fine_predictions),
                        train_coarse_ground_truth=np.array(train_coarse_ground_truths),
                        train_coarse_prediction=np.array(train_coarse_predictions),
                        num_fine_grain_classes=len(data_preprocessing.fine_grain_classes_str),
                        num_coarse_grain_classes=len(data_preprocessing.coarse_grain_classes_str)))

                train_fine_accuracies += [training_fine_accuracy]
                train_coarse_accuracies += [training_coarse_accuracy]

                train_total_losses += [total_running_loss / num_batches]

                Y_pred_fine = torch.cat(all_Y_pred_fine, dim=0).to(device)
                Y_pred_coarse = torch.cat(all_Y_pred_coarse, dim=0).to(device)

                ltn_support.compute_sat_testing_value(
                    logits_to_predicate=logits_to_predicate,
                    train_pred_fine_batch=Y_pred_fine,
                    train_pred_coarse_batch=Y_pred_coarse,
                    train_true_fine_batch=torch.tensor(
                        data_preprocessing.train_true_fine_data).to(device),
                    train_true_coarse_batch=torch.tensor(
                        data_preprocessing.train_true_coarse_data).to(device),
                    original_train_pred_fine_batch=torch.tensor(
                        self.pred_data['train']['original']['fine']).to(device),
                    original_train_pred_coarse_batch=torch.tensor(
                        self.pred_data['train']['original']['coarse']).to(device),
                    original_secondary_train_pred_fine_batch=torch.tensor(
                        self.pred_data['secondary_model']['train']['fine']).to(device),
                    original_secondary_train_pred_coarse_batch=torch.tensor(
                        self.pred_data['secondary_model']['train']['coarse']).to(device),
                    error_detection_rules=self.error_detection_rules,
                    device=device
                )

                scheduler.step()

                print('#' * 100)

        return train_fine_predictions, train_coarse_predictions

    def run_learning_pipeline(self,
                              EDCR_epoch_num=0):
        print('Started learning pipeline...\n')
        self.print_metrics(test=False, prior=True)

        for g in data_preprocessing.granularities.values():
            self.learn_detection_rules(g=g)

        print('\nRule learning completed\n')

        fine_tuners, loaders, devices, num_fine_grain_classes, num_coarse_grain_classes = (
            vit_pipeline.initiate(
                batch_len=self.batch_size,
                combined=self.combined,
                pretrained_path=self.pretrain_path,
                debug=False,
                get_indices=True))

        if self.correction_model is None:
            self.correction_model = fine_tuners[0]

        with context_handlers.ClearSession():
            self.fine_tune_combined_model(
                fine_tuner=self.correction_model,
                device=devices[0],
                loaders=loaders,
                loss=self.loss,
            )

        print('#' * 100)

    def run_evaluating_pipeline(self):
        _, loaders, devices, num_fine_grain_classes, num_coarse_grain_classes = (
            vit_pipeline.initiate(batch_len=self.batch_size,
                                  lrs=[self.lr],
                                  combined=self.combined,
                                  debug=False,
                                  evaluation=True))

        device = devices[0]
        loader = loaders['test']
        self.correction_model.to(device)
        self.correction_model.eval()

        fine_predictions = []
        coarse_predictions = []

        fine_ground_truths = []
        coarse_ground_truths = []

        print(f'Testing {self.correction_model} on {device}...')

        all_Y_pred_fine = []
        all_Y_pred_coarse = []

        with torch.no_grad():
            if utils.is_local():
                from tqdm import tqdm
                gen = tqdm(enumerate(loader), total=len(loader))
            else:
                gen = enumerate(loader)

            for i, data in gen:
                X, Y_true_fine, Y_true_coarse = data[0].to(device), data[1].to(device), data[3].to(device)

                Y_pred = self.correction_model(X)
                Y_pred_fine = Y_pred[:, :len(data_preprocessing.fine_grain_classes_str)]
                Y_pred_coarse = Y_pred[:, len(data_preprocessing.fine_grain_classes_str):]

                all_Y_pred_fine.append(Y_pred_fine)
                all_Y_pred_coarse.append(Y_pred_coarse)

                predicted_fine = torch.max(Y_pred_fine, 1)[1]
                predicted_coarse = torch.max(Y_pred_coarse, 1)[1]

                fine_ground_truths += Y_true_fine.tolist()
                coarse_ground_truths += Y_true_coarse.tolist()

                fine_predictions += predicted_fine.tolist()
                coarse_predictions += predicted_coarse.tolist()

        fine_accuracy, coarse_accuracy = (
            vit_pipeline.get_and_print_metrics(pred_fine_data=fine_predictions,
                                               pred_coarse_data=coarse_predictions,
                                               loss=self.loss,
                                               true_fine_data=fine_ground_truths,
                                               true_coarse_data=coarse_ground_truths,
                                               test=True))

        Y_pred_fine = torch.cat(all_Y_pred_fine, dim=0)
        Y_pred_coarse = torch.cat(all_Y_pred_coarse, dim=0)

        logits_to_predicate = ltn.Predicate(ltn_support.LogitsToPredicate()).to(ltn.device)

        ltn_support.compute_sat_testing_value(
            logits_to_predicate=logits_to_predicate,
            train_pred_fine_batch=Y_pred_fine,
            train_pred_coarse_batch=Y_pred_coarse,
            train_true_fine_batch=torch.tensor(
                data_preprocessing.train_true_fine_data).to(device),
            train_true_coarse_batch=torch.tensor(
                data_preprocessing.train_true_coarse_data).to(device),
            original_train_pred_fine_batch=torch.tensor(
                self.pred_data['train']['original']['fine']).to(device),
            original_train_pred_coarse_batch=torch.tensor(
                self.pred_data['train']['original']['coarse']).to(device),
            original_secondary_train_pred_fine_batch=torch.tensor(
                self.pred_data['secondary_model']['train']['fine']).to(device),
            original_secondary_train_pred_coarse_batch=torch.tensor(
                self.pred_data['secondary_model']['train']['coarse']).to(device),
            error_detection_rules=self.error_detection_rules,
            device=device
        )


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

        edcr.run_learning_pipeline()
        edcr.run_evaluating_pipeline()
