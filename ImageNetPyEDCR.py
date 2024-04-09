import json
import os
import torch.utils.data
import context_handlers
import copy
import numpy as np
import torch
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import accuracy_score
from datasets import load_dataset

from PyEDCR import EDCR
import data_preprocessing
import vit_pipeline
import typing
import config as configuration
import neural_evaluation
import neural_fine_tuning
import neural_metrics
import models
import utils

preprocessor = data_preprocessing.DataPreprocessor(data_str='imagenet')

imagenet100_folder_path = 'data/ImageNet100'

with open(f'{imagenet100_folder_path}/Labels.json', 'r') as f:
    labels = json.load(f)

print(labels)
coarse_grain_classes_list = [
    'Bird', 'Snake', 'Spider', 'Small Fish', 'Turtle', 'Lizard', 'Crab', 'Shark'
]

fine_grain_classes_dict = {
    'n01818515': 'macaw',
    'n01537544': 'indigo bunting, indigo finch, indigo bird, Passerina cyanea',
    'n02007558': 'flamingo',
    'n02002556': 'white stork, Ciconia ciconia',
    'n01614925': 'bald eagle, American eagle, Haliaeetus leucocephalus',
    'n01582220': 'magpie',
    'n01806143': 'peacock',
    'n01795545': 'black grouse',
    'n01531178': 'goldfinch, Carduelis carduelis',
    'n01622779': 'great grey owl, great gray owl, Strix nebulosa',
    'n01833805': 'hummingbird',
    'n01740131': 'night snake, Hypsiglena torquata',
    'n01735189': 'garter snake, grass snake',
    'n01755581': 'diamondback, diamondback rattlesnake, Crotalus adamanteus',
    'n01751748': 'sea snake',
    'n01729977': 'green snake, grass snake',
    'n01729322': 'hognose snake, puff adder, sand viper',
    'n01734418': 'king snake, kingsnake',
    'n01728572': 'thunder snake, worm snake, Carphophis amoenus',
    'n01739381': 'vine snake',
    'n01756291': 'sidewinder, horned rattlesnake, Crotalus cerastes',
    'n01773797': 'garden spider, Aranea diademata',
    'n01775062': 'wolf spider, hunting spider',
    'n01773549': 'barn spider, Araneus cavaticus',
    'n01774384': 'black widow, Latrodectus mactans',
    'n01774750': 'tarantula',
    'n01440764': 'tench, Tinca tinca',
    'n01443537': 'goldfish, Carassius auratus',
    'n01667778': 'terrapin',
    'n01667114': 'mud turtle',
    'n01664065': 'loggerhead, loggerhead turtle, Caretta caretta',
    'n01665541': 'leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea',
    'n01687978': 'agama',
    'n01677366': 'common iguana, iguana, Iguana iguana',
    'n01695060': 'Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis',
    'n01685808': 'whiptail, whiptail lizard',
    'n01978287': 'Dungeness crab, Cancer magister',
    'n01986214': 'hermit crab',
    'n01978455': 'rock crab, Cancer irroratus',
    'n01491361': 'tiger shark, Galeocerdo cuvieri',
    'n01484850': 'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias',
    'n01494475': 'hammerhead, hammerhead shark'
}

coarse_to_fine_dict = {
    'Bird': ['Macaw',
             'Indigo Bunting',
             'Flamingo',
             'White Stork',
             'Bald Eagle',
             'Magpie',
             'Peacock',
             'Black Grouse',
             'Goldfinch',
             'Great Grey Owl',
             'Hummingbird'],
    'Snake': ['Snake1',
              'Snake2',
              'Snake3',
              'Snake4',
              'Snake5',
              'Snake6',
              'Snake7',
              'Snake8',
              'Snake9',
              'Snake10'],
    'Spider': ['Garden Spider',
               'Wolf Spider',
               'Barn Spider',
               'Black Widow',
               'Tarantula'],
    'Small Fish': ['Tench', 'Goldfish'],
    'Turtle': ['Terrapin',
               'Mud Turtle',
               'Loggerhead Turtle',
               'Leatherback Turtle'],
    'Lizard': ['Agama', 'Common Iguana', 'Komodo Dragon', 'Whiptail Lizard'],
    'Crab': ['Dungeness Crab', 'Hermit Crab', 'Rock Crab'],
    'Shark': ['Tiger Shark', 'Great White Shark', 'Hammerhead Shark']}

fine_to_coarse_dict = {
    'Macaw': 'Bird',
    'Indigo Bunting': 'Bird',
    'Flamingo': 'Bird',
    'White Stork': 'Bird',
    'Bald Eagle': 'Bird',
    'Magpie': 'Bird',
    'Peacock': 'Bird',
    'Black Grouse': 'Bird',
    'Goldfinch': 'Bird',
    'Great Grey Owl': 'Bird',
    'Hummingbird': 'Bird',
    'Snake1': 'Snake',
    'Snake2': 'Snake',
    'Snake3': 'Snake',
    'Snake4': 'Snake',
    'Snake5': 'Snake',
    'Snake6': 'Snake',
    'Snake7': 'Snake',
    'Snake8': 'Snake',
    'Snake9': 'Snake',
    'Snake10': 'Snake',
    'Garden Spider': 'Spider',
    'Wolf Spider': 'Spider',
    'Barn Spider': 'Spider',
    'Black Widow': 'Spider',
    'Tarantula': 'Spider',
    'Tench': 'Small Fish',
    'Goldfish': 'Small Fish',
    'Terrapin': 'Turtle',
    'Mud Turtle': 'Turtle',
    'Loggerhead Turtle': 'Turtle',
    'Leatherback Turtle': 'Turtle',
    'Agama': 'Lizard',
    'Common Iguana': 'Lizard',
    'Komodo Dragon': 'Lizard',
    'Whiptail Lizard': 'Lizard',
    'Dungeness Crab': 'Crab',
    'Hermit Crab': 'Crab',
    'Rock Crab': 'Crab',
    'Tiger Shark': 'Shark',
    'Great White Shark': 'Shark',
    'Hammerhead Shark': 'Shark'
}

id2label = {ID: label for ID, label in enumerate(labels)}
label2id = {label: ID for ID, label in enumerate(labels)}

model_name = "facebook/dinov2-large"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name, id2label=id2label, label2id=label2id)

# make sure to use the appropriate image mean, std and interpolation
# of the inference processor
mean = processor.image_mean
std = processor.image_std
interpolation = processor.resample

# for training, we use some image transformations from Torchvision
# feel free to use other libraries like Albumentations or Kornia here
train_transform = Compose([
    RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=interpolation),
    RandomHorizontalFlip(p=0.5),
    ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4)),
    ToTensor(),
    Normalize(mean=mean, std=std),
])


def prepare(batch, mode="train"):
    # get images
    images = batch["image"]

    # prepare for the model
    if mode == "train":
        images = [train_transform(image.convert("RGB")) for image in images]
        pixel_values = torch.stack(images)
    elif mode == "test":
        pixel_values = processor(images, return_tensors="pt").pixel_values
    else:
        raise ValueError(f"Mode {mode} not supported")

    inputs = {"pixel_values": pixel_values, "labels": torch.tensor(batch["label"])}

    return inputs


# see https://huggingface.co/docs/datasets/image_dataset to load your own custom dataset
dataset = load_dataset("timm/oxford-iiit-pet")

# set num_proc equal to the number of CPU cores on your machine
# see https://docs.python.org/3/library/multiprocessing.html#multiprocessing.cpu_count
train_dataset = dataset["train"].map(prepare, num_proc=os.cpu_count(), batched=True, batch_size=20,
                                     fn_kwargs={"mode": "train"})
eval_dataset = dataset["test"].map(prepare, num_proc=os.cpu_count(), batched=True, batch_size=20,
                                   fn_kwargs={"mode": "test"})

train_dataset.set_format("torch")
eval_dataset.set_format("torch")


# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    accuracy = accuracy_score(y_pred=predictions, y_true=eval_pred.label_ids)
    return {"accuracy": accuracy}


from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    f"{model_name}-finetuned-oxford",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    return {"pixel_values": pixel_values, "labels": torch.tensor([example["label"] for example in examples])}


trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

trainer.train()


class EDCR_Imagenet100_experiment(EDCR):
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
        super().__init__(
            data='imagenet',
            main_model_name=main_model_name,
            combined=combined,
            loss=loss,
            lr=lr,
            original_num_epochs=num_epochs,
            epsilon=epsilon,
            K_train=K_train,
            K_test=K_test,
            include_inconsistency_constraint=include_inconsistency_constraint,
            secondary_model_name=secondary_model_name)
        self.batch_size = config.batch_size
        self.scheduler_gamma = config.scheduler_gamma
        self.num_imagenet_epoch = config.num_imagenet_epoch
        self.num_edcr_epoch = config.num_edcr_epoch
        self.scheduler_step_size = num_epochs
        self.pretrain_path = config.main_pretrained_path
        self.correction_model = {}
        self.num_models = config.num_models
        self.get_fraction_of_example_with_label = config.get_fraction_of_example_with_label

        self.formatted_removed_label = ",".join(f"{key},{value}"
                                                for key, value in self.get_fraction_of_example_with_label.items())

        if self.pretrain_path is None or not os.path.exists(self.pretrain_path):
            raise FileNotFoundError('Need to specify pretrain model!')

        self.fine_tuners, self.loaders, self.devices, _, _ = (
            vit_pipeline.initiate(
                data=self.data,
                lrs=[self.lr],
                combined=self.combined,
                debug=False,
                pretrained_path=self.pretrain_path,
                get_indices=True,
                train_eval_split=0.8,
                get_fraction_of_example_with_label=self.get_fraction_of_example_with_label))

        self.baseline_model = self.fine_tuners[0]

        neural_evaluation.evaluate_combined_model(fine_tuner=self.baseline_model,
                                                  loaders=self.loaders,
                                                  loss='BCE',
                                                  device=self.devices[0],
                                                  split='test')

    def fine_tune_and_evaluate_combined_model(self,
                                              fine_tuner: models.FineTuner,
                                              device: torch.device,
                                              loaders: dict[str, torch.utils.data.DataLoader],
                                              loss: str,
                                              mode: str,
                                              epoch: int = 0,
                                              optimizer=None,
                                              scheduler=None):
        fine_tuner.to(device)
        fine_tuner.train()
        loader = loaders[mode]
        num_batches = len(loader)
        train_or_test = 'train' if mode != 'test' else 'test'

        print(f'\n{mode} {fine_tuner} with {len(fine_tuner)} parameters for {self.num_imagenet_epoch} epochs '
              f'using lr={self.lr} on {device}...')
        print('#' * 100 + '\n')

        with context_handlers.TimeWrapper():
            total_running_loss = 0.0

            fine_predictions = []
            coarse_predictions = []

            fine_ground_truths = []
            coarse_ground_truths = []

            batches = neural_fine_tuning.get_fine_tuning_batches(train_loader=loader,
                                                                 num_batches=num_batches,
                                                                 debug=False)

            for batch_num, batch in batches:
                with context_handlers.ClearCache(device=device):
                    X, Y_fine_grain, Y_coarse_grain, indices = (
                        batch[0].to(device), batch[1].to(device), batch[3].to(device), batch[4])

                    Y_fine_grain_one_hot = torch.nn.functional.one_hot(
                        Y_fine_grain, num_classes=preprocessor.num_fine_grain_classes)
                    Y_coarse_grain_one_hot = torch.nn.functional.one_hot(
                        Y_coarse_grain, num_classes=preprocessor.num_coarse_grain_classes)

                    Y_combine = torch.cat(tensors=[Y_fine_grain_one_hot, Y_coarse_grain_one_hot], dim=1).float()

                    # currently we have many option to get prediction, depend on whether fine_tuner predict
                    # fine / coarse or both
                    Y_pred = fine_tuner(X)

                    Y_pred_fine_grain = Y_pred[:, :preprocessor.num_fine_grain_classes]
                    Y_pred_coarse_grain = Y_pred[:, preprocessor.num_coarse_grain_classes:]

                    if mode == 'train' and optimizer is not None and scheduler is not None:
                        optimizer.zero_grad()

                        if loss == 'BCE':
                            criterion = torch.nn.BCEWithLogitsLoss()
                            batch_total_loss = criterion(Y_pred, Y_combine)

                        if loss == "soft_marginal":
                            criterion = torch.nn.MultiLabelSoftMarginLoss()
                            batch_total_loss = criterion(Y_pred, Y_combine)

                        neural_metrics.print_post_batch_metrics(batch_num=batch_num,
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

                    del X, Y_fine_grain, Y_coarse_grain, indices, Y_pred_fine_grain, Y_pred_coarse_grain

        fine_accuracy, coarse_accuracy = neural_metrics.get_and_print_post_epoch_metrics(
            epoch=epoch,
            num_epochs=self.num_imagenet_epoch,
            train_fine_ground_truth=np.array(fine_ground_truths),
            train_fine_prediction=np.array(fine_predictions),
            train_coarse_ground_truth=np.array(coarse_ground_truths),
            train_coarse_prediction=np.array(coarse_predictions))

        if mode == 'train':
            scheduler.step()

        print('#' * 100)

        return fine_accuracy, coarse_accuracy, fine_predictions, coarse_predictions

    def run_learning_pipeline(self,
                              model_index: int):

        self.correction_model[model_index] = copy.deepcopy(self.baseline_model)

        print('Started learning pipeline...\n')

        for g in preprocessor.granularities.values():
            self.learn_detection_rules(g=g)

        perceived_examples_with_errors = set()
        for g in preprocessor.granularities.values():
            perceived_examples_with_errors = perceived_examples_with_errors.union(set(
                np.where(self.get_predictions(test=False, g=g, stage='post_detection') == -1)[0]))

        perceived_examples_with_errors = np.array(list(perceived_examples_with_errors))

        print(utils.red_text(f'\nNumber of perceived train errors: {len(perceived_examples_with_errors)} / '
                             f'{self.T_train}\n'))

        print('\nRule learning completed\n')
        print(f'\nStarted train model from the error for model {model_index}...\n')

        fine_tuners, loaders, devices, _, _ = (
            vit_pipeline.initiate(
                data=self.data,
                lrs=[self.lr],
                combined=self.combined,
                debug=False,
                pretrained_path=self.pretrain_path,
                get_indices=True,
                train_eval_split=0.8,
                error_indices=perceived_examples_with_errors,
                get_fraction_of_example_with_label=self.get_fraction_of_example_with_label))

        train_fine_accuracies = []
        train_coarse_accuracies = []

        optimizer = torch.optim.Adam(params=self.correction_model[model_index].parameters(),
                                     lr=self.lr)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                    step_size=self.scheduler_step_size,
                                                    gamma=self.scheduler_gamma)
        for epoch in range(self.num_imagenet_epoch):
            with context_handlers.ClearSession():

                self.fine_tune_and_evaluate_combined_model(
                    fine_tuner=self.correction_model[model_index],
                    device=self.devices[0],
                    loaders=self.loaders,
                    loss=self.loss,
                    epoch=epoch,
                    mode='train',
                    optimizer=optimizer,
                    scheduler=scheduler
                )

                training_fine_accuracy, training_coarse_accuracy, _, _ = self.fine_tune_and_evaluate_combined_model(
                    fine_tuner=self.correction_model[model_index],
                    device=self.devices[0],
                    loaders=self.loaders,
                    loss=self.loss,
                    epoch=epoch,
                    mode='train_eval'
                )

                train_fine_accuracies += [training_fine_accuracy]
                train_coarse_accuracies += [training_coarse_accuracy]

                slicing_window_last = (sum(train_fine_accuracies[-3:]) + sum(train_coarse_accuracies[-3:])) / 6
                slicing_window_before_last = (sum(train_fine_accuracies[-4:-2]) + sum(
                    train_coarse_accuracies[-4:-2])) / 6

                if epoch >= 6 and slicing_window_last <= slicing_window_before_last:
                    break

        print(f'\nfinish train and eval model {model_index}!\n')

        print('#' * 100)

    def run_evaluating_pipeline(self,
                                model_index: int):

        print(f'\nStarted testing Imagenet 100 model {model_index}...\n')

        _, _, fine_predictions, coarse_prediction = self.fine_tune_and_evaluate_combined_model(
            fine_tuner=self.correction_model[model_index],
            device=self.devices[0],
            loaders=self.loaders,
            loss=self.loss,
            mode='test'
        )
        return fine_predictions, coarse_prediction

    def get_majority_vote(self,
                          predictions: dict[int, tuple[list, list]],
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
        for idx in range(self.num_models):
            for edcr_epoch in range(self.num_edcr_epoch):
                self.run_learning_pipeline(model_index=idx)
            fine_prediction[idx], coarse_prediction[idx] = self.run_evaluating_pipeline(model_index=idx)

        print("\nGot all the prediction from test model!\n")

        final_fine_prediction = self.get_majority_vote(fine_prediction,
                                                       g=data_preprocessing.granularities['fine'])
        final_coarse_prediction = self.get_majority_vote(coarse_prediction,
                                                         g=data_preprocessing.granularities['coarse'])

        np.save(f"combined_results/Imagenet100_EDCR_result/vit_b_16_test_fine_{self.loss}_"
                f"lr_{self.lr}_batch_size_{self.batch_size}_num_EDCR_epoch_{self.num_edcr_epoch}_"
                f"num_Imagenet100_epoch_{self.num_imagenet_epoch}_num_model_{self.num_models}_"
                f"remove_label_{self.formatted_removed_label}.npy",
                np.array(final_fine_prediction))

        np.save(f"combined_results/Imagenet100_EDCR_result/vit_b_16_test_coarse_{self.loss}_"
                f"lr_{self.lr}_batch_size_{self.batch_size}_num_EDCR_epoch_{self.num_edcr_epoch}_"
                f"num_Imagenet100_epoch_{self.num_imagenet_epoch}_num_model_{self.num_models}_"
                f"remove_label_{self.formatted_removed_label}.npy",
                np.array(final_coarse_prediction))

        neural_metrics.get_and_print_metrics(pred_fine_data=np.array(final_fine_prediction),
                                             pred_coarse_data=np.array(final_coarse_prediction),
                                             loss=self.loss,
                                             true_fine_data=data_preprocessing.get_ground_truths(
                                                 test=True,
                                                 g=data_preprocessing.granularities['fine']),
                                             true_coarse_data=data_preprocessing.get_ground_truths(
                                                 test=True,
                                                 g=data_preprocessing.granularities['coarse']),
                                             test=True)


if __name__ == '__main__':
    epsilons = [0.1 * i for i in range(2, 3)]
    test_bool = False
    main_pretrained_path = configuration

    for eps in epsilons:
        print('#' * 25 + f'eps = {eps}' + '#' * 50)
        edcr = EDCR_Imagenet100_experiment(
            epsilon=eps,
            main_model_name=configuration.vit_model_names[0],
            combined=configuration.combined,
            loss=configuration.loss,
            lr=configuration.lr,
            num_epochs=configuration.num_epochs,
            include_inconsistency_constraint=configuration.include_inconsistency_constraint,
            secondary_model_name=configuration.secondary_model_name,
            config=configuration)
        edcr.run_evaluating_pipeline_all_models()
