import json
import os
import torch
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize
from transformers import AutoImageProcessor, AutoModelForImageClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import numpy as np
from transformers import TrainingArguments, Trainer


def prepare(batch: dict,
            train_transform,
            processor,
            mode: str = "train",
            ):
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


# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    accuracy = accuracy_score(y_pred=predictions, y_true=eval_pred.label_ids)
    return {"accuracy": accuracy}


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    return {"pixel_values": pixel_values, "labels": torch.tensor([example["label"] for example in examples])}


def fine_tune_image_net(imagenet100_folder_path: str = 'ImageNet100',
                        model_name: str = "facebook/dinov2-large",
                        lr: float = 5e-5,
                        num_train_epochs: int = 3,
                        batch_size: int = 32):
    with open(f'{imagenet100_folder_path}/Labels.json', 'r') as f:
        labels = json.load(f)

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

    id2label = {ID: label for ID, label in enumerate(labels)}
    label2id = {label: ID for ID, label in enumerate(labels)}

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name,
                                                            id2label=id2label,
                                                            label2id=label2id)

    # for training, we use some image transformations from Torchvision
    # feel free to use other libraries like Albumentations or Kornia here
    train_transform = Compose([
        RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=processor.resample),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4)),
        ToTensor(),
        Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

    # see https://huggingface.co/docs/datasets/image_dataset to load your own custom dataset
    dataset = load_dataset("timm/oxford-iiit-pet")

    # set num_proc equal to the number of CPU cores on your machine
    # see https://docs.python.org/3/library/multiprocessing.html#multiprocessing.cpu_count
    train_dataset = dataset["train"].map(prepare,
                                         num_proc=os.cpu_count(),
                                         batched=True,
                                         batch_size=batch_size,
                                         fn_kwargs={"mode": "train",
                                                    "train_transform": train_transform,
                                                    "processor": processor})
    eval_dataset = dataset["test"].map(prepare,
                                       num_proc=os.cpu_count(),
                                       batched=True,
                                       batch_size=batch_size,
                                       fn_kwargs={"mode": "test",
                                                  "train_transform": train_transform,
                                                  "processor": processor
                                                  })

    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")

    args = TrainingArguments(
        output_dir=f"{model_name}-finetuned-oxford",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=4,
        num_train_epochs=num_train_epochs,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        # use_cpu=True
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    trainer.train()


if __name__ == '__main__':
    fine_tune_image_net()
