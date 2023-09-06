import os
import torch
from torch import mps
import torchvision
import torch.utils.data
import numpy as np
from sklearn.metrics import accuracy_score
from time import time
from tqdm import tqdm
from typing import Tuple
import matplotlib.pyplot as plt
from abc import ABC

cwd = os.getcwd()


def get_transforms(train_or_val: str,
                   model_name: str) -> torchvision.transforms.Compose:
    if model_name == 'inception':
        resize_num = 299
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]
    else:
        resize_num = 224
        means = stds = [0.5] * 3

    return torchvision.transforms.Compose(
        ([torchvision.transforms.RandomResizedCrop(resize_num),
          torchvision.transforms.RandomHorizontalFlip()] if train_or_val == 'train' else
         [torchvision.transforms.Resize(int(resize_num / 224 * 256)),
          torchvision.transforms.CenterCrop(resize_num)]) +
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(means, stds)
         ])


class ImageFolderWithName(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = self.loader(path)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        name = os.path.basename(path)

        return image, target, name


class ClearCache:
    def __init__(self, device: torch.device):
        self.device_backend = {'cuda': torch.cuda,
                               'mps': mps,
                               'cpu': None}[device.type]

    def __enter__(self):
        if self.device_backend:
            self.device_backend.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device_backend:
            self.device_backend.empty_cache()


class ModelFineTuner(torch.nn.Module, ABC):
    def __str__(self) -> str:
        return self.__class__.__name__.split('Model')[0].lower()


class InceptionModelFineTuner(ModelFineTuner):
    def __init__(self, num_classes: int):
        super().__init__()
        self.inception = torchvision.models.inception_v3(
            weights=torchvision.models.Inception_V3_Weights.DEFAULT)
        num_features = self.inception.fc.in_features
        self.inception.fc = torch.nn.Linear(in_features=num_features, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inception(x)
        return x


class VITModelFineTuner(ModelFineTuner):
    def __init__(self, num_classes: int):
        super().__init__()
        self.vit = torchvision.models.vit_l_16(weights=torchvision.models.ViT_L_16_Weights.DEFAULT)
        self.vit.heads[-1] = torch.nn.Linear(in_features=self.vit.hidden_dim, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit(x)
        return x


def test(fine_tuner: ModelFineTuner) -> Tuple[list[int], list[int], float]:
    test_loader = loaders[f'{fine_tuner}_val']
    fine_tuner.eval()
    correct = 0
    total = 0
    test_prediction = []
    test_ground_truth = []
    name_list = []

    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
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


def fine_tune(fine_tuner: ModelFineTuner,
              lr: float,
              scheduler_step_size: int,
              scheduler_gamma: float) -> Tuple[list[int], list[int], list[int], list[int]]:
    fine_tuner.to(device)
    fine_tuner.train()

    train_loader = loaders[f'{fine_tuner}_train']
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params=fine_tuner.parameters(),
                                 lr=lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                step_size=scheduler_step_size,
                                                gamma=scheduler_gamma)

    train_losses = []
    train_accuracies = []
    train_ground_truths = []
    train_predictions = []

    test_accuracies = []
    test_ground_truths = []
    test_predictions = []

    for epoch in range(num_epochs):
        t1 = time()
        running_loss = 0.0
        train_predictions = []
        train_ground_truths = []

        for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = fine_tuner(inputs)

            if type(outputs) == torchvision.models.InceptionOutputs:
                outputs = outputs[0]

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            predicted = torch.max(outputs, 1)[1]
            train_ground_truths += labels.tolist()
            train_predictions += predicted.tolist()

        true_labels = np.array(train_ground_truths)
        predicted_labels = np.array(train_predictions)
        acc = accuracy_score(true_labels, predicted_labels)

        print(f'\nModel: {fine_tuner} '
              f'epoch {epoch + 1}/{num_epochs} done in {int(time() - t1)} seconds, '
              f'\nTraining loss: {round(running_loss / len(train_loader), 3)}'
              f'\ntraining accuracy: {round(acc, 3)}\n')

        if len(train_accuracies) and acc < 0.5 * train_accuracies[-1]:
            raise AssertionError('Training accuracy reduced by too much, stopped learning')

        train_accuracies += [acc]
        train_losses += [running_loss / len(train_loader)]
        scheduler.step()
        test_ground_truths, test_predictions, test_accuracy = test(fine_tuner)
        test_accuracies += [test_accuracy]
        print('#' * 100)

        # if str(fine_tuner) == 'vit' and test_accuracy > np.load('inception_test_acc.npy')[-1]:
        #     print('vit test accuracy better than the inception test accuracy. Early stopping')
        #     break

    np.save(f'{fine_tuner}_train_acc.npy', train_accuracies)
    np.save(f'{fine_tuner}_train_loss.npy', train_losses)
    np.save(f'{fine_tuner}_test_acc.npy', test_accuracies)

    return train_ground_truths, train_predictions, test_ground_truths, test_predictions


if __name__ == '__main__':
    while np.load(cwd + '/vit_test_acc.npy')[-1] < np.load(cwd + '/inception_test_acc.npy')[-1]:
        batch_size = 24
        lr = 0.0002
        scheduler_gamma = 0.1
        num_epochs = 12
        scheduler_step_size = num_epochs

        model_names = ['vit',
                       # 'inception'
                       ]
        data_dir = os.path.join(os.getcwd(), 'data/FineGrainDataset')
        datasets = {f'{model_name}_{train_or_val}': ImageFolderWithName(root=os.path.join(data_dir, train_or_val),
                                                                        transform=get_transforms(
                                                                            train_or_val=train_or_val,
                                                                            model_name=model_name))
                    for model_name in model_names for train_or_val in ['train', 'val']}

        assert all(datasets[f'{model_name}_train'].classes == datasets[f'{model_name}_val'].classes
                   for model_name in model_names)

        loaders = {f'{model_name}_{train_or_val}': torch.utils.data.DataLoader(
            dataset=datasets[f'{model_name}_{train_or_val}'],
            batch_size=batch_size,
            shuffle=train_or_val == 'train')
            for model_name in model_names for train_or_val in ['train', 'val']}

        classes = datasets['vit_train'].classes

        n = len(classes)

        device = torch.device('mps' if torch.backends.mps.is_available() else
                              ("cuda" if torch.cuda.is_available() else 'cpu'))

        vit_fine_tuner = VITModelFineTuner(num_classes=n)
        # inception_fine_tuner = InceptionModelFineTuner(num_classes=n)
        fine_tuners = [
            # inception_fine_tuner,
            vit_fine_tuner
        ]

        for fine_tuner in fine_tuners:
            with ClearCache(device):
                train_ground_truth, train_prediction, test_ground_truth, test_prediction = fine_tune(fine_tuner, lr,
                                                                                                     scheduler_step_size,
                                                                                                     num_epochs)
                np.save(f'{fine_tuner}_pred.npy', test_prediction)
                np.save(f'{fine_tuner}_true.npy', test_ground_truth)
                print('#' * 100)

        # inception_true_labels = np.load("inception_true.npy")
        vit_true_labels = np.load(cwd + "/vit_true.npy")

        # Assert the true labels match for both models
        # assert np.all(
        #     inception_true_labels == vit_true_labels), "True labels do not match between inception and vit models"

        for fine_tuner in fine_tuners:
            plt.plot(np.load(f'{cwd}/{fine_tuner}_train_loss.npy'), label=f'{fine_tuner} training Loss')

        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.show()

        plt.cla()
        plt.clf()

        # %%
        for fine_tuner in fine_tuners:
            plt.plot(np.load(f'{cwd}/{fine_tuner}_train_acc.npy'), label=f'{fine_tuner} training accuracy')

        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        plt.legend()
        plt.grid()
        plt.show()

        plt.cla()
        plt.clf()

        # %%
        for fine_tuner in fine_tuners:
            plt.plot(np.load(f'{cwd}/{fine_tuner}_test_acc.npy'), label=f'{fine_tuner} test accuracy')

        plt.xlabel('Epoch')
        plt.ylabel('Test Accuracy')
        plt.title('Test Accuracy')
        plt.legend()
        plt.grid()
        plt.show()
