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
from pathlib import Path
import sys
import timm

batch_size = 24
lr = 2e-4
scheduler_gamma = 0.1
num_epochs = 2
cwd = Path(__file__).parent.resolve()
scheduler_step_size = num_epochs

vit_model_names = {0: 'b_16',
                   1: 'b_32',
                   2: 'l_16',
                   3: 'l_32',
                   4: 'h_14'}

vit_model_indices = [3]
train_folder_name = 'train'
test_folder_name = 'test'


def get_transforms(train_or_val: str,
                   model_name: str) -> torchvision.transforms.Compose:
    if model_name == 'inception':
        resize_num = 299
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]
    else:
        resize_num = 518 if model_name == 'h_14' else 224
        means = stds = [0.5] * 3

    return torchvision.transforms.Compose(
        ([torchvision.transforms.RandomResizedCrop(resize_num),
          torchvision.transforms.RandomHorizontalFlip()] if train_or_val == train_folder_name else
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


def is_running_in_colab() -> bool:
    """
    Check if the code is running in Google Colab.
    Returns:
        True if running in Google Colab, False otherwise.
    """

    return 'google.colab' in sys.modules


class ClearSession:
    def __init__(self):
        self.colab = False
        if is_running_in_colab():
            from google.colab import drive

            # Mount Google Drive
            self.drive = drive
            self.drive.mount('/content/drive')
            self.colab = True

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.colab:
            self.drive.flush_and_unmount()


class FineTuner(torch.nn.Module, ABC):
    def __str__(self) -> str:
        return self.__class__.__name__.split('Fine')[0].lower()

    def __len__(self) -> int:
        return sum(p.numel() for p in self.parameters())


class InceptionV3FineTuner(FineTuner):
    def __init__(self, num_classes: int):
        super().__init__()
        self.inception = torchvision.models.inception_v3(
            weights=torchvision.models.Inception_V3_Weights.DEFAULT)
        num_features = self.inception.fc.in_features
        self.inception.fc = torch.nn.Linear(in_features=num_features, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inception(x)
        return x


class VITFineTuner(FineTuner):
    def __init__(self,
                 vit_model_index: int,
                 num_classes: int):
        super().__init__()
        self.vit_model_name = vit_model_names[vit_model_index]
        vit_model = eval(f'torchvision.models.vit_{self.vit_model_name}')
        vit_weights = eval(f"torchvision.models.ViT_{'_'.join([s.upper() for s in self.vit_model_name.split('_')])}"
                           f"_Weights.DEFAULT")
        self.vit = vit_model(weights=vit_weights)
        self.vit.heads[-1] = torch.nn.Linear(in_features=self.vit.hidden_dim,
                                             out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit(x)
        return x

    def __str__(self):
        return f'{super().__str__()}_{self.vit_model_name}'


class InceptionResNetV2FineTuner(FineTuner):
    def __init__(self, num_classes: int):
        super().__init__()
        self.inception = timm.create_model('inception_resnet_v2',
                                           pretrained=True,
                                           num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inception(x)
        return x


def test(fine_tuner: FineTuner) -> Tuple[list[int], list[int], float]:
    test_loader = loaders[f'{fine_tuner}_{test_folder_name}']
    fine_tuner.eval()
    correct = 0
    total = 0
    test_prediction = []
    test_ground_truth = []
    name_list = []

    print(f'Started testing {fine_tuner} on {device}...')

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


def fine_tune(fine_tuner: FineTuner) -> Tuple[list[int], list[int], list[int], list[int]]:
    fine_tuner.to(device)
    fine_tuner.train()

    train_loader = loaders[f'{fine_tuner}_{train_folder_name}']
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

    print(f'Started fine-tuning {fine_tuner} on {device}...')

    for epoch in range(num_epochs):
        print(f'Started epoch {epoch + 1}/{num_epochs}...')
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

        print(f'\nModel: {fine_tuner} with {len(fine_tuner)} parameters\n'
              f'epoch {epoch + 1}/{num_epochs} done in {int(time() - t1)} seconds, '
              f'\nTraining loss: {round(running_loss / len(train_loader), 3)}'
              f'\ntraining accuracy: {round(acc, 3)}\n')

        # if len(train_accuracies) and acc < 0.5 * train_accuracies[-1]:
        #     raise AssertionError('Training accuracy reduced by too much, stopped learning')

        train_accuracies += [acc]
        train_losses += [running_loss / len(train_loader)]
        scheduler.step()
        test_ground_truths, test_predictions, test_accuracy = test(fine_tuner)
        test_accuracies += [test_accuracy]
        print('#' * 100)

        # if str(fine_tuner).__contains__('vit') and test_accuracy > 0.8:
        #     print('vit test accuracy better than the inception test accuracy. Early stopping')
        #     break
    colab_path = '/content/drive/My Drive/' if is_running_in_colab() else ''

    np.save(f"{colab_path}{fine_tuner}_train_acc.npy", train_accuracies)
    np.save(f"{colab_path}{fine_tuner}_train_loss.npy", train_losses)

    test_accuracies_filename = Path(f"{colab_path}{fine_tuner}_test_acc.npy")
    if (not Path.is_file(test_accuracies_filename)) or (test_accuracies[-1] > np.load(test_accuracies_filename)[-1]):
        np.save(test_accuracies_filename, test_accuracies)

    return train_ground_truths, train_predictions, test_ground_truths, test_predictions


if __name__ == '__main__':
    model_names = [f'vit_{vit_model_names[vit_model_index]}' for vit_model_index in vit_model_indices]
    data_dir = Path.joinpath(cwd, '.')
    datasets = {f'{model_name}_{train_or_val}': ImageFolderWithName(root=os.path.join(data_dir, train_or_val),
                                                                    transform=get_transforms(train_or_val=train_or_val,
                                                                                             model_name=model_name))
                for model_name in model_names for train_or_val in [train_folder_name, test_folder_name]}

    # assert all(datasets[f'{model_name}_{train_folder_name}'].classes ==
    #            datasets[f'{model_name}_{test_folder_name}'].classes
    #            for model_name in model_names)

    loaders = {f"{model_name}_{train_or_val}": torch.utils.data.DataLoader(
        dataset=datasets[f'{model_name}_{train_or_val}'],
        batch_size=batch_size,
        shuffle=train_or_val == train_folder_name)
        for model_name in model_names for train_or_val in [train_folder_name, test_folder_name]}

    classes = datasets[f'{model_names[0]}_{train_folder_name}'].classes

    n = len(classes)

    device = torch.device('mps' if torch.backends.mps.is_available() else
                          ("cuda" if torch.cuda.is_available() else 'cpu'))

    fine_tuners = [eval(
        f"{'VIT' if model_name.__contains__('vit') else 'Inception' if model_name == 'inception' else 'InceptionResNetV2'}"
        f"FineTuner({'vit_model_index=model_index ,' if model_name.__contains__('vit') else ''}num_classes=n)")
        for model_index, model_name in zip(vit_model_indices, model_names)]

    for fine_tuner in fine_tuners:
        with ClearCache(device):
            colab_path = '/content/drive/My Drive/' if is_running_in_colab() else ''
            with ClearSession():
                train_ground_truth, train_prediction, \
                    test_ground_truth, test_prediction = fine_tune(fine_tuner)
                np.save(f"{colab_path}{fine_tuner}_pred.npy", test_prediction)
                np.save(f"{colab_path}{fine_tuner}_true.npy", test_ground_truth)
                torch.save(fine_tuner.state_dict(), f'{fine_tuner}.pth')
                print('#' * 100)

    for fine_tuner in fine_tuners:
        vit_train_loss = np.load(Path.joinpath(cwd, f'{fine_tuner}_train_loss.npy'))
        plt.plot(vit_train_loss, label=f'{fine_tuner} training Loss')

    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    plt.cla()
    plt.clf()

    for fine_tuner in fine_tuners:
        vit_train_acc = np.load(Path.joinpath(cwd, f'{fine_tuner}_train_acc.npy'))
        plt.plot(vit_train_acc, label=f'{fine_tuner} training accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.grid()
    plt.show()

    plt.cla()
    plt.clf()

    for fine_tuner in fine_tuners:
        vit_test_acc = np.load(f'{fine_tuner}_test_acc.npy')
        plt.plot(vit_test_acc, label=f'{fine_tuner} test accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy')
    plt.legend()
    plt.grid()
    plt.show()
