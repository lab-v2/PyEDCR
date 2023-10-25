import os
import torch

if torch.backends.mps.is_available():
    from torch import mps

import torchvision
import torch.utils.data
import numpy as np
from sklearn.metrics import accuracy_score
from time import time
from typing import Tuple, Union
# import matplotlib.pyplot as plt
import abc
from datetime import timedelta
# from tqdm import tqdm
from pathlib import Path
import sys
# import timm
import re

batch_size = 24
lrs = [1e-5, 5e-5, 1e-6, 5e-6]
scheduler_gamma = 0.1
num_epochs = 4
vit_model_names = {
    # 0: 'b_16',
    #                1: 'b_32',
    #                2: 'l_16',
    #                3: 'l_32',
                   4: 'h_14'
}
cwd = Path(__file__).parent.resolve()
scheduler_step_size = num_epochs

# vit_model_indices = list(vit_model_names.keys())
train_folder_name = 'train'
test_folder_name = 'test'


def format_seconds(seconds):
    # Create a timedelta object with the given seconds
    time_delta = timedelta(seconds=seconds)

    # Use the total_seconds() method to get the total number of seconds
    total_seconds = time_delta.total_seconds()

    # Use divmod to get the hours and minutes
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Create the formatted string
    if hours > 0:
        return f"{int(hours)} hour{'s' if hours > 1 else ''}"
    elif minutes > 0:
        return f"{int(minutes)} minute{'s' if minutes > 1 else ''}"
    else:
        return f"{int(seconds)} second{'s' if seconds > 1 else ''}"


# Function to create a directory if it doesn't exist
def create_directory(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f'Created {directory}')


def get_transforms(train_or_val: str,
                   model_name: str) -> torchvision.transforms.Compose:
    if model_name == 'inception_v3':
        resize_num = 299
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]
    else:
        resize_num = 518 if 'h_14' in model_name else 224
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


class Context(abc.ABC):
    @abc.abstractmethod
    def __enter__(self):
        pass

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class ClearCache(Context):
    def __init__(self,
                 device: torch.device):
        self.device_backend = {'cuda': torch.cuda,
                               'mps': mps if torch.backends.mps.is_available() else None,
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


def is_local() -> bool:
    return Path(__file__).parent.parent.name == 'PycharmProjects'


colab_path = '/content/drive/My Drive/' if is_running_in_colab() else ''
results_path = fr'{colab_path}results/'
create_directory(results_path)


class ClearSession(Context):
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


class FineTuner(torch.nn.Module, abc.ABC):
    def __init__(self,
                 num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def __str__(self) -> str:
        return re.sub(pattern=r'([a-z])([A-Z0-9])',
                      repl=r'\1_\2',
                      string=self.__class__.__name__.split('Fine')[0]).lower()

    def __len__(self) -> int:
        return sum(p.numel() for p in self.parameters())


class InceptionV3FineTuner(FineTuner):
    def __init__(self,
                 num_classes: int):
        super().__init__(num_classes=num_classes)
        self.inception = torchvision.models.inception_v3(
            weights=torchvision.models.Inception_V3_Weights.DEFAULT)
        num_features = self.inception.fc.in_features
        self.inception.fc = torch.nn.Linear(in_features=num_features, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inception(x)
        return x


class VITFineTuner(FineTuner):
    def __init__(self,
                 vit_model: Union[int, str],
                 num_classes: int):
        super().__init__(num_classes=num_classes)
        if isinstance(vit_model, int):
            self.vit_model_index = vit_model
            self.vit_model_name = vit_model_names[vit_model]
        else:
            self.vit_model_index = list(vit_model_names.keys())[
                list(vit_model_names.values()).index(vit_model.split('vit_')[-1])]
            self.vit_model_name = vit_model

        vit_model = eval(f'torchvision.models.{self.vit_model_name}')
        vit_weights = eval(
            f"torchvision.models.ViT_{'_'.join([s.upper() for s in self.vit_model_name.split('vit_')[-1].split('_')])}"
            f"_Weights.DEFAULT")
        self.vit = vit_model(weights=vit_weights)
        self.vit.heads[-1] = torch.nn.Linear(in_features=self.vit.hidden_dim,
                                             out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit(x)
        return x

    def __str__(self):
        return self.vit_model_name


# class InceptionResNetV2FineTuner(FineTuner):
#     def __init__(self, num_classes: int):
#         super().__init__(num_classes=num_classes)
#         self.inception = timm.create_model('inception_resnet_v2',
#                                            pretrained=True,
#                                            num_classes=num_classes)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.inception(x)
#         return x


def test(fine_tuner: FineTuner,
         loaders,
         device) -> Tuple[list[int], list[int], float]:
    test_loader = loaders[f'{fine_tuner}_{test_folder_name}']
    fine_tuner.eval()
    correct = 0
    total = 0
    test_prediction = []
    test_ground_truth = []
    name_list = []

    print(f'Started testing {fine_tuner} on {device}...')

    with torch.no_grad():
        try:
            from tqdm import tqdm
            gen = tqdm(enumerate(test_loader), total=len(test_loader))
        except:
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
              device,
              loaders):
    fine_tuner.to(device)
    fine_tuner.train()

    train_loader = loaders[f'{fine_tuner}_{train_folder_name}']
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
            with ClearCache(device=device):
                print(f'Started epoch {epoch + 1}/{num_epochs}...')
                t1 = time()
                running_loss = 0.0
                train_predictions = []
                train_ground_truths = []

                try:
                    from tqdm import tqdm
                    gen = tqdm(enumerate(train_loader, 0), total=len(train_loader))
                except:
                    gen = enumerate(train_loader, 0)

                for i, X_Y in gen:
                    X, Y = X_Y[0].to(device), X_Y[1].to(device)
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

                true_labels = np.array(train_ground_truths)
                predicted_labels = np.array(train_predictions)
                acc = accuracy_score(true_labels, predicted_labels)

                print(f'\nModel: {fine_tuner} with {len(fine_tuner)} parameters\n'
                      f'epoch {epoch + 1}/{num_epochs} done in {format_seconds(int(time() - t1))}, '
                      f'\nTraining loss: {round(running_loss / len(train_loader), 3)}'
                      f'\ntraining accuracy: {round(acc, 3)}\n')

                train_accuracies += [acc]
                train_losses += [running_loss / len(train_loader)]
                scheduler.step()
                test_ground_truths, test_predictions, test_accuracy = test(fine_tuner=fine_tuner,
                                                                           loaders=loaders,
                                                                           device=device)
                test_accuracies += [test_accuracy]
                print('#' * 100)

                np.save(f"{results_path}{fine_tuner}_train_acc_lr{lr}_e{epoch}.npy", train_accuracies)
                np.save(f"{results_path}{fine_tuner}_train_loss_lr{lr}_e{epoch}.npy", train_losses)

                np.save(f"{results_path}{fine_tuner}_test_acc_lr{lr}_e{epoch}.npy", test_accuracies)
                np.save(f"{results_path}{fine_tuner}_test_pred_lr{lr}_e{epoch}.npy", test_predictions)

        torch.save(fine_tuner.state_dict(), f"{fine_tuner}_lr{lr}.pth")

        if not os.path.exists(f"{results_path}test_true.npy"):
            np.save(f"{results_path}test_true.npy", test_ground_truths)


# class Plot(Context):
#     def __init__(self,
#                  fig_sizes: tuple = None):
#         if fig_sizes:
#             plt.figure(figsize=fig_sizes)
#
#     def __enter__(self):
#         plt.cla()
#         plt.clf()
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         plt.show()
#         plt.cla()
#         plt.clf()


def main():
    print(F'Learning rates: {lrs}')
    model_names = (
        # ['inception_v3', 'inception_resnet_v2'] +
        [f'vit_{vit_model_name}' for vit_model_name in vit_model_names.values()])
    print(f'Models: {model_names}')
    data_dir = Path.joinpath(cwd, '.')
    datasets = {f'{model_name}_{train_or_val}': ImageFolderWithName(root=os.path.join(data_dir, train_or_val),
                                                                    transform=get_transforms(train_or_val=train_or_val,
                                                                                             model_name=model_name))
                for model_name in model_names for train_or_val in [train_folder_name, test_folder_name]}

    assert all(datasets[f'{model_name}_{train_folder_name}'].classes ==
               datasets[f'{model_name}_{test_folder_name}'].classes
               for model_name in model_names)

    loaders = {f"{model_name}_{train_or_val}": torch.utils.data.DataLoader(
        dataset=datasets[f'{model_name}_{train_or_val}'],
        batch_size=batch_size,
        shuffle=train_or_val == train_folder_name)
        for model_name in model_names for train_or_val in [train_folder_name, test_folder_name]}

    classes = datasets[f'{model_names[0]}_{train_folder_name}'].classes
    print(f'Classes: {classes}')
    n = len(classes)

    device = torch.device('mps' if torch.backends.mps.is_available() else
                          ("cuda" if torch.cuda.is_available() else 'cpu'))
    print(f'Using {device}')

    all_fine_tuners = ({'inception_v3': InceptionV3FineTuner,
                        # 'inception_resnet_v2': InceptionResNetV2FineTuner
                        } |
                       {f'vit_{vit_model_name}': VITFineTuner for vit_model_name in list(vit_model_names.values())})

    fine_tuners = []
    for model_name in model_names:
        fine_tuners_constructor = all_fine_tuners[model_name]
        fine_tuner = fine_tuners_constructor(*tuple([model_name, n] if 'vit' in model_name else [n]))
        fine_tuners += [fine_tuner]

    print(f'Fine tuners: {[str(ft) for ft in fine_tuners]}')
    for fine_tuner in fine_tuners:
        # try:
        print(f'Initiating {fine_tuner}')

        with ClearSession():
            fine_tune(fine_tuner=fine_tuner,
                      device=device,
                      loaders=loaders)

            print('#' * 100)
    # except:
    #     continue
    # with Plot():
    #     for fine_tuner in fine_tuners:
    #         vit_train_loss = np.load(Path.joinpath(cwd, f'{fine_tuner}_train_loss.npy'))
    #         plt.plot(vit_train_loss, label=f'{fine_tuner} training Loss')
    #
    #     plt.title('Training Loss')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.legend()
    #     plt.grid()
    #
    # with Plot():
    #     for fine_tuner in fine_tuners:
    #         vit_train_acc = np.load(Path.joinpath(cwd, f'{fine_tuner}_train_acc.npy'))
    #         plt.plot(vit_train_acc, label=f'{fine_tuner} training accuracy')
    #
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Accuracy')
    #     plt.title('Training Accuracy')
    #     plt.legend()
    #     plt.grid()
    #
    # with Plot():
    #     for fine_tuner in fine_tuners:
    #         vit_test_acc = np.load(f'{fine_tuner}_test_acc.npy')
    #         plt.plot(vit_test_acc, label=f'{fine_tuner} test accuracy')
    #
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Test Accuracy')
    #     plt.title('Test Accuracy')
    #     plt.legend()
    #     plt.grid()


if __name__ == '__main__':
    main()
