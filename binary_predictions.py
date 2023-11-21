import os
import torch
from torch.utils import data
import numpy as np
import torchvision
from sklearn.metrics import accuracy_score

from tqdm import tqdm
import typing
from PIL import Image
import multiprocessing
import json
import cv2
import timm


# Dataset class to capture filenames during testing
class NamedImageFolder(torchvision.datasets.folder.ImageFolder):
    def __init__(self,
                 root: str,
                 transform: typing.Optional[torchvision.transforms.Compose] = None):
        super().__init__(root=root,
                         transform=transform)
        self.imgs = []
        self.filenames = []

        for subdir in os.listdir(root):
            subdir_path = os.path.join(root, subdir)
            if os.path.isdir(subdir_path):
                for filename in os.listdir(subdir_path):
                    img_path = os.path.join(subdir_path, filename)
                    self.imgs += [img_path]
                    self.filenames += [f'{subdir}/{filename}']

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,
                    idx: int):
        img_path = self.imgs[idx]
        image = Image.open(img_path)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        filename = self.filenames[idx]

        return image, filename


def compute_means_stds(image_folder):
    """
    Compute the overall means and standard deviations of each channel (R, G, and B) for a folder of images.

    :param image_folder: Path to the main folder containing 'train' and 'test' subfolders.
    :return: A dictionary containing overall means and standard deviations for each channel (R, G, and B).
    """
    data_split = ('train', 'test')
    channel_names = ('r', 'g', 'b')

    # Initialize dictionaries to store means and standard deviations for each channel
    overall_means = {channel: [] for channel in channel_names}
    overall_stds = {channel: [] for channel in channel_names}

    # Iterate through 'train' and 'test' data splits
    for split in data_split:
        split_folder = os.path.join(image_folder, split)

        # Get a list of class subfolders
        class_folders = [f for f in os.listdir(split_folder) if os.path.isdir(os.path.join(split_folder, f))]

        # Iterate through each class subfolder
        for class_folder in class_folders:
            class_folder_path = os.path.join(split_folder, class_folder)

            # Get a list of image file names in the class subfolder
            image_files = [f for f in os.listdir(class_folder_path) if
                           f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

            # Loop through each image in the class subfolder
            for image_file in image_files:
                # Load the image using OpenCV
                image_path = os.path.join(class_folder_path, image_file)
                image = cv2.imread(image_path)

                # Convert the image to float32 for numerical stability
                image = image.astype(np.float32) / 255.0

                # Compute means and standard deviations for each channel
                mean, std = cv2.meanStdDev(image)
                mean = mean.squeeze()
                std = std.squeeze()

                # Append the values to the respective dictionaries
                for i, channel in enumerate(channel_names):
                    overall_means[channel].append(mean[i])
                    overall_stds[channel].append(std[i])

    # Compute the overall mean and standard deviation for each channel
    overall_means = {channel: np.mean(overall_means[channel]) for channel in channel_names}
    overall_stds = {channel: np.mean(overall_stds[channel]) for channel in channel_names}

    return ([overall_means['r'], overall_means['g'], overall_means['b']],
            [overall_stds['r'], overall_stds['g'], overall_stds['b']])


def create_data_loaders(data_dir: str,
                        batch_size: int,
                        named_dataset: bool = False) -> typing.Dict[str, torch.utils.data.DataLoader]:
    means, stds = compute_means_stds(data_dir)
    transform = {
        'train': torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(means, stds)
        ]),
        'test': torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(means, stds)
        ]),
    }

    dataset_class = NamedImageFolder if named_dataset else torchvision.datasets.folder.ImageFolder
    datasets = {split: dataset_class(root=os.path.join(data_dir, split),
                                     transform=transform[split])
                for split in transform.keys()}

    dset_loaders = {
        split: torch.utils.data.DataLoader(dataset=datasets[split],
                                           batch_size=batch_size,
                                           shuffle=split == 'train')
        for split in transform.keys()}

    return dset_loaders


class BinaryViTModel(torch.nn.Module):
    def __init__(self):
        super(BinaryViTModel, self).__init__()

        # Load a pre-trained Vision Transformer model, without the classifier head
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = torch.nn.Linear(self.vit.head.in_features, 2)  # Binary classification

    @classmethod
    def load_pretrained_weights(cls,
                                model_save_path: str):
        model = cls()

        try:
            model.load_state_dict(torch.load(model_save_path))
            print(f"Loaded pre-trained weights from {model_save_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load pre-trained weights from {model_save_path}: {str(e)}")

        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit(x)
        return x


def train_ViT(train_loader: torch.utils.data.DataLoader,
              model_save_path: str,
              model: torch.nn.Module,
              device: torch.device,
              num_epochs: typing.Optional[int] = 25,
              learning_rate: typing.Optional[float] = 0.00001):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                step_size=7,
                                                gamma=0.1)

    train_loss = []
    train_acc = []

    model.to(device)
    model.train()

    for epoch in range(num_epochs):

        running_loss = 0.0
        prediction = []
        ground_truth = []

        with torch.set_grad_enabled(True):
            for i, data in enumerate(tqdm(train_loader), 0):
                inputs, labels = data[0].to(device), data[1].to(device)

                if optimizer:
                    optimizer.zero_grad()

                outputs = model(inputs)

                if type(outputs) == torchvision.models.InceptionOutputs:
                    outputs = outputs[0]

                loss = criterion(outputs, labels)

                if optimizer:
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                ground_truth += labels.tolist()
                prediction += predicted.tolist()

        true_labels = np.array(ground_truth)
        predicted_labels = np.array(prediction)
        np.save('predicted.npy', predicted_labels)
        acc = accuracy_score(true_labels, predicted_labels)
        train_loss.append(running_loss / len(train_loader))
        train_acc.append(acc)

        print(f'Epoch {epoch + 1}\nloss: {train_loss[-1]:.3f}\naccuracy: {train_acc[-1]:.3f}')
        scheduler.step()
        print()

    torch.save(model.state_dict(), model_save_path)
    print(f'Model weights saved at: {model_save_path}')


def test_ViT(test_loader: torch.utils.data.DataLoader,
             model: torch.nn.Module,
             device: torch.device,
             class_name: str = None):
    running_loss = 0.0
    prediction_dict = {}

    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader), start=0):
            inputs, labels, filenames = data[0].to(device), data[1].to(device), data[2]

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            # Store predictions in a dictionary with filenames as keys
            for filename, pred in zip(filenames, predicted.tolist()):
                prediction_dict[filename] = pred

    # Calculate accuracy
    true_labels = np.array(labels.tolist())
    predicted_labels = np.array(predicted.tolist())
    acc = accuracy_score(true_labels, predicted_labels)

    # Print test results
    print(f'Test Results\nloss: {running_loss:.3f}\naccuracy: {acc:.3f}')

    # Save the prediction dictionary to a JSON file
    with open(f'{class_name}_predictions.json', 'w') as json_file:
        json.dump(prediction_dict, json_file, indent=4)


def test_ViT_without_accuracy(test_loader: torch.utils.data.DataLoader,
                              model: torch.nn.Module,
                              device: torch.device,
                              class_name: str = None):
    model.to(device)
    model.eval()

    predictions_dict = {}

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader), 0):
            inputs, filenames_batch = data[0].to(device), data[1]  # The second element is filenames

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            predictions_batch = predicted.tolist()
            for filename, prediction in zip(filenames_batch, predictions_batch):
                predictions_dict[filename] = prediction

    # Save the dictionary to a JSON file
    with open(f'{class_name}_predictions.json', 'w') as json_file:
        json.dump(predictions_dict, json_file, indent=4)


def load_or_train_ViT(data_dir: str,
                      model_save_path: str,
                      num_epochs: typing.Optional[int] = 25,
                      learning_rate: typing.Optional[float] = 0.00001,
                      batch_size: typing.Optional[int] = 32,
                      class_name: str = None):
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')

    test_data_dir = os.path.join(data_dir, 'test')

    is_dataset_binary_vehicles = (os.path.exists(os.path.join(test_data_dir, 'vehicles'))
                                  and os.path.exists(os.path.join(test_data_dir, 'Non-vehicles')))

    dset_loaders = create_data_loaders(data_dir=data_dir,
                                       batch_size=batch_size,
                                       named_dataset=not is_dataset_binary_vehicles)

    if os.path.isfile(model_save_path):
        model = BinaryViTModel.load_pretrained_weights(model_save_path)
        test_loader = dset_loaders['test']

        if is_dataset_binary_vehicles:
            test_ViT(test_loader=test_loader,
                     model=model,
                     device=device,
                     class_name=class_name)
        else:
            test_ViT_without_accuracy(test_loader=test_loader,
                                      model=model,
                                      device=device,
                                      class_name=class_name)
    else:
        if is_dataset_binary_vehicles:
            model = BinaryViTModel()

            train_ViT(train_loader=dset_loaders['train'],
                      model_save_path=model_save_path,
                      model=model,
                      device=device,
                      num_epochs=num_epochs,
                      learning_rate=learning_rate)

            print("Model training complete.")

            test_ViT(test_loader=dset_loaders['test'],
                     model=model,
                     device=device)
        else:
            raise RuntimeError("Missing model")


def run_model(data_dir, model_save_path, class_name):
    print(f"Running model for class {class_name}")
    # Your code to load or train the ViT model goes here.
    # For example, you might call load_or_train_ViT function with the parameters.
    load_or_train_ViT(data_dir=data_dir, model_save_path=model_save_path, class_name=class_name)
    print(f"Model for class {class_name} completed")


def worker_init(args_tuple):
    # Unpack the arguments tuple
    run_model(*args_tuple)


if __name__ == "__main__":
    class_names = [
        # 'Air_Defense',
        'BMD',
        # 'BMP',
        # 'BTR',
        'MT_LB',
        # 'SPA',
        # 'Tank'
    ]

    args_list = [('coarse',
                  fr'binary_classifiers/{class_name}_model.pth', class_name)
                 for class_name in class_names]

    with multiprocessing.Pool(processes=10) as pool:
        # Map the worker_init function to the list of argument tuples
        pool.map(worker_init, args_list)

    print("All models have been run.")
