import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision.models import inception_v3, Inception_V3_Weights, vit_l_32, ViT_L_32_Weights
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from time import time

batch_size = 26
resize = (299, 299)

# Load Data

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(max(resize)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    'val': transforms.Compose([
        # Higher scale-up for inception
        transforms.Resize(int(max(resize) / 224 * 256)),
        transforms.CenterCrop(max(resize)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class ImageFolderWithName(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        name = os.path.basename(path)
        return image, target, name


data_dir = os.getcwd() + '/data/FineGrainDataset'
datasets = {x: ImageFolderWithName(os.path.join(data_dir, x), data_transforms[x])
            for x in ['train', 'val']}
loaders = {x: DataLoader(datasets[x],
                         batch_size=batch_size,
                         shuffle=True) for x in ['train', 'val']}
sizes = {x: len(datasets[x]) for x in ['train', 'val']}
train_loader, test_loader = loaders['train'], loaders['val']

assert datasets['train'].classes == datasets['val'].classes
dset_classes = datasets['train'].classes
dset_classes


class VITModel(nn.Module):
    def __init__(self, num_classes):
        super(VITModel, self).__init__()
        self.vit = vit_l_32(weights=ViT_L_32_Weights.DEFAULT)
        num_features = self.vit.heads[-1].in_features
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.vit(x)
        x = self.fc(x)
        return x


vit_model = VITModel(num_classes=24)

for model in [vit_model]:
    model_name = model.__class__.__name__

    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=lr)
    scheduler_step_size = 7
    scheduler_gamma = 0.1
    scheduler = StepLR(optimizer=optimizer,
                       step_size=scheduler_step_size,
                       gamma=scheduler_gamma)
    train_loss = []
    train_acc = []
    train_precision = []
    train_recall = []
    prediction = []
    ground_truth = []
    model.to(device)

    num_epochs = 25

    for epoch in range(num_epochs):
        t1 = time()
        running_loss = 0.0
        prediction = []
        ground_truth = []
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs[0], labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs[0], 1)
            ground_truth += labels.tolist()
            prediction += predicted.tolist()

        print(f'Model: {model_name} '
              f'Epoch {epoch + 1}/{num_epochs} done in {time() - t1} sec, '
              f'\nloss: {round(running_loss / len(train_loader), 3)}')
        true_labels = np.array(ground_truth)
        predicted_labels = np.array(prediction)
        acc = accuracy_score(true_labels, predicted_labels)
        print(f'accuracy: {round(acc, 3)}')
        train_acc.append(acc)
        train_loss.append(running_loss / len(train_loader))
        scheduler.step()
        print()

    np.save(f'{model_name}_train_acc.npy', train_acc)
    np.save(f'{model_name}_train_loss.npy', train_loss)
