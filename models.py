import re
import os
import abc
import torch
import torchvision
from typing import Union


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
        self.inception.fc = torch.nn.Linear(in_features=num_features,
                                            out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inception(x)
        return x


class VITFineTuner(FineTuner):
    def __init__(self,
                 vit_model: Union[int, str],
                 vit_model_names: list,
                 num_classes: int):
        super().__init__(num_classes=num_classes)
        if isinstance(vit_model, int):
            self.vit_model_index = vit_model
            self.vit_model_name = vit_model_names[vit_model]
        else:
            self.vit_model_index = vit_model_names[vit_model_names.index(vit_model)]
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



class ImageFolderWithName(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = self.loader(path)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        name = os.path.basename(path)
        folder_path = os.path.basename(os.path.dirname(path))

        return image, target, f'{folder_path}/{name}'
