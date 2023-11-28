from __future__ import annotations

import re
import abc
import torch
import torchvision


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


class VITFineTuner(FineTuner):
    def __init__(self,
                 vit_model_name: str,
                 num_classes: int,
                 predefined: bool = False
                 ):
        super().__init__(num_classes=num_classes)
        self.vit_model_name = vit_model_name

        vit_model = eval(f'torchvision.models.{self.vit_model_name}')
        vit_weights = eval(
            f"torchvision.models.ViT_{'_'.join([s.upper() for s in self.vit_model_name.split('vit_')[-1].split('_')])}"
            f"_Weights.DEFAULT") if predefined else None
        self.vit = vit_model(weights=vit_weights)
        self.vit.heads[-1] = torch.nn.Linear(in_features=self.vit.hidden_dim,
                                             out_features=num_classes)

    @classmethod
    def with_predefined_weights(cls,
                                vit_model_name: str,
                                num_classes: int,
                                predefined_weights_path: str,
                                device: torch.device) -> VITFineTuner:
        instance = cls(vit_model_name=vit_model_name,
                       num_classes=num_classes,
                       predefined=True)
        predefined_weights = torch.load(predefined_weights_path,
                                        map_location=device)

        # Remove 'vit.' prefix from keys
        new_predefined_weights = {}
        for key, value in predefined_weights.items():
            new_key = key.replace('vit.', '')
            new_predefined_weights[new_key] = value

        instance.vit.load_state_dict(new_predefined_weights)

        return instance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit(x)
        return x

    def __str__(self) -> str:
        return self.vit_model_name

