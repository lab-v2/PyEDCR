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
                 num_classes: int):
        super().__init__(num_classes=num_classes)
        self.vit_model_name = vit_model_name

        vit_model = eval(f'torchvision.models.{self.vit_model_name}')
        vit_weights = eval(
            f"torchvision.models.ViT_{'_'.join([s.upper() for s in self.vit_model_name.split('vit_')[-1].split('_')])}"
            f"_Weights.DEFAULT")
        self.vit = vit_model(weights=vit_weights)
        self.vit.heads[-1] = torch.nn.Linear(in_features=self.vit.hidden_dim,
                                             out_features=num_classes)

    @classmethod
    def from_pretrained(cls,
                        vit_model_name: str,
                        classes_num: int,
                        pretrained_path: str,
                        device: torch.device):
        instance = cls(vit_model_name, classes_num)
        predefined_weights = torch.load(pretrained_path,
                                        map_location=device)

        if 'model_state_dict' in predefined_weights.keys():
            predefined_weights = predefined_weights['model_state_dict']

        new_predefined_weights = {}
        for key, value in predefined_weights.items():
            new_key = key.replace('vit.', '')
            new_predefined_weights[new_key] = value

        instance.vit.load_state_dict(new_predefined_weights)

        return instance

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X_dim = [batches_num, channels_num, pixels_num, pixels_num]
        # Y_dim = [batch_num, classes_num]

        Y_pred = self.vit(X)

        return Y_pred

    def __str__(self):
        return self.vit_model_name
