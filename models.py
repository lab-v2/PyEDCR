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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit(x)
        return x

    def __str__(self):
        return self.vit_model_name


class LearnedHierarchicalWeightedLoss(torch.nn.Module):
    def __init__(self,
                 num_fine_grain_classes: int,
                 num_coarse_grain_classes: int
                 ):
        super(LearnedHierarchicalWeightedLoss, self).__init__()
        self.minimal_alpha = num_fine_grain_classes / (num_fine_grain_classes + num_coarse_grain_classes)
        self.alpha = torch.nn.Parameter(torch.Tensor([self.minimal_alpha]), requires_grad=True)

    def forward(self,
                fine_loss: torch.Tensor,
                coarse_loss: torch.Tensor) -> torch.Tensor:
        self.alpha.data = torch.clamp(self.alpha.data, min=self.minimal_alpha, max=1)

        return self.alpha * fine_loss + (1 - self.alpha) * coarse_loss
