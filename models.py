import re
import abc
import torch
import torchvision


class FineTuner(torch.nn.Module, abc.ABC):
    """
    Base class for fine-tuning transformers for image classification tasks.
    """
    def __init__(self,
                 num_classes: int):
        """
        Initializes the FineTuner with the number of classes.

        :param num_classes: The number of output classes for classification.
        """
        super().__init__()
        self.num_classes = num_classes

    def __str__(self) -> str:
        return re.sub(pattern=r'([a-z])([A-Z0-9])',
                      repl=r'\1_\2',
                      string=self.__class__.__name__.split('Fine')[0]).lower()

    def __len__(self) -> int:
        return sum(p.numel() for p in self.parameters())


class VITFineTuner(FineTuner):
    """
    This class inherits from `FineTuner` to provide specific functionalities for
    fine-tuning vision transformer (ViT) models.
    """
    def __init__(self,
                 vit_model_name: str,
                 num_classes: int):
        """
        Initializes the VITFineTuner with a pre-trained ViT model and number of classes.

        :param vit_model_name: The name of the pre-trained ViT model to use (e.g., 'vit_base_patch16').
        :param num_classes: The number of output classes for classification.
        """
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
        """
        Loads a pre-trained VITFineTuner model from a specified path.

        :param vit_model_name: The name of the pre-trained ViT model used during training.
        :param classes_num: The number of output classes for the loaded model.
        :param pretrained_path: The path to the saved pre-trained model checkpoint.
        :param device: The device (CPU or GPU) to load the model onto.

        :return: An instance of VITFineTuner loaded with pre-trained weights.
        """
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
        """
        Performs a forward pass through the fine-tuned ViT model for prediction.

        :param X: Input tensor of image data (batch_num, channels_num, height, width).

        :return: Predicted class probabilities for each input image (batch_num, classes_num).
        """

        Y_pred = self.vit(X)

        return Y_pred

    def __str__(self):
        return self.vit_model_name
