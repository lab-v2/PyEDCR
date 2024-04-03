import re
import abc
import torch
import torchvision
import typing

import data_preprocessing



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
        vit_model = getattr(torchvision.models, vit_model_name)

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


def get_filepath(model_name: typing.Union[str, FineTuner],
                 test: bool,
                 loss: str,
                 lr: typing.Union[str, float],
                 pred: bool,
                 combined: bool = True,
                 l: data_preprocessing.Label = None,
                 epoch: int = None,
                 granularity: str = None,
                 lower_prediction_index: int = None) -> str:
    """
    Constructs the file path to the model output / ground truth data.

    :param l:
    :param lower_prediction_index:
    :param model_name: The name of the model or `FineTuner` object.
    :param combined: Whether the model are individual or combine one.
    :param test: Whether the data is getting from testing or training set.
    :param granularity: The granularity level.
    :param loss: The loss function used during training.
    :param lr: The learning rate used during training.
    :param pred: Whether the data is a prediction from neural or ground truth
    :param epoch: The epoch number (optional, only for training data).
    :return: The generated file path.
    """
    epoch_str = f'_e{epoch - 1}' if epoch is not None else ''
    granularity_str = f'_{granularity}' if granularity is not None else ''
    test_str = 'test' if test else 'train'
    pred_str = 'pred' if pred else 'true'
    folder_str = 'binary' if l is not None else ('combined' if combined else 'individual')
    lower_prediction_index_str = f'_lower_{lower_prediction_index}' if lower_prediction_index is not None else ''
    lower_prediction_folder_str = 'lower_prediction/' if lower_prediction_index is not None else ''
    l_str = f'_{l}' if l is not None else ''

    return (f"{folder_str}_results/{lower_prediction_folder_str}"
            f"{model_name}_{test_str}{granularity_str}_{pred_str}_{loss}_lr{lr}{epoch_str}"
            f"{lower_prediction_index_str}{l_str}.npy")