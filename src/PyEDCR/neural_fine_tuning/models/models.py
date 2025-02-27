import argparse
import numpy as np
import abc

import torch
import torch.optim
import torchvision

try:
    from PyEDCR.neural_fine_tuning.models.bn_fusion import fuse_bn_recursively
    from src_files.models import create_model
    from src_files.models.tresnet.tresnet import InplacABN_to_ABN
except ModuleNotFoundError:
    pass

from PyEDCR.PyEDCR import data_preprocessor


class FineTuner(torch.nn.Module, abc.ABC):
    """
    Base class for fine-tuning transformers for image classification tasks.
    """

    def __init__(self,
                 model_name: str,
                 num_classes: int):
        """
        Initializes the FineTuner with the number of classes.

        :param num_classes: The number of output classes for classification.
        """
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

    @classmethod
    def from_pretrained(cls,
                        model_name: str,
                        num_classes: int,
                        pretrained_path: str,
                        device: torch.device):
        pass

    def __str__(self) -> str:
        return self.model_name

    def __len__(self) -> int:
        return sum(p.numel() for p in self.parameters())


class EfficientNetV2FineTuner(FineTuner):
    def __init__(self,
                 model_name: str,
                 num_classes: int):
        super().__init__(model_name=model_name,
                         num_classes=num_classes)
        self.efficient_net_v2_model_name = model_name
        efficient_net_v2_model = getattr(torchvision.models, model_name)

        efficient_net_v2_weights = getattr(getattr(
            torchvision.models,
            f"EfficientNet_V2_{model_name.split('efficientnet_v2_')[-1].upper()}_Weights"),
            'DEFAULT')

        self.efficient_net_v2 = efficient_net_v2_model(weights=efficient_net_v2_weights)
        self.output_layer = torch.nn.Linear(in_features=self.efficient_net_v2.classifier[-1].out_features,
                                            out_features=num_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.efficient_net_v2(X)
        X = self.output_layer(X)

        return X


class DINOV2FineTuner(FineTuner):
    def __init__(self,
                 model_name: str,
                 num_classes: int):
        super().__init__(model_name=model_name,
                         num_classes=num_classes)
        self.model_size = model_name.split('dinov2_vit')[-1][0]
        in_features = {'s': 384,
                       'm': 768,
                       'l': 1024}[self.model_size]

        self.transformer = torch.hub.load(repo_or_dir='facebookresearch/dinov2',
                                          model=model_name)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=num_classes))

    @classmethod
    def from_pretrained(cls,
                        model_name: str,
                        num_classes: int,
                        pretrained_path: str,
                        device: torch.device):
        """
        Loads a pre-trained DINO V2 model from a specified path.

        :param model_name: The name of the pre-trained ViT model used during training.
        :param num_classes: The number of output classes for the loaded model.
        :param pretrained_path: The path to the saved pre-trained model checkpoint.
        :param device: The device (CPU or GPU) to load the model onto.

        :return: An instance of VITFineTuner loaded with pre-trained weights.
        """
        instance = cls(model_name, num_classes)
        predefined_weights = torch.load(pretrained_path,
                                        map_location=device)
        transformer_weights = {'.'.join(k.split('.')[1:]): v for k, v in predefined_weights.items()
                               if k.split('.')[0] == 'transformer'}
        instance.transformer.load_state_dict(transformer_weights)

        classifier_weights = {'.'.join(k.split('.')[1:]): v for k, v in predefined_weights.items()
                              if k.split('.')[0] == 'classifier'}
        instance.classifier.load_state_dict(classifier_weights)

        return instance

    def forward(self,
                X: torch.Tensor) -> torch.Tensor:
        X = self.transformer(X)
        X = self.transformer.norm(X)
        X = self.classifier(X)

        return X


class MultiHeadAttentionBlock(torch.nn.Module):
    def __init__(self, num_classes, embed_size, num_heads):
        super(MultiHeadAttentionBlock, self).__init__()
        self.linear = torch.nn.Linear(num_classes, embed_size)  # Embedding the one-hot vector
        self.attention = torch.nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads)
        self.norm = torch.nn.LayerNorm(embed_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # Assume x is batch of indices [batch_size, 1]
        x = torch.nn.functional.one_hot(x.long(), num_classes=self.linear.in_features).float()  # One-hot encode
        x = self.linear(x)  # Now [batch_size, embed_size]
        x = x.unsqueeze(1)  # Add sequence length dimension [batch_size, 1, embed_size]
        attn_output, _ = self.attention(x, x, x)
        attn_output = self.norm(attn_output + x)
        return self.relu(attn_output.squeeze(1))


class ErrorDetector(FineTuner):
    def __init__(self,
                 model_name: str,
                 num_classes: int,
                 preprocessor: data_preprocessor.FineCoarseDataPreprocessor,
                 pretrained_path: str = None,
                 device=None):
        super().__init__(model_name=model_name, num_classes=num_classes)
        self.preprocessor = preprocessor

        # Initialize the DINOV2FineTuner to get prediction
        if pretrained_path is not None:
            if 'dino' in model_name:
                self.transformer = DINOV2FineTuner.from_pretrained(
                    model_name, num_classes, pretrained_path, device=device)
            elif 'vit' in model_name:
                self.transformer = VITFineTuner.from_pretrained(
                    model_name, num_classes, pretrained_path, device=device)
        else:
            self.transformer = DINOV2FineTuner(model_name, num_classes)

        self.normalize_function_fine = torch.nn.Sigmoid()
        self.normalize_function_coarse = torch.nn.Sigmoid()

    @classmethod
    def from_pretrained(cls,
                        model_name: str,
                        num_classes: int,
                        pretrained_path: str,
                        device: torch.device,
                        preprocessor: data_preprocessor.FineCoarseDataPreprocessor = None):
        """
        Loads a pre-trained DINO V2 model from a specified path.

        :param preprocessor: The preprocessor object used to preprocess the input data.
        :param model_name: The name of the pre-trained ViT model used during training.
        :param num_classes: The number of output classes for the loaded model.
        :param pretrained_path: The path to the saved pre-trained model checkpoint.
        :param device: The device (CPU or GPU) to load the model onto.

        :return: An instance of VITFineTuner loaded with pre-trained weights.
        """
        instance = cls(model_name, num_classes,
                       pretrained_path=pretrained_path, device=device, preprocessor=preprocessor)

        return instance

    def forward(self, X_image: torch.Tensor, X_base_model_prediction: torch.Tensor) -> torch.Tensor:
        image_features = self.transformer(X_image)

        fine_prediction = self.normalize_function_fine(image_features[:, :self.preprocessor.num_fine_grain_classes])
        coarse_prediction = self.normalize_function_coarse(image_features[:, self.preprocessor.num_fine_grain_classes:])

        fine_prediction_from_previous_model = X_base_model_prediction[:, :self.preprocessor.num_fine_grain_classes]
        coarse_prediction_from_previous_model = X_base_model_prediction[:, self.preprocessor.num_fine_grain_classes:]

        error_fine_grain_class = torch.sum(fine_prediction_from_previous_model * fine_prediction, dim=1)
        error_coarse_grain_class = torch.sum(coarse_prediction_from_previous_model * coarse_prediction, dim=1)

        error_probability = 1 - error_coarse_grain_class * error_fine_grain_class

        return error_probability


class VITFineTuner(FineTuner):
    """
    This class inherits from `FineTuner` to provide specific functionalities for
    fine-tuning vision transformer (ViT) models.
    """

    def __init__(self,
                 model_name: str,
                 num_classes: int,
                 weights: str = 'DEFAULT'):
        """
        Initializes the VITFineTuner with a pre-trained ViT model and number of classes.

        :param model_name: The name of the pre-trained ViT model to use (e.g., 'vit_base_patch16').
        :param num_classes: The number of output classes for classification.
        """
        super().__init__(model_name=model_name,
                         num_classes=num_classes)
        self.vit_model_name = model_name
        vit_model = getattr(torchvision.models, model_name)

        vit_weights = getattr(getattr(
            torchvision.models,
            f"ViT_{'_'.join([s.upper() for s in self.vit_model_name.split('vit_')[-1].split('_')])}_Weights"),
            weights)
        self.vit = vit_model(weights=vit_weights)
        self.vit.heads[-1] = torch.nn.Linear(in_features=self.vit.hidden_dim,
                                             out_features=num_classes)

    @classmethod
    def from_pretrained(cls,
                        model_name: str,
                        num_classes: int,
                        pretrained_path: str,
                        device: torch.device):
        """
        Loads a pre-trained VITFineTuner model from a specified path.

        :param model_name: The name of the pre-trained ViT model used during training.
        :param num_classes: The number of output classes for the loaded model.
        :param pretrained_path: The path to the saved pre-trained model checkpoint.
        :param device: The device (CPU or GPU) to load the model onto.

        :return: An instance of VITFineTuner loaded with pre-trained weights.
        """
        instance = cls(model_name, num_classes)
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

    def forward(self,
                X: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the fine-tuned ViT model for prediction.
        :param X: Input tensor of image data (batch_num, channels_num, height, width).
        :return: Predicted class probabilities for each input image (batch_num, classes_num).
        """
        return self.vit(X)


class VITFineTunerLTN(FineTuner):
    """
    This class inherits from `FineTuner` to provide specific functionalities for
    fine-tuning vision transformer (ViT) models.
    """

    def __init__(self,
                 model_name: str,
                 num_classes: int,
                 weights: str = 'DEFAULT'):
        """
        Initializes the VITFineTuner with a pre-trained ViT model and number of classes.

        :param model_name: The name of the pre-trained ViT model to use (e.g., 'vit_base_patch16').
        :param num_classes: The number of output classes for classification.
        """
        super().__init__(model_name=model_name,
                         num_classes=num_classes)
        self.vit_model_name = model_name[:-4]
        vit_model = getattr(torchvision.models, model_name[:-4])

        vit_weights = getattr(getattr(
            torchvision.models,
            f"ViT_{'_'.join([s.upper() for s in self.vit_model_name.split('vit_')[-1].split('_')])}_Weights"),
            weights)
        self.vit = vit_model(weights=vit_weights)
        self.vit.heads[-1] = torch.nn.Linear(in_features=self.vit.hidden_dim,
                                             out_features=self.vit.hidden_dim)
        self.elu = torch.nn.ELU()
        self.fc_main = torch.nn.Linear(in_features=self.vit.hidden_dim,
                                       out_features=num_classes)
        self.fc_branch = torch.nn.Linear(in_features=self.vit.hidden_dim,
                                         out_features=1024)

    @classmethod
    def from_pretrained(cls,
                        model_name: str,
                        num_classes: int,
                        pretrained_path: str,
                        device: torch.device):
        """
        Loads a pre-trained VITFineTuner model from a specified path.

        :param model_name: The name of the pre-trained ViT model used during training.
        :param num_classes: The number of output classes for the loaded model.
        :param pretrained_path: The path to the saved pre-trained model checkpoint.
        :param device: The device (CPU or GPU) to load the model onto.

        :return: An instance of VITFineTuner loaded with pre-trained weights.
        """
        instance = cls(model_name, num_classes)
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

    def forward(self,
                X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the fine-tuned ViT model for prediction.
        :param X: Input tensor of image data (batch_num, channels_num, height, width).
        :return: Predicted class probabilities for each input image (batch_num, classes_num).
        """
        base_output = self.vit(X)
        main_output = self.elu(self.fc_main(base_output))
        branch_output = self.elu(self.fc_branch(base_output))

        return main_output, branch_output


class DINOV2FineTunerLTN(FineTuner):
    """
    This class inherits from `FineTuner` to provide specific functionalities for
    fine-tuning vision transformer (ViT) models.
    """

    def __init__(self,
                 model_name: str,
                 num_classes: int):
        super().__init__(model_name=model_name,
                         num_classes=num_classes)
        self.model_size = model_name.split('dinov2_vit')[-1][0]
        in_features = {'s': 384,
                       'm': 768,
                       'l': 1024}[self.model_size]

        self.transformer = torch.hub.load(repo_or_dir='facebookresearch/dinov2',
                                          model=model_name)
        # self.classifier = torch.nn.Sequential(
        #     torch.nn.Linear(in_features=in_features, out_features=256),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(in_features=256, out_features=num_classes))
        self.elu = torch.nn.ELU()
        self.fc_main = torch.nn.Linear(in_features=in_features,
                                       out_features=num_classes)
        self.fc_branch = torch.nn.Linear(in_features=in_features,
                                         out_features=1024)

    @classmethod
    def from_pretrained(cls,
                        model_name: str,
                        num_classes: int,
                        pretrained_path: str,
                        device: torch.device):
        """
        Loads a pre-trained DINO V2 model from a specified path.

        :param model_name: The name of the pre-trained ViT model used during training.
        :param num_classes: The number of output classes for the loaded model.
        :param pretrained_path: The path to the saved pre-trained model checkpoint.
        :param device: The device (CPU or GPU) to load the model onto.

        :return: An instance of VITFineTuner loaded with pre-trained weights.
        """
        instance = cls(model_name, num_classes)
        predefined_weights = torch.load(pretrained_path,
                                        map_location=device)
        transformer_weights = {'.'.join(k.split('.')[1:]): v for k, v in predefined_weights.items()
                               if k.split('.')[0] == 'transformer'}
        instance.transformer.load_state_dict(transformer_weights)

        fc_main_weights = {'.'.join(k.split('.')[1:]): v for k, v in predefined_weights.items()
                           if k.split('.')[0] == 'fc_main'}
        fc_branch_weights = {'.'.join(k.split('.')[1:]): v for k, v in predefined_weights.items()
                             if k.split('.')[0] == 'fc_branch'}

        instance.fc_main.load_state_dict(fc_main_weights)
        instance.fc_branch.load_state_dict(fc_branch_weights)

        return instance

    def forward(self,
                X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the fine-tuned ViT model for prediction.
        :param X: Input tensor of image data (batch_num, channels_num, height, width).
        :return: Predicted class probabilities for each input image (batch_num, classes_num).
        """
        base_output = self.transformer.norm(self.transformer(X))
        main_output = self.elu(self.fc_main(base_output))
        branch_output = self.elu(self.fc_branch(base_output))

        return main_output, branch_output


class TResnetFineTuner(FineTuner):
    """
    This class inherits from `FineTuner` to provide specific functionalities for
    fine-tuning TResnet models.
    """

    def __init__(self,
                 tresnet_model_name: str,
                 num_classes: int,
                 preprocessor: data_preprocessor.FineCoarseDataPreprocessor = None):
        """
        Initializes the TResnetFineTuner with a pre-trained TResnet model and number of classes.

        :param tresnet_model_name: The name of the pre-trained TResnet model to use (e.g., 'tresnet_m').
        :param num_classes: The number of output classes for classification.
        """
        super().__init__(model_name=tresnet_model_name,
                         num_classes=num_classes)
        self.model = None
        self.preprocessor = preprocessor

    @classmethod
    def from_pretrained(cls,
                        model_name: str,
                        num_classes: int,
                        pretrained_path: str,
                        device: torch.device,
                        preprocessor: data_preprocessor.FineCoarseDataPreprocessor = None,
                        ):
        """
        Loads a pre-trained TResnetFineTuner model from a specified path.

        :param preprocessor: The preprocessor object used to preprocess the input data.
        :param model_name: The name of the pre-trained TResnet model used during training.
        :param num_classes: The number of output classes for the loaded model.
        :param pretrained_path: The path to the saved pre-trained model checkpoint.
        :param device: The device (CPU or GPU) to load the model onto.

        :return: An instance of TResnetFineTuner loaded with pre-trained weights.
        """
        instance = cls(model_name, num_classes)
        # Fixed configuration
        parser = argparse.ArgumentParser(description='PyTorch Open Image infer')
        parser.add_argument('--num-classes', default=9605, type=int)
        parser.add_argument('--model-path', type=str, default=pretrained_path)
        parser.add_argument('--pic-path', type=str, default='./pics/000000000885.jpg')
        parser.add_argument('--model-name', type=str, default=model_name)
        parser.add_argument('--image-size', type=int, default=224)
        # parser.add_argument('--dataset-type', type=str, default='MS-COCO')
        parser.add_argument('--th', type=float, default=0.75)
        parser.add_argument('--top-k', type=float, default=20)
        # ML-Decoder
        parser.add_argument('--use-ml-decoder', default=1, type=int)
        parser.add_argument('--num-of-groups', default=200, type=int)
        parser.add_argument('--decoder-embedding', default=768, type=int)
        parser.add_argument('--zsl', default=0, type=int)

        # Setup model
        args = parser.parse_args()

        # Setup model
        print('creating model {}...'.format(args.model_name))
        instance.model = create_model(args, load_head=True)
        state = torch.load(args.model_path, map_location=device)
        instance.model.load_state_dict(state['model'], strict=True)

        # eliminate BN for faster inference ###########
        instance.model = InplacABN_to_ABN(instance.model)
        instance.model = fuse_bn_recursively(instance.model)
        instance.model.to(device)
        #######################################################

        instance.classes_list = np.array(list(state['idx_to_class'].values()))
        instance.preprocessor = preprocessor

        return instance

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the fine-tuned TResnet model for prediction.

        :param X: Input tensor of image data (batch_num, channels_num, height, width).

        :return: Predicted class probabilities for each input image (batch_num, classes_num).
        """
        output = torch.squeeze(torch.sigmoid(self.model(X)))
        np_output = output.cpu().detach().numpy()

        class_positions = np.array([np.where(self.classes_list == cls)[0]
                                    for cls in self.preprocessor.fine_grain_classes_str])
        fine_grain_classes_prediction = np_output[:, class_positions]

        class_positions = np.array([np.where(self.classes_list == cls)[0]
                                    for cls in self.preprocessor.coarse_grain_classes_str])
        coarse_grain_classes_prediction = np_output[:, class_positions]

        fine_and_coarse_output = np.concatenate(
            [fine_grain_classes_prediction, coarse_grain_classes_prediction],
            axis=1)
        return torch.squeeze(torch.tensor(fine_and_coarse_output))
