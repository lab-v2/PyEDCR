import os

import backbone_pipeline
import combined_fine_tuning
import neural_evaluation
import utils

if utils.is_local():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import numpy as np
import typing
from tqdm.contrib.concurrent import process_map

import data_preprocessing
import PyEDCR


class Error_detection_model(PyEDCR.EDCR):
    def __init__(self,
                 data_str: str,
                 main_model_name: str,
                 combined: bool,
                 loss: str,
                 lr: typing.Union[str, float],
                 original_num_epochs: int,):
        super(Error_detection_model, self).__init__(data_str=data_str,
                                                    main_model_name=main_model_name,
                                                    combined=combined,
                                                    loss=loss,
                                                    lr=lr,
                                                    original_num_epochs=original_num_epochs,
                                                    epsilon=0.1)

    def learn_error_binary_model(self,
                                 model_name: str,
                                 lr: typing.Union[float, str],
                                 pretrained_path: str = None):
        preprocessor, fine_tuners, loaders, devices = backbone_pipeline.initiate(
            data_str=self.data_str,
            model_name=model_name,
            preprocessor=self.preprocessor,
            lr=lr,
            fine_predictions=self.get_predictions(test=False, g=self.preprocessor.granularities['fine']),
            coarse_predictions=self.get_predictions(test=False, g=self.preprocessor.granularities['coarse']),
            train_eval_split=0.8,
            pretrained_path=pretrained_path
        )

        combined_fine_tuning.fine_tune_combined_model(
            preprocessor=preprocessor,
            lr=lr,
            fine_tuner=fine_tuners[0],
            device=devices[0],
            loaders=loaders,
            loss='error_BCE',
            save_files=False,
            evaluate_on_train_eval=True,
            evaluate_on_test=True,
            num_epochs=10,
            data_str=data_str,
            model_name=main_model_name
        )

    def evaluate_error_binary_model(self,
                                    model_name: str,
                                    lr: typing.Union[float, str],
                                    error_fine_prediction: np.array,
                                    error_coarse_prediction: np.array):
        preprocessor, fine_tuners, loaders, devices = backbone_pipeline.initiate(
            data_str=self.data_str,
            model_name=model_name,
            preprocessor=self.preprocessor,
            lr=lr,
            fine_predictions=self.get_predictions(test=False, g=self.preprocessor.granularities['fine']),
            coarse_predictions=self.get_predictions(test=False, g=self.preprocessor.granularities['coarse']),
            train_eval_split=0.8,
        )

        neural_evaluation.evaluate_binary_model(
            fine_tuner=fine_tuners[0],
            loaders=loaders,
            loss='BCE',
            device=devices[0],
            split='test',
            preprocessor=preprocessor,
            error_fine_prediction=error_fine_prediction,
            error_coarse_prediction=error_coarse_prediction
        )


if __name__ == '__main__':
    data_str = 'military_vehicles'
    main_model_name = new_model_name = 'vit_b_16'
    main_lr = new_lr = binary_lr = 0.0001
    original_num_epochs = 20
    pretrained_path = ''

    # data_str = 'imagenet'
    # main_model_name = new_model_name = 'dinov2_vits14'
    # main_lr = new_lr = binary_lr = 0.000001
    # original_num_epochs = 8

    # data_str = 'openimage'
    # main_model_name = new_model_name = 'tresnet_m'
    # main_lr = new_lr = 0.000001
    # original_num_epochs = 0

    edcr = Error_detection_model(data_str=data_str,
                                 main_model_name=main_model_name,
                                 combined=True,
                                 loss='BCE',
                                 lr=main_lr,
                                 original_num_epochs=original_num_epochs,
                                 )

    # edcr.learn_error_binary_model(model_name=main_model_name,
    #                               lr=new_lr,
    #                               pretrained_path=pretrained_path)

    error_fine_prediction = np.load('')
    error_coarse_prediction = np.load('')
    edcr.evaluate_error_binary_model(model_name=main_model_name,
                                     lr=main_lr,
                                     error_fine_prediction=error_fine_prediction,
                                     error_coarse_prediction=error_coarse_prediction)
