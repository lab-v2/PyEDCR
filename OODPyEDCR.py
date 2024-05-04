import os
import utils

if utils.is_local():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import numpy as np
import typing
from tqdm.contrib.concurrent import process_map

import data_preprocessing
import PyEDCR


class OODPyEDCR(PyEDCR.EDCR):
    def __init__(self,
                 data_str: str,
                 main_model_name: str,
                 combined: bool,
                 loss: str,
                 lr: typing.Union[str, float],
                 original_num_epochs: int,
                 epsilon: typing.Union[str, float],
                 remove_label: list[data_preprocessing.Label] = None,
                 sheet_index: int = None,
                 binary_l_strs: typing.List[str] = [],
                 binary_num_epochs: int = None,
                 binary_lr: typing.Union[str, float] = None,
                 maximize_ratio: bool = True):
        super(OODPyEDCR, self).__init__(data_str=data_str,
                                        main_model_name=main_model_name,
                                        combined=combined,
                                        loss=loss,
                                        lr=lr,
                                        original_num_epochs=original_num_epochs,
                                        epsilon=epsilon,
                                        sheet_index=sheet_index,
                                        binary_l_strs=binary_l_strs,
                                        binary_num_epochs=binary_num_epochs,
                                        binary_lr=binary_lr,
                                        maximize_ratio=maximize_ratio,
                                        remove_label=remove_label)

    def run_learning_pipeline(self,
                              multi_threading: bool = True):
        print('Started learning pipeline...\n')

        for g in data_preprocessing.DataPreprocessor.granularities.values():
            self.learn_detection_rules(g=g,
                                       multi_threading=multi_threading)
            self.apply_detection_rules(test=False,
                                       g=g)

        print('\nRule learning completed\n')


def work_on_value(args):
    (epsilon_index,
     epsilon,
     data_str,
     main_model_name,
     main_lr,
     original_num_epochs,
     binary_l_strs,
     binary_lr,
     binary_num_epochs,
     new_model_name,
     new_lr,
     maximize_ratio,
     remove_label
     ) = args

    print('#' * 25 + f'labels to remove = {(l.str for l in remove_label)}, eps = {epsilon}' + '#' * 50)
    edcr = OODPyEDCR(data_str=data_str,
                     epsilon=epsilon,
                     sheet_index=epsilon_index,
                     main_model_name=main_model_name,
                     combined=True,
                     loss='BCE',
                     lr=main_lr,
                     original_num_epochs=original_num_epochs,
                     binary_l_strs=binary_l_strs,
                     binary_lr=binary_lr,
                     binary_num_epochs=binary_num_epochs,
                     maximize_ratio=maximize_ratio,
                     remove_label=remove_label
                     )
    edcr.run_learning_pipeline(multi_threading=True)
    edcr.run_error_detection_application_pipeline(test=True,
                                                  print_results=False,
                                                  save_to_google_sheets=True)


def simulate_for_values(total_number_of_points: int = 10,
                        min_value: float = 0.1,
                        max_value: float = 0.3,
                        remove_label: list[data_preprocessing.Label] = None,
                        multi_process: bool = True,
                        binary_l_strs: typing.List[str] = [],
                        binary_lr: typing.Union[str, float] = None,
                        binary_num_epochs: int = None,
                        maximize_ratio: bool = True,):
    datas = [(i,
              round(epsilon, 3),
              data_str,
              main_model_name,
              main_lr,
              original_num_epochs,
              binary_l_strs,
              binary_lr,
              binary_num_epochs,
              new_model_name,
              new_lr,
              maximize_ratio,
              remove_label
              ) for i, epsilon in np.linspace(start=min_value, stop=max_value, num=total_number_of_points)]

    if multi_process:
        processes_num = 2
        process_map(work_on_value,
                    datas,
                    max_workers=processes_num)
    for data in datas:
        work_on_value(data)


if __name__ == '__main__':
    # data_str = 'military_vehicles'
    # main_model_name = new_model_name = 'vit_b_16'
    # main_lr = new_lr = binary_lr = 0.0001
    # original_num_epochs = 20
    # binary_num_epochs = 10
    # max_num_train_images_per_class = 500

    data_str = 'imagenet'
    main_model_name = new_model_name = 'dinov2_vits14'
    main_lr = new_lr = binary_lr = 0.000001
    original_num_epochs = 8
    binary_num_epochs = 5
    max_num_train_images_per_class = 1300

    # data_str = 'openimage'
    # main_model_name = new_model_name = 'tresnet_m'
    # main_lr = new_lr = 0.000001
    # original_num_epochs = 0

    binary_l_strs = list({f.split(f'e{binary_num_epochs - 1}_')[-1].replace('.npy', '')
                          for f in os.listdir('binary_results')
                          if f.startswith(f'{data_str}_{main_model_name}')})

    maximize_ratio = True

    preprocessor = data_preprocessing.DataPreprocessor(data_str)
    fg_l = list(preprocessor.fine_grain_labels.values())

    # Define label you want to remove here (fine grain only)
    remove_label_str = []
    remove_label_dict = [label for label in fg_l if label.l_str in remove_label_str]

    simulate_for_values(
        total_number_of_points=10,
        min_value=0.1,
        max_value=0.3,
        remove_label=remove_label_dict,
        binary_l_strs=binary_l_strs,
        binary_lr=binary_lr,
        binary_num_epochs=binary_num_epochs,
        maximize_ratio=maximize_ratio,
    )
