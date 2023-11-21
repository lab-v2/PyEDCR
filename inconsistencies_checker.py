import numpy as np
import EDCR_pipeline
import os
import torch

from models import VITFineTuner
from data_preprocessing import get_datasets, get_loaders, granularities
from vit_pipeline import cwd, batch_size

best_coarse_main_model = 'vit_b_16'
best_coarse_main_lr = '5e-05'
best_coarse_secondary_model = 'vit_l_16'
best_coarse_secondary_lr = '1e-05'
best_coarse_folder = (f'main_coarse_{best_coarse_main_model}_lr{best_coarse_main_lr}_secondary_'
                      f'{best_coarse_secondary_model}_lr{best_coarse_secondary_lr}')
best_coarse_results = np.load(rf'{EDCR_pipeline.figs_folder}{best_coarse_folder}/results.npy')
coarse_test_true = np.load(os.path.join(EDCR_pipeline.data_folder, f'test_true_coarse.npy'))

best_fine_main_model = 'vit_l_16'
best_fine_main_lr = '1e-06'
best_fine_secondary_model = 'vit_b_16'
best_fine_secondary_lr = '1e-06'
best_fine_folder = (f'main_fine_{best_fine_main_model}_lr{best_fine_main_lr}_secondary_'
                    f'{best_fine_secondary_model}_lr{best_fine_secondary_lr}')
best_fine_results = np.load(rf'{EDCR_pipeline.figs_folder}{best_fine_folder}/results.npy')
fine_test_true = np.load(os.path.join(EDCR_pipeline.data_folder, f'test_true.npy'))


def m(fine_batch_num: int,
      fine_data,
      coarse_test_loader):
    fines_to_coarses = {}
    fine_images, fine_labels, fine_names = fine_data[0], fine_data[1], fine_data[2]

    for curr_fine_image_num, (fine_image, fine_name) in enumerate(zip(fine_images, fine_names)):
        for coarse_batch_num, coarse_data in enumerate(coarse_test_loader):
            coarse_images, coarse_labels, coarse_names = coarse_data[0], coarse_data[1], coarse_data[2]
            for curr_coarse_image_num, (coarse_image, coarse_name) in enumerate(zip(coarse_images, coarse_names)):
                if torch.all(coarse_image == fine_image):
                    fine_index = fine_batch_num * batch_size + curr_fine_image_num
                    coarse_index = coarse_batch_num * batch_size + curr_coarse_image_num
                    fines_to_coarses[fine_index] = coarse_index
                    print(f'{fine_index}: {coarse_index}')
                    break
            if fine_name in fines_to_coarses:
                break


    return fines_to_coarses


def worker_init(args_tuple):
    fines_to_coarses = m(*args_tuple)
    return fines_to_coarses


if __name__ == "__main__":
    fine_tuners = {}
    loaders = {}

    for granularity in granularities.values():
        train_folder_name = f'train_{granularity}'
        test_folder_name = f'test_{granularity}'
        vit_model_names = [best_coarse_main_model]
        datasets, n = get_datasets(model_names=vit_model_names,
                                   cwd=cwd,
                                   train_folder_name=train_folder_name,
                                   test_folder_name=test_folder_name)
        granularity_fine_tuners = [VITFineTuner(model_name, vit_model_names, n) for model_name in vit_model_names]
        granularity_loaders = get_loaders(datasets=datasets,
                                          batch_size=batch_size,
                                          model_names=vit_model_names,
                                          train_folder_name=train_folder_name,
                                          test_folder_name=test_folder_name)

        fine_tuners[granularity] = granularity_fine_tuners
        loaders[granularity] = granularity_loaders

    fine_fine_tuner = fine_tuners['fine'][0]
    coarse_fine_tuner = fine_tuners['coarse'][0]

    fine_test_loader = loaders['fine'][f'{fine_fine_tuner}_test_fine']
    coarse_test_loader = loaders['coarse'][f'{coarse_fine_tuner}_test_coarse']

    args_list = [(fine_batch_num, fine_data, coarse_test_loader)
                 for fine_batch_num, fine_data in enumerate(fine_test_loader)]

    for fine_batch_num, fine_data in enumerate(fine_test_loader):
        worker_init((fine_batch_num, fine_data, coarse_test_loader))

    # with mp.Pool() as pool:
    #     res = pool.map(worker_init, args_list)

    # with open('res.json', 'w') as json_file:
    #     json.dump(res, json_file)

