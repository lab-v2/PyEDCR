import typing
import torch
import torch.utils.data
from sklearn.metrics import accuracy_score

import models
import utils


def test(fine_grain_model: models.FineTuner,

         loaders: dict[str, torch.utils.data.DataLoader],
         device: torch.device,
         test_folder_name: str) -> typing.Tuple[list[int], list[int], float]:
    test_loader = loaders[f'{fine_tuner}_{test_folder_name}']
    fine_tuner.eval()
    correct = 0
    total = 0
    test_prediction = []
    test_ground_truth = []
    name_list = []

    print(f'Started testing {fine_tuner} on {device}...')

    with torch.no_grad():
        if utils.is_local():
            from tqdm import tqdm
            gen = tqdm(enumerate(test_loader), total=len(test_loader))
        else:
            gen = enumerate(test_loader)

        for i, data in gen:
            pred_temp = []
            truth_temp = []
            name_temp = []
            images, labels, names = data[0].to(device), data[1].to(device), data[2]
            outputs = fine_tuner(images)
            predicted = torch.max(outputs.data, 1)[1]
            test_ground_truth += labels.tolist()
            test_prediction += predicted.tolist()
            name_list += names  # Collect the name values
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pred_temp += predicted.tolist()
            truth_temp += labels.tolist()
            name_temp += names

    test_accuracy = round(accuracy_score(y_true=test_ground_truth, y_pred=test_prediction), 3)
    print(f'\nTest accuracy: {test_accuracy}')

    return test_ground_truth, test_prediction, test_accuracy