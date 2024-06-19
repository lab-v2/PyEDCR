import typing
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import data_preprocessing


# Define the neural network
class MLP(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=2 * output_size)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def plot_metrics_dynamic(losses: typing.List[float],
                         train_accuracies: typing.List[typing.List[float]],
                         test_accuracies: typing.List[typing.List[float]]):
    plt.figure(figsize=(14, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(losses) + 1), losses, label='Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid()

    # Plot Training and Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1),
             [ta[0] for ta in train_accuracies], label='Train GT Accuracy', color='green')
    plt.plot(range(1, len(train_accuracies) + 1),
             [ta[1] for ta in train_accuracies], label='Train Pred Accuracy', color='lime')
    plt.plot(range(1, len(test_accuracies) + 1),
             [va[0] for va in test_accuracies], label='Validation GT Accuracy', color='red')
    plt.plot(range(1, len(test_accuracies) + 1),
             [va[1] for va in test_accuracies], label='Validation Pred Accuracy', color='orange')

    plt.yticks(np.arange(0, 1.1, 0.1))

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.grid()
    plt.show()


def calculate_accuracy_and_predictions(model,
                                       data,
                                       targets,
                                       num_classes):
    model.eval()
    with torch.no_grad():
        outputs = model(data)
        predicted_gt = torch.argmax(outputs[:, :num_classes], dim=1)
        predicted_pred = torch.argmax(outputs[:, num_classes:2 * num_classes], dim=1)

        correct_gt = (predicted_gt == targets[:, 0]).sum().item()
        correct_pred = (predicted_pred == targets[:, 1]).sum().item()

        accuracy_gt = correct_gt / targets.size(0)
        accuracy_pred = correct_pred / targets.size(0)

    model.train()
    return accuracy_gt, accuracy_pred, predicted_gt, predicted_pred


def train_confuse_NN(num_epochs: int):
    data_str = 'COX'
    main_model_name = 'main_model'
    df = pd.read_csv('COX/dataset_may_2024.csv')
    data_with_gt = df[df['gt'].notna()]
    num_classes = data_with_gt['gt'].nunique()

    # Selecting the features (p0 to p31) and the targets (gt and pred)
    features = [f'p{i}' for i in range(32)]
    X = data_with_gt[features].values

    Y_gt, gt_labels = pd.factorize(data_with_gt['gt'])
    Y_pred, pred_labels = pd.factorize(data_with_gt['pred'])

    Y = np.stack((Y_gt, Y_pred), axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    assert np.all(Y_train[:, 0] == np.load(data_preprocessing.get_filepath(data_str=data_str,
                                                                           model_name=main_model_name,
                                                                           test=False,
                                                                           pred=False)))
    assert np.all(Y_train[:, 1] == np.load(data_preprocessing.get_filepath(data_str=data_str,
                                                                           model_name=main_model_name,
                                                                           test=False,
                                                                           pred=True)))
    assert np.all(Y_test[:, 0] == np.load(data_preprocessing.get_filepath(data_str=data_str,
                                                                          model_name=main_model_name,
                                                                          test=True,
                                                                          pred=False)))
    assert np.all(Y_test[:, 1] == np.load(data_preprocessing.get_filepath(data_str=data_str,
                                                                          model_name=main_model_name,
                                                                          test=True,
                                                                          pred=True)))

    np.save(data_preprocessing.get_filepath(data_str=data_str,
                                            model_name=main_model_name,
                                            test=False,
                                            pred=False),
            Y_train[:, 0])
    np.save(data_preprocessing.get_filepath(data_str=data_str,
                                            model_name=main_model_name,
                                            test=False,
                                            pred=True),
            Y_train[:, 1])

    np.save(data_preprocessing.get_filepath(data_str=data_str,
                                            model_name=main_model_name,
                                            test=True,
                                            pred=False),
            Y_test[:, 0])
    np.save(data_preprocessing.get_filepath(data_str=data_str,
                                            model_name=main_model_name,
                                            test=True,
                                            pred=True),
            Y_test[:, 1])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = MLP(input_size=X_train.shape[1],
                output_size=num_classes)
    criterion = nn.CrossEntropyLoss()

    lr = 0.5
    # Using SGD with momentum
    optimizer = optim.SGD(params=model.parameters(),
                          lr=lr,
                          momentum=0.9)

    # Using ReduceLROnPlateau scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                     mode='min',
                                                     factor=0.1,
                                                     patience=10)

    losses = []
    train_accuracies = []
    test_accuracies = []

    print("Starting training...")

    inputs = torch.tensor(X_train, dtype=torch.float32)
    targets = torch.tensor(Y_train, dtype=torch.long)

    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss_gt = criterion(outputs[:, :num_classes], targets[:, 0])
        loss_pred = criterion(outputs[:, num_classes:], targets[:, 1])
        loss = loss_gt * 2 + loss_pred  # Adjust the weight of GT loss
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step(loss)

        losses.append(loss.item())

        # Calculate accuracies
        train_accuracy_gt, train_accuracy_pred = calculate_accuracy_and_predictions(
            model, torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.long), num_classes)[:2]
        test_accuracy_gt, test_accuracy_pred = calculate_accuracy_and_predictions(
            model, torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.long), num_classes)[:2]

        train_accuracy = [train_accuracy_gt, train_accuracy_pred]
        test_accuracy = [test_accuracy_gt, test_accuracy_pred]
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        if epoch % 100 == 0 or epoch == num_epochs:  # Update plot every 100 epochs or at the end
            print(
                f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}, '
                f'Train Accuracy (GT, Pred): ({train_accuracy_gt:.4f}, {train_accuracy_pred:.4f}), '
                f'Test Accuracy (GT, Pred): ({test_accuracy_gt:.4f}, {test_accuracy_pred:.4f})')
            plot_metrics_dynamic(losses, train_accuracies, test_accuracies)

    # Save final predictions
    train_accuracy_gt, train_accuracy_pred, train_pred_gt, train_pred_pred = calculate_accuracy_and_predictions(
        model, torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.long), num_classes)
    test_accuracy_gt, test_accuracy_pred, test_pred_gt, test_pred_pred = calculate_accuracy_and_predictions(
        model, torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.long), num_classes)

    train_predictions = ['{:02}{:02}'.format(gt, pred)
                         for gt, pred in zip(train_pred_gt.numpy(), train_pred_pred.numpy())]
    test_predictions = ['{:02}{:02}'.format(gt, pred)
                        for gt, pred in zip(test_pred_gt.numpy(), test_pred_pred.numpy())]

    np.save(data_preprocessing.get_filepath(data_str='COX',
                                            model_name='MLP',
                                            test=False,
                                            loss='CE',
                                            lr=lr,
                                            pred=True,
                                            epoch=num_epochs), np.array(train_predictions))
    np.save(data_preprocessing.get_filepath(data_str='COX',
                                            model_name='MLP',
                                            test=True,
                                            loss='CE',
                                            lr=lr,
                                            pred=True,
                                            epoch=num_epochs), np.array(test_predictions))
    print("Training complete.")


if __name__ == "__main__":
    train_confuse_NN(num_epochs=500)
