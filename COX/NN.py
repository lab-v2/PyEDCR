import typing

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output


# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=2 * output_size)

    def forward(self, x):
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
             [ta[1] for ta in train_accuracies], label='Train Pred Accuracy', color='green')
    plt.plot(range(1, len(test_accuracies) + 1),
             [va[0] for va in test_accuracies], label='Validation GT Accuracy', color='red')
    plt.plot(range(1, len(test_accuracies) + 1),
             [va[1] for va in test_accuracies], label='Validation Pred Accuracy', color='red')

    plt.yticks(np.arange(0, 1.1, 0.1))

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.grid()
    plt.show()
    clear_output(wait=True)
    display(plt.gcf())


def calculate_accuracy(model,
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

        # print(f'Accuracy for gt: {accuracy_gt:.4f}')
        # print(f'Accuracy for pred: {accuracy_pred:.4f}')

    model.train()
    return accuracy_gt, accuracy_pred, predicted_gt.numpy(), predicted_pred.numpy()


def train_confuse_NN(num_epochs: int):
    df = pd.read_csv('dataset_may_2024.csv')
    data_with_gt = df[df['gt'].notna()]
    num_classes = data_with_gt['gt'].nunique()

    # Selecting the features (p0 to p31) and the targets (gt and pred)
    features = [f'p{i}' for i in range(32)]
    X = data_with_gt[features].values

    Y_gt, _ = pd.factorize(data_with_gt['gt'])
    Y_pred, _ = pd.factorize(data_with_gt['pred'])
    Y = np.stack((Y_gt, Y_pred), axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SimpleNN(input_size=X_train.shape[1], output_size=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    losses = []
    train_accuracies = []
    test_accuracies = []
    final_train_predictions = []
    final_test_predictions = []

    for epoch in range(1, num_epochs + 1):
        inputs = torch.tensor(X_train, dtype=torch.float32)
        targets = torch.tensor(Y_train, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, num_classes), targets.view(-1))
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # Calculate accuracies
        train_accuracy_gt, train_accuracy_pred, train_pred_gt, train_pred_pred = calculate_accuracy(
            model, torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.long), num_classes)
        test_accuracy_gt, test_accuracy_pred, test_pred_gt, test_pred_pred = calculate_accuracy(
            model, torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.long), num_classes)

        if epoch == num_epochs:
            final_train_predictions = ['{:02}{:02}'.format(gt, pred)
                                       for gt, pred in zip(train_pred_gt, train_pred_pred)]
            final_test_predictions = ['{:02}{:02}'.format(gt, pred)
                                      for gt, pred in zip(test_pred_gt, test_pred_pred)]


        train_accuracy = [train_accuracy_gt, train_accuracy_pred]
        test_accuracy = [test_accuracy_gt, test_accuracy_pred]
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        if epoch % 100 == 0:  # Update plot every 100 epochs
            print(
                f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}, '
                f'Train Accuracy: {train_accuracy}, '
                f'Test Accuracy: {test_accuracy}')
            plot_metrics_dynamic(losses, train_accuracies, test_accuracies)

    # Final plot
    plot_metrics_dynamic(losses, train_accuracies, test_accuracies)

    # Save predictions
    np.save('train_predictions.npy', np.array(final_train_predictions, dtype=int))
    np.save('test_predictions.npy', np.array(final_test_predictions, dtype=int))


if __name__ == "__main__":
    train_confuse_NN(num_epochs=1000)
