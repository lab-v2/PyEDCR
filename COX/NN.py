import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

df = pd.read_csv('dataset_may_2024.csv')

# Selecting the features (p0 to p31) and the targets (gt and pred)
features = [f'p{i}' for i in range(32)]
X = df[features].values
y_gt, y_gt_labels = pd.factorize(df['gt'])
y_pred, y_pred_labels = pd.factorize(df['pred'])
y = np.column_stack((y_gt, y_pred))

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting the data into train and test sets with an 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Determine the number of unique classes in gt and pred
num_classes_gt = len(y_gt_labels)
num_classes_pred = len(y_pred_labels)
total_classes = num_classes_gt + num_classes_pred


# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3_gt = nn.Linear(32, num_classes_gt)
        self.fc3_pred = nn.Linear(32, num_classes_pred)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        output_gt = self.fc3_gt(x)
        output_pred = self.fc3_pred(x)
        return output_gt, output_pred


# Initialize the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs_gt, outputs_pred = model(X_train_tensor)

    # Concatenate the outputs and targets along the batch dimension
    outputs_combined = torch.cat((outputs_gt, outputs_pred), dim=0)
    targets_combined = torch.cat((y_train_tensor[:, 0], y_train_tensor[:, 1]), dim=0)

    # Calculate the loss
    loss = criterion(outputs_combined, targets_combined)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluating the model
model.eval()
with torch.no_grad():
    outputs_gt, outputs_pred = model(X_test_tensor)

    _, predicted_gt = torch.max(outputs_gt, 1)
    _, predicted_pred = torch.max(outputs_pred, 1)

    correct_gt = (predicted_gt == y_test_tensor[:, 0]).sum().item()
    correct_pred = (predicted_pred == y_test_tensor[:, 1]).sum().item()

    accuracy_gt = correct_gt / y_test_tensor.size(0)
    accuracy_pred = correct_pred / y_test_tensor.size(0)

    print(f'Accuracy for gt: {accuracy_gt:.4f}')
    print(f'Accuracy for pred: {accuracy_pred:.4f}')