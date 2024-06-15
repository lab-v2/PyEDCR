import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_random_forest():
    df = pd.read_csv('dataset_may_2024.csv')
    data_with_gt = df[df['gt'].notna()]

    # Selecting the features (p0 to p31) and the targets (gt and pred)
    features = [f'p{i}' for i in range(32)]
    X = data_with_gt[features].values

    Y_gt, _ = pd.factorize(data_with_gt['gt'])
    Y_pred, _ = pd.factorize(data_with_gt['pred'])
    Y = np.stack((Y_gt, Y_pred), axis=1)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model and calculate accuracy
    model.fit(X_train, Y_train[:, 0])
    train_accuracy = accuracy_score(Y_train[:, 0], model.predict(X_train))
    val_accuracy = accuracy_score(Y_val[:, 0], model.predict(X_val))

    print(f'Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}')


if __name__ == "__main__":
    print("Starting training...")
    train_random_forest()
    print("Training complete.")
