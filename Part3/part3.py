from typing import Any

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import pickle
import math

# we load all the datasets of Part 3
with open("datasets/part3_train_dataset.data", "rb") as train_input_file:
    x_train, y_train = pickle.load(train_input_file)

with open("datasets/part3_validation_dataset.data", "rb") as validation_input_file:
    x_validation, y_validation = pickle.load(validation_input_file)

with open("datasets/part3_test_dataset.data", "rb") as test_input_file:
    x_test, y_test = pickle.load(test_input_file)


# We first shuffle the training data
indices = list(range(len(x_train)))
np.random.shuffle(indices)
x_train = x_train[indices]
y_train = y_train[indices]

# We rescale pixel values to [0, 1] range
x_train = x_train.astype(np.float32) / 255.0
train_mean = np.mean(x_train, axis=0)
train_std = np.std(x_train, axis=0, keepdims=True)
# then we standardize the feature values so that they have mean=0, std=1
x_train = (x_train - train_mean) / train_std

# by utilizing the mean and std calcuted from the training dataset
# we rescale and standardize the validation dataset
x_validation = x_validation.astype(np.float32) / 255.0
x_validation = (x_validation - train_mean) / train_std

# by utilizing the mean and std calcuted from the training dataset
# we rescale and standardize the test dataset
x_test = x_test.astype(np.float32) / 255.0
x_test = (x_test - train_mean) / train_std

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# and converting them into Pytorch tensors in order to be able to work with Pytorch
X_train = torch.from_numpy(x_train).to(device)
L_train = torch.from_numpy(y_train).to(torch.long).to(device)

X_validation = torch.from_numpy(x_validation).to(device)
L_validation = torch.from_numpy(y_validation).to(torch.long).to(device)

X_test = torch.from_numpy(x_test).to(device)
L_test = torch.from_numpy(y_test).to(torch.long).to(device)

configs = [
    {
        "hidden_layers": [16, 8],
        "activation_function": nn.Sigmoid,
        "optimizer": torch.optim.SGD,
        "learning_rate": 0.1,
        "epochs": 200,
    }
]


class BugClassifier(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, activation_function):
        super().__init__()
        layers = []
        for layer in hidden_layers:
            layers.append(nn.Linear(input_dim, layer))
            layers.append(activation_function())
            input_dim = layer
        layers.append(nn.Linear(input_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    train_data: torch.Tensor,
    validation_data: torch.Tensor,
    train_labels,
    validation_labels,
):
    criterion = torch.nn.CrossEntropyLoss()
    accuracy = 0
    train_losses = []
    validation_losses = []
    validation_outputs = torch.zeros((len(validation_data), 3))
    for i in range(epochs):
        optimizer.zero_grad()
        outputs = model(train_data)
        train_loss = criterion(outputs, train_labels)
        train_losses.append(train_loss.item())
        train_loss.backward()
        optimizer.step()
        with torch.no_grad():
            validation_outputs = model(validation_data)
            validation_loss = criterion(validation_outputs, validation_labels)
            validation_losses.append(validation_loss.item())
    corrects = (
        (validation_labels == torch.argmax(validation_outputs, dim=1)).sum().item()
    )
    accuracy = corrects / len(validation_labels) * 100
    return accuracy, train_losses, validation_losses


confidence_intervals = torch.zeros(len(configs))
mean_accuracies = torch.zeros(len(configs))

for idx, config in enumerate(configs):
    accuracies = torch.zeros(10)
    average_train_losses = torch.zeros(config["epochs"])
    average_validation_losses = torch.zeros(config["epochs"])

    for i in range(10):  # bir configi 10 kere denediğimiz yer
        classifier = BugClassifier(
            30000,
            config["hidden_layers"],
            100,
            config["activation_function"],
        ).to(device)
        accuracy, train_losses, validation_losses = train(
            classifier,
            config["optimizer"](classifier.parameters(), lr=config["learning_rate"]),
            config["epochs"],
            X_train,
            X_validation,
            L_train,
            L_validation,
        )
        accuracies[i] = accuracy
        average_train_losses += torch.tensor(train_losses)
        average_validation_losses += torch.tensor(validation_losses)

    average_train_losses /= 10
    average_validation_losses /= 10
    mean = torch.mean(accuracies)
    mean_accuracies[idx] = mean
    std = torch.std(accuracies)
    confidence_interval = 1.96 * std / math.sqrt(10)
    confidence_intervals[idx] = confidence_interval

    print(
        f"Config {idx}: Mean Validation Acc: {mean.item():.2f}% ± {confidence_interval.item():.2f}%"
    )
    plt.plot(average_train_losses, label="Average Train Loss")
    plt.plot(average_validation_losses, label="Average Validation Loss")
    plt.legend()
    plt.show()

best_config = configs[torch.argmax(mean_accuracies)]
X_combined = torch.cat([X_train, X_validation], dim=0)
L_combined = torch.cat([L_train, L_validation], dim=0)

test_accuracies = torch.zeros(10)

for i in range(10):
    best_model = BugClassifier(
        30000,
        best_config["hidden_layers"],
        100,
        best_config["activation_function"],
    ).to(device)
    optimizer = best_config["optimizer"](
        best_model.parameters(), lr=best_config["learning_rate"]
    )

    # Train on X_combined and evaluate on X_test
    test_acc, _, _ = train(
        best_model,
        optimizer,
        best_config["epochs"],
        X_combined,
        X_test,
        L_combined,
        L_test,
    )
    test_accuracies[i] = test_acc

test_mean = torch.mean(test_accuracies).item()
test_std = torch.std(test_accuracies).item()
test_ci = 1.96 * test_std / math.sqrt(10)

print(f"Final Test Accuracy: {test_mean:.2f}% ± {test_ci:.2f}%")
