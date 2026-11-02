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

# gpu cpu seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# and converting them into Pytorch tensors in order to be able to work with Pytorch
# tensorleri device'a gönder
X_train = torch.from_numpy(x_train).to(device)
L_train = torch.from_numpy(y_train).to(torch.long).to(device)

X_validation = torch.from_numpy(x_validation).to(device)
L_validation = torch.from_numpy(y_validation).to(torch.long).to(device)

X_test = torch.from_numpy(x_test).to(device)
L_test = torch.from_numpy(y_test).to(torch.long).to(device)


# hidden layers: [256], [512, 256]
# activation functions: Sigmoid, ReLU
# learning rates: 0.001, 0.0001
# epochs: 100, 200
configs = [
    {
        "hidden_layers": [256],
        "activation_function": nn.Sigmoid,
        "learning_rate": 0.001,
        "epochs": 100,
    },
    {
        "hidden_layers": [256],
        "activation_function": nn.Sigmoid,
        "learning_rate": 0.001,
        "epochs": 200,
    },
    {
        "hidden_layers": [256],
        "activation_function": nn.Sigmoid,
        "learning_rate": 0.0001,
        "epochs": 100,
    },
    {
        "hidden_layers": [256],
        "activation_function": nn.Sigmoid,
        "learning_rate": 0.0001,
        "epochs": 200,
    },
    {
        "hidden_layers": [256],
        "activation_function": nn.ReLU,
        "learning_rate": 0.001,
        "epochs": 100,
    },
    {
        "hidden_layers": [256],
        "activation_function": nn.ReLU,
        "learning_rate": 0.001,
        "epochs": 200,
    },
    {
        "hidden_layers": [256],
        "activation_function": nn.ReLU,
        "learning_rate": 0.0001,
        "epochs": 100,
    },
    {
        "hidden_layers": [256],
        "activation_function": nn.ReLU,
        "learning_rate": 0.0001,
        "epochs": 200,
    },
    {
        "hidden_layers": [512, 256],
        "activation_function": nn.Sigmoid,
        "learning_rate": 0.001,
        "epochs": 100,
    },
    {
        "hidden_layers": [512, 256],
        "activation_function": nn.Sigmoid,
        "learning_rate": 0.001,
        "epochs": 200,
    },
    {
        "hidden_layers": [512, 256],
        "activation_function": nn.Sigmoid,
        "learning_rate": 0.0001,
        "epochs": 100,
    },
    {
        "hidden_layers": [512, 256],
        "activation_function": nn.Sigmoid,
        "learning_rate": 0.0001,
        "epochs": 200,
    },
    {
        "hidden_layers": [512, 256],
        "activation_function": nn.ReLU,
        "learning_rate": 0.001,
        "epochs": 100,
    },
    {
        "hidden_layers": [512, 256],
        "activation_function": nn.ReLU,
        "learning_rate": 0.001,
        "epochs": 200,
    },
    {
        "hidden_layers": [512, 256],
        "activation_function": nn.ReLU,
        "learning_rate": 0.0001,
        "epochs": 100,
    },
    {
        "hidden_layers": [512, 256],
        "activation_function": nn.ReLU,
        "learning_rate": 0.0001,
        "epochs": 200,
    },
]

NUMBER_OF_RUNS = 10


def config_to_string(config):
    return (
        f"hidden_layers={config['hidden_layers']}, "
        f"activation={config['activation_function'].__name__}, "
        f"lr={config['learning_rate']}, epochs={config['epochs']}"
    )


# activation function'dan instance üretmek gerekiyor
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


def calculate_accuracy(model, data, labels):  # labels: [2,5,1,0,...]
    with torch.no_grad():
        predictions = torch.argmax(model(data), dim=1)  # [2, 0, ...]
    return (predictions == labels).sum().item() / len(labels) * 100


def confidence_interval(scores):
    return 1.96 * torch.std(scores).item() / math.sqrt(len(scores))


def train(
    model: torch.nn.Module,
    learning_rate: float,
    epochs: int,
    train_data: torch.Tensor,
    validation_data: torch.Tensor,
    train_labels,
    validation_labels,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    train_losses = torch.zeros(epochs)
    validation_losses = torch.zeros(epochs)
    for i in range(epochs):
        optimizer.zero_grad()
        outputs = model(train_data)
        train_loss = criterion(outputs, train_labels)
        train_losses[i] = train_loss.item()
        train_loss.backward()
        optimizer.step()
        with torch.no_grad():
            validation_outputs = model(validation_data)
            validation_losses[i] = criterion(
                validation_outputs, validation_labels
            ).item()
    train_accuracy = calculate_accuracy(model, train_data, train_labels)
    validation_accuracy = calculate_accuracy(model, validation_data, validation_labels)
    return train_accuracy, validation_accuracy, train_losses, validation_losses


confidence_intervals = torch.zeros(len(configs))
mean_validation_accuracies = torch.zeros(len(configs))

for idx, config in enumerate(configs):
    avg_train_losses = torch.zeros(config["epochs"])
    avg_validation_losses = torch.zeros(config["epochs"])
    runs_train_accuracies = torch.zeros(NUMBER_OF_RUNS)
    runs_validation_accuracies = torch.zeros(NUMBER_OF_RUNS)

    for i in range(NUMBER_OF_RUNS):
        model = BugClassifier(
            30000, config["hidden_layers"], 100, config["activation_function"]
        ).to(device)
        train_accuracy, validation_accuracy, train_losses, validation_losses = train(
            model,
            config["learning_rate"],
            config["epochs"],
            X_train,
            X_validation,
            L_train,
            L_validation,
        )
        avg_train_losses += train_losses
        avg_validation_losses += validation_losses
        runs_train_accuracies[i] = train_accuracy
        runs_validation_accuracies[i] = validation_accuracy

    mean_train_accuracy = torch.mean(runs_train_accuracies)
    mean_validation_accuracy = torch.mean(runs_validation_accuracies)
    train_ci = confidence_interval(runs_train_accuracies)
    validation_ci = confidence_interval(runs_validation_accuracies)
    avg_train_losses /= NUMBER_OF_RUNS
    avg_validation_losses /= NUMBER_OF_RUNS

    mean_validation_accuracies[idx] = mean_validation_accuracy
    confidence_intervals[idx] = validation_ci

    print(f"Config {idx}: {config_to_string(config)}")
    print(
        f"    Mean Train Acc: {mean_train_accuracy.item():.2f}% ± {train_ci:.2f}% | "
        f"Mean Validation Acc: {mean_validation_accuracy.item():.2f}% ± "
        f"{confidence_intervals[idx].item():.2f}%"
    )

    # SRM'leri dosyaya yaz
    plt.figure()
    plt.plot(avg_train_losses, label="Average Train Loss")
    plt.plot(avg_validation_losses, label="Average Validation Loss")
    plt.title(f"Config {idx}: {config_to_string(config)}", fontsize=7)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.savefig(f"srm_config_{idx}.png")
    plt.close()

# en iyi config validation acc en yüksek olan
best_idx = int(torch.argmax(mean_validation_accuracies))
best_config = configs[best_idx]
print(f"Best config: Config {best_idx}: {config_to_string(best_config)}")

# train ve validation datasetlerini birleştirip en iyi modelle deneme
X_combined = torch.cat([X_train, X_validation], dim=0)
L_combined = torch.cat([L_train, L_validation], dim=0)

combined_train_accuracies = torch.zeros(NUMBER_OF_RUNS)
test_accuracies = torch.zeros(NUMBER_OF_RUNS)

for i in range(NUMBER_OF_RUNS):
    best_model = BugClassifier(
        30000,
        best_config["hidden_layers"],
        100,
        best_config["activation_function"],
    ).to(device)

    combined_train_acc, test_acc, _, _ = train(
        best_model,
        best_config["learning_rate"],
        best_config["epochs"],
        X_combined,
        X_test,
        L_combined,
        L_test,
    )
    combined_train_accuracies[i] = combined_train_acc
    test_accuracies[i] = test_acc

combined_train_accuracy_mean = torch.mean(combined_train_accuracies).item()
combined_train_ci = confidence_interval(combined_train_accuracies)
test_mean = torch.mean(test_accuracies).item()
test_ci = confidence_interval(test_accuracies)

print(
    f"Final Train (train + validation) Accuracy: "
    f"{combined_train_accuracy_mean:.2f}% ± {combined_train_ci:.2f}%"
)
print(f"Final Test Accuracy: {test_mean:.2f}% ± {test_ci:.2f}%")
