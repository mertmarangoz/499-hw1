import torch
import numpy as np
import pickle

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
x_train = x_train.astype(np.float32)/255.0
train_mean = np.mean(x_train, axis=0)
train_std = np.std(x_train, axis=0, keepdims=True)
# then we standardize the feature values so that they have mean=0, std=1
x_train = (x_train-train_mean)/train_std

# by utilizing the mean and std calcuted from the training dataset
# we rescale and standardize the validation dataset
x_validation = x_validation.astype(np.float32)/255.0
x_validation = (x_validation-train_mean)/train_std

# by utilizing the mean and std calcuted from the training dataset
# we rescale and standardize the test dataset
x_test = x_test.astype(np.float32) / 255.0
x_test = (x_test-train_mean)/train_std

# and converting them into Pytorch tensors in order to be able to work with Pytorch
X_train = torch.from_numpy(x_train)
L_train = torch.from_numpy(y_train).to(torch.long)

X_validation = torch.from_numpy(x_validation)
L_validation = torch.from_numpy(y_validation).to(torch.long)

X_test = torch.from_numpy(x_test)
L_test = torch.from_numpy(y_test).to(torch.long)
