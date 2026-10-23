import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

def forward_pass(w1, b1, w2, b2, w3, b3, w4, b4, input_data):
    """
    The network consists of 7 input nodes, 256 first hidden layer nodes, 128 second hidden layer nodes, 64 third hidden layer nodes, and 1 output layer node
    The activation function of the hidden layers is selu.
    Here you are expected to perform all the required operations for a forward pass over the network with the given dataset
    """
    ...

# we load all training, validation, and test datasets for the regression task
with open("datasets/part2_regression_train_dataset.data", "rb") as train_input_file:
    train_dataset, train_label = pickle.load(train_input_file)
with open("datasets/part2_regression_validation_dataset.data", "rb") as validation_input_file:
    validation_dataset, validation_label = pickle.load(validation_input_file)
with open("datasets/part2_regression_test_dataset.data", "rb") as test_input_file:
    test_dataset, test_label = pickle.load(test_input_file)

# In order to be able to work with Pytorch, all datasets (and labels/ground truth) should be converted into a tensor
# since the datasets are already available as numpy arrays, we simply create tensors from them via torch.from_numpy()

train_dataset = torch.from_numpy(train_dataset)
train_label = torch.from_numpy(train_label)

validation_dataset = torch.from_numpy(validation_dataset)
validation_label = torch.from_numpy(validation_label)

test_dataset = torch.from_numpy(test_dataset)
test_label = torch.from_numpy(test_label)

# we shuffle the training dataset by preserving instance-label relationship
indices = list(range(len(train_dataset)))
np.random.shuffle(indices)
train_dataset = train_dataset[indices]
train_label = train_label[indices]

# You are expected to create and initialize the parameters of the network
# Please do not forget to specify requires_grad=True for all parameters since they need to be trainable.

# W1 defines the parameters between the input layer and the first hidden layer
# Here you are expected to initialize W1 via the Normal distribution (mean=0, std=(2.0/(7+256))**0.5).
W1 = . . .
...
# b1 defines the bias parameters for the first hidden layer
# Here you are expected to initialize b1 via the Normal distribution (mean=0, std=(2.0/(1+256))**0.5).
b1 = . . .
...
# W2 defines the parameters between the first hidden layer and the second hidden layer
# Here you are expected to initialize W2 via the Normal distribution (mean=0, std=(2.0/(256+128))**0.5).
W2 = . . .
...
# b2 defines the bias parameters for the second hidden layer
# Here you are expected to initialize b2 via the Normal distribution (mean=0, std=(2.0/(1+128))**0.5).
b2 = . . .
...
# W3 defines the parameters between the second hidden layer and the third hidden layer
# Here you are expected to initialize W3 via the Normal distribution (mean=0, std=(2.0/(128+64))**0.5).
W3 = . . .
...

# b3 defines the bias parameters for the third hidden layer
# Here you are expected to initialize b3 via the Normal distribution (mean=0, std=(2.0/(1+64))**0.5).
b3 = . . .
...

# W4 defines the parameters between the third hidden layer and the output layer
# Here you are expected to initialize W4 via the Normal distribution (mean=0, std=(2.0/(64+1))**0.5).
W4 = . . .
...

# b4 defines the bias parameters for the output layer
# Here you are expected to initialize b4 via the Normal distribution (mean=0, std=(2.0/(1+1))**0.5).
b4 = . . .
...


# These arrays will store the loss values incurred at every training iteration
iteration_array = []
train_loss_array = []
validation_loss_array = []

# We utilize the stochastic gradient descent optimizer of PyTorch
# W1, b1, W2, b2, W3, b3 are the trainable parameters of the neural network
optimizer = torch.optim.SGD([W1, b1, W2, b2, W3, b3, W4, b4], lr=0.0001)


# We are going to perform the backpropagation algorithm 'ITERATION' times over the training dataset
# After each pass, we are calculating the average/mean squared error (MSE) loss over the validation dataset.
ITERATION = 1500
for iteration in range(1, ITERATION+1):
    iteration_array.append(iteration+1)
   
    # we need to zero all the stored gradient values calculated from the previous backpropagation step.
    optimizer.zero_grad()
    # Using the forward_pass function, we are performing a forward pass over the network with the training data   
    train_predictions = forward_pass(W1, b1, W2, b2, W3, b3, W4, b4, train_dataset)
    
    # Here you are expected to calculate the MEAN squared error loss with respect to the network predictions and the training ground truth
    train_mse_loss = ...
    
    train_loss_array.append(train_mse_loss.item())

    # We initiate the gradient calculation procedure to get gradient values with respect to the calculated loss 
    train_mse_loss.backward()
    # After the gradient calculation, we update the neural network parameters with the calculated gradients.  
    optimizer.step()
    with torch.no_grad():
        validation_predictions = forward_pass(W1, b1, W2, b2, W3, b3, W4, b4, validation_dataset)
        # Here you are expected to calculate the average/mean squared error loss for the validation datasets by using the validation dataset ground truth.
        validation_mse_loss = ...
        validation_loss_array.append(validation_mse_loss.item())
    print("Iteration : %d - Train MSE Loss %.4f - Validation MSE Loss : %.2f" % (iteration, train_mse_loss.item(), validation_mse_loss.item()))

# after completing the training, we calculate our network's mean squared error score on the test dataset...
# Again, here we don't need to perform any gradient-related operations, so we are using torch.no_grad() function.
with torch.no_grad():
    test_predictions = forward_pass(W1, b1, W2, b2, W3, b3, W4, b4, test_dataset)
    # Here you are expected to calculate the network's MSE on the test dataset...
    test_loss = ...
    print("Test MSE loss : %.4f" % test_loss.item())
    print("Test RMSE loss : %.4f" % (test_loss.item()**0.5))    
# We plot the loss versus iteration graph for both datasets (training and validation)
plt.plot(iteration_array, train_loss_array, label="Train Loss")
plt.plot(iteration_array, validation_loss_array, label="Validation Loss")
plt.legend()
plt.show()





