import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

# this function returns the neural network output for a given dataset and set of parameters
def forward_pass(w1, b1, w2, b2, w3, b3, w4, b4, input_data):
    """
    The network consists of 8 input layer nodes, 64 first hidden layer nodes, 32 second hidden layer nodes, 16 third hidden layer nodes, and 3 output layer nodes
    The activation function of the hidden layers is leaky relu.
    The output layer should apply the softmax function to obtain posterior probability distribution. And the function should return this distribution
    Here you are expected to perform all the required operations for a forward pass over the network with the given dataset
    """
    ...
    "Your implementation goes here"
    ...

# we load all training, validation, and test datasets for the classification task
with open("datasets/part2_classification_train_dataset.data", "rb") as train_input_file:
    train_dataset, train_label = pickle.load(train_input_file)
with open("datasets/part2_classification_validation_dataset.data", "rb") as validation_input_file:
    validation_dataset, validation_label = pickle.load(validation_input_file)
with open("datasets/part2_classification_test_dataset.data", "rb") as test_input_file:
    test_dataset, test_label = pickle.load(test_input_file)

# In order to be able to work with Pytorch, all datasets (and labels/ground truth) should be converted into a tensor
# since the datasets are already available as numpy arrays, we simply create tensors from them via torch.from_numpy()
train_dataset = torch.from_numpy(train_dataset)
train_label = torch.from_numpy(train_label)

validation_dataset = torch.from_numpy(validation_dataset)
validation_label = torch.from_numpy(validation_label)

test_dataset = torch.from_numpy(test_dataset)
test_label = torch.from_numpy(test_label)

# You are expected to create and initialize the parameters of the network

# W1 defines the parameters between the input layer and the first hidden layer
# Here you are expected to initialize W1 via the Normal distribution (mean=0, std=(2.0/(8+64))**0.5).
W1 = . . .
...
# b1 defines the bias parameters for the first hidden layer
# Here you are expected to initialize b1 via the Normal distribution (mean=0, std=(2.0/(1+64))**0.5).
b1 = . . .
...
# W2 defines the parameters between the first hidden layer and the second hidden layer
# Here you are expected to initialize W2 via the Normal distribution (mean=0, std=(2.0/(64+32))**0.5).
W2 = . . .
...
# b2 defines the bias parameters for the second hidden layer
# Here you are expected to initialize b2 via the Normal distribution (mean=0, std=(2.0/(1+32))**0.5).
b2 = . . .
...
# W3 defines the parameters between the second hidden layer and the third hidden layer
# Here you are expected to initialize W3 via the Normal distribution (mean=0, std=(2.0/(32+16))**0.5).
W3 = . . .
...

# b3 defines the bias parameters for the third hidden layer
# Here you are expected to initialize b3 via the Normal distribution (mean=0, std=(2.0/(1+16))**0.5).
b3 = . . .
...

# W4 defines the parameters between the third hidden layer and the output layer
# Here you are expected to initialize W4 via the Normal distribution (mean=0, std=(2.0/(16+3))**0.5).
W4 = . . .
...

# b4 defines the bias parameters for the output layer
# Here you are expected to initialize b4 via the Normal distribution (mean=0, std=(2.0/(1+3))**0.5).
b4 = . . .
...

# We utilize the Adam optimizer of PyTorch
# W1, b1, W2, b2, W3, b3, W4, b4 are the trainable parameters of the neural network
optimizer = torch.optim.Adam([W1, b1, W2, b2, W3, b3, W4, b4], lr=0.0001)

# These arrays will store the loss values incurred at every training iteration
iteration_array = []
train_loss_array = []
validation_loss_array = []

# We are going to perform the backpropagation algorithm 'ITERATION' times over the training dataset
# After each pass, we are calculating the average/mean cross entropy loss over the validation dataset along with accuracy scores on both datasets.
ITERATION = 1000
for iteration in range(1, ITERATION+1):
    iteration_array.append(iteration)

    
    # we need to zero all the stored gradient values calculated from the previous backpropagation step.
    optimizer.zero_grad()
    # Using the forward_pass function, we are performing a forward pass over the network with the training data
    train_predictions = forward_pass(W1, b1, W2, b2, W3, b3, W4, b4, train_dataset)
    
    # here you are expected to calculate the MEAN cross-entropy loss with respect to the network predictions and the training label
    train_mean_crossentropy_loss = ...
    
    train_loss_array.append(train_mean_crossentropy_loss.item())

    # We initiate the gradient calculation procedure to get gradient values with respect to the calculated loss 
    train_mean_crossentropy_loss.backward()
    # After the gradient calculation, we update the neural network parameters with the calculated gradients.
    optimizer.step()

    # after each epoch on the training data we are calculating the loss and accuracy scores on the validation dataset
    # with torch.no_grad() disables gradient operations, since during testing the validation dataset, we don't need to perform any gradient operations
    with torch.no_grad():
        # Here you are expected to calculate the accuracy score on the training dataset by using the training labels.
        train_predictions = forward_pass(W1, b1, W2, b2, W3, b3, W4, b4, train_dataset)

        train_accuracy = ...

        validation_predictions = forward_pass(W1, b1, W2, b2, W3, b3, W4, b4, validation_dataset)
        
        # Here you are expected to calculate the average/mean cross entropy loss for the validation dataset by using the validation dataset labels.
        validation_mean_crossentropy_loss = ...

        validation_loss_array.append(validation_mean_crossentropy_loss.item())

        # Similarly, here, you are expected to calculate the accuracy score on the validation dataset
        validation_accuracy = ...
    
    print("Iteration : %d - Train Loss %.4f - Train Accuracy : %.2f - Validation Loss : %.4f Validation Accuracy : %.2f" % (iteration, train_mean_crossentropy_loss.item(), train_accuracy, validation_mean_crossentropy_loss.item(), validation_accuracy))


# after completing the training, we calculate our network's accuracy score on the test dataset...
# Again, here we don't need to perform any gradient-related operations, so we are using torch.no_grad() function.
with torch.no_grad():
    test_predictions = forward_pass(W1, b1, W2, b2, W3, b3, W4, b4, test_dataset)
    # Here you are expected to calculate the network's accuracy score on the test dataset...
    test_accuracy = ...
    print("Test accuracy : %.2f" % (test_accuracy.item()))

# We plot the loss versus iteration graph for both datasets (training and validation)
plt.plot(iteration_array, train_loss_array, label="Train Loss")
plt.plot(iteration_array, validation_loss_array, label="Validation Loss")
plt.legend()
plt.show()





