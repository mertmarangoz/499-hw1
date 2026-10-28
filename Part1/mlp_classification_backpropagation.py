import pickle
import numpy as np

def sigmoid(x: np.ndarray):
    return 1.0/(1.0+np.exp(-x))


def softmax(x: np.array):
    exped = np.exp(x)
    return exped/np.sum(exped, axis=1, keepdims=True)


class MLPClassifier:
    def __init__(self, learning_rate: float, epoch_number: int):
        # Please do not make any changes in this constructor

        # Weights are initialized temporarily for size clarification
        # These values are overwritten via the initialize_parameter function call belove.
        self.A = np.zeros((2, 5))  # of size (2, 5)
        self.A_bias = np.zeros((1, 5))  # of size (1, 5), a row vector (1x5 matrix)
        
        self.B = np.zeros((5, 3))  # of size (4, 3)
        self.B_bias = np.zeros((1, 3))  # of size (1, 3), a row vector (1x3) matrix

        self.C = np.zeros((3, 2))  # of size (3, 2)
        self.C_bias = np.zeros((1, 2))  # of size (1, 2), a row vector (1x2) matrix


        self.D = np.zeros((2, 4))  # of size (2, 4)
        self.D_bias = np.zeros((1, 4))  # of size (1, 4), a row vector (1x4) matrix

        self.learning_rate = learning_rate
        self.epoch_number = epoch_number

        self.initialize_parameters()

    def initialize_parameters(self):
        # Please do not make any changes in this function

        # In order to be able to obtain the same results during training
        # start with fixed initial weights
        with open("datasets/part1_classification_initial_weights.data", "rb") as input_file:
            self.A, self.A_bias, self.B, self.B_bias, self.C, self.C_bias, self.D, self.D_bias = pickle.load(input_file)


    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: two-dimensional data numpy array fed to the network, size: (# of data instances, # of features)
        :return: tuple of four numpy arrays as the outputs of the hidden and output layers, the forward propagation result for the given data, sizes: (# of data instances, 5) (first hidden layer output), (# of data instances, 3) (second hidden layer output), (# of data instances, 2) (third hidden layer output), (# of data instances, 4) (output layer output)

        Please implement the forward propagation procedure for given data instances here
        """
        first_hidden_layer_output, second_hidden_layer_output, third_hidden_layer_output, output_layer_output = None, None, None, None
        ...
        first_hidden_layer_output = sigmoid(np.matmul(x, self.A) + self.A_bias)
        second_hidden_layer_output = sigmoid(np.matmul(first_hidden_layer_output, self.B) + self.B_bias)
        third_hidden_layer_output = sigmoid(np.matmul(second_hidden_layer_output, self.C) + self.C_bias)
        output_layer_output = softmax(np.matmul(third_hidden_layer_output, self.D) + self.D_bias)
        ...
        return first_hidden_layer_output, second_hidden_layer_output, third_hidden_layer_output, output_layer_output

    def train(self, data_instances: np.ndarray, labels: np.ndarray):
        # Please only implement the part asked in this function
        # And please do not make any other changes on the already provided code pieces
        for iteration_count in range(1, self.epoch_number+1):

            for data_index in range(len(data_instances)):
                x = data_instances[data_index].reshape(1, -1) # convert into a row vector, (1x2) matrix
                label = labels[data_index] # 1D array with 4 elements: [l0, l1, l2, l3]
                first_hidden_layer_output, second_hidden_layer_output, third_hidden_layer_output, output_layer_output = self.forward(x)

                A_update = np.zeros_like(self.A)
                A_bias_update = np.zeros_like(self.A_bias)

                B_update = np.zeros_like(self.B)
                B_bias_update = np.zeros_like(self.B_bias)

                C_update = np.zeros_like(self.C)
                C_bias_update = np.zeros_like(self.C_bias)
                
                D_update = np.zeros_like(self.D)
                D_bias_update = np.zeros_like(self.D_bias)
                """
                    Please calculate the weight update rules for A, A_bias, B, B_bias, C, C_bias, D, and D_bias matrices here
                    using the "x", "first_hidden_layer_output", "second_hidden_layer_output", "third_hidden_layer_output", "output_layer_output" and "label" variables defined above
                    
                    The amount of weight changes should be stored in "A_update", "A_bias_update", "B_update", "B_bias_update", "C_update", "C_bias_update", "D_update", and "D_bias_update" variables.
                """
                ...
                delta_O =  output_layer_output - label.reshape(1, -1) #  1x4 [dO1 dO2 dO3 dO4]
                D_bias_update = delta_O
                D_update =  np.matmul(third_hidden_layer_output.T, delta_O)  # 2x4 []
                
                delta_T = np.matmul(delta_O, self.D.T) * (third_hidden_layer_output * (1 - third_hidden_layer_output)) # 1x2 [dT1 dT2]
                C_bias_update = delta_T
                C_update = np.matmul(second_hidden_layer_output.T, delta_T)
                
                delta_S = np.matmul(delta_T, self.C.T) * (second_hidden_layer_output * (1 - second_hidden_layer_output))
                B_bias_update = delta_S
                B_update = np.matmul(first_hidden_layer_output.T, delta_S)
                
                delta_F = np.matmul(delta_S, self.B.T) * (first_hidden_layer_output * (1 - first_hidden_layer_output))
                A_bias_update = delta_F
                A_update = np.matmul(x.T, delta_F)
                
                ...

                # After finding update values we are performing the weight updates
                self.D = self.D - self.learning_rate*D_update
                self.D_bias = self.D_bias - self.learning_rate*D_bias_update

                self.C = self.C - self.learning_rate*C_update
                self.C_bias = self.C_bias - self.learning_rate*C_bias_update

                self.B = self.B - self.learning_rate*B_update
                self.B_bias = self.B_bias - self.learning_rate*B_bias_update

                self.A = self.A - self.learning_rate*A_update
                self.A_bias = self.A_bias - self.learning_rate*A_bias_update

            # After each epoch on the dataset, calculate the Mean Cross Entropy loss and accuracy with the dataset.
            correct, wrong = 0, 0
            total_loss_value = 0.0
            for data_index in range(len(data_instances)):
                x = data_instances[data_index].reshape(1, -1)  # convert into a row vector, 1x4 matrix
                label = labels[data_index]
                _, _,_, output = self.forward(x)
                loss = 0
                max_index = None
                max_value = -float('inf')
                for i in range(len(output[0])):
                    loss += label[i]*np.log(output[0][i])
                    if output[0][i] > max_value:
                        max_value = output[0][i]
                        max_index = i
                total_loss_value += -loss
                if label[max_index] == 1:
                    correct += 1
                else:
                    wrong += 1
            mean_ce = total_loss_value/len(data_instances)
            accuracy = correct/(correct+wrong)*100
            print(f"Epoch Number: {iteration_count} - Training Mean CE: {mean_ce:.3f} - Training Accuracy: {accuracy:.2f}")

with open("datasets/part1_classification_dataset.data", "rb") as input_file:
    X, L = pickle.load(input_file)
mlp = MLPClassifier(learning_rate=0.01, epoch_number=50)
mlp.train(X, L)
