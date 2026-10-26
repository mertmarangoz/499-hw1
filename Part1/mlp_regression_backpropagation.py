import pickle
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x: np.ndarray):
    return 1.0/(1.0+np.exp(-x))

class MLPRegressor:
    def __init__(self, learning_rate: float, epoch_number: int):
        # Please do not make any changes in this constructor

        # Weights are initialized temporarily for size clarification
        # These values are overwritten via the initialize_parameter function call belove.
        self.A = np.zeros((3, 4))  # of size (3, 4)
        self.A_bias = np.zeros((1, 4))  # of size (1, 4), a row vector (1x4 matrix)
        
        self.B = np.zeros((4, 4))  # of size (4, 4)
        self.B_bias = np.zeros((1, 4))  # of size (1, 4), a row vector (1x4) matrix

        self.C = np.zeros((4, 3))  # of size (4, 3)
        self.C_bias = np.zeros((1, 3))  # of size (1, 3), a row vector (1x3) matrix

        self.D = np.zeros((3, 1))  # of size (3, 1)
        self.D_bias = np.zeros((1, 1))  # of size (1, 1), a scalar (1x1) matrix

        self.learning_rate = learning_rate
        self.epoch_number = epoch_number

        self.initialize_parameters()

    def initialize_parameters(self):
        # Please do not make any changes in this function

        # In order to be able to obtain the same results during training
        # start with fixed initial weights
        with open("datasets/part1_regression_initial_weights.data", "rb") as input_file:
            self.A, self.A_bias, self.B, self.B_bias, self.C, self.C_bias, self.D, self.D_bias = pickle.load(input_file)


    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: two-dimensional data numpy array fed to the network, size: (# of data instances, # of features)
        :return: tuple of four numpy arrays as the outputs of the hidden and output layers, the forward propagation result for the given data, sizes: (# of data instances, 4) (first hidden layer output), (# of data instances, 4) (second hidden layer output), (# of data instances, 3) (third hidden layer output), (# of data instances, 1) (output layer output)

        Please implement the forward propagation procedure for given data instances here
        """
        first_hidden_layer_output, second_hidden_layer_output, third_hidden_layer_output, output_layer_output = None, None, None, None
        ...
        first_hidden_layer_output = sigmoid(np.matmul(x, self.A) + self.A_bias)
        second_hidden_layer_output = sigmoid(np.matmul(first_hidden_layer_output, self.B) + self.B_bias)
        third_hidden_layer_output = sigmoid(np.matmul(second_hidden_layer_output, self.C) + self.C_bias)
        output_layer_output = np.matmul(third_hidden_layer_output, self.D) + self.D_bias
        ...
                
        return first_hidden_layer_output, second_hidden_layer_output, third_hidden_layer_output, output_layer_output

    def train(self, data_instances: np.ndarray, labels: np.ndarray):
        # Please only implement the part asked in this function
        # And please do not make any other changes on the already provided code pieces
        for iteration_count in range(1, self.epoch_number+1):

            for data_index in range(len(data_instances)):
                x = data_instances[data_index].reshape(1, -1)  # convert into a row vector, (1x2) matrix  --> 1x2 değil 1x3
                label = labels[data_index] # 1D array
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
                delta_O = -2 * (label - output_layer_output)                # (1,1)
                D_bias_update = delta_O                                     # (1,1)
                D_update = np.matmul(third_hidden_layer_output.T, delta_O)  # (3,1)
                
                delta_T = np.matmul(delta_O, self.D.T) * (third_hidden_layer_output * (1 - third_hidden_layer_output))  # (1,3)
                C_bias_update = delta_T
                C_update = np.matmul(second_hidden_layer_output.T, delta_T)  # (4, 3)
                                    
                delta_S = np.matmul(delta_T, self.C.T) * (second_hidden_layer_output * (1 - second_hidden_layer_output)) # (1,4)
                B_bias_update = delta_S
                B_update = np.matmul(first_hidden_layer_output.T, delta_S)  # (4, 4)

                delta_F = np.matmul(delta_S, self.B.T) * (first_hidden_layer_output * (1 - first_hidden_layer_output)) # (1,4)
                A_bias_update = delta_F
                A_update = np.matmul(x.T, delta_F)  # (3, 4)

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

            total_loss_value = 0.0
            for data_index in range(len(data_instances)):
                x = data_instances[data_index].reshape(1, -1) # convert into a row vector, 1x4 matrix
                label = labels[data_index]
                _,_,_ ,  output = self.forward(x)
                total_loss_value += (label[0]-output[0][0])**2

            mean_mse = total_loss_value/len(data_instances)
            print(f"Epoch Number: {iteration_count} - Training Mean SE: {mean_mse:.5f}")


X, L = pickle.load(open("datasets/part1_regression_dataset.data", "rb"))
mlp = MLPRegressor(learning_rate=0.01, epoch_number=50)
mlp.train(X, L)