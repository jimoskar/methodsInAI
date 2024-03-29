import numpy as np
import numpy as np
import pickle
import os


class Layer:
    def __init__(self, size, output_dim):
        self.size = size
        self.W = np.random.randn(size, output_dim)
        self.b = np.random.randn(output_dim)
        self.inp = None # Input to layer.
        self.a = None # Activation of input to layer.
        self.delta = None # The delta used in back-prop.

class NeuralNetwork:
    """Implement/make changes to places in the code that contains #TODO."""

    def __init__(self, input_dim: int, hidden_layer: bool) -> None:
        """
        Initialize the feed-forward neural network with the given arguments.
        :param input_dim: Number of features in the dataset.
        :param hidden_layer: Whether or not to include a hidden layer.
        :return: None.
        """

        # --- PLEASE READ --
        # Use the parameters below to train your feed-forward neural network.

        # Number of hidden units if hidden_layer = True.
        self.hidden_units = 25

        # This parameter is called the step size, also known as the learning rate (lr).
        # See 18.6.1 in AIMA 3rd edition (page 719).
        # This is the value of α on Line 25 in Figure 18.24.
        self.lr = 1e-3

        # Line 6 in Figure 18.24 says "repeat".
        # This is the number of times we are going to repeat. This is often known as epochs.
        self.epochs = 400

        # We are going to store the data here.
        # Since you are only asked to implement training for the feed-forward neural network,
        # only self.x_train and self.y_train need to be used. You will need to use them to implement train().
        # The self.x_test and self.y_test is used by the unit tests. Do not change anything in it.
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None

        # TODO: Make necessary changes here. For example, assigning the arguments "input_dim" and "hidden_layer" to
        # variables and so forth.
        self.input_dim = input_dim

        if hidden_layer:
            self.layers = [Layer(self.input_dim, self.hidden_units), Layer(self.hidden_units, 1), Layer(1, 0)]
        else:
            self.layers = [Layer(self.input_dim, 1), Layer(1, 0)]

        

    def load_data(self, file_path: str = os.path.join(os.getcwd(), 'data_breast_cancer.p')) -> None:
        """
        Do not change anything in this method.

        Load data for training and testing the model.
        :param file_path: Path to the file 'data_breast_cancer.p' downloaded from Blackboard. If no arguments is given,
        the method assumes that the file is in the current working directory.

        The data have the following format.
                   (row, column)
        x: shape = (number of examples, number of features)
        y: shape = (number of examples)
        """
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            self.x_train, self.y_train = data['x_train'], data['y_train']
            self.x_test, self.y_test = data['x_test'], data['y_test']
    
    def sigma(self, x):
        return 1/(1 + np.exp(-x))
    
    def sigma_der(self, x):
        return self.sigma(x) * (1 - self.sigma(x))

    def train(self) -> None:
        """Run the backpropagation algorithm to train this neural network"""
        # TODO: Implement the back-propagation algorithm outlined in Figure 18.24 (page 734) in AIMA 3rd edition.
        # Only parts of the algorithm need to be implemented since we are only going for one hidden layer.

        # Line 6 in Figure 18.24 says "repeat".
        # We are going to repeat self.epochs times as written in the __init()__ method.

        # Line 27 in Figure 18.24 says "return network". Here you do not need to return anything as we are coding
        # the neural network as a class

        for i in range(self.epochs):
            counter = 0
            for j, x in enumerate(self.x_train):
                #print("COUNT:")
                #print(counter)
                counter += 1
                t = self.y_train[j]

                # Forward feeding
                self.layers[0].a = self.layers[0].inp = x
                for k in range(1, len(self.layers)):
                    #print(self.layers[k - 1].W.shape)
                    #print(self.layers[k - 1].a.shape)
                    #print(self.layers[k-1].a)
                    inp = self.layers[k - 1].W.T @ self.layers[k - 1].a + self.layers[k - 1].b
                    self.layers[k].inp = inp
                    self.layers[k].a = self.sigma(inp)

                # Back-propagation
                self.layers[-1].delta = self.sigma_der(self.layers[-1].inp) * (t - self.layers[-1].a)
                bias_delta = self.sigma_der(self.layers[-1].inp) * (t- self.layers[-1].a)

              
                for l in range(len(self.layers) - 2, -1, -1):
                    # Updating weights
                    #print(l)
                    #print(self.layers[l].W.shape)
                    #print(self.layers[l].a.shape)
                    #print(self.layers[l + 1].delta.shape)
                    delta_mat = np.tile(self.layers[l + 1].delta.T, (self.layers[l].size, 1))
                    a_mat = np.tile(self.layers[l].a, (self.layers[l].size, 1))
                    print(self.layers[l + 1].delta.shape)
                    print(delta_mat.shape)
                    print(a_mat.shape)

                    self.layers[l].W += self.lr * a_mat.T @ delta_mat 
             
                    # Updating bias
                    #print(self.layers[l].b.shape)
                    #print(self.layers[l].inp.shape)

                    self.layers[l].b -= self.lr * bias_delta
                    if l > 0:
                        print("in here")
                        print(self.layers[l].inp.shape)
                        print(self.layers[l].W.shape)
                        print("end")

                        self.layers[l].delta = self.sigma_der(self.layers[l].inp) * self.layers[l].W[:,0] * self.layers[l + 1].delta
                        bias_delta = bias_delta * np.ones(self.layers[l].size) * self.sigma_der(self.layers[l].inp)



    def predict(self, x: np.ndarray) -> float:
        """
        Given an example x we want to predict its class probability.
        For example, for the breast cancer dataset we want to get the probability for cancer given the example x.
        :param x: A single example (vector) with shape = (number of features)
        :return: A float specifying probability which is bounded [0, 1].
        """
        # TODO: Implement the forward pass.

        for l in self.layers[:-1]:
            print("b:")
            print(l.b.shape)
            print("W:")
            print(l.W.shape)
            x = self.sigma(l.W.T @ x + l.b)
        print("X:")
        print(x)
    
        return x

A = np.ones((25, 1)) * np.ones(25)
print(A.shape)
NN = NeuralNetwork(30, True)
NN.load_data()
NN.train()