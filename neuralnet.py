################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Eric Yang Yu, Ajit Kumar, Savyasachi
# Winter 2022
################################################################################

import numpy as np
import math


class Activation:
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta=1.0)
    """

    def __init__(self, activation_type="sigmoid"):
        """
        Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU"]:
            raise NotImplementedError("%s is not implemented." % (activation_type))

        # Type of non-linear activation.
        self.activation_type = activation_type
        # Placeholder for input. This will be used for computing gradients.
        self.x = None

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        """
        self.x = a
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

    def backward(self, delta):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        return grad * delta

    def sigmoid(self, x):
        """
        Implement the sigmoid activation here.
        """
        return 1 / (1 + np.exp(-1 * x))

    def tanh(self, x):
        """
        Implement tanh here.
        """
        return np.tanh(x)

    def ReLU(self, x):
        """
        Implement ReLU here.
        """
        return np.maximum(0, x)

    def grad_sigmoid(self):
        """
        Compute the gradient for sigmoid here.
        """
        return self.sigmoid(self.x) * (1 - self.sigmoid(self.x))

    def grad_tanh(self):
        """
        Compute the gradient for tanh here.
        """
        return 1 - (self.tanh(self.x)) ** 2

    def grad_ReLU(self):
        """
        Compute the gradient for ReLU here.
        """
        return (self.x > 0) * 1


class Layer:
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        >>> fully_connected_layer = Layer(1024, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta)
    """

    def __init__(self, in_units, out_units, config):
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        self.config = config
        self.w = math.sqrt(2 / in_units) * np.random.randn(in_units,
                                                           out_units)  # You can experiment with initialization.
        self.b = np.zeros((1, out_units))  # Create a placeholder for Bias
        self.x = None  # Save the input to forward in this
        self.a = None  # Save the output of forward pass in this (without activation)

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this
        self.weight_decay = 0
        self.m_w = 0
        self.m_b = 0

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        Compute the forward pass through the layer here.
        Do not apply activation here.
        Return self.a
        """
        self.x = x
        self.a = self.x @ self.w + self.b
        return self.a
    
    def L1(self, w):
        w_1 = np.where(w < 0, -1, w)
        w_2 = np.where(w_1 > 0, 1, w_1)
        return w_2
    
    def L2(self, w):
        return 2 * w

    def backward(self, delta, gamma, regularization = False, L = 'L1'):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        if regularization:
            if L == 'L1':
                self.weight_decay = self.L1(self.w)
            else:
                self.weight_decay = self.L2(self.w)
        prev_delta = delta @ self.w.T
        self.d_x = delta @ self.w.T
        self.d_w = -(self.x.T @ delta)
        self.d_b = -np.sum(delta, axis = 0)
        self.m_w = gamma * self.m_w + (1 - gamma) * self.d_w
        self.m_b = gamma * self.m_b + (1 - gamma) * self.d_b
        alpha = self.config['learning_rate']
        if (self.config['momentum']):
            self.w = self.w - alpha * self.m_w / self.config['batch_size'] \
            - alpha * self.config['L2_penalty'] * self.weight_decay
            self.b = self.b - alpha * self.m_b
        else:
            self.w = self.w - alpha * self.d_w / self.config['batch_size']
            self.b = self.b - alpha * self.d_b
        return prev_delta


class NeuralNetwork:
    """
    Create a Neural Network specified by the input configuration.

    Example:
        >>> net = NeuralNetwork(config)
        >>> output = net(input)
        >>> net.backward()
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []  # Store all layers in this list.
        self.x = None  # Save the input to forward in this
        self.y = None  # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable
        self.deltas = []
        self.config = config

        # Add layers specified by layer_specs.
        for i in range(len(self.config['layer_specs']) - 1):
            self.layers.append(Layer(self.config['layer_specs'][i], self.config['layer_specs'][i + 1], self.config))
            if i < len(self.config['layer_specs']) - 2:
                self.layers.append(Activation(self.config['activation']))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """
        self.x = x
        self.targets = targets
        for layer in self.layers:
            self.x = layer(self.x)
        self.y = self.softmax(self.x)
        if targets is None:
            return self.y, None
        return self.y, self.loss(self.y, self.targets)

    def backward(self, regularization = False, L = 'L1'):
        """
        Implement backpropagation here.
        Call backward methods of individual layer's.
        """
        delta = self.targets - self.y
        self.deltas = [delta]
        for i in range(len(self.layers) - 1, -1, -1):
            if isinstance(self.layers[i], Layer):
                if regularization:
                    delta = self.layers[i].backward(delta, self.config['momentum_gamma'], regularization, L)
                else:
                    delta = self.layers[i].backward(delta, self.config['momentum_gamma'])
            else:
                delta = self.layers[i].backward(delta)
            self.deltas.append(delta)
        return delta

    def softmax(self, x):
        """
        Implement the softmax function here.
        Remember to take care of the overflow condition.
        """
        return np.exp(x - np.max(x, axis = 1, keepdims = True)) \
               / np.sum(np.exp(x - np.max(x, axis = 1, keepdims = True)), axis = 1).reshape(-1,1) 

    def loss(self, logits, targets):
        """
        compute the categorical cross-entropy loss and return it.
        """
        return -np.sum(targets * np.log(logits + 1e-20)) / len(targets)
