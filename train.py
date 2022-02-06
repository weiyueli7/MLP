################################################################################
# CSE 151b: Programming Assignment 2
# Code snippet by Eric Yang Yu, Ajit Kumar, Savyasachi
# Winter 2022
################################################################################
from data import write_to_file
from neuralnet import *


def train(x_train, y_train, x_val, y_val, config, experiment=None):
    """
    Train your model here using batch stochastic gradient descent and early stopping. Use config to set parameters
    for training like learning rate, momentum, etc.

    Args:
        x_train: The train patterns
        y_train: The train labels
        x_val: The validation set patterns
        y_val: The validation set labels
        config: The configs as specified in config.yaml
        experiment: An optional dict parameter for you to specify which experiment you want to run in train.

    Returns:
        5 things:
            training and validation loss and accuracies - 1D arrays of loss and accuracy values per epoch.
            best model - an instance of class NeuralNetwork. You can use copy.deepcopy(model) to save the best model.
    """
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
    best_model = None

    model = NeuralNetwork(config=config)

    # return train_acc, val_acc, train_loss, val_loss, best_model
    raise NotImplementedError('Train not implemented')


def test(model, x_test, y_test):
    """
    Does a forward pass on the model and returns loss and accuracy on the test set.

    Args:
        model: The trained model to run a forward pass on.
        x_test: The test patterns.
        y_test: The test labels.

    Returns:
        Loss, Test accuracy
    """
    # return loss, accuracy
    raise NotImplementedError('Test not implemented')


def train_mlp(x_train, y_train, x_val, y_val, x_test, y_test, config):
    """
    This function trains a single multi-layer perceptron and plots its performances.

    NOTE: For this function and any of the experiments, feel free to come up with your own ways of saving data
            (i.e. plots, performances, etc.). A recommendation is to save this function's data and each experiment's
            data into separate folders, but this part is up to you.
    """
    # train the model
    train_acc, valid_acc, train_loss, valid_loss, best_model = \
        train(x_train, y_train, x_val, y_val, config)

    test_loss, test_acc = test(best_model, x_test, y_test)

    print("Config: %r" % config)
    print("Test Loss", test_loss)
    print("Test Accuracy", test_acc)

    # DO NOT modify the code below.
    data = {'train_loss': train_loss, 'val_loss': valid_loss, 'train_acc': train_acc, 'val_acc': valid_acc,
            'best_model': best_model, 'test_loss': test_loss, 'test_acc': test_acc}

    write_to_file('./results.pkl', data)


def activation_experiment(x_train, y_train, x_val, y_val, x_test, y_test, config):
    """
    This function tests all the different activation functions available and then plots their performances.
    """
    raise NotImplementedError('Activation Experiment not implemented')


def topology_experiment(x_train, y_train, x_val, y_val, x_test, y_test, config):
    """
    This function tests performance of various network topologies, i.e. making
    the graph narrower and wider by halving and doubling the number of hidden units.

    Then, we change number of hidden layers to 2 of equal size instead of 1, and keep
    number of parameters roughly equal to the number of parameters of the best performing
    model previously.
    """
    raise NotImplementedError('Topology Experiment not implemented')


def regularization_experiment(x_train, y_train, x_val, y_val, x_test, y_test, config):
    """
    This function tests the neural network with regularization.
    """
    raise NotImplementedError('Regularization Experiment not implemented')


def check_gradients(x_train, y_train, config):
    """
    Check the network gradients computed by back propagation by comparing with the gradients computed using numerical
    approximation.
    """
    raise NotImplementedError('Check Gradients Experiment not implemented')
