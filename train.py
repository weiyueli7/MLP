################################################################################
# CSE 151b: Programming Assignment 2
# Code snippet by Eric Yang Yu, Ajit Kumar, Savyasachi
# Winter 2022
################################################################################
from data import *
from neuralnet import *
import matplotlib.pyplot as plt

def train(x_train, y_train, x_val, y_val, config, regularization = False, L = 'L1'):
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
    best_model_loss = float("inf")
    
    model = NeuralNetwork(config=config)
    
    patience = 0
    
    count_epoch = 0
    
    for epoch in tqdm(range(config['epochs'])):
        
        count_epoch += 1
        
        x_train, y_train = shuffle((x_train, y_train))
        
        x_val, y_val = shuffle((x_val, y_val))
        
        for batch in generate_minibatches((x_train, y_train), config['batch_size']):
            
            pred, loss = model.forward(batch[0], batch[1])
            
            if regularization:
            
                model.backward(regularization, L)
                
            else:
                
                model.backward()
    
        train_pred, train_losses = model.forward(x_train, y_train)
        
        train_accuracy = accuracy(train_pred, y_train)
        
        train_loss.append(train_losses)
        
        train_acc.append(train_accuracy)
        
        val_pred, val_losses = model.forward(x_val, y_val)
        
        val_accuracy = accuracy(val_pred, y_val)

        val_loss.append(val_losses)
        
        val_acc.append(val_accuracy)
        
        if val_losses < best_model_loss:
            
            best_model_loss = val_losses
            
            best_model = copy.deepcopy(model)
            
        if config['early_stop']:
        
            if val_losses > best_model_loss:
            
                patience += 1
            
                if patience == config['early_stop_epoch']:
                
                    break
            
            else:
                
                patience = 0
        
    return train_acc, val_acc, train_loss, val_loss, best_model                 
    


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
    test_pred, test_loss = model.forward(x_test, y_test)
    
    acc = np.mean(np.argmax(test_pred, axis = 1) == onehot_decode(y_test))
    
    return test_loss, acc

def accuracy(pred, target):
    return np.mean(np.argmax(pred, axis = 1) == onehot_decode(target))


def train_mlp(x_train, y_train, x_val, y_val, x_test, y_test, config, regularization = False, L = 'L1'):
    """
    This function trains a single multi-layer perceptron and plots its performances.

    NOTE: For this function and any of the experiments, feel free to come up with your own ways of saving data
            (i.e. plots, performances, etc.). A recommendation is to save this function's data and each experiment's
            data into separate folders, but this part is up to you.
    """
    # train the model
    train_acc, valid_acc, train_loss, valid_loss, best_model = \
        train(x_train, y_train, x_val, y_val, config, regularization, L)

    test_loss, test_acc = test(best_model, x_test, y_test)

    print("Config: %r" % config)
    print("Train Loss: ", train_loss[-1])
    print("Validation Loss: ", valid_loss[-1])
    print("Validation Accuracy: ", valid_acc[-1])
    print("Test Loss: ", test_loss)
    print("Test Accuracy: ", test_acc)

    # DO NOT modify the code below.
    data = {'train_loss': train_loss, 'val_loss': valid_loss, 'train_acc': train_acc, 'val_acc': valid_acc,
            'best_model': best_model, 'test_loss': test_loss, 'test_acc': test_acc}

    #write_to_file('./results.pkl', data)

    data_plt, label_plt = data['train_acc'], "Training Accuracy"
    data2_plt, label2_plt = data['val_acc'], "Validation Accuracy"
    data3_plt, label3_plt = data['train_loss'], "Training Loss"
    data4_plt, label4_plt = data['val_loss'], "Validation Loss"
    xlabel_plt, ylabel_plt = "Epochs", "Accuracy"
    xlabel2_plt, ylabel2_plt = "Epochs", "Loss"

    # 1st plot
    fig, ax = plt.subplots()
    ax.plot(data_plt, label = label_plt)
    ax.plot(data2_plt, label = label2_plt)
    plt.xlabel(xlabel_plt)
    plt.ylabel(ylabel_plt)
    plt.title(str(config['learning_rate']).replace('.', '') + '_' + \
                config['activation'] + '_' + str(config['early_stop'])[0].lower() + '_acc')
    legend = ax.legend(loc='lower right')
    plt.savefig( str(config['learning_rate']).replace('.', '') + '_' + \
        config['activation'] + '_' + str(config['early_stop'])[0].lower() + "_" + \
        str(config['layer_specs'])[1:-1].replace(", ", "_") + '_acc.png')
    
    # 2nd plot
    fig, ax = plt.subplots()
    ax.plot(data3_plt, label = label3_plt)
    ax.plot(data4_plt, label = label4_plt)
    plt.xlabel(xlabel2_plt)
    plt.ylabel(ylabel2_plt)
    plt.title(str(config['learning_rate']).replace('.', '') + '_' + \
                config['activation'] + '_' + str(config['early_stop'])[0].lower() + '_loss')
    legend = ax.legend(loc='upper right')
    plt.savefig( str(config['learning_rate']).replace('.', '') + '_' + \
            config['activation'] + '_' + str(config['early_stop'])[0].lower() + "_" + \
            str(config['layer_specs'])[1:-1].replace(", ", "_") + '_loss.png')

    #write_to_file('./results.pkl', data)
    return data


def activation_experiment(x_train, y_train, x_val, y_val, x_test, y_test, config):
    """
    This function tests all the different activation functions available and then plots their performances.
    """
    activations = ['sigmoid', 'tanh', 'ReLU']
    config_copy = config
    for activation in activations:
        config_copy['activation'] = activation
        train_mlp(x_train, y_train, x_val, y_val, x_test, y_test, config_copy)

def topology_experiment(x_train, y_train, x_val, y_val, x_test, y_test, config):
    """
    This function tests performance of various network topologies, i.e. making
    the graph narrower and wider by halving and doubling the number of hidden units.

    Then, we change number of hidden layers to 2 of equal size instead of 1, and keep
    number of parameters roughly equal to the number of parameters of the best performing
    model previously.
    """
    units = [64, 128, 256]
    config_copy = config
    for unit in units:
        config_copy['layer_specs'][1] = unit
        train_mlp(x_train, y_train, x_val, y_val, x_test, y_test, config_copy)
    double_hidden = [784, 203, 203, 10]
    config_copy['layer_specs'] = double_hidden
    train_mlp(x_train, y_train, x_val, y_val, x_test, y_test, config_copy)


def regularization_experiment(x_train, y_train, x_val, y_val, x_test, y_test, config, L = 'L1'):
    """
    This function tests the neural network with regularization.
    """
    config_copy = config
    config_copy['epochs'] = 110
    train_mlp(x_train, y_train, x_val, y_val, x_test, y_test, config_copy, True, L)


def check_gradients(x_train, y_train, adjust, config, random_row):
    """
    Check the network gradients computed by back propagation by comparing with the gradients computed using numerical
    approximation.
    """
    model = NeuralNetwork(load_config('config.yaml'))
    layer = model.layers[2]
    save = copy.deepcopy(layer.b[0][random_row])
    
    layer.b[0][random_row] += adjust
    loss_one = model(x_train, y_train)[1]
    print(loss_one)
    
    layer.b[0][random_row] = save
    layer.b[0][random_row] -= adjust
    loss_two = model(x_train, y_train)[1]
    print(loss_two)
    
    numeric = (loss_one - loss_two) / (2 * adjust)
    print("numerical approximation: " + str(numeric))
    
    layer.b[0][random_row] = save
    model(x_train, y_train)
    model.backward()
    backward_result = layer.d_b[random_row]
    print("backpropagation gradient: " + str(backward_result))
    
    diff = abs(backward_result - numeric)
    print("absolute error: " + str(diff))
    return diff
