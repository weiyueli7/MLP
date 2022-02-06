################################################################################
# CSE 151b: Programming Assignment 2
# Code snippet by Eric Yang Yu, Ajit Kumar, Savyasachi
# Winter 2022
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################
import argparse
import matplotlib.pyplot as plt
from data import *
from train import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mlp', dest='train_mlp', action='store_true', default=False,
                        help='Train a single multi-layer perceptron using configs provided in config.yaml')
    parser.add_argument('--check_gradients', dest='check_gradients', action='store_true', default=False,
                        help='Check the network gradients computed by comparing the gradient computed using'
                             'numerical approximation with that computed as in back propagation.')
    parser.add_argument('--regularization', dest='regularization', action='store_true', default=False,
                        help='Experiment with weight decay added to the update rule during training.')
    parser.add_argument('--activation', dest='activation', action='store_true', default=False,
                        help='Experiment with different activation functions for hidden units.')
    parser.add_argument('--topology', dest='topology', action='store_true', default=False,
                        help='Experiment with different network topologies.')
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    # Load the configuration.
    config = load_config("./config.yaml")

    # Load the data
    train_data, (x_test, y_test) = load_data(), load_data(train=False)

    # Create validation set out of training data.
    x_train, y_train, x_val, y_val = None, None, None, None

    # Any pre-processing on the datasets goes here.
    x_train = z_score_normalize(load_data(True)[0])[0]
    y_train = one_hot_encoding(load_data(True)[1], num_classes=10)
    x_test = z_score_normalize(load_data(False)[0])[0]
    y_test = one_hot_encoding(load_data(False)[1], num_classes=10)
    x_train, y_train = shuffle((x_train, y_train))
    x_test, y_test = shuffle((x_test, y_test))
    ind = int(0.8 * len(x_train))

    # Run the writeup experiments here
    if args.train_mlp:
        train_mlp(x_train, y_train, x_val, y_val, x_test, y_test, config)
    if args.check_gradients:
        check_gradients(x_train, y_train, config)
    if args.regularization:
        regularization_experiment(x_train, y_train, x_val, y_val, x_test, y_test, config)
    if args.activation:
        activation_experiment(x_train, y_train, x_val, y_val, x_test, y_test, config)
    if args.topology:
        topology_experiment(x_train, y_train, x_val, y_val, x_test, y_test, config)


    print("Find Best Model\n_____________________________________________________________________")

    candidate = train_mlp(x_train[0:ind], y_train[0:ind], x_train[ind:], \
                      y_train[ind:], x_test, y_test, load_config('config.yaml'))

    # print("Activation Experiments\n______________________________________________________________")
    # data_act = activation_experiment(x_train[0:ind], y_train[0:ind], x_train[ind:], \
    #                   y_train[ind:], x_test, y_test, load_config('config.yaml'))
    
    # # plot the graphs
    # # activation_functions = {0: 'sigmoid', 1: 'tanh', 2: 'ReLU'}
    # # for i in range(len(data_act)):
    # #     data_plt, label_plt = data_act[i]['train_acc'], "Training Accuracy"
    # #     data2_plt, label2_plt = data_act[i]['val_acc'], "Validation Accuracy"
    # #     data3_plt, label3_plt = data_act[i]['train_loss'], "Training Loss"
    # #     data4_plt, label4_plt = data_act[i]['val_loss'], "Validation Loss"
    # #     xlabel_plt, ylabel_plt = "Epochs", "Accuracy"
    # #     xlabel2_plt, ylabel2_plt = "Epochs", "Loss"
    
    # #     # 1st plot
    # #     fig, ax = plt.subplots()
    # #     ax.plot(data_plt, label = label_plt)
    # #     ax.plot(data2_plt, label = label2_plt)
    # #     plt.xlabel(xlabel_plt)
    # #     plt.ylabel(ylabel_plt)
    # #     plt.title('Training vs Validation Accuracy Activation_' + str(i + 1))
    # #     legend = ax.legend(loc='lower right')
    # #     plt.savefig( str(config['learning_rate']).replace('.', '') + '_' + \
    # #         activation_functions[i] + '_' + str(config['early_stop'])[0].lower() + "_" + \
    # #             str(config['layer_specs'])[1:-1].replace(", ", "_") + '_acc.png')
    
    # #     # 2nd plot
    # #     fig, ax = plt.subplots()
    # #     ax.plot(data3_plt, label = label3_plt)
    # #     ax.plot(data4_plt, label = label4_plt)
    # #     plt.xlabel(xlabel2_plt)
    # #     plt.ylabel(ylabel2_plt)
    # #     plt.title('Training vs Validation Loss Activation_' + str(i + 1))
    # #     legend = ax.legend(loc='lower right')
    # #     plt.savefig( str(config['learning_rate']).replace('.', '') + '_' + \
    # #         activation_functions[i] + '_' + str(config['early_stop'])[0].lower() + "_" + \
    # #         str(config['layer_specs'])[1:-1].replace(", ", "_") +'_loss.png')
    
    # print("Topology Experiments\n________________________________________________________________")
    # data_top = topology_experiment(x_train[0:ind], y_train[0:ind], x_train[ind:], 
    #                   y_train[ind:], x_test, y_test, load_config('config.yaml'))

    # tops = {0: 'half', 1: 'one', 2: 'dub', 3:'dub_layer'}
    # # for i in range(len(data_top)):
    # #     data_plt, label_plt = data_top[i]['train_acc'], "Training Accuracy"
    # #     data2_plt, label2_plt = data_top[i]['val_acc'], "Validation Accuracy"
    # #     data3_plt, label3_plt = data_top[i]['train_loss'], "Training Loss"
    # #     data4_plt, label4_plt = data_top[i]['val_loss'], "Validation Loss"
    # #     xlabel_plt, ylabel_plt = "Epochs", "Accuracy"
    # #     xlabel2_plt, ylabel2_plt = "Epochs", "Loss"
    
    # #     # 1st plot
    # #     fig, ax = plt.subplots()
    # #     ax.plot(data_plt, label = label_plt)
    # #     ax.plot(data2_plt, label = label2_plt)
    # #     plt.xlabel(xlabel_plt)
    # #     plt.ylabel(ylabel_plt)
    # #     plt.title('Training vs Validation Accuracy Activation_' + str(i + 1))
    # #     legend = ax.legend(loc='lower right')
    # #     plt.savefig( str(config['learning_rate']).replace('.', '') + '_' + \
    # #         tops[i] + '_' + str(config['early_stop'])[0].lower() + "_" +\
    # #          str(config['layer_specs'])[1:-1].replace(", ", "_") + '_acc.png')
    
    # #     # 2nd plot
    # #     fig, ax = plt.subplots()
    # #     ax.plot(data3_plt, label = label3_plt)
    # #     ax.plot(data4_plt, label = label4_plt)
    # #     plt.xlabel(xlabel2_plt)
    # #     plt.ylabel(ylabel2_plt)
    # #     plt.title('Training vs Validation Loss Activation_' + str(i + 1))
    # #     legend = ax.legend(loc='lower right')
    # #     plt.savefig( str(config['learning_rate']).replace('.', '') + '_' + \
    # #         tops[i] + '_' + str(config['early_stop'])[0].lower() + "_" + \
    # #         str(config['layer_specs'])[1:-1].replace(", ", "_") + '_loss.png')
