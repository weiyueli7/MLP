
# Optimization and Evaluation of Multi-layer Neural Networks: Exploring Regularization, Learning Rates, and Topologies

A report is aviliable [here](report.pdf).

## Description

In this work, we implement a multi-layer neural network equipped with forward and backward propagation, various regularization techniques, and momentum-based optimization. Our objective was to classify Japanese Hiragana handwritten characters from the [KMNIST dataset](https://github.com/rois-codh/kmnist), employing softmax as the output layer. One-fold cross-validation was utilized to evaluate the model, coupled with the integration of regularization techniques. Our most efficient model leveraged ReLU activations and achieved an accuracy of **0.8688**. Subsequent architecture adjustments, including layer count and hidden unit modifications, yielded a test set accuracy of **0.8626**.

## Getting Started

### Dependencies

* Python3
* `Numpy`
* `Matplotlib`
* `tqdm`


### Files
* `config.yaml` is the file where you could set-up your desired configuration including the number of layers and number of hidden neurons in each layer, the type of non-linear activation function, the learning rate, the batch size, the number of epochs, if the model will early-stop, the patience for early-stop, the L2 penalty, if the model will use momentum, and the momentum $\gamma$. Default values specified below.
* `data.py` is the file where we have pre processed the data. We have implemented some helper methods for one-hot encoding, normalizing, shuffling the dataset, and so on.
* `main.py` is the ONLY file you need to run to execute our model.
* `nueralnet.py` is the file where were have implemented the structure of the nueral network including the classes Layer, Activation, and NueralNetwork where we have integrated three classes together.
* `train.py` is the file that all training happened. It will also plot our data of training/validation accuracy/loss.

### Executing program

* Go to the correct directory.
* In your terminal, run `python3 main.py` for default:
    * layer_specs: [784, 128, 10]
    * activation: "tanh"
    * learning_rate: 0.005
    * batch_size: 128
    * epochs: 100
    * early_stop: False
    * early_stop_epoch: 5
    * L2_penalty: 0.0001
    * momentum: True
    * momentum_gamma: 0.9
* You may adjust the default values by editing the `config.yaml`: 

If run properly, you should be able to see the progress bar of training, as well as their associated results.


## Help

If the program takes too slow to execute. Please only run one part at a time and comment out the rest of the parts in `main.py`.


## Authors

Contributors names and contact info (alphabetical order):

* Kong, Linghang
    * l3kong@ucsd.edu
* Li, Weiyue
    * wel019@ucsd.edu
* Li, Yi
    * yil115@ucsd.edu 
