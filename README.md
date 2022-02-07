---
title: Image Classification in Practice: Implementing Multi-layer Neural Network Using Numpy
author: Weiyue Li
date: 02/06/2022
---

# Image Classification in Practice: Implementing Multi-layer Neural Network Using Numpy

Source code for CSE 151B Winter 2022 PA2.

## Description

We continued with the Japanese Hiragana handwriting dataset from Programming Assignment 1. We preprocessed the data by converting all images from 2 dimensional arrays of size (28,28) into one-dimensional arrays of size 784. We applied z-score normalization to the samples and shuffled them before the training process. The data was split into 80% training data and 20% validation data. For saving time purposes, we have only used one-fold cross validation according to the write-up. After fine-tuning, changing number of layers and experimenting with different network topologies, we were able to obtain 0.8626 accuracy on our best model on the validation set.

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

## Acknowledgments

We appriciate the help from course Piazza and TA/Tutor office hours, as well as the knowledge gained from Professor [Garrison W. Cottrell](https://cseweb.ucsd.edu/~gary/)'s lectures.