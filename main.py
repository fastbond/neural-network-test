import numpy as np
from neuralnetwork import NeuralNetwork
from layers import *
from activations import *
from lossfuncs import *

import time

#import tensorflow as tf
#print(tf.__version__)
#print(tf.config.list_physical_devices('GPU'))
#from tensorflow.python.keras import layers

# This is being done because can't import it properly in PyCharm...
# Can access normally:
#   tensorflow.keras
# But PyCharm fails to recognize when importing(although code runs):
#   from tensorflow import keras
#mnist = tf.keras.datasets.mnist
# Moved out of np_utils
#to_categorical = tf.keras.utils.to_categorical

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.datasets import mnist
from keras.utils import np_utils


np.set_printoptions(suppress=True)


def simple_test():
    x_train = np.array([[0,0], [0,1], [1,0], [1,1]])
    y_train = np.array([[0], [1], [1], [0]])

    model = NeuralNetwork((x_train.shape[1:]))
    model.add_layer(FullyConnectedLayer(3))
    model.add_layer(ActivationLayer(tanh, tanh_d))
    model.add_layer(FullyConnectedLayer(1))
    model.add_layer(ActivationLayer(tanh, tanh_d))
    model.set_loss_function(mse, mse_d)

    model.train(x_train, y_train, 0.1, 1000, batch_size=1)

    pred = model.predict(x_train)
    print(pred)


def test():
    # X is of shape (samples, 28, 28) with each value being [0-255] (greyscale)
    # Y is of shape (samples,) with values [0-9]
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape to (samples, w*h), EXCEPT
    # Actually (samples, 1, w*h) for matrix multiplication reasons in first layer backprop
    # Normalization/scaling makes a big difference here(may depend on final layer)
    x_train = x_train.reshape((x_train.shape[0], 1, np.prod(x_train.shape[1:]))) / 255
    x_test = x_test.reshape((x_test.shape[0], 1, np.prod(x_test.shape[1:]))) / 255
    # x_train = x_train.reshape((x_train.shape[0], np.prod(x_train.shape[1:]))) / 255
    # x_test = x_test.reshape((x_test.shape[0], np.prod(x_test.shape[1:]))) / 255

    y_train = np_utils.to_categorical(y_train)
    #y_train = to_categorical(y_train)#np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    #y_test = to_categorical(y_test)#np_utils.to_categorical(y_test)

    act = sigmoid
    act_d = sigmoid_d

    input_shape = x_train.shape[1:]
    model = NeuralNetwork(input_shape)
    model.add_layer(FullyConnectedLayer(100))
    model.add_layer(ActivationLayer(act, act_d))
    model.add_layer(FullyConnectedLayer(100))
    model.add_layer(ActivationLayer(act, act_d))
    model.add_layer(FullyConnectedLayer(10))
    model.add_layer(ActivationLayer(act, act_d))
    # model.add_layer(ActivationLayer(sigmoid, sigmoid_d))
    model.set_loss_function(mse, mse_d)

    # np.random.seed(10)
    model.train(x_train[:1000], y_train[0:1000], 0.1, 50, batch_size=1)
    # model.train(x_train[:1000], y_train[0:1000], 0.1*32, 500, batch_size=32)

    np.set_printoptions(precision=2)
    n_y = 10
    predicts = model.predict(x_test[:n_y])
    for i in range(n_y):
        print("Predict={:d}  True={:d}".format(np.argmax(predicts[i]), np.argmax(y_test[i])))


def test_CNN():
    # X is of shape (samples, 28, 28) with each value being [0-255] (greyscale)
    # Y is of shape (samples,) with values [0-9]
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape X to add depth channels(1 in this case)
    x_train = np.expand_dims(x_train, axis=1)
    x_test = np.expand_dims(x_test, axis=1)

    # Convert true 0-9 values to a binary categorical format
    # Ex. [ 0-9 ] to [[0 or 1] of length 10]
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    n_train = 10

    input_shape = x_train.shape[1:]
    model = NeuralNetwork(input_shape)
    model.add_layer(ConvolutionalLayer(num_kernels=4, kernel_size=3, strides=1))
    model.add_layer(ActivationLayer(sigmoid, sigmoid_d))
    model.add_layer(ConvolutionalLayer(num_kernels=4, kernel_size=2, strides=1))
    model.add_layer(ActivationLayer(sigmoid, sigmoid_d))
    model.add_layer(ConvolutionalLayer(num_kernels=2, kernel_size=3, strides=1))
    model.add_layer(ActivationLayer(sigmoid, sigmoid_d))
    model.add_layer(FlattenLayer())
    model.add_layer(FullyConnectedLayer(10))
    model.add_layer(ActivationLayer(sigmoid, sigmoid_d))
    model.set_loss_function(mse, mse_d)

    t = time.time()
    model.train(x_train[:n_train], y_train[:n_train], 0.3, epochs=200, batch_size=1)
    print(f'Train duration: {time.time() - t}')

    np.set_printoptions(precision=2)
    #n_y = 10
    #predicts = model.predict(x_test[:n_y])
    #for i in range(n_y):
    #    print("Predict={:d}  True={:d}".format(np.argmax(predicts[i]), np.argmax(y_test[i])))

    t = time.time()
    predicts = model.predict(x_test)
    accuracy = sum((np.argmax(predicts[i]) == np.argmax(y_test[i]) for i in range(len(predicts)))) / len(predicts)
    print(f'Accuracy: {accuracy}')
    print(f'Predict duration: {time.time() - t}')



def test_CNN2():
    layer = ConvolutionalLayer(num_kernels=5, kernel_size=2, strides=1)
    img = np.array([np.reshape(range(1,10),(3,3)),np.reshape(range(9),(3,3))])
    #img = np.moveaxis(img, 0, -1)
    print(f'Image shape: {img.shape}')
    print(f'Image : \n{img}')
    #print(img[:,:,0])
    #print(img[:,:,1])

    layer.build(input_shape=img.shape)
    #print(layer.kernels)
    #layer.weights = np.ones(layer.weights.shape)
    for k in range(len(layer.weights)):
        layer.weights[k] = np.full(layer.weights[k].shape, k-2)

    print(f'Input shape: {layer.input_shape}')
    print(f'Output shape: {layer.output_shape}')
    img_batched = np.array([img])
    print(f'Batched image : \n{img_batched}')
    print(f'Batched image shape: {img_batched.shape}')
    output = layer.forward_prop(img_batched)
    print(f'Output: \n{output}')
    print(f'Output shape: {output.shape}')

    act_layer = ActivationLayer(sigmoid, sigmoid_d)
    act_layer.build(layer.output_shape)
    output2 = act_layer.forward_prop(output)
    print(f'Activation Output: \n{output2}')
    print(f'Activation Output shape: {output2.shape}')

    flatten = FlattenLayer()
    flatten.build(output2.shape)
    flattened = flatten.forward_prop(output2)
    print(f'Flatten Output: \n{flattened}')
    print(f'Flatten Output shape: {flattened.shape}')


    '''act = sigmoid(output)
    dact = sigmoid_d(output)

    print(act)
    print(act.shape)

    print(dact)
    print(dact.shape)'''



#simple_test()
#test()
test_CNN()
#test_CNN2()