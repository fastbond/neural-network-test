from abc import ABC, abstractmethod
import numpy as np
from typing import Iterable

class Layer(ABC):
    def __init__(self):
        self.input_shape = None
        self.output_shape = None

    # Used to set expected input, output dimensions once adjacent layers are known,
    # as well as construct weight matrices
    @abstractmethod
    def build(self, input_shape=None, output_shape=None):
        raise NotImplementedError

    @abstractmethod
    def forward_prop(self, input):
        raise NotImplementedError

    @abstractmethod
    def backprop(self, error):
        raise NotImplementedError

    def update(self, lr):
        return


# aka Dense Layer
class FullyConnectedLayer(Layer):
    def __init__(self, neurons, input_shape=None, weight_range=(-0.5,0.5)):
        super().__init__()
        self.neurons = neurons
        self.input_shape = input_shape
        self.output_shape = (neurons)
        self.weight_range = weight_range
        self.weights = None
        self.bias = None
        self.grad_weights = None
        self.bias_weights = None
        self.num_samples_used = 0

    def build(self, input_shape=None, output_shape=None):
        # Output shape is equal to neurons for standard dense layer
        self.input_shape = input_shape or self.input_shape
        self.output_shape = (self.neurons)
        self.init_weights()

    # Weight initialization
    #   See https://www.analyticsvidhya.com/blog/2021/05/how-to-initialize-weights-in-neural-networks/
    #   -Small magnitude is recommended
    #   -Heuristics are good
    #   -Just randomizing with a range of 1
    def init_weights(self):
        min_w = self.weight_range[0]
        max_w = self.weight_range[1]
        input_neurons = np.prod(self.input_shape)
        self.weights = np.random.uniform(min_w, max_w, (input_neurons, self.neurons))
        self.bias = np.zeros((1, self.neurons))
        # Matrices for summed weight gradients during backprop
        # Used to store gradients for post-backprop GD update
        self.grad_weights = np.zeros(self.weights.shape)
        self.grad_bias = np.zeros(self.bias.shape)
        # Stores number of samples adding to current gradient sum matrices
        self.num_samples_used = 0

    def forward_prop(self, inputs):
        # Reshape input tensor to an appropriate shape
        # (num_samples, 1, M) where M is number of features per sample
        # Makes each sample a single feature vector
        # Ex: input is output from a Conv layer
        inputs = np.reshape(inputs,(inputs.shape[0], 1, np.prod(inputs.shape[1:])))

        # Y = XW + B, where
        #   X is vector of inputs
        #   W is matrix of weights
        #   B is column of biases
        outputs = np.matmul(inputs, self.weights) + self.bias
        self.inputs = inputs

        # Track amount of samples included in this batch so far
        # Required for averaging sum of sample gradients in update
        self.num_samples_used += inputs.shape[0]
        return outputs

    # dE_dY is of shape (1, neurons)
    def backprop(self, dE_dY):
        # Compute gradient
        # Recall Y = XW + B, where
        #   X is vector of inputs
        #   W is matrix of weights
        #   B is column of biases
        # So, for a given weight wij(neuron i, weight from input j)
        #   yi = xj*wij + bi
        #   dyi/dwij = xj*1 + 0 = xj
        # so
        #   dE/dwij = dE/dyi * dyi/dwij             = dE/dyi * xj
        #   dE/dbi  = dE/dyi * dyi/dbi = dE/dyi * 1 = dE/dyi
        # (inputs, outputs) = (inputs, 1) . (1, outputs)
        transpose = self.inputs.swapaxes(-1,-2)
        dE_dW = np.matmul(transpose, dE_dY)
        dE_dB = dE_dY

        # For each layer, have matrix of weight/bias derivatives matching weight dimensions
        # Add onto it for each sample, then divide by batch size for avg deriv
        self.grad_weights += np.sum(dE_dW, axis=0)
        self.grad_bias += np.sum(dE_dB, axis=0)

        # Pass along error gradient(dE_dX)
        # Y(output) of previous layer is this layer's X(input)
        #   dE/dxj = dE/dyi * dyi/dxj               = dE/dyi * wij
        # (1, inputs) = (1, outputs) . (outputs, inputs)
        dE_dX = np.dot(dE_dY, self.weights.T)
        return dE_dX

    def update(self, lr):
        # Average summed weight gradients by dividing by number of samples in batch
        self.grad_weights /= self.num_samples_used
        self.grad_bias /= self.num_samples_used

        # Update via gradient descent
        self.weights  = self.weights - (lr * self.grad_weights)
        self.bias = self.bias - (lr * self.grad_bias)

        # Reset gradient sums, batch size count for next batch
        self.grad_weights = np.zeros(self.weights.shape)
        self.grad_bias = np.zeros(self.bias.shape)
        self.num_samples_used = 0


class ActivationLayer(Layer):
    def __init__(self, activation_func, derivative_func):
        super().__init__()
        self.activation = activation_func
        self.derivative = derivative_func

    def build(self, input_shape=None, output_shape=None):
        if input_shape is not None:
            self.input_shape = input_shape
        self.output_shape = self.input_shape

    def forward_prop(self, inputs):
        self.inputs = inputs
        #print(f'Activation Layer input shape: {self.inputs.shape}')
        outputs = self.activation(inputs)
        self.outputs = outputs
        #print(f'Activation Layer output shape: {self.outputs.shape}')
        return outputs

    # dE_dY = (1, outputs)
    # derivative(inputs) is another vector of (1, inputs)
    # |inputs| = |outputs| since just applying function to each
    # TODO: Is this derivative correct? Should it be of outputs or inputs?
    def backprop(self, dE_dY):
        dY_dX = self.derivative(self.outputs)#self.inputs) #
        dE_dX = dE_dY * dY_dX
        return dE_dX

    def update(self, lr):
        return


# Assumes 2D for now(single color channel - greyscale)
# Kernel size logic: https://stackoverflow.com/questions/57438922/different-size-filters-in-the-same-layer-with-tensorflow-2-0
# https://towardsdatascience.com/convolutional-neural-network-ii-a11303f807dc
# https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/
# https://towardsdatascience.com/convolutional-neural-network-ii-a11303f807dc
# https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks
# https://d2l.ai/chapter_convolutional-neural-networks/channels.html
# https://cs231n.github.io/convolutional-networks/#conv
# https://arxiv.org/pdf/1603.07285v1.pdf
class ConvolutionalLayer(Layer):
    def __init__(self, num_kernels, kernel_size, strides, padding=None, channels_first=True):
        super().__init__()
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.channels_first = channels_first
        # Kernel aka weights aka parameters
        self.weights = None
        self.bias = None

    # Input shape is (batch_size x rows x cols x channels aka depth)?
    # https://miro.medium.com/v2/resize:fit:720/format:webp/0*-zjHFGVymDMb9XAZ.png
    def build(self, input_shape=None, output_shape=None):
        self.input_shape = input_shape or self.input_shape
        output_width = self.input_shape[1] - self.kernel_size + 1
        output_height = self.input_shape[2] - self.kernel_size + 1
        self.output_shape = (self.num_kernels, output_width, output_height)

        # https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks
        input_channels = self.input_shape[0]
        #self.weights = np.random.uniform(-0.5, 0.5, (self.num_kernels, self.kernel_size, self.kernel_size))
        self.weights = np.random.uniform(-0.5, 0.5, (self.num_kernels, input_channels, self.kernel_size, self.kernel_size))

        # Every filter has one bias? Or it it one per filter per channel?
        #https://datascience.stackexchange.com/a/73936
        self.bias = np.zeros(self.num_kernels)


    # channels == input feature maps
    # input should be of shape (batch_size, width, height, channels)
    # output should be shape (batch_size, width', height', num_filters)
    # TODO: Correct output shape, but not sure combining channels properly per filter
    def forward_prop(self, input):
        batch_size = input.shape[0]
        kernels = self.num_kernels  # also output channels/feature maps
        width = input.shape[2]
        height = input.shape[3]
        # Assuming stride of 1 in both directions
        convolution = np.zeros(
            (batch_size,
             kernels,
             width - self.kernel_size + 1,
             height - self.kernel_size + 1)
        )

        #print(f'Input dims: {input.shape}')
        #print(f'Kernels dims: {self.kernels.shape}')
        # w and h will be indices in output matrix/feature map
        # TODO: Process all batches at once
        #  also condense calls
        #  also its just wrong
        for batch in range(batch_size):
            for h in range(height - self.kernel_size + 1):
                for w in range(width - self.kernel_size + 1):
                    '''print("--POSITION--")
                    print(w)
                    print(h)
                    print("------------")'''
                    input_section = input[batch, :, w: w+self.kernel_size, h: h+self.kernel_size]
                    '''print(f'Input section: \n{input_section}')
                    print(f'Kernels: \n{self.weights}')
                    print(f'Kernels shape: {self.weights.shape}')
                    print(f'Input section shape: {input_section.shape}')'''
                    for k in range(len(self.weights)):
                        # products = np.matmul(input_section, self.kernels)
                        # For each k, (c, x, y) * (c, x, y)
                        products = input_section * self.weights[k]  # * for element-wise multiplication
                        sums = np.sum(products)  # , axis=(1, 2))
                        '''print(f'Products: \n{products}')
                        print(f'Products shape: \n{products.shape}')
                        print(f'Sums: \n{sums}')
                        print(f'Sums shape: \n{sums.shape}')'''
                        convolution[batch, k, w, h] = sums

        #convolution += self.bias
        #print(convolution.shape)

        return convolution

    # Error aka dE_dY
    # (batch_size, width', height', num_filters)
    def backprop(self, dE_dY):
        print(dE_dY.shape)
        print("backprop")

        dE_dk = np.zeros(self.kernels.shape)
        for patch, h, w in self.patches_generator(self.image):
            for f in range(self.kernel_num):
                dE_dk[f] += patch * dE_dY[h, w, f]
        # Update the parameters
        self.kernels -= alpha * dE_dk
        return dE_dk

    def update(self, lr):
        return

'''
# Extract image height and width
image_h, image_w = image.shape
# Initialize the convolution output volume of the correct size
convolution_output = np.zeros((image_h - self.kernel_size + 1, image_w - self.kernel_size + 1, self.kernel_num))
# Unpack the generator
for patch, h, w in self.patches_generator(image):
    # Perform convolution for each patch
    convolution_output[h, w] = np.sum(patch * self.kernels, axis=(1, 2))
return convolution_output
'''


class FlattenLayer(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape=None, output_shape=None):
        self.input_shape = input_shape
        print(f'Flatten input shape: {self.input_shape}')
        #self.output_shape = (input_shape[0], 1, np.prod(self.input_shape[1:]))
        self.output_shape = (1, np.prod(self.input_shape[:]))
        print(f'Flatten output shape: {self.output_shape}')

    def forward_prop(self, inputs):
        outputs = np.reshape(inputs, self.output_shape)
        return outputs

    def backprop(self, dE_dY):
        dE_dY = np.reshape(dE_dY, self.input_shape)
        return dE_dY

    def update(self, lr):
        return