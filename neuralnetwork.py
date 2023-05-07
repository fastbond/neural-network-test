import numpy as np
import math
from layers import Layer

# Currently only implements a linear network structure
# input_shape does not include batch size
# input_shape needs to be refactored.  Currently takes an int
class NeuralNetwork():
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.layers = []

    def add_layer(self, layer: Layer):
        input_shape = self.layers[-1].output_shape if len(self.layers) > 0 else self.input_shape
        layer.build(input_shape=input_shape)
        self.layers.append(layer)

    def set_loss_function(self, loss_func, loss_deriv):
        self.loss_func = loss_func
        self.loss_deriv = loss_deriv

    # Lacks error checking on input dimensions
    def train(self, X, Y, lr, epochs, batch_size=None):
        num_samples = X.shape[0]
        if batch_size is None or batch_size > num_samples:
            batch_size = num_samples
        num_batches = math.ceil(num_samples / batch_size)

        # NOTE: this method is quick and dirty and will only work for a linear network
        for epoch in range(epochs):
            error = 0

            # Should probably shuffle batches(or samples?)...
            for i in range(0, num_batches, 1):
                X_batch = X[batch_size * i: batch_size * (i + 1)]
                Y_batch = Y[batch_size * i: batch_size * (i + 1)]

                output = X_batch
                for layer in self.layers:
                    output = layer.forward_prop(output)

                Y_batch = Y_batch.reshape(output.shape)

                # Compute reported error(loss)
                # In Keras, reported losses are the average of per sample losses in each batch
                # Assumption: error function returns avg error of samples within batch
                # Multiply by number of samples in batch, then later divide by total number of samples
                # This accounts for variable batch size
                error += self.loss_func(output, Y_batch) * X_batch.shape[0]

                error_gradient = self.loss_deriv(output, Y_batch)
                for layer in reversed(self.layers):
                    error_gradient = layer.backprop(error_gradient)

                # Update using the computed weight gradients
                for layer in self.layers:
                    layer.update(lr)

            # Divide total error by number of samples for per-sample mean error
            error /= len(X)

            print("Epoch {:d}: {:f}".format(epoch, error))

    def predict(self, X):
        print(X.shape)
        output = X
        for layer in self.layers:
            output = layer.forward_prop(output)
        return output

