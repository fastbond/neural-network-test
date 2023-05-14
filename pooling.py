from layers import Layer
import numpy as np

class MaxPooling(Layer):
    def __init__(self, filter_size, channels_first=True):
        super().__init__()
        self.filter_size = filter_size
        self.channels_first = channels_first
        self.outputs = None

    def build(self, input_shape=None, output_shape=None):
        self.input_shape = input_shape
        channels = input_shape[0]
        self.output_shape = (channels, *(i - self.filter_size + 1 for i in input_shape[1:]))

    def forward_prop(self, inputs):
        batch_size = inputs.shape[0]
        channels = inputs.shape[1]
        width = inputs.shape[2]
        height = inputs.shape[3]
        outputs = np.zeros((batch_size, *self.output_shape))

        # Potentially need to save which indices produced maximums here for passing along during backprop
        for h in range(height - self.filter_size + 1):
            for w in range(width - self.filter_size + 1):
                maxes = np.amax(inputs[:, :, w:w+self.filter_size, h:h+self.filter_size], axis=(-2,-1))
                outputs[:, :, w, h] = maxes
                #print(maxes.shape)
                #print(outputs[:, :, w, h].shape)

        # Store to check which produced max during backprop
        self.outputs = outputs

        return outputs

    def backprop(self, dE_dY):
        # (batch, input_shape)
        batch_size = dE_dY.shape[0]


        dE_dX = np.zeros((batch_size, *self.input_shape))



        return dE_dX

    def update(self, lr):
        return