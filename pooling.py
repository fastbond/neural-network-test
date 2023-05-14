from layers import Layer
import numpy as np

class MaxPooling(Layer):
    def __init__(self, filter_size, channels_first=True):
        super().__init__()
        self.filter_size = filter_size
        self.channels_first = channels_first
        self.outputs = None
        self.inputs = None

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
        self.inputs = inputs
        self.outputs = outputs

        return outputs

    # (batch_size, channels, output_width, output_height)
    def backprop(self, dE_dY):
        batch_size = dE_dY.shape[0]
        channels = dE_dY.shape[1]
        output_width = dE_dY.shape[2]
        output_height = dE_dY.shape[3]
        input_width = self.input_shape[-2]
        input_height = self.input_shape[-1]

        dE_dX = np.zeros((batch_size, *self.input_shape))

        # TODO: This is very very slow
        # All dE_dx are 0 except those which contributed to maximum in outputs
        # A single input x can contribute to multiple outputs(covered by multiple filters)
        for batch in range(batch_size):
            for c in range(channels):
                for h in range(output_height - self.filter_size + 1):
                    for w in range(output_width - self.filter_size + 1):
                        section_max = self.outputs[batch][c][w][h]
                        for kw in range(self.filter_size):
                            for kh in range(self.filter_size):
                                if self.inputs[batch][c][w+kw][h+kh] == section_max:
                                    dE_dX[batch][c][w+kw][h+kh] += dE_dY[batch][c][w][h]

        return dE_dX

    def update(self, lr):
        return