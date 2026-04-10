#!/usr/bin/env python3
import tensorflow as tf
import tensorflow.keras as K
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))
    from Conv1DBlock import Conv1DBlock
    from ConvSE import ConvSE
else:
    from Model.SleepStageBlock.Model_tf.model.SIStaging.stager.layers import ConvSE
    from Model.SleepStageBlock.Model_tf.model.SIStaging.stager.layers import Conv1DBlock
    
__all__ = [
    "ConvFS",
]

class ConvFS(K.layers.Layer):
    """
    `ConvFS` layer used to convolve input data.
    """

    def __init__(self, params_fs, params_norm, params_series, **kwargs):
        """
        Initialize `ConvFS` object.
        :param n_filters: (3[list],) - The dimensionality of the output space.
        :param kernel_sizes: (3[list],) - The length of the 1D convolution window.
        :param dilation_rates: (3[list],) - The dilation rate to use for dilated convolution.
        :param kwargs: The arguments related to initialize `tf.keras.layers.Layer`-style object.
        """
        # First call super class init function to set up `K.layers.Layer`
        # style model and inherit it's functionality.
        super(ConvFS, self).__init__(**kwargs)

        # Initialize parameters.
        self.params_fs = params_fs
        self.params_norm = params_norm
        self.params_series = params_series
    """
    network funcs
    """
    # def build func
    def build(self, input_shape):
        """
        Build the network on the first call of `call`.
        :param input_shape: tuple - The shape of input data.
        """
        # Initialize the first component of `ConvFS`.
        
        self.conv_se = ConvSE(self.params_fs)
        self.conv_norm = Conv1DBlock(self.params_norm)
        self.conv_seq_1 = Conv1DBlock(self.params_series)
        self.conv_seq_2 = Conv1DBlock(self.params_series)
        self.conv_seq_3 = Conv1DBlock(self.params_series)
        self.fs_drop = K.layers.Dropout(rate=0.5)
        # Build super to set up `K.layers.Layer`-style model and inherit it's network.
        super(ConvFS, self).build(input_shape)

    # def call func
    def call(self, inputs):
        """
        Forward layers in `ConvFS` to get the final result.
        :param inputs: (batch_size, seq_len, n_input_channels) - The input data.
        :return outputs: (batch_size, seq_len, n_output_channels) - The convolved data.
        """
        # Execute the first component of `ConvFS`.
        # outputs - (batch_size, seq_len, n_filters[0])
        X = self.conv_se(inputs)
        X = self.conv_norm(X)
        output = self.conv_seq_1(X)
        output = self.conv_seq_2(output)
        output = self.fs_drop(self.conv_seq_3(output))
        output = output + X
        return output


if __name__ == "__main__":
    # macro
    batch_size = 16; seq_len = 850; n_input_channels = 320
    n_filters = [320, 320, 640]; kernel_sizes = [3, 3, 3]; dilation_rates = [1, 2, 2]

    # Instantiate ConvFS.
    cb_inst = ConvFS(n_filters, kernel_sizes, dilation_rates)
    # Initialize input data.
    inputs = tf.random.normal((batch_size, seq_len, n_input_channels), dtype=tf.float32)
    # Forward layers in `cb_inst`.
    outputs = cb_inst(inputs)

