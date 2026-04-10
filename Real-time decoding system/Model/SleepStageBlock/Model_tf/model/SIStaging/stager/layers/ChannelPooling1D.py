#!/usr/bin/env python3
import tensorflow as tf
import tensorflow.keras as K
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))

__all__ = [
    "ChannelPooling1D",
]

class ChannelPooling1D(K.layers.Layer):
    """
    `ChannelPooling1D` layer used to do channel-wise pooling over input data.
    """

    def __init__(self, n_filters=1, use_bias=True, **kwargs):
        """
        Initialize `ChannelPooling1D` object.

        Args:
            *: The parameters of `K.layers.Conv1D` layer.

        Returns:
            None
        """
        # First call super class init function to set up `K.layers.Layer`
        # style model and inherit it's functionality.
        super(ChannelPooling1D, self).__init__(**kwargs)

        # Initialize parameters.
        self.n_filters = n_filters; self.use_bias = use_bias

    """
    network funs
    """
    # def build func
    def build(self, input_shape):
        """
        Build the network on the first call of `call`.

        Args:
            input_shape: tuple - The shape of input data.

        Returns:
            None
        """
        # Initialize the 1x1-convolution layer.
        self.conv = K.layers.Conv1D(
            # Modified `Conv1D` layer parameters.
            self.n_filters, 1, use_bias=self.use_bias,
            # Default `Conv1D` layer parameters.
            strides=1, padding="valid", data_format="channels_last", dilation_rate=1, groups=1, activation=None,
            kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None,
            bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None
        )
        # Initialize the flatten layer.
        self.flatten = K.layers.Flatten()
        # Build super to set up `K.layers.Layer`-style model and inherit it's network.
        super(ChannelPooling1D, self).build(input_shape)

    # def call func
    def call(self, inputs):
        """
        Forward layers in ``ChannelPooling1D` to get the final result.

        Args:
            inputs: (batch_size, seq_len, n_channels) - The input data.

        Returns:
            outputs: (batch_size, n_filters * seq_len) - The channel-wise pooling data.
        """
        # Execute 1D channel-wise pooling.
        # outputs - (batch_size, n_filters * seq_len)
        outputs = self.flatten(self.conv(inputs))
        # Return the final `outputs`.
        return outputs

if __name__ == "__main__":
    # macro
    batch_size = 16; seq_len = 160; n_channels = 204; n_filters = 2

    # Instantiate Conv1DBlock.
    cp_inst = ChannelPooling1D(n_filters)
    # Initialize input data.
    inputs = tf.random.normal((batch_size, seq_len, n_channels), dtype=tf.float32)
    # Forward layers in `cp_inst`.
    outputs = cp_inst(inputs)

