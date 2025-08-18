#!/usr/bin/env python3
import tensorflow as tf
import tensorflow.keras as K
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))

__all__ = [
    "ResidualConv1D",
]

class ResidualConv1D(K.layers.Layer):
    """
    `ResidualConv1D` layer used to convolve input data.
    """

    def __init__(self, n_filters, kernel_size, strides=1, dilation_rate=1, use_bias=True, **kwargs):
        """
        Initialize `ResidualConv1D` object.

        Args:
            *: The parameters of `K.layers.Conv1D` layer.

        Returns:
            None
        """
        # First call super class init function to set up `K.layers.Layer`
        # style model and inherit it's functionality.
        super(ResidualConv1D, self).__init__(**kwargs)

        # Initialize parameters.
        self.n_filters = n_filters; self.kernel_size = kernel_size; self.strides = strides
        self.dilation_rate = dilation_rate; self.use_bias = use_bias

    """
    network funcs
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
        # Initialize `n_channels` from `input_shape`.
        # Note: We have assumed that data_format is `channels_last`!
        self.n_channels = input_shape[-1]
        # Initialize the convolution layer.
        self.conv = K.layers.Conv1D(
            # Modified `Conv1D` layer parameters.
            self.n_filters, self.kernel_size, strides=self.strides, padding="same",
            dilation_rate=self.dilation_rate, use_bias=self.use_bias,
            # Default `Conv1D` layer parameters.
            data_format="channels_last", groups=1, activation=None,
            kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None,
            bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None
        )
        # If the dimensions of input is not equal to the dimensions of output, we have to project the shortcut connection.
        # Note: We set `kernel_size` to `(1, 1)`, there is no difference whether we set `padding` to `same` or not!
        self.shortcut = K.layers.Conv1D(
            # Modified `Conv2D` layer parameters.
            self.n_filters, 1, padding="same",
            # Default `Conv2D` layer parameters.
            strides=1, data_format="channels_last", dilation_rate=1, groups=1, activation=None, use_bias=True,
            kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None,
            bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None
        ) if self.n_channels != self.n_filters else tf.identity
        # Build super to set up `K.layers.Layer`-style model and inherit it's network.
        super(ResidualConv1D, self).build(input_shape)

    # def call func
    def call(self, inputs):
        """
        Forward layers in ``ResidualConv1D` to get the final result.

        Args:
            inputs: (batch_size, seq_len, n_channels) - The input data.

        Returns:
            outputs: (batch_size, seq_len, n_filters) - The convolved data.
        """
        # Execute the convolution layer, then add the original inputs.
        # outputs - (batch_size, seq_len, n_filters)
        outputs = self.conv(inputs) + self.shortcut(inputs)
        # Return the final `outputs`.
        return outputs

if __name__ == "__main__":
    # macro
    batch_size = 16; seq_len = 160; n_channels = 204; n_filters = 256; kernel_size = 3

    # Instantiate Conv1DBlock.
    rc_inst = ResidualConv1D(n_filters, kernel_size)
    # Initialize input data.
    inputs = tf.random.normal((batch_size, seq_len, n_channels), dtype=tf.float32)
    # Forward layers in `rc_inst`.
    outputs = rc_inst(inputs)


