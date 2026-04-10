#!/usr/bin/env python3
import tensorflow as tf
import tensorflow.keras as K
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))

__all__ = [
    "Conv1DBlock",
]

class Conv1DBlock(K.layers.Layer):
    """
    `Conv1DBlock` layer used to convolve input data.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `Conv1DBlock` object.
        :param n_filters: (3[list],) - The dimensionality of the output space.
        :param kernel_sizes: (3[list],) - The length of the 1D convolution window.
        :param dilation_rates: (3[list],) - The dilation rate to use for dilated convolution.
        :param kwargs: The arguments related to initialize `tf.keras.layers.Layer`-style object.
        """
        # First call super class init function to set up `K.layers.Layer`
        # style model and inherit it's functionality.
        super(Conv1DBlock, self).__init__(**kwargs)

        # Initialize parameters.
        self.filters = params['filters']
        self.kernel = params['kernel']
        self.stride = params['stride']



    """
    network funcs
    """
    # def build func
    def build(self, input_shape):
        """
        Build the network on the first call of `call`.
        :param input_shape: tuple - The shape of input data.
        """
        # Initialize the first component of `Conv1DBlock`.
        self.conv1 = K.layers.Conv1D(self.filters, kernel_size=self.kernel, padding="same",strides = self.stride,
            data_format="channels_first", name="Conv1D_0", groups=1, activation=None, use_bias=True,
            kernel_initializer="he_uniform", 
            bias_initializer=None, 
            # kernel_initializer=K.initializers.random_normal(mean=0., stddev=0.01), 
            # bias_initializer=K.initializers.constant(value=0.01),
            kernel_regularizer=None,
            bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None
        )
        self.bn1 = K.layers.BatchNormalization(axis=1)
        # Build super to set up `K.layers.Layer`-style model and inherit it's network.
        super(Conv1DBlock, self).build(input_shape)

    # def call func
    def call(self, inputs):
        """
        Forward layers in `Conv1DBlock` to get the final result.
        :param inputs: (batch_size, seq_len, n_input_channels) - The input data.
        :return outputs: (batch_size, seq_len, n_output_channels) - The convolved data.
        """
        # Execute the first component of `Conv1DBlock`.
        # outputs - (batch_size, seq_len, n_filters[0])
        outputs = self.conv1(inputs=inputs)
        outputs = K.activations.relu(self.bn1(inputs=outputs))
        return outputs

# def glu func
def glu(x, axis=-1):
    """
    GLU activation function, `glu(x) = x[0] * sigmoid(x[1])`.
    :param x: (batch_size, seq_len, n_input_channels) - The input data.
    :param axis: int - The axis along which we perform glu activation.
    :return x: (batch_size, seq_len, n_input_channels // 2) - The glu activation applied to x.
    """
    # Split the original data into two parts.
    a, b = tf.split(x, num_or_size_splits=2, axis=axis)
    # Execute glu activation.
    x = tf.multiply(a, tf.sigmoid(b))
    # Return the final `x`.
    return x

if __name__ == "__main__":
    # macro
    batch_size = 16; seq_len = 850; n_input_channels = 320
    n_filters = [320, 320, 640]; kernel_sizes = [3, 3, 3]; dilation_rates = [1, 2, 2]

    # Instantiate Conv1DBlock.
    cb_inst = Conv1DBlock(n_filters, kernel_sizes, dilation_rates)
    # Initialize input data.
    inputs = tf.random.normal((batch_size, seq_len, n_input_channels), dtype=tf.float32)
    # Forward layers in `cb_inst`.
    outputs = cb_inst(inputs)

