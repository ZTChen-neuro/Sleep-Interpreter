#!/usr/bin/env python3
import tensorflow as tf
import tensorflow.keras as K
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))

__all__ = [
    "ConvSE",
]

class ConvSE(K.layers.Layer):
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
        super(ConvSE, self).__init__(**kwargs)

        # Initialize parameters.
        self.filters = params["filters"]
        self.kernel = params["kernel"]
        self.stride = params["stride"]
        self.ratio = params["ratio"]
        self.filters2 = self.filters // self.ratio



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
        self.conv = K.layers.Conv1D(self.filters, kernel_size=self.kernel, padding="same",strides = self.stride,
            data_format="channels_first", name="Conv1D_0",groups=1, activation=None, use_bias=True,
            kernel_initializer="he_uniform", 
            bias_initializer=None, 
            # kernel_initializer=K.initializers.random_normal(mean=0., stddev=0.01), 
            # bias_initializer=K.initializers.constant(value=0.01),
            kernel_regularizer=None,
            bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None
        )
        self.bn = K.layers.BatchNormalization(axis=1)
        self.avgpool = K.layers.GlobalAveragePooling1D(data_format='channels_first')
        self.fc = K.layers.Dense(self.filters2, 
                                #  kernel_initializer="he_uniform",
                                kernel_initializer=K.initializers.random_normal(mean=0., stddev=0.01), 
                                bias_initializer=K.initializers.constant(value=0.01),
                                 use_bias=False)
        self.fc2 = K.layers.Dense(self.filters, 
                                #   kernel_initializer="he_uniform",
                                kernel_initializer=K.initializers.random_normal(mean=0., stddev=0.01), 
                                bias_initializer=K.initializers.constant(value=0.01),
                                  use_bias=False)
        # Build super to set up `K.layers.Layer`-style model and inherit it's network.
        super(ConvSE, self).build(input_shape)

    # def call func
    def call(self, inputs):
        """
        Forward layers in `Conv1DBlock` to get the final result.
        :param inputs: (batch_size, seq_len, n_input_channels) - The input data.
        :return outputs: (batch_size, seq_len, n_output_channels) - The convolved data.
        """
        # Execute the first component of `Conv1DBlock`.
        # outputs - (batch_size, seq_len, n_filters[0])
        X = inputs
        b, c, _ = X.shape
        v = tf.reshape(self.avgpool(X), (b, c))
        v = self.fc2(K.activations.relu(self.fc(v)))
        v = tf.reshape(v, (b, self.filters, 1))
        v = K.activations.sigmoid(v)

        X = self.conv(X)
        X = self.bn(X)
        X = K.activations.relu(X)
        out = X * v
        return out

if __name__ == "__main__":
    # macro
    batch_size = 16; seq_len = 850; n_input_channels = 320
    n_filters = [320, 320, 640]; kernel_sizes = [3, 3, 3]; dilation_rates = [1, 2, 2]

    # Instantiate Conv1DBlock.
    cb_inst = ConvSE(n_filters, kernel_sizes, dilation_rates)
    # Initialize input data.
    inputs = tf.random.normal((batch_size, seq_len, n_input_channels), dtype=tf.float32)
    # Forward layers in `cb_inst`.
    outputs = cb_inst(inputs)

