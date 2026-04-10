#!/usr/bin/env python3
import tensorflow as tf
import tensorflow.keras as K
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))

__all__ = [
    "SubjectBlock_ft",
]

class SubjectBlock_ft(K.layers.Layer):
    """
    `SubjectBlock_ft` layer used to transform each channel with specified subject id.
    """

    def __init__(self, n_output_channels, **kwargs):
        """
        Initialize `SubjectBlock_ft` object.
        :param kwargs: The arguments related to initialize `tf.keras.layers.Layer`-style object.
        """
        # First call super class init function to set up `K.layers.Layer`
        # style model and inherit it's functionality.
        super(SubjectBlock_ft, self).__init__(**kwargs)
        self.n_output_channels = n_output_channels

    """
    network funcs
    """
    # def build func
    def build(self, input_shape):
        """
        Build the network on the first call of `call`.
        :param input_shape: tuple - The shape of input data.
        """
        # Initialize Conv1D.
        self.conv1d_layer = K.layers.Conv1D(self.n_output_channels[1], kernel_size=9, padding = "same", dilation_rate = 1,
            kernel_initializer=K.initializers.random_normal(mean=0., stddev=0.01),
            bias_initializer=K.initializers.constant(value=0.01),
            # Default `Dense` layer parameters.
            activation=None, use_bias=True, kernel_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None, bias_constraint=None, trainable=False)
        self.drop = K.layers.Dropout(rate=0.)
        self.bm = K.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer="zeros",
            gamma_initializer="ones", moving_mean_initializer="zeros", moving_variance_initializer="ones",
            beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None
        )
        self.conv1d_layer2 = K.layers.Conv1D(self.n_output_channels[1], kernel_size=9, padding = "same", dilation_rate = 2,
            kernel_initializer=K.initializers.random_normal(mean=0., stddev=0.01),
            bias_initializer=K.initializers.constant(value=0.01),
            # Default `Dense` layer parameters.
            activation=None, use_bias=True, kernel_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None, bias_constraint=None, trainable=False)
        self.drop2 = K.layers.Dropout(rate=0.)
        self.bm2 = K.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer="zeros",
            gamma_initializer="ones", moving_mean_initializer="zeros", moving_variance_initializer="ones",
            beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None
        )

        self.dense_layer1 = K.layers.Dense(
            # Modified `Dense` layer parameters.
            units=256,
            kernel_initializer=K.initializers.random_normal(mean=0., stddev=0.01),
            bias_initializer=K.initializers.constant(value=0.01),
            # Default `Dense` layer parameters.
            activation=None, use_bias=True, kernel_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None, bias_constraint=None
        )
        self.dense_layer2 = K.layers.Dense(
            # Modified `Dense` layer parameters.
            units=128,
            kernel_initializer=K.initializers.random_normal(mean=0., stddev=0.01),
            bias_initializer=K.initializers.constant(value=0.01),
            # Default `Dense` layer parameters.
            activation=None, use_bias=True, kernel_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None, bias_constraint=None
        )
        super(SubjectBlock_ft, self).build(input_shape)

    # def call func
    def call(self, inputs):
        """
        Forward layers in `SubjectBlock_ft` to get the final result.
        :param inputs: The input data.
        :return outputs: (batch_size, seq_len, n_output_channels) - The subject-transformed data.
        """
        # X - (batch_size, seq_len, n_input_channels)
        X = inputs
        outputs = self.conv1d_layer(inputs=X)
        outputs = self.bm(outputs)

        outputs = self.conv1d_layer2(inputs=outputs) + outputs
        outputs = self.bm2(outputs)
        
        outputs = self.dense_layer1(inputs=outputs)
        outputs = self.dense_layer2(outputs)
        return outputs


