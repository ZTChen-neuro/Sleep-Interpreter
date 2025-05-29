#!/usr/bin/env python3
import copy as cp
import tensorflow as tf
import tensorflow.keras as K
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.getcwd()))
    from layers import *
else:
    from .layers import *

__all__ = [
    "stager",
]

class stager(K.Model):
    """
    `stager` model specified for eeg sleep staging.
    """

    def __init__(self, fs, weight, **kwargs):
        """
        Initialize `stager` object.

        Args:
            fs: sampling rate of input EEG data
            weight: Model training preference values for different sleep stages
            kwargs: The arguments related to initialize `tf.keras.Model`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `K.Model`
        # style model and inherit it's functionality.
        super(stager, self).__init__(**kwargs)

        self.n_labels = 4
        self.weight = weight
        
        self.params_fs1 = {"filters": 16, "kernel": int(fs / 1),"stride": int(fs / 8),"ratio": 2}
        self.params_fs2 = {"filters": 16, "kernel": int(fs / 2),"stride": int(fs / 8),"ratio": 2}
        self.params_fs3 = {"filters": 16, "kernel": int(fs / 4),"stride": int(fs / 8),"ratio": 2}
        self.params_fs4 = {"filters": 16, "kernel": int(fs / 8),"stride": int(fs / 8),"ratio": 2}
        self.params_fs5 = {"filters": 16, "kernel": int(fs * 2),"stride": int(fs / 8),"ratio": 2}
        self.params_fs6 = {"filters": 16, "kernel": int(fs * 4),"stride": int(fs / 8),"ratio": 2}

        self.params_n1 = {"filters": 32,"kernel": 10,"stride": 1}
        self.params_n2 = {"filters": 32,"kernel": 10,"stride": 1}
        self.params_n3 = {"filters": 32,"kernel": 10,"stride": 1}
        self.params_n4 = {"filters": 32,"kernel": 10,"stride": 1}
        self.params_n5 = {"filters": 32,"kernel": 10,"stride": 1}
        self.params_n6 = {"filters": 32,"kernel": 10,"stride": 1}

        self.params_s5 = {"filters": 32,"kernel": 5,"stride": 1}
        
        self.params_pool = {"filters": 128, "kernel": 7,"stride": 7,"ratio": 2}

        self.params_s1_1 = {"filters": 256,"kernel": 3,"stride": 1}
        self.params_s1_2 = {"filters": 192,"kernel": 3,"stride": 1}
        self.params_s1_3 = {"filters": 128,"kernel": 3,"stride": 1}
        
        self.params_pool2 = {"filters": 64, "kernel": 5,"stride": 5,"ratio": 2}
        # Flatten
        self.linear = {"filters": 4}
        # Create trainable vars.
        self._init_trainable()

    """
    init funcs
    """
    # def _init_trainable func
    def _init_trainable(self):
        """
        Initialize trainable variables.

        Args:
            None

        Returns:
            None
        """
        self.conv_fs1 = ConvFS(self.params_fs1, self.params_n1, self.params_s5)
        self.conv_fs2 = ConvFS(self.params_fs2, self.params_n2, self.params_s5)
        self.conv_fs3 = ConvFS(self.params_fs3, self.params_n3, self.params_s5)
        self.conv_fs4 = ConvFS(self.params_fs4, self.params_n4, self.params_s5)
        self.conv_fs5 = ConvFS(self.params_fs5, self.params_n5, self.params_s5)
        self.conv_fs6 = ConvFS(self.params_fs6, self.params_n6, self.params_s5)
        self.pool_drop = K.layers.Dropout(rate=0.5)
        self.series_drop = K.layers.Dropout(rate=0.5)

        self.conv_pool = ConvSE(self.params_pool)
        self.conv_series = K.models.Sequential()
        self.conv_series.add(Conv1DBlock(self.params_s1_1))
        self.conv_series.add(Conv1DBlock(self.params_s1_2))
        self.conv_series.add(Conv1DBlock(self.params_s1_3))
        
        self.conv_pool2 = ConvSE(self.params_pool2)
        self.flatten = K.layers.Flatten()
        self.fc = K.layers.Dense(self.linear["filters"], activation="softmax",
                                #  kernel_initializer="he_normal",
                                 kernel_initializer=K.initializers.random_normal(mean=0., stddev=0.01), 
                                bias_initializer=K.initializers.constant(value=0.01))
        self.fc2 = K.layers.Dense(64, activation='relu',
                                #  kernel_initializer="he_normal",
                                 kernel_initializer=K.initializers.random_normal(mean=0., stddev=0.01), 
                                bias_initializer=K.initializers.constant(value=0.01))
        
        self.lstm = K.layers.LSTM(256, kernel_initializer='he_uniform',dropout=0.,
                                recurrent_dropout=0.)
        
        self.conv_block = K.models.Sequential()
        self.conv_block.add(K.layers.Conv1D(
            # Modified `Conv1D` layer parameters.
            filters=240, kernel_size=10, padding="same", dilation_rate=1,
            # Default `Conv1D` layer parameters.
            strides=1, data_format="channels_last", groups=1, activation=None, use_bias=True,
            kernel_initializer="he_uniform", bias_initializer="zeros", kernel_regularizer=None,
            bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None
        ))
        self.conv_block.add(K.layers.Conv1D(
            # Modified `Conv1D` layer parameters.
            filters=240, kernel_size=20, padding="same", dilation_rate=2, #2
            # Default `Conv1D` layer parameters.
            strides=1, data_format="channels_last", groups=1, activation=None, use_bias=True,
            kernel_initializer="he_uniform", bias_initializer="zeros", kernel_regularizer=None,
            bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None
        ))
        self.conv_block.add(K.layers.Dropout(rate=0.5))
        self.conv_block.add(K.layers.BatchNormalization(
            # Default `BatchNormalization` parameters.
            axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer="zeros",
            gamma_initializer="ones", moving_mean_initializer="zeros", moving_variance_initializer="ones",
            beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None
        ))
        self.conv_block2 = K.models.Sequential()
        self.conv_block2.add(K.layers.Conv1D(
            # Modified `Conv1D` layer parameters.
            filters=240, kernel_size=9, padding="same", dilation_rate=4,
            # Default `Conv1D` layer parameters.
            strides=1, data_format="channels_last", groups=1, activation=None, use_bias=True,
            kernel_initializer="he_uniform", bias_initializer="zeros", kernel_regularizer=None,
            bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None
        ))
        self.conv_block2.add(K.layers.Conv1D(
            # Modified `Conv1D` layer parameters.
            filters=240, kernel_size=9, padding="same", dilation_rate=1, #2
            # Default `Conv1D` layer parameters.
            strides=1, data_format="channels_last", groups=1, activation=None, use_bias=True,
            kernel_initializer="he_uniform", bias_initializer="zeros", kernel_regularizer=None,
            bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None
        ))
        self.conv_block2.add(K.layers.Dropout(rate=0.5))
        self.conv_block2.add(K.layers.BatchNormalization(
            # Default `BatchNormalization` parameters.
            axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer="zeros",
            gamma_initializer="ones", moving_mean_initializer="zeros", moving_variance_initializer="ones",
            beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None
        ))

        self.conv_block3 = K.models.Sequential()
        self.conv_block3.add(K.layers.Conv1D(
            # Modified `Conv1D` layer parameters.
            filters=240, kernel_size=9, padding="same", dilation_rate=2,
            # Default `Conv1D` layer parameters.
            strides=1, data_format="channels_last", groups=1, activation=None, use_bias=True,
            kernel_initializer="he_uniform", bias_initializer="zeros", kernel_regularizer=None,
            bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None
        ))
        self.conv_block3.add(K.layers.Conv1D(
            # Modified `Conv1D` layer parameters.
            filters=240, kernel_size=9, padding="same", dilation_rate=4, #2
            # Default `Conv1D` layer parameters.
            strides=1, data_format="channels_last", groups=1, activation=None, use_bias=True,
            kernel_initializer="he_uniform", bias_initializer="zeros", kernel_regularizer=None,
            bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None
        ))
        self.conv_block3.add(K.layers.Dropout(rate=0.5))
        self.conv_block3.add(K.layers.BatchNormalization(
            # Default `BatchNormalization` parameters.
            axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer="zeros",
            gamma_initializer="ones", moving_mean_initializer="zeros", moving_variance_initializer="ones",
            beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None
        ))

    """
    network funcs
    """
    # def call func
    def call(self, inputs, training=None, mask=None):
        """
        Forward `stager` to get the final predictions.

        Args:
            inputs: tuple - The input data.
            training: Boolean or boolean scalar tensor, indicating whether to run
                the `Network` in training mode or inference mode.
            mask: A mask or list of masks. A mask can be either a tensor or None (no mask).

        Returns:
            outputs: (batch_size, n_labels) - The output labels.
            loss: tf.float32 - The corresponding loss.
        """
        # Initialize components of inputs.
        # X - (batch_size, n_channels, seq_len)
        # true_label - (batch_size, 4)
        # label - (batch_size,)
        X = inputs

        X1 = self.conv_fs1(X)
        X2 = self.conv_fs2(X)
        X3 = self.conv_fs3(X)
        X4 = self.conv_fs4(X)
        X5 = self.conv_fs5(X)
        X6 = self.conv_fs6(X)
        X = tf.concat([X1, X2, X3, X4, X5, X6], axis=1)
        X = self.conv_block(X)
        X = self.conv_block2(X) + X
        X = self.conv_block3(X) + X

        X = self.pool_drop(self.conv_pool(X))
        X_s = self.series_drop(self.conv_series(X))
        output = X + X_s
        output = self.conv_pool2(output)
        output = tf.transpose(output, perm = [0,2,1])
        output = self.lstm(output)

        # Calculate the final prediction `output`.
        output = self.fc(self.fc2(output))

        return output


