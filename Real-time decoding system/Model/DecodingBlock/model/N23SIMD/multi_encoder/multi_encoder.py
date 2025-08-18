#!/usr/bin/env python3
import copy as cp
import tensorflow as tf
import tensorflow.keras as K
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.getcwd()))
    print(os.path.join(os.getcwd()))
    from layers import *
else:
    from .layers import *

__all__ = [
    "multi_encoder",
]

class multi_encoder(K.Model):
    """
    `multi_encoder` model, with considering time information.
    """

    def __init__(self, mode='normal', **kwargs):
        """
        Initialize `multi_encoder` object.

        Args:
            kwargs: The arguments related to initialize `tf.keras.Model`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `K.Model`
        # style model and inherit it's functionality.
        super(multi_encoder, self).__init__(**kwargs)
        self.n_labels = 15
        self.d_model = 128
        # The maximum length of element sequence.
        self.max_len = 80
        # The depth of encoder.
        self.encoder_depth = 6
        self.basic_depth = 12
        # The number of attention heads.
        self.n_heads = 4
        self.basic_heads = 6
        # The dimensions of attention head.
        self.basic_head = 64
        self.d_head = 64
        # The dropout probability of attention weights.
        self.mha_dropout_prob = 0.
        # The dimensions of the hidden layer in ffn.
        self.d_ff = 1024
        self.basic_ff = 1024
        # The dropout probability of the hidden layer in ffn.
        self.ff_dropout_prob = [0.5, 0.5]
        # The dimensions of the hidden layer in fc block.
        self.d_fc = 128
        # The dropout probability of the hidden layer in fc block.
        self.fc_dropout_prob = 0.
        self.mode = mode
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
        # Initialize the embedding layer for input.
        self.sleep_emb_input = SleepSubjectBlock([204, 256, 256], self.mode)
        self.sleep_emb_pos = PositionEmbedding(200)
        self.image_encoder = K.models.Sequential(layers=[TransformerBlock(
            n_heads=3, d_head=64, mha_dropout_prob=0.,
            d_ff=256, ff_dropout_prob=[0., 0.]) for _ in range(3)
        ], name="encoder")
        self.image_encoder.add(K.layers.BatchNormalization(
            # Default `BatchNormalization` parameters.
            axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer="zeros",
            gamma_initializer="ones", moving_mean_initializer="zeros", moving_variance_initializer="ones",
            beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None
        ))
        self.audio_encoder = K.models.Sequential(layers=[TransformerBlock(
            n_heads=5, d_head=64, mha_dropout_prob=0.,
            d_ff=256, ff_dropout_prob=[0., 0.]) for _ in range(5)
        ], name="encoder")
        self.audio_encoder.add(K.layers.BatchNormalization(
            # Default `BatchNormalization` parameters.
            axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer="zeros",
            gamma_initializer="ones", moving_mean_initializer="zeros", moving_variance_initializer="ones",
            beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None
        ))
        self.sleep_encoder = K.models.Sequential(layers=[TransformerBlock(
            n_heads=self.basic_heads, d_head=self.basic_head, mha_dropout_prob=self.mha_dropout_prob,
            d_ff=self.basic_ff, ff_dropout_prob=self.ff_dropout_prob) for _ in range(self.basic_depth)
        ], name="encoder")
        self.sleep_encoder.add(K.layers.BatchNormalization(
            # Default `BatchNormalization` parameters.
            axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer="zeros",
            gamma_initializer="ones", moving_mean_initializer="zeros", moving_variance_initializer="ones",
            beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None
        ))
        # Initialize fc block.
        self.image_feature_block = K.models.Sequential()
        self.image_feature_block.add(K.layers.Dense(
            # Modified `Dense` layer parameters.
            units=256, activation=None, 
            kernel_initializer=K.initializers.random_normal(mean=0., stddev=0.01),
            bias_initializer=K.initializers.constant(value=0.01),
            # Default `Dense` layer parameters.
            use_bias=True, kernel_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None, bias_constraint=None
        ))
        self.image_feature_block.add(K.layers.BatchNormalization(
            # Default `BatchNormalization` parameters.
            axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer="zeros",
            gamma_initializer="ones", moving_mean_initializer="zeros", moving_variance_initializer="ones",
            beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None
        ))
        self.audio_feature_block = K.models.Sequential()
        self.audio_feature_block.add(K.layers.Dense(
            # Modified `Dense` layer parameters.
            units=256, activation=None, 
            kernel_initializer=K.initializers.random_normal(mean=0., stddev=0.01),
            bias_initializer=K.initializers.constant(value=0.01),
            # Default `Dense` layer parameters.
            use_bias=True, kernel_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None, bias_constraint=None
        ))
        self.audio_feature_block.add(K.layers.BatchNormalization(
            # Default `BatchNormalization` parameters.
            axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer="zeros",
            gamma_initializer="ones", moving_mean_initializer="zeros", moving_variance_initializer="ones",
            beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None
        ))
        self.sleep_feature_block = K.models.Sequential()
        self.sleep_feature_block.add(K.layers.Dense(
            # Modified `Dense` layer parameters.
            units=256, activation=None, 
            kernel_initializer=K.initializers.random_normal(mean=0., stddev=0.01),
            bias_initializer=K.initializers.constant(value=0.01),
            # Default `Dense` layer parameters.
            use_bias=True, kernel_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None, bias_constraint=None
        ))
        self.sleep_feature_block.add(K.layers.BatchNormalization(
            # Default `BatchNormalization` parameters.
            axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer="zeros",
            gamma_initializer="ones", moving_mean_initializer="zeros", moving_variance_initializer="ones",
            beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None
        ))
        
        self.sleep_classification_block = K.models.Sequential(name="fc")
        self.sleep_classification_block.add(K.layers.Dropout(rate=0.5))
        self.sleep_classification_block.add(K.layers.Flatten())
        self.sleep_classification_block.add(K.layers.Dense(
            # Modified `Dense` layer parameters.
            units=128, activation="relu", 
            kernel_initializer=K.initializers.random_normal(mean=0., stddev=0.01),
            bias_initializer=K.initializers.constant(value=0.01),
            # Default `Dense` layer parameters.
            use_bias=True, kernel_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None, bias_constraint=None
        ))
        self.sleep_classification_block.add(K.layers.BatchNormalization(
            # Default `BatchNormalization` parameters.
            axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer="zeros",
            gamma_initializer="ones", moving_mean_initializer="zeros", moving_variance_initializer="ones",
            beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None
        ))
        self.sleep_classification_block.add(K.layers.Dense(
            # Modified `Dense` layer parameters.
            units=self.n_labels,
            kernel_initializer=K.initializers.random_normal(mean=0., stddev=0.01),
                bias_initializer=K.initializers.constant(value=0.01),
                # Default `Dense` layer parameters.
                activation=None, use_bias=True, kernel_regularizer=None, bias_regularizer=None,
                activity_regularizer=None, kernel_constraint=None, bias_constraint=None
        ))
        self.sleep_classification_block.add(K.layers.Softmax(axis=-1))

    """
    network funcs
    """
    # def call func
    def call(self, inputs, training=None, mask=None):
        """
        Forward `multi_encoder` to get the final predictions.

        Args:
            inputs: tuple - The input data.
            training: Boolean or boolean scalar tensor, indicating whether to run
                the `Network` in training mode or inference mode.

        Returns:
            outputs: (batch_size, n_labels) - The output labels.
            loss: tf.float32 - The corresponding loss.
        """
        # Initialize components of inputs.
        sleep_data = inputs

        sleep_emb = self.sleep_emb_pos(self.sleep_emb_input(sleep_data))
        sleep_emb = self.sleep_encoder(sleep_emb)
        sleep_emb_1 = self.audio_encoder(sleep_emb) + sleep_emb
        sleep_emb_2 = self.image_encoder(sleep_emb) + sleep_emb
        
        sleep = self.sleep_feature_block(sleep_emb)
        sleep2 = self.sleep_feature_block(sleep_emb_1)
        sleep3 = self.sleep_feature_block(sleep_emb_2)
        
        # Calculate the binary cross entropy loss.
        # loss - tf.float32
        sleep_pred1 = self.sleep_classification_block(sleep)
        sleep_pred2 = self.sleep_classification_block(sleep2)
        sleep_pred3 = self.sleep_classification_block(sleep3)
        sleep_pred = (sleep_pred1 + sleep_pred2 + sleep_pred3) / 3.0
        
        return sleep_pred
    
