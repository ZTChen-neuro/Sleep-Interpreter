#!/usr/bin/env python3
import tensorflow as tf
import tensorflow.keras as K
from keras import regularizers
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))

__all__ = [
    "LossLayer",
]

class LossLayer(K.layers.Layer):
    """
    `LossLayer` layer used to calculate contrastive loss.
    """

    def __init__(self, d_contra, loss_mode, **kwargs):
        """
        Initialize `LossLayer` object.

        Args:
            d_contra: int - The dimension of contrastive space after projection layer.
            loss_mode: str - The mode of loss calculation.
            kwargs: The arguments related to initialize `tf.keras.layers.Layer`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `K.layers.Layer`
        # style model and inherit it's functionality.
        super(LossLayer, self).__init__(**kwargs)

        # Initialize parameters.
        self.d_contra = d_contra; self.loss_mode = loss_mode

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
        # Initialize `y_shape` & `z_shape`.
        y_shape, z_shape, true_label_shape = input_shape
        # Initialize temperature variables according to `loss_mode`.
        if self.loss_mode == "clip":
            self.tau = tf.Variable(0.25, trainable=False, name="tau")
        elif self.loss_mode == "clip_orig":
            self.t = tf.Variable(2.0, trainable=True, name="t")
        # Initialize projection layers for `Y` & `Z`.
        self.proj_y = K.models.Sequential(name="proj_y")
        self.proj_y.add(K.layers.Dense(
            # Modified `Dense` layer parameters.
            self.d_contra, activation="gelu", kernel_initializer="he_uniform",#"gelu"
            kernel_regularizer=K.regularizers.l2(l2=0.01),
            # Defaullt `Dense` layer parameters.
            use_bias=True, bias_initializer="zeros", bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None, bias_constraint=None
        ))
        self.proj_y.add(K.layers.Flatten())
        self.proj_z = K.models.Sequential(name="proj_z")
        self.proj_z.add(K.layers.Flatten())
        # Build super to set up `K.layers.Layer`-style model and inherit it's network.
        super(LossLayer, self).build(input_shape)

    # def call func
    def call(self, inputs):
        """
        Forward layers in `LossLayer` to get the final result.

        Args:
            inputs: (2[list],) - The input data, including [Y,Z].

        Returns:
            loss: tf.float32 - The corresponding contrastive loss.
            prob_matrix: (batch_size, batch_size) - The un-normalized probability matrix.
        """
        # Initialize `Y` & `Z` from `inputs`.
        Z, Y, true_label = inputs
        # Use `proj_*` layers to get the embeddings.
        # emb_[y,z] - (batch_size, d_contra)
        emb_y = tf.linalg.normalize(self.proj_y(Y), ord="euclidean", axis=1)[0]
        emb_z = tf.linalg.normalize(self.proj_z(Z), ord="euclidean", axis=1)[0]
        # Calculate `loss` and related matrices according to `loss_mode`.
        if self.loss_mode == "clip":
            # Calculate `loss_matrix` from `emb_y` and `emb_z`.
            loss_matrix = tf.cast(tf.exp(tf.matmul(emb_z, tf.transpose(emb_y)) / self.tau), dtype = tf.float32)
            # Calculate `loss_z` & `loss_y` from `loss_matrix`, which is `z`x`y`.
            # loss_[y,z] - (batch_size,), loss - tf.float32
            labels = tf.matmul(true_label, tf.transpose(true_label))
            labels = labels / tf.reduce_sum(labels, axis=-1)
            loss_z = tf.squeeze(tf.subtract(tf.math.log(tf.reduce_sum(loss_matrix, axis=0, keepdims=True)),
                tf.math.log(tf.reduce_sum(tf.multiply(loss_matrix, labels), axis=0, keepdims=True))))
            loss_y = tf.squeeze(tf.subtract(tf.math.log(tf.reduce_sum(loss_matrix, axis=1, keepdims=True)),
                tf.math.log(tf.reduce_sum(tf.multiply(loss_matrix, labels), axis=1, keepdims=True))))
            loss = (tf.reduce_mean(loss_z) + tf.reduce_mean(loss_y)) / 2
        elif self.loss_mode == "clip_orig":
            # Calculate `loss_matrix` from `emb_y` and `emb_z`.
            # loss_matrix - (batch_size, batch_size)
            loss_matrix = tf.matmul(emb_z, tf.transpose(emb_y)) * tf.exp(self.t)
            # Calculate `loss_z` & `loss_y` from `loss_matrix`, which is `z`x`y`.
            # loss_[y,z] - (batch_size,), loss - tf.float32
            labels = tf.matmul(true_label, tf.transpose(true_label))
            labels = labels / tf.reduce_sum(labels, axis=-1)
            loss_z = tf.nn.softmax_cross_entropy_with_logits(logits=loss_matrix, labels=labels, axis=0)
            
            loss_y = tf.nn.softmax_cross_entropy_with_logits(logits=loss_matrix, labels=labels, axis=1)
            
            loss = (tf.reduce_mean(loss_z) + tf.reduce_mean(loss_y)) / 2
        # Return the final `loss` & `prob_matrix`.
        return loss, loss_matrix

