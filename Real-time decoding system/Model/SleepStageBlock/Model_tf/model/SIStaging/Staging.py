#!/usr/bin/env python3
import copy as cp
import tensorflow as tf
import tensorflow.keras as K
import os
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))
    from stager import *
else:
    from .stager import *

__all__ = [
    "Staging",
]

class Staging(K.Model):
    """
    `Staging` model, with considering time information.
    """

    def __init__(self, fs, weights, **kwargs):
        """
        Initialize `Staging` object.

        Args:
            fs: sampling rate of input EEG data
            weights: Model training preference value list for different sleep stages
            kwargs: The arguments related to initialize `tf.keras.Model`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `K.Model`
        # style model and inherit it's functionality.
        super(Staging, self).__init__(**kwargs)
        self.individual_models = []
        self.ensemble_models = 6
        for i in range(self.ensemble_models):
            model_item = stager(fs, weights[i])
            self.individual_models.append(model_item)


    """
    network funcs
    """
    # def call func
    def call(self, inputs, training=None, mask=None):
        """
        Forward `Staging` to get the final predictions.

        Args:
            inputs: tuple - The input data.
            training: Boolean or boolean scalar tensor, indicating whether to run
                the `Network` in training mode or inference mode.
        Returns:
            outputs: (batch_size, n_labels) - The output labels.
            loss: tf.float32 - The corresponding loss.
        """
        # Initialize components of inputs.
        X = inputs
        result_matrix = []
        for model in self.individual_models:
            stage_pred = model((X), training=training)
            result_matrix.append(stage_pred)
        result_matrix = tf.reduce_mean(result_matrix, axis = 0)
        result_matrix = result_matrix.numpy()
        return result_matrix


