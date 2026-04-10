#!/usr/bin/env python3
import copy as cp
import tensorflow as tf
import tensorflow.keras as K
import os
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))
    from multi_encoder import *
else:
    from .multi_encoder import *
import utils.model

__all__ = [
    "SIMD",
]

class SIMD(K.Model):
    """
    `SIMD` model, with considering time information.
    """

    def __init__(self, finetune=False,**kwargs):
        """
        Initialize `SIMD` object.

        Args:
            kwargs: The arguments related to initialize `tf.keras.Model`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `K.Model`
        # style model and inherit it's functionality.
        super(SIMD, self).__init__(**kwargs)
        self.individual_models = []
        self.ensemble_models = 5
        self.finetune = finetune
        for i in range(self.ensemble_models):
            model_item = multi_encoder(self.finetune)
            self.individual_models.append(model_item)


    """
    network funcs
    """
    # def call func
    def call(self, inputs, training=None, mask=None):
        """
        Forward `SIMD` to get the final predictions.

        Args:
            inputs: tuple - The input data.
            training: Boolean or boolean scalar tensor, indicating whether to run
                the `Network` in training mode or inference mode.
        Returns:
            outputs: (batch_size, n_labels) - The output labels.
            loss: tf.float32 - The corresponding loss.
        """
        # Initialize components of inputs.
        image_data = inputs[0] ; audio_data = inputs[1] ; sleep_data = inputs[2] ; true_label = inputs[3]
        result_matrix = []; Loss = []
        for model in self.individual_models:
            loss,sleep_pred = model((image_data,audio_data,sleep_data,true_label))
            result_matrix.append(sleep_pred)
            Loss.append(loss)
        result_matrix = tf.reduce_mean(result_matrix, axis = 0)
        Loss = tf.reduce_mean(Loss, axis = 0)
        
        result_matrix = result_matrix.numpy()
        Loss = Loss.numpy()
        return Loss,result_matrix


