#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))

__all__ = [
    "PositionEmbedding",
]

class PositionEmbedding(K.layers.Layer):
    """
    Sinusoidal positional encoding for non-recurrent neural networks.
    """

    def __init__(self, max_len, **kwargs):
        """
        Initialize `PositionEmbedding` object.

        Args:
            max_len: int - The maximum length of the element sequence.
            kwargs: The arguments related to initialize `tf.keras.layers.Layer`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `K.layers.Layer`
        # style model and inherit it's functionality.
        super(PositionEmbedding, self).__init__(**kwargs)

        # Initialize parameters.
        self.max_len = max_len

    """
    network funcs
    """
    # def build func
    def build(self, input_shape):
        """
        Build the network on the first call of `call`.

        Args:
            input_shape: tuple - The shape of input data, e.g. (batch_size, seq_len, d_model).

        Returns:
            None
        """
        # Initialize `d_model` from `input_shape`.
        self.d_model = input_shape[-1]; assert self.d_model % 2 == 0
        # Empty `position_encodings` matrix.
        # position_encodings - (max_len, d_model))
        position_encodings = np.zeros((self.max_len, self.d_model), dtype=np.float32)
        # Get the indexes of available positions (i.e. within `max_len`).
        # position_idxs - (max_len, 1)
        position_idxs = np.expand_dims(np.arange(0, self.max_len, dtype=np.float32), axis=-1)
        # Get the divide term, i.e. $(1e4)*exp(\frac{-2i}{d_model})$.
        # div_term - (d_model//2,)
        div_term = np.exp(np.arange(0, self.d_model, 2, dtype=np.float32) * -(np.log(1e4) / self.d_model))
        # $PE_{p,2i} = sin\Bigg(\frac{p}{10000^{\frac{2i}{d_{model}}}}\Bigg)$.
        position_encodings[:,0::2] = np.sin(position_idxs * div_term)
        # $PE_{p,2i + 1} = cos\Bigg(\frac{p}{10000^{\frac{2i}{d_{model}}}}\Bigg)$
        position_encodings[:,1::2] = np.cos(position_idxs * div_term)
        # Set `position_encodings` as constant, i.e. not trainable.
        self.position_encodings = tf.constant(position_encodings, dtype=tf.float32)

    # def call func
    def call(self, emb):
        """
        Forward layers in `PositionEmbedding` to get the position-embedded result.

        Args:
            emb: (batch_size, seq_len, d_model) - The sequence of elements.

        Returns:
            emb: (batch_size, seq_len, d_model) - The sequence of position-embedded elements.
        """
        # Get the position embeddings `pe` according to the `seq_len`.
        # pos_emb - (seq_len, d_model)
        pos_emb = self.position_encodings[:emb.shape[1],:]
        # Add `pos_emb` to `emb` to get the position-embedded embedding.
        # Note: We have to make sure that `emb` is 0-mean 1-var distribution.
        # If we apply layer normalization over `emb`, `emb` is 0-mean 1/sqrt(d_model)-var
        # distribution, i.e. we have to multiply `emb` with `sqrt(d_model)`.
        emb = emb + tf.expand_dims(pos_emb, axis=0)
        # Return the final `emb`.
        return emb

if __name__ == "__main__":
    # Initialize macros.
    batch_size = 64; max_len = 600; d_model = 256
    # Instantiate `PositionEmbedding`.
    pe_inst = PositionEmbedding(max_len=max_len)
    # Forward `pe_inst` with 0s.
    emb = pe_inst(tf.zeros((batch_size, max_len, d_model), dtype=tf.float32))

