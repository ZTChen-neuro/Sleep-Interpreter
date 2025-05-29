#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))

__all__ = [
    "MultiHeadAttention",
]

class MultiHeadAttention(K.layers.Layer):
    """
    `MultiHeadAttention` computes the scaled multi-headed attention.
    """

    def __init__(self, n_heads, d_head, dropout_prob, use_bias=True, **kwargs):
        """
        Initialize `MultiHeadAttention` object.

        Args:
            n_heads: int - The number of attention heads.
            d_head: int - The dimensions of attention head.
            dropout_prob: float - The probability of dropout.
            use_bias: bool - The flag indicates whether use bias.

        Returns:
            None
        """
        # First call super class init function to set up `K.layers.Layer`
        # style model and inherit it's functionality.
        super(MultiHeadAttention, self).__init__(**kwargs)

        # Initialize parameters.
        self.n_heads = n_heads
        self.d_head = d_head
        self.dropout_prob = dropout_prob
        self.use_bias = use_bias

    """
    network funcs
    """
    # def build func
    def build(self, input_shape):
        """
        Build the network on the first call of `call`.

        Args:
            input_shape: tuple - The shape of input data, e.g. (batch_size, seq_len, d_input).

        Returns:
            None
        """
        # Note: As we use tuple as input, we cannot initialize `d_input` from `input_shape`.
        # Initialize the query & key & value transformation matrices (perhaps w. bias).
        # W_[q,k,v] - (batch_size, seq_len, d_input) -> (batch_size, seq_len, n_heads, d_head)
        self.W_q = MHAMatrix(n_heads=self.n_heads, d_head=self.d_head, use_bias=self.use_bias)
        self.W_k = MHAMatrix(n_heads=self.n_heads, d_head=self.d_head, use_bias=self.use_bias)
        self.W_v = MHAMatrix(n_heads=self.n_heads, d_head=self.d_head, use_bias=self.use_bias)
        # Initialize the dropout layer.
        self.dropout = K.layers.Dropout(self.dropout_prob, noise_shape=None, seed=None)

    def call(self, embs):
        """
        Forward layers in `MultiHeadAttention` to get the single-head attention result.

        Args:
            embs: tuple - The embeddings containing emb_[q,k,v], each element is (batch_size, seq_len, d_input).

        Returns:
            emb: (batch_size, seq_len, n_heads, d_head) - The single-head attention embedding.
        """
        # Initialize `emb_q` & `emb_k` & `emb_v` from `embs`.
        # emb_[q,k,v] - (batch_size, seq_len, d_input)
        emb_q, emb_k, emb_v = embs
        # Prepare query & key & value for attention computation.
        # emb_[q,k,v] - (batch_size, n_heads, seq_len, d_head)
        emb_q = tf.transpose(self.W_q(emb_q), perm=[0,2,1,3])
        emb_k = tf.transpose(self.W_k(emb_k), perm=[0,2,1,3])
        emb_v = tf.transpose(self.W_v(emb_v), perm=[0,2,1,3])
        # Calculate attention scores from `emb_q` and `emb_k`, and scale the attention scores.
        # attention - (batch_size, n_heads, seq_len, seq_len)
        attention = tf.nn.softmax(tf.matmul(emb_q, tf.transpose(emb_k, perm=[0,1,3,2])) / np.sqrt(self.d_head), axis=-1)
        # Apply dropout to the calculated attention scores.
        attention = self.dropout(attention)
        # Multiple `emb_v` to get the single-head attention embedding.
        # emb - (batch_size, seq_len, n_heads, d_head)
        emb = tf.transpose(tf.matmul(attention, emb_v), perm=[0,2,1,3])
        # Concatenate multiple heads.
        # emb - (batch_size, seq_len, n_heads * d_head)
        emb = tf.reshape(emb, (*emb.shape[:-2], -1))
        # Return the final `emb`.
        return emb

class MHAMatrix(K.layers.Layer):
    """
    `MHAMatrix` model does a linear transformation and splits the vector into given number of heads
    for multi-head attention. This is used to transform key, query, and value vectors.
    """

    def __init__(self, n_heads, d_head, use_bias=True, **kwargs):
        """
        Initialize `MHAMatrix` object.

        Args:
            n_heads: int - The number of attention heads.
            d_head: int - The dimensions of attention head.
            use_bias: bool - The flag indicates whether use bias.

        Returns:
            None
        """
        # First call super class init function to set up `K.layers.Layer`
        # style model and inherit it's functionality.
        super(MHAMatrix, self).__init__(**kwargs)

        # Initialize parameters.
        self.n_heads = n_heads
        self.d_head = d_head
        self.use_bias = use_bias

    """
    network funcs
    """
    def build(self, input_shape):
        """
        Build the network on the first call of `call`.

        Args:
            input_shape: tuple - The shape of input data, e.g. (batch_size, seq_len, d_input).

        Returns:
            None
        """
        # Initialize `d_input` from `input_shape`.
        self.d_input = input_shape[-1]
        # Initialize the transformation matrix (perhaps w. bias).
        # W - (batch_size, seq_len, d_input) -> (batch_size, seq_len, n_heads * d_head)
        self.W = K.layers.Dense(
            # Modified `Dense` layer parameters.
            self.n_heads * self.d_head, use_bias=self.use_bias,
            kernel_initializer=K.initializers.random_normal(mean=0., stddev=0.01),
            bias_initializer=K.initializers.constant(value=0.01),
            # Default `Dense` layer parameters.
            activation=None, kernel_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None, bias_constraint=None
        )

    # def call func
    def call(self, emb):
        """
        Forward layers in `MHAMatrix` to get the linear transformed result.

        Args:
            emb: (batch_size, seq_len, d_input) - The input embedding.

        Returns:
            emb: (batch_size, seq_len, n_heads, d_head) - The linear transformed embedding.
        """
        # Get the shape of head from `emb`.
        # head_shape - tuple, should be (batch_size, seq_len)
        head_shape = emb.shape[:-1]
        # Linearly transform `emb` using `W`.
        # emb - (batch_size, seq_len, n_heads, d_head)
        emb = tf.reshape(self.W(emb), (*head_shape, self.n_heads, self.d_head))
        # Return the final `emb`.
        return emb

if __name__ == "__main__":
    # Initialize macros.
    batch_size = 64; seq_len = 600; d_input = 256; n_heads = 16; d_head = 64; dropout_prob = 0.4; use_bias = True
    # Instantiate `MultiHeadAttention`.
    mha_inst = MultiHeadAttention(n_heads=n_heads, d_head=d_head, dropout_prob=dropout_prob, use_bias=use_bias)
    # Forward `mha_inst` with random input.
    emb = mha_inst((tf.random.uniform((batch_size, seq_len, d_input)) for _ in range(3)))

