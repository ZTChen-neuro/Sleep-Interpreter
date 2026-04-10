#!/usr/bin/env python3
import tensorflow as tf
import tensorflow.keras as K
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))
    from MultiHeadAttention import MultiHeadAttention
    from FeedForward import FeedForward
else:
    from model.N23SISD.sleep_encoder.layers import MultiHeadAttention
    from model.N23SISD.sleep_encoder.layers import FeedForward

__all__ = [
    "TransformerBlock",
]

class TransformerBlock(K.layers.Layer):
    """
    `TransformerBlock` acts as an encoder layer or a decoder layer.
    """

    def __init__(self, n_heads, d_head, mha_dropout_prob, d_ff, ff_dropout_prob, **kwargs):
        """
        Initialize `TransformerBlock` object.

        Args:
            n_heads: int - The number of attention heads in `mha` block.
            d_head: int - The dimensions of attention head in `mha` block.
            mha_dropout_prob: float - The dropout probability in `mha` block.
            d_ff: int - The dimensions of the hidden layer in `ffn` block.
            ff_dropout_prob: float - The dropout probability in `ffn` block.

        Returns:
            None
        """
        # First call super class init function to set up `K.layers.Layer`
        # style model and inherit it's functionality.
        super(TransformerBlock, self).__init__(**kwargs)

        # Initialize parameters.
        self.n_heads = n_heads; self.d_head = d_head; self.mha_dropout_prob = mha_dropout_prob
        self.d_ff = d_ff; self.ff_dropout_prob = ff_dropout_prob

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
        self.d_model = input_shape[-1]
        # Initialize `mha` block.
        # mha - (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads * d_head)
        self.mha = MultiHeadAttention(n_heads=self.n_heads, d_head=self.d_head, dropout_prob=self.mha_dropout_prob)
        # Initialize the fully connected layer after `mha` block.
        # fc_mha - (batch_size, seq_len, n_heads * d_head) -> (batch_size, seq_len, d_model)
        self.fc_mha = K.layers.Dense(
            # Modified `Dense` layer parameters.
            self.d_model, use_bias=True,
            kernel_initializer=K.initializers.random_normal(mean=0., stddev=0.01),
            bias_initializer=K.initializers.constant(value=0.01),
            # Default `Dense` layer parameters.
            activation=None, kernel_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None, bias_constraint=None
        )
        # Initialize the layer-normalization layer after `mha` block.
        self.layernorm_mha = K.layers.LayerNormalization(
            # Modified `LayerNormalization` layer parameters.
            epsilon=1e-5,
            # Default `LayerNormalization` layer parameters.
            axis=-1, center=True, scale=True, beta_initializer="zeros", gamma_initializer="ones",
            beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None
        )
        # Initialize `ffn` block.
        # ffn - (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        self.ffn = FeedForward(d_ff=self.d_ff, dropout_prob=self.ff_dropout_prob)
        # Initialize the layer-normalization layer after `ffn` block.
        self.layernorm_ffn = K.layers.LayerNormalization(
            # Modified `LayerNormalization` layer parameters.
            epsilon=1e-5,
            # Default `LayerNormalization` layer parameters.
            axis=-1, center=True, scale=True, beta_initializer="zeros", gamma_initializer="ones",
            beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None
        )

    # def call func
    def call(self, emb):
        """
        Forward layers in `TransformerBlock` to get the mha-ffn transformed result.

        Args:
            emb: (batch_size, seq_len, d_model) - The input embedding.

        Returns:
            emb: (batch_size, seq_len, d_model) - The mha-ffn transformed embedding.
        """
        # Get the mha transformed embedding.
        # emb - (batch_size, seq_len, d_model)
        emb = self.layernorm_mha(self.fc_mha(self.mha((emb, emb, emb))) + emb)
        # Get the ffn transformed embedding.
        # emb - (batch_size, seq_len, d_model)
        emb = self.layernorm_ffn(self.ffn(emb) + emb)
        # Return the final `emb`.
        return emb

if __name__ == "__main__":
    # Initialize macros.
    batch_size = 64; seq_len = 600; d_model = 256
    n_heads = 16; d_head = 64; mha_dropout_prob = 0.4; d_ff = 1024; ff_dropout_prob = [0.4, 0.4]
    # Instantiate `TransformerBlock`.
    tb_inst = TransformerBlock(n_heads, d_head, mha_dropout_prob, d_ff, ff_dropout_prob)
    # Forward `tb_inst` with random input.
    emb = tb_inst(tf.random.uniform((batch_size, seq_len, d_model)))

