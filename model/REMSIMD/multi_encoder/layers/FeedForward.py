#!/usr/bin/env python3
import tensorflow as tf
import tensorflow.keras as K
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))

__all__ = [
    "FeedForward",
]

class FeedForward(K.layers.Layer):
    """
    Position-wise feedforward network. FFN consists of two fully connected layers.
    Number of dimensions in the hidden layer $d_{ff}$, is generally set to around
    four times that of the token embedding $d_{model}$. So it is sometime also
    called the expand-and-contract network.
    """

    def __init__(self, d_ff, dropout_prob, use_bias=[True, True], use_bias_gate=None, **kwargs):
        """
        Initialize `FeedForward` object.

        Args:
            d_ff: int - The number of features in the hidden layer of the FFN.
            dropout_prob: float - The dropout probability for the hidden layer.
            use_bias: (2[list],) - The flags indicate whether the fully connected layers have a learnable bias.
            use_bias_gate: bool - The flag indicates whether the fully connected layer for the gate have a learnable bias.

        Returns:
            None
        """
        # First call super class init function to set up `K.layers.Layer`
        # style model and inherit it's functionality.
        super(FeedForward, self).__init__(**kwargs)

        # Initialize parameters.
        self.d_ff = d_ff; self.dropout_prob = dropout_prob; self.use_bias = use_bias
        self.is_gated = (use_bias_gate is not None); self.use_bias_gate = use_bias_gate

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
        # Initialize the fully connected layers.
        # fc1 - (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff)
        self.fc1 = K.layers.Dense(
            # Modified `Dense` layer parameters.
            self.d_ff, use_bias=self.use_bias[0], activation="relu",
            kernel_initializer=K.initializers.random_normal(mean=0., stddev=0.01),
            bias_initializer=K.initializers.constant(value=0.01),
            # Default `Dense` layer parameters.
            kernel_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None, bias_constraint=None
        )
        # fc2 - (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        self.fc2 = K.layers.Dense(
            # Modified `Dense` layer parameters.
            self.d_model, use_bias=self.use_bias[1],
            kernel_initializer=K.initializers.random_normal(mean=0., stddev=0.01),
            bias_initializer=K.initializers.constant(value=0.01),
            # Default `Dense` layer parameters.
            activation=None, kernel_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None, bias_constraint=None
        )
        # Initialize the dropout layer.
        self.dropout1 = K.layers.Dropout(self.dropout_prob[0], noise_shape=None, seed=None)
        self.dropout2 = K.layers.Dropout(self.dropout_prob[1], noise_shape=None, seed=None)
        # Initialize the gate layer.
        # gate - (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff)
        self.gate = K.layers.Dense(
            # Modified `Dense` layer parameters.
            self.d_ff, use_bias=self.use_bias_gate,
            kernel_initializer=K.initializers.random_normal(mean=0., stddev=0.01),
            bias_initializer=K.initializers.constant(value=0.01),
            # Default `Dense` layer parameters.
            activation=None, kernel_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None, bias_constraint=None
        ) if self.is_gated else None

    # def call func
    def call(self, emb):
        """
        Forward layers in `FeedForward` to get the MLP transformed result.

        Args:
            emb: (batch_size, seq_len, d_model) - The input embedding.

        Returns:
            emb: (batch_size, seq_len, d_model) - The MLP transformed embedding.
        """
        # Get the activation of the hidden layer.
        # emb - (batch_size, seq_len, d_ff)
        emb = self.fc1(emb) * self.gate(emb) if self.is_gated else self.fc1(emb)
        # Apply dropout the hidden layer.
        emb = self.dropout1(emb)
        # Get the activation of the final layer.
        # emb - (batch_size, seq_len, d_model)
        emb = self.fc2(emb)
        # Apply dropout the final layer.
        emb = self.dropout2(emb)
        # Return the final `emb`.
        return emb

if __name__ == "__main__":
    # Initialize macros.
    batch_size = 64; seq_len = 600; d_model = 256; d_ff = 1024; dropout_prob = [0.4, 0.4]; use_bias = [True, True]
    # Instantiate `FeedForward`.
    ff_inst = FeedForward(d_ff=d_ff, dropout_prob=dropout_prob, use_bias=use_bias)
    # Forward `ff_inst` with random input.
    emb = ff_inst(tf.random.uniform((batch_size, seq_len, d_model)))

