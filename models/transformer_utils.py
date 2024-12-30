import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------------------------------------------------
# Transformer Block Utility
# ---------------------------------------------------------

def transformer_block(x, num_heads=4, ff_dim=128, dropout=0.1):
    """Defines a single transformer block with multi-head attention and feed-forward network."""
    # Multi-Head Attention
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1])(x, x)
    attn_output = layers.Dropout(dropout)(attn_output)
    out1 = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

    # Feed-Forward Network
    ffn = models.Sequential([
        layers.Dense(ff_dim, activation='relu'),
        layers.Dense(x.shape[-1])
    ])
    ffn_output = ffn(out1)
    ffn_output = layers.Dropout(dropout)(ffn_output)
    out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    return out2

# ---------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------

def positional_encoding(length, depth):
    """Generates positional encoding for input sequences."""
    depth = depth // 2
    positions = tf.range(length)[:, tf.newaxis]  # (seq, 1)
    depths = tf.range(depth)[tf.newaxis, :] / tf.cast(depth, tf.float32)  # (1, depth)

    angle_rates = 1 / (10000 ** depths)
    angle_rads = positions * angle_rates  # (seq, depth)

    pos_encoding = tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)  # (seq, 2*depth)
    return pos_encoding

class AddPositionalEncoding(layers.Layer):
    """Custom Keras layer to add positional encoding to the input."""
    def __init__(self):
        super(AddPositionalEncoding, self).__init__()

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        d_model = tf.shape(inputs)[2]
        pos_encoding = positional_encoding(seq_len, d_model)
        return inputs + pos_encoding
