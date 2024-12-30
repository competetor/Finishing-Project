import tensorflow as tf
from tensorflow.keras import layers, models
from models.transformer_utils import transformer_block

# ---------------------------------------------------------
# WGAN-GP Models
# ---------------------------------------------------------

def create_generator(time_steps=100, latent_dim=100, num_classes=10, embedding_dim=50):
    noise_input = tf.keras.Input(shape=(latent_dim,))
    label_input = tf.keras.Input(shape=(), dtype=tf.int32)

    label_emb = layers.Embedding(num_classes, embedding_dim)(label_input)
    label_emb_flat = layers.Flatten()(label_emb)

    combined = layers.Concatenate()([noise_input, label_emb_flat])
    x = layers.Dense(time_steps * 3, activation='relu')(combined)
    x = layers.Reshape((time_steps, 3))(x)

    # Match SSL encoder architecture
    x = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu', name='gen_conv1')(x)
    x = layers.Conv1D(128, kernel_size=5, padding='same', activation='relu', name='gen_conv2')(x)

    # Transformer blocks
    x = transformer_block(x, num_heads=4, ff_dim=128, dropout=0.1)
    x = transformer_block(x, num_heads=4, ff_dim=128, dropout=0.1)

    x = layers.Dense(3)(x)

    model = models.Model([noise_input, label_input], x, name='generator')
    return model

def create_discriminator(time_steps=100, num_classes=10, embedding_dim=50):
    data_input = tf.keras.Input(shape=(time_steps, 3))
    label_input = tf.keras.Input(shape=(), dtype=tf.int32)

    label_emb = layers.Embedding(num_classes, embedding_dim)(label_input)
    label_emb_expanded = tf.tile(tf.expand_dims(label_emb, 1), [1, time_steps, 1])
    combined = layers.Concatenate()([data_input, label_emb_expanded])

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(combined)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1)(x)

    model = models.Model([data_input, label_input], out, name='discriminator')
    return model

# ---------------------------------------------------------
# Gradient Penalty
# ---------------------------------------------------------

def gradient_penalty(discriminator, real_data, fake_data, labels):
    batch_size = tf.shape(real_data)[0]
    alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator([interpolated, labels], training=True)
    grads = gp_tape.gradient(pred, interpolated)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

# ---------------------------------------------------------
# WGAN-GP Training Loop
# ---------------------------------------------------------

class WGAN_GP(tf.keras.Model):
    def __init__(self, generator, discriminator, gp_weight=10.0, d_steps=5):
        super(WGAN_GP, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.gp_weight = gp_weight
        self.d_steps = d_steps

    def compile(self, g_optimizer, d_optimizer):
        super(WGAN_GP, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    def train_step(self, batch_data):
        real_data, real_labels = batch_data
        batch_size = tf.shape(real_data)[0]

        # Train discriminator
        for _ in range(self.d_steps):
            noise = tf.random.normal([batch_size, 100])
            with tf.GradientTape() as d_tape:
                fake_data = self.generator([noise, real_labels], training=True)
                d_real = self.discriminator([real_data, real_labels], training=True)
                d_fake = self.discriminator([fake_data, real_labels], training=True)
                gp = gradient_penalty(self.discriminator, real_data, fake_data, real_labels)
                d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real) + self.gp_weight * gp

            d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        # Train generator
        noise = tf.random.normal([batch_size, 100])
        with tf.GradientTape() as g_tape:
            fake_data = self.generator([noise, real_labels], training=True)
            d_fake = self.discriminator([fake_data, real_labels], training=True)
            g_loss = -tf.reduce_mean(d_fake)

        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}

# ---------------------------------------------------------
# SSL Integration with Generator
# ---------------------------------------------------------

def integrate_ssl_weights_naturally(ssl_encoder, generator):
    # Extract weights from SSL encoder's first two Conv1D layers
    ssl_conv1_weights = ssl_encoder.layers[1].get_weights()
    ssl_conv2_weights = ssl_encoder.layers[2].get_weights()

    # Match them to generator's 'gen_conv1' and 'gen_conv2'
    for layer in generator.layers:
        if layer.name == 'gen_conv1':
            layer.set_weights(ssl_conv1_weights)
        elif layer.name == 'gen_conv2':
            layer.set_weights(ssl_conv2_weights)

# ---------------------------------------------------------
# Utility Function for Training
# ---------------------------------------------------------

def train_wgan_gp(generator, discriminator, dataset, epochs=10000, gp_weight=10.0, d_steps=5, latent_dim=100):
    wgan = WGAN_GP(generator, discriminator, gp_weight=gp_weight, d_steps=d_steps)
    wgan.compile(
        g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9),
        d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
    )

    wgan.fit(dataset, epochs=epochs)
    return wgan
