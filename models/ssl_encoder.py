import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
from models.augmentation import random_augment

def nt_xent_loss(z_i, z_j, temperature=0.07):
    z_i = tf.math.l2_normalize(z_i, axis=1)
    z_j = tf.math.l2_normalize(z_j, axis=1)
    batch_size = tf.shape(z_i)[0]

    representations = tf.concat([z_i, z_j], axis=0)
    sim_matrix = tf.matmul(representations, representations, transpose_b=True)

    mask = tf.eye(2 * batch_size)
    mask = 1 - mask
    exp_sim = tf.exp(sim_matrix / temperature) * mask
    log_prob = sim_matrix / temperature - tf.math.log(tf.reduce_sum(exp_sim, axis=1, keepdims=True))

    positives = tf.range(batch_size)
    pos_pairs = tf.concat([tf.stack([positives, positives + batch_size], axis=1),
                           tf.stack([positives + batch_size, positives], axis=1)], axis=0)
    pos_sims = tf.gather_nd(log_prob, pos_pairs)
    loss = -tf.reduce_mean(pos_sims)
    return loss

def create_ssl_encoder(input_shape=(100, 3), embedding_dim=128):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(inputs)
    x = layers.Conv1D(128, kernel_size=5, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(embedding_dim)(x)
    model = models.Model(inputs, outputs, name='ssl_encoder')
    return model

def ssl_pretrain(X, batch_size=64, lr=0.001, epochs=100, embedding_dim=128, temperature=0.07, val_split=0.1):
    n = len(X)
    val_size = int(n * val_split)
    X_train = X[:-val_size]
    X_val = X[-val_size:]

    train_dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(len(X_train)).batch(batch_size, drop_remainder=True)
    val_dataset = tf.data.Dataset.from_tensor_slices(X_val).batch(batch_size, drop_remainder=True)

    encoder = create_ssl_encoder(embedding_dim=embedding_dim)
    optimizer = optimizers.Adam(lr)

    @tf.function
    def augment_batch(x_batch):
        x_i = tf.numpy_function(lambda arr: np.stack([random_augment(x) for x in arr]), [x_batch], tf.float32)
        x_j = tf.numpy_function(lambda arr: np.stack([random_augment(x) for x in arr]), [x_batch], tf.float32)
        x_i.set_shape([batch_size, X.shape[1], X.shape[2]])
        x_j.set_shape([batch_size, X.shape[1], X.shape[2]])
        return x_i, x_j

    @tf.function
    def train_step(x_batch):
        x_i, x_j = augment_batch(x_batch)
        with tf.GradientTape() as tape:
            z_i = encoder(x_i, training=True)
            z_j = encoder(x_j, training=True)
            loss = nt_xent_loss(z_i, z_j, temperature=temperature)
        grads = tape.gradient(loss, encoder.trainable_variables)
        optimizer.apply_gradients(zip(grads, encoder.trainable_variables))
        return loss

    for ep in range(epochs):
        train_losses = [train_step(x_batch).numpy() for x_batch in train_dataset]
        print(f"SSL Epoch {ep + 1}/{epochs}, Train Loss: {np.mean(train_losses):.4f}")

    return encoder
