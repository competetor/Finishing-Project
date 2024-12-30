import tensorflow as tf
from models.ssl_encoder import ssl_pretrain
from models.wgan_gp import create_generator, create_discriminator, integrate_ssl_weights_naturally, train_wgan_gp
from utils.data_utils import load_and_preprocess_data
from utils.oversampling import simple_oversample
import numpy as np

# ---------------------------------------------------------
# Main Pipeline for Training
# ---------------------------------------------------------

def main_pipeline(data_dir, num_classes, epochs_ssl=100, epochs_wgan=5000, batch_size=32):
    """Pipeline to run SSL pretraining and WGAN-GP training."""

    print("[INFO] Loading and preprocessing data...")
    X_labeled, y_labeled, X_unlabeled, le = load_and_preprocess_data(data_dir)

    print("[INFO] Balancing labeled dataset...")
    X_labeled, y_labeled = simple_oversample(X_labeled, y_labeled)

    X_ssl = tf.concat([X_labeled, X_unlabeled], axis=0) if len(X_unlabeled) > 0 else X_labeled

    print("[INFO] Starting SSL pretraining...")
    ssl_encoder = ssl_pretrain(X_ssl, batch_size=batch_size, epochs=epochs_ssl, embedding_dim=128, temperature=0.07, val_split=0.1)

    # Evaluate SSL encoder here
    # This step can include training a linear classifier on the embeddings if needed

    print("[INFO] Preparing WGAN-GP models...")
    generator = create_generator(time_steps=X_labeled.shape[1], latent_dim=100, num_classes=num_classes, embedding_dim=50)
    discriminator = create_discriminator(time_steps=X_labeled.shape[1], num_classes=num_classes, embedding_dim=50)

    integrate_ssl_weights_naturally(ssl_encoder, generator)

    print("[INFO] Starting WGAN-GP training...")
    dataset = tf.data.Dataset.from_tensor_slices((X_labeled, y_labeled)).shuffle(len(X_labeled)).batch(batch_size)
    train_wgan_gp(generator, discriminator, dataset, epochs=epochs_wgan, gp_weight=10.0, d_steps=5, latent_dim=100)

    print("[INFO] Generating synthetic samples...")
    class_counts = tf.math.bincount(y_labeled)
    max_count = tf.reduce_max(class_counts).numpy()

    X_aug_list = [X_labeled]
    y_aug_list = [y_labeled]

    for cls in range(num_classes):
        diff = max_count - class_counts[cls].numpy()
        if diff > 0:
            noise = tf.random.normal([diff, 100])
            labels = tf.constant([cls] * diff, dtype=tf.int32)
            fake_data = generator([noise, labels], training=False)
            X_aug_list.append(fake_data.numpy())
            y_aug_list.append(labels.numpy())

    X_final = np.concatenate(X_aug_list)
    y_final = np.concatenate(y_aug_list)

    print("[INFO] Training completed.")
    return X_final, y_final, le

if __name__ == "__main__":
    data_dir = "../cow_data"  
    num_classes = 13          # Number of target classes in our dataset
    X_final, y_final, label_encoder = main_pipeline(data_dir, num_classes)
