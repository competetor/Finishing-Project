import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------
# Data Loading and Preprocessing
# ---------------------------------------------------------

def load_and_preprocess_data(data_dir, time_steps=100, stride=10):
    """Load CSV files and preprocess data for SSL and WGAN."""
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    dfs = [pd.read_csv(f) for f in all_files]
    data = pd.concat(dfs, ignore_index=True)

    data['Label'] = data['Label'].astype(str)
    data['Label'] = data['Label'].fillna('BLN')

    # Extract features and labels
    X_raw = data[['AccX', 'AccY', 'AccZ']].values
    labels_raw = data['Label'].values

    # Normalize to [-1, 1]
    X_min, X_max = X_raw.min(), X_raw.max()
    X_norm = 2 * (X_raw - X_min) / (X_max - X_min) - 1

    segments_labeled = []
    labels_labeled = []
    segments_unlabeled = []  # For BLN and unlabeled data

    for start in range(0, len(X_norm) - time_steps, stride):
        segment = X_norm[start:start + time_steps]
        segment_labels = labels_raw[start:start + time_steps]

        unique_lbls, counts_lbls = np.unique(segment_labels, return_counts=True)
        maj_label = unique_lbls[np.argmax(counts_lbls)]

        # If majority label is BLN, treat as unlabeled
        if maj_label == 'BLN':
            segments_unlabeled.append(segment)
        else:
            segments_labeled.append(segment)
            labels_labeled.append(maj_label)

    # Convert to arrays
    X_labeled = np.array(segments_labeled)
    y_labeled = np.array(labels_labeled)
    X_unlabeled = np.array(segments_unlabeled) if len(segments_unlabeled) > 0 else np.empty((0, time_steps, 3))

    # Encode non-BLN labels
    le = LabelEncoder()
    y_labeled = le.fit_transform(y_labeled)

    return X_labeled, y_labeled, X_unlabeled, le

# ---------------------------------------------------------
# Oversampling for Class Imbalance
# ---------------------------------------------------------

def simple_oversample(X, y):
    """Oversample minority classes to match the largest class count."""
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    X_res, y_res = [], []
    for c in classes:
        idxs = np.where(y == c)[0]
        count_c = len(idxs)
        # Add all samples of class c
        X_res.append(X[idxs])
        y_res.append(y[idxs])
        # Oversample if needed
        if count_c < max_count:
            diff = max_count - count_c
            add_idxs = np.random.choice(idxs, size=diff, replace=True)
            X_res.append(X[add_idxs])
            y_res.append(y[add_idxs])
    X_res = np.concatenate(X_res, axis=0)
    y_res = np.concatenate(y_res, axis=0)

    # Shuffle to avoid large identical blocks
    perm = np.random.permutation(len(X_res))
    X_res = X_res[perm]
    y_res = y_res[perm]
    return X_res, y_res

# ---------------------------------------------------------
# SSL Dataset Preparation
# ---------------------------------------------------------

def prepare_ssl_dataset(X_labeled, X_unlabeled):
    """Combine labeled and unlabeled data for SSL."""
    if X_unlabeled.size > 0:
        X_ssl = np.concatenate([X_labeled, X_unlabeled], axis=0)
    else:
        X_ssl = X_labeled
    return X_ssl