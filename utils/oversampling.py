import numpy as np

# ---------------------------------------------------------
# Oversampling Utilities
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

def targeted_oversample(X, y, target_class, factor=2):
    """Oversample a specific class by a given factor."""
    idxs = np.where(y == target_class)[0]
    additional_samples = np.random.choice(idxs, size=(len(idxs) * (factor - 1)), replace=True)
    X_new = np.concatenate([X, X[additional_samples]], axis=0)
    y_new = np.concatenate([y, y[additional_samples]], axis=0)

    # Shuffle the dataset
    perm = np.random.permutation(len(X_new))
    X_new = X_new[perm]
    y_new = y_new[perm]

    return X_new, y_new

def balanced_oversample(X, y, min_classes=None):
    """Balance all classes to have at least min_classes samples."""
    classes, counts = np.unique(y, return_counts=True)
    if min_classes is None:
        min_classes = counts.max()

    X_res, y_res = [], []
    for c, count_c in zip(classes, counts):
        idxs = np.where(y == c)[0]
        X_res.append(X[idxs])
        y_res.append(y[idxs])

        if count_c < min_classes:
            diff = min_classes - count_c
            add_idxs = np.random.choice(idxs, size=diff, replace=True)
            X_res.append(X[add_idxs])
            y_res.append(y[add_idxs])

    X_res = np.concatenate(X_res, axis=0)
    y_res = np.concatenate(y_res, axis=0)

    # Shuffle the dataset
    perm = np.random.permutation(len(X_res))
    X_res = X_res[perm]
    y_res = y_res[perm]

    return X_res, y_res