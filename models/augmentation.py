import numpy as np

def jitter(x, std=0.01):
    return x + np.random.normal(0, std, x.shape).astype(np.float32)

def gaussian_noise(x, std=0.01):
    return x + np.random.normal(0, std, x.shape).astype(np.float32)

def timeshift(x, max_shift=5):
    length = x.shape[0]
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift > 0:
        x = np.pad(x, ((shift, 0), (0, 0)), mode='constant')[:length]
    elif shift < 0:
        x = np.pad(x, ((0, -shift), (0, 0)), mode='constant')[-length:]
    return x

def crop(x, crop_min=0.8):
    crop_ratio = np.random.uniform(crop_min, 1.0)
    length = x.shape[0]
    new_length = int(crop_ratio * length)
    start = np.random.randint(0, length - new_length + 1)
    cropped = x[start:start + new_length]
    return np.pad(cropped, ((0, length - new_length), (0, 0)), mode='constant')

def random_augment(x):
    if np.random.rand() < 0.5: x = jitter(x)
    if np.random.rand() < 0.5: x = gaussian_noise(x)
    if np.random.rand() < 0.5: x = timeshift(x)
    if np.random.rand() < 0.5: x = crop(x)
    return x
