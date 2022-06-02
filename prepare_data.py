import numpy as np
from templates import load_feature_tuple


def pad_feature_width(feature, feature_width):
    diff = feature_width - feature.shape[1]
    if diff > 0:
        return np.pad(feature, pad_width=((0, 0), (0, diff)), mode='constant')
    elif diff < 0:
        return np.delete(feature, np.arange(feature_width, feature.shape[1]), axis=1)
    return feature


def extract_feature(file_name, n_mfcc, feature_width):
    image = np.concatenate(load_feature_tuple(file_name, n_mfcc), axis=0)
    image = pad_feature_width(image, feature_width)
    image = np.transpose(image)
    return image
