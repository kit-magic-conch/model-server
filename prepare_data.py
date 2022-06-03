import numpy as np
from templates import load_feature_tuple


def extract_features(file_name, n_mfcc, feature_width):
    full_feature = np.concatenate(load_feature_tuple(file_name, n_mfcc), axis=0)
    if full_feature.shape[1] < feature_width:
        raise

    features = np.empty((0, feature_width, full_feature.shape[0]))

    for i in range(0, full_feature.shape[1], 500):
        feature = np.transpose(full_feature[:, i:i+150])
        feature = feature.reshape((1,) + feature.shape)
        features = np.append(features, feature, axis=0)

    return features
