import numpy as np
import librosa
from keras.models import load_model


n_mfcc = 128
num_rows = 150
num_columns = 130

input_shape = (num_rows, num_columns)


def pad_features(feature):
    pad_width = num_rows - feature.shape[1]
    if pad_width > 0:
        feature = np.pad(feature, pad_width=((0, 0), (0, pad_width)), mode='constant')
    elif pad_width < 0:
        feature = np.delete(feature, np.arange(num_rows, feature.shape[1]), axis=1)
    return feature


def extract_features(file_name):
    try:
        y, sr = librosa.load(file_name)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

        image = pad_features(mfcc)
        image = np.append(image, pad_features(spectral_centroid), axis=0)
        image = np.append(image, pad_features(spectral_rolloff), axis=0)
        image = np.transpose(image)

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        print(e)
        return None

    return image


def predict_one(file_name):
    prediction_feature = extract_features(file_name).reshape((1, ) + input_shape)
    model = load_model('saved_models/model.h5')

    predicted_proba = model.predict(prediction_feature)[0]
    predicted_class = np.argmax(predicted_proba)

    return predicted_class


def recognize_emotion(filename):
    return ['NEGATIVE', 'NEUTRAL', 'POSITIVE'][predict_one(filename)]
