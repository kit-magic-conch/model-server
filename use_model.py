import numpy as np
from keras.models import load_model

from prepare_data import extract_feature


def predict_one(file_name, model_path, n_mfcc, input_shape, label_classes):
    prediction_feature = extract_feature(file_name, n_mfcc, input_shape[0]).reshape((1, ) + input_shape)
    model = load_model(model_path)

    predicted_proba = model.predict(prediction_feature)[0]
    predicted_class = np.argmax(predicted_proba)

    return label_classes[predicted_class]
