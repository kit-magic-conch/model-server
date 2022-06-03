import numpy as np
from keras.models import load_model

from prepare_data import extract_features


def predict_one(file_name, model_path, n_mfcc, input_shape, label_classes):
    try:
        prediction_features = extract_features(file_name, n_mfcc, input_shape[0])
    except:
        return label_classes[1]
    model = load_model(model_path)

    predicted_proba_vector = model.predict(prediction_features)
    predicted_class_vector = np.argmax(predicted_proba_vector, axis=1)
    unique, counts = np.unique(predicted_class_vector, return_counts=True)
    count = dict(zip(unique, counts))
    return label_classes[max(count, key=count.get)]
