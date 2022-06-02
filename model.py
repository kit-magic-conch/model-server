from use_model import predict_one


n_mfcc = 128
num_rows = feature_width = 150
num_columns = 130
input_shape = (num_rows, num_columns)


def recognize_emotion(filename):
    return predict_one(filename, 'saved_models/model.h5', n_mfcc, input_shape, ['NEGATIVE', 'NEUTRAL', 'POSITIVE'])
