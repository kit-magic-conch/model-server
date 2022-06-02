import librosa


def load_feature_tuple(file_name, n_mfcc):
    y, sr = librosa.load(file_name)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    return mfcc, spectral_centroid, spectral_rolloff
