import numpy as np
from typing import List
from joblib import load
import librosa


class AudioClassifier:
    def __init__(self, AUDIO_CLASSIFIER_MODEL='audio_classifier_model.pkl') -> None:
        self.model = load(AUDIO_CLASSIFIER_MODEL)
        self.labels = {"neutral": 0, "anger": 0, "happiness": 0, "sadness": 0}
        self.labels_id = {0: "anger", 1: "happiness",
                          2: "sadness", 3: "neutral"}

    def spikes_metric(self, data):
        if len(data.shape) != 1:
            data = np.concatenate(data)
        mean = np.mean(data)
        std = np.std(data)
        threshold = mean + std * 2 / 100
        num_spikes = 0
        for value in data:
            if value >= threshold:
                num_spikes += 1

        return num_spikes

    def extract_features(self, data, is_file_name=True, sr=16000) -> List:
        if is_file_name:
            y, sr = librosa.load(data, res_type='kaiser_fast')
        else:
            y = data
        zcr = librosa.feature.zero_crossing_rate(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)

        return np.array([np.std(librosa.feature.chroma_stft(y=y, sr=sr)), np.mean(zcr), np.min(zcr),
                         np.var(librosa.feature.melspectrogram(
                             y=y, sr=sr, n_mels=32)), self.spikes_metric(spec_cent),
                         np.min(spec_cent), np.var(
                             librosa.feature.spectral_bandwidth(y=y, sr=sr)),
                         np.mean(librosa.feature.spectral_contrast(
                             y=y, sr=sr)), np.var(mfcc[0]), np.max(mfcc[0]),
                         np.var(mfcc[1]), np.var(mfcc[2]), np.var(
            mfcc[3]), np.max(mfcc[4]), np.max(mfcc[5]),
            np.min(mfcc[6]), np.var(mfcc[8]), np.max(
                mfcc[8]), np.max(mfcc[9]), np.max(mfcc[12]),
            np.var(mfcc[13]), np.var(mfcc[14]), np.min(mfcc[16]), np.min(mfcc[18])], np.float64).reshape(1, -1)

    def predict(self, audio_file):
        classified_labels = self.labels.copy()
        classified_labels[
            self.labels_id[
                self.model.predict(self.extract_features(audio_file))[0]
            ]
        ] = 1
        return classified_labels
    
    def predict_segment(self, segment, numeric_output=False):
        prediction_numeric = self.model.predict(self.extract_features(segment, is_file_name=False))[0]

        if numeric_output:
            return prediction_numeric
        else:
            return self.labels_id[prediction_numeric]


if __name__ == "__main__":
    AUDIO_CLASSIFIER_MODEL = 'audio_classifier_model.pkl'
    model = AudioClassifier(AUDIO_CLASSIFIER_MODEL)
    audio_file_test = "../../IEMOCAP_Dataset/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav"
    print(model.predict(audio_file_test))
