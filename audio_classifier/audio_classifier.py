import numpy as np
from typing import List
from joblib import load
import librosa
import noisereduce as nr
from scipy.stats import kurtosis


class AudioClassifier:
    def __init__(self, AUDIO_CLASSIFIER_MODEL='traditional_model.pkl') -> None:
        self.model = load(AUDIO_CLASSIFIER_MODEL)
        self.labels = {"neutral": 0, "anger": 0, "happiness": 0, "sadness": 0}
        self.labels_id = {0: "anger", 1: "happiness",
                          2: "sadness", 3: "neutral"}

    def spikes(self, data):
        if len(data.shape) != 1:
            data = np.concatenate(data)
        mean = np.mean(data)
        std = np.std(data)
        threshold = mean + np.abs(std) * 2 / 100
        num_spikes = 0
        for value in data:
            if value >= threshold:
                num_spikes += 1
        num_spikes = num_spikes / len(data)
        return num_spikes

    def preprocess_audio(self, y, sr):
        y = nr.reduce_noise(y=y, sr=sr, n_fft=2048, hop_length=512, prop_decrease=.75, time_constant_s=1)
        y, _ = librosa.effects.trim(y, top_db=30)
        return y

    def extract_features(self, data, is_file_name=True, sr=16000) -> List:
        if is_file_name:
            y, sr = librosa.load(data, sr=16000)
        else:
            y = data

        y = self.preprocess_audio(y, sr)

        mfcc = librosa.feature.mfcc(y=y, sr=sr)

        zcr = librosa.feature.zero_crossing_rate(y=y)
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=32)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)

        return np.array([np.min(zcr), self.spikes(zcr), np.var(mel_spect), self.spikes(mel_spect),
            np.percentile(chroma_stft, 0.25), self.spikes(chroma_stft), np.mean(spec_bw), np.max(spec_bw),
            np.percentile(librosa.feature.rms(y=y), 0.25), np.var(mfcc[0]), np.var(mfcc[2]), np.max(mfcc[4]),
            np.var(mfcc[4]), np.median(mfcc[4]), self.spikes(mfcc[5]), np.percentile(mfcc[6], 0.75),
            np.max(mfcc[6]), np.var(mfcc[7]), np.sum(mfcc[9]), np.max(mfcc[9]), np.percentile(mfcc[10], 0.75),
            np.max(mfcc[10]), np.sum(mfcc[11]), kurtosis(mfcc[11]), np.mean(mfcc[12]), np.mean(mfcc[14]),
            self.spikes(mfcc[15]), kurtosis(mfcc[16]), np.mean(mfcc[16]), kurtosis(mfcc[17]),
            self.spikes(mfcc[18]), np.mean(mfcc[18]), np.mean(mfcc[19])], np.float64).reshape(1, -1)

    def predict(self, audio_file):
        return self.labels_id[self.model.predict(self.extract_features(audio_file))[0]]

    def predict_proba(self, audio_file):
        proba = self.model.predict_proba(self.extract_features(audio_file))[0]
        return {"neutral": proba[3], "anger": proba[0], "happiness": proba[1], "sadness": proba[2]}

    def predict_segment(self, segment, numeric_output=False):
        prediction_numeric = self.model.predict(self.extract_features(segment, is_file_name=False))[0]

        if numeric_output:
            return prediction_numeric
        else:
            return self.labels_id[prediction_numeric]
    

    def predict_features(self, features):
        return self.model.predict(features)


if __name__ == "__main__":
    AUDIO_CLASSIFIER_MODEL = 'traditional_model.pkl'
    model = AudioClassifier(AUDIO_CLASSIFIER_MODEL)
    audio_file_test = "../../IEMOCAP_Dataset/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav"
    print(model.predict(audio_file_test))
    print(model.predict_proba(audio_file_test))
