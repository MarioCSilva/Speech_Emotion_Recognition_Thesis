import numpy as np
from typing import List, Union
from joblib import load
import librosa
import noisereduce as nr
from scipy.stats import kurtosis


class SERClassifier:
    def __init__(self, SER_CLASSIFIER_MODEL: str = 'traditional_model.pkl') -> None:
        """
        Constructor for SERClassifier class.
        Load the SER classifier model from a file and initialize label dictionaries.

        Args:
            SER_CLASSIFIER_MODEL (str): Path to the SER classifier model file.
        """
        self.model = load(SER_CLASSIFIER_MODEL)
        self.labels = {"neutral": 0, "anger": 0, "happiness": 0, "sadness": 0}
        self.labels_id = {0: "anger", 1: "happiness",
                          2: "sadness", 3: "neutral"}

    def spikes(self, data: np.ndarray) -> float:
        """
        Compute the proportion of spikes (values above a threshold) in the given data.

        Args:
            data (np.ndarray): Input data.

        Returns:
            float: Proportion of spikes in the input data.
        """
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

    def preprocess_audio(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply noise reduction and trimming to the given audio signal.

        Args:
            y (np.ndarray): Audio signal.
            sr (int): Sampling rate of the audio signal.

        Returns:
            np.ndarray: Preprocessed audio signal.
        """
        y = nr.reduce_noise(y=y, sr=sr, n_fft=2048, hop_length=512,
                            prop_decrease=.75, time_constant_s=1)
        y, _ = librosa.effects.trim(y, top_db=30)
        return y

    def extract_features(self, data: Union[str, np.ndarray], is_file_name: bool = True, sr: int = 16000) -> List:
        """
        Extract the SER features from the given audio signal.

        Args:
            data (Union[str, np.ndarray]): Audio file name or signal.
            is_file_name (bool): Whether the input data is a file name or signal.
            sr (int): Sampling rate of the audio signal.

        Returns:
            List: List of feature values.
        """
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
                         np.percentile(chroma_stft, 0.25), self.spikes(
                             chroma_stft), np.mean(spec_bw), np.max(spec_bw),
                         np.percentile(librosa.feature.rms(y=y), 0.25), np.var(
                             mfcc[0]), np.var(mfcc[2]), np.max(mfcc[4]),
                         np.var(mfcc[4]), np.median(mfcc[4]), self.spikes(
                             mfcc[5]), np.percentile(mfcc[6], 0.75),
                         np.max(mfcc[6]), np.var(mfcc[7]), np.sum(mfcc[9]), np.max(
            mfcc[9]), np.percentile(mfcc[10], 0.75),
            np.max(mfcc[10]), np.sum(mfcc[11]), kurtosis(
                mfcc[11]), np.mean(mfcc[12]), np.mean(mfcc[14]),
            self.spikes(mfcc[15]), kurtosis(mfcc[16]), np.mean(
                mfcc[16]), kurtosis(mfcc[17]),
            self.spikes(mfcc[18]), np.mean(mfcc[18]), np.mean(mfcc[19])], np.float64).reshape(1, -1)

    def predict(self, audio_file):
        """
        Predicts the emotion label for a given audio file.

        Args:
            audio_file (str): The path to the audio file to predict.

        Returns:
            str: The predicted emotion label (one of "neutral", "anger", "happiness", "sadness").
        """
        return self.labels_id[self.model.predict(self.extract_features(audio_file))[0]]

    def predict_proba(self, audio_file):
        """
        Predicts the probabilities of each emotion label for a given audio file.

        Args:
            audio_file (str): The path to the audio file to predict.

        Returns:
            dict: A dictionary containing the predicted probabilities for each emotion label ("neutral", "anger", "happiness", "sadness").
        """
        proba = self.model.predict_proba(self.extract_features(audio_file))[0]
        return {"neutral": proba[3], "anger": proba[0], "happiness": proba[1], "sadness": proba[2]}

    def predict_segment(self, segment, return_proba=False):
        """
        Predicts the emotion label or probabilities for a given audio segment.

        Args:
            segment (np.ndarray): A numpy array containing the audio segment.
            return_proba (bool): Whether to return the probabilities instead of the predicted label.

        Returns:
            str or dict: The predicted emotion label (one of "neutral", "anger", "happiness", "sadness") if `return_proba` is False, 
            otherwise a dictionary containing the predicted probabilities for each emotion label ("neutral", "anger", "happiness", "sadness").
        """
        proba = self.model.predict_proba(
            self.extract_features(segment, is_file_name=False))[0]
        proba = {"neutral": proba[3], "anger": proba[0],
                 "happiness": proba[1], "sadness": proba[2]}

        if return_proba:
            return proba
        else:
            return max(proba, key=proba.get)

    def predict_features(self, features):
        """
        Predicts the emotion label for a given feature vector.

        Args:
            features (np.ndarray): A numpy array containing the feature vector.

        Returns:
            str: The predicted emotion label (one of "neutral", "anger", "happiness", "sadness").
        """
        return self.model.predict(features)


if __name__ == "__main__":
    SER_CLASSIFIER_MODEL = 'traditional_model.pkl'
    model = SERClassifier(SER_CLASSIFIER_MODEL)
    audio_file_test = "../../IEMOCAP_Dataset/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav"
    print(model.predict(audio_file_test))
    print(model.predict_proba(audio_file_test))
