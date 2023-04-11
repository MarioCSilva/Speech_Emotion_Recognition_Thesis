from xmlrpc.client import Boolean
import librosa
import io
import numpy as np
from PIL import Image
from typing import List, Union
from joblib import load
import noisereduce as nr
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
from keras.utils import img_to_array
import tensorflow as tf


class SERClassifier:
    def __init__(self, traditional_ser: bool = True, stratified: bool = False) -> None:
        """
        Constructor for SERClassifier class.
        Load the SER classifier model from a file and initialize label dictionaries.

        Args:
            traditional_ser (bool): Define the traditional model for classying segments
                                    if True, or the deep learning model if False.
            stratified (bool): Set to use the models resulting of the data stratification study.
        """
        self.traditional_ser = traditional_ser
        if traditional_ser:
            self.model = load(
                f"C:/Users/Chico/Desktop/VADER/Audio_Sentiment_Analysis/ser_tools/{'stratified_' if stratified else ''}traditional_model.pkl")
        else:
            self.model = tf.keras.models.load_model(
                f"C:/Users/Chico/Desktop/VADER/Audio_Sentiment_Analysis/ser_tools/{'stratified_' if stratified else ''}dl_model.h5")

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

    def extract_trad_features(self, data: Union[str, np.ndarray], is_file_name: bool = True, sr: int = 16000) -> List:
        """
        Extract the SER traditional features from the given audio file or signal.

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

    def extract_dl_features(self, data: Union[str, np.ndarray], is_file_name: bool = True, sr: int = 16000) -> List:
        """
        Extract the SER deep learning features from the given audio file or signal.

        Args:
            data (Union[str, np.ndarray]): Audio file name or signal.
            is_file_name (bool): Whether the input data is a file name or signal.
            sr (int): Sampling rate of the audio signal.

        Returns:
            List: Spectrogram image array with (1, 224, 224, 3) shape.
        """
        if is_file_name:
            y, sr = librosa.load(data, sr=16000)
        else:
            y = data

        y = self.preprocess_audio(y, sr)

        fig = plt.figure(dpi=100)
        ax = fig.add_subplot()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        spec = librosa.amplitude_to_db(
            np.abs(librosa.stft(y,  n_fft=2048, hop_length=512)), ref=np.max)
        librosa.display.specshow(
            spec, sr=sr, hop_length=512, ax=ax, cmap="viridis_r")
        buf = io.BytesIO()
        fig.savefig(buf, bbox_inches='tight', pad_inches=0, dpi=100)
        buf.seek(0)
        fig.clf()
        plt.close(fig)
        img = Image.open(buf).convert('RGB').resize((224, 224), Image.NEAREST)

        return tf.keras.applications.resnet50.preprocess_input(
            img_to_array(img)).reshape((1, 224, 224, 3))

    def predict(self, audio_file):
        """
        Predicts the emotion label for a given audio file.

        Args:
            audio_file (str): The path to the audio file to predict.

        Returns:
            str: The predicted emotion label (one of "neutral", "anger", "happiness", "sadness").
        """
        return self.labels_id[self.model.predict(self.extract_trad_features(audio_file))[0]]

    def predict_proba(self, audio_file):
        """
        Predicts the probabilities of each emotion label for a given audio file.

        Args:
            audio_file (str): The path to the audio file to predict.

        Returns:
            dict: A dictionary containing the predicted probabilities for each emotion label ("neutral", "anger", "happiness", "sadness").
        """
        proba = self.model.predict_proba(
            self.extract_trad_features(audio_file))[0]
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
        if self.traditional_ser:
            proba = self.model.predict_proba(
                self.extract_trad_features(segment, is_file_name=False))[0]
        else:
            proba = self.model.predict(
                self.extract_dl_features(segment, is_file_name=False), verbose=0)[0]

        proba = {"anger": proba[0], "happiness": proba[1],
                 "sadness": proba[2], "neutral": proba[3]}

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
