from Audio_Sentiment_Analysis.ser_tools.ser_classifier import SERClassifier
import torch
import numpy as np
import librosa
import sys
import os
sys.path.append(os.path.abspath('./../../../'))


class SERPipeline:
    def __init__(
        self,
        FORMAT=np.float32,  # Data type of audio samples
        SAMPLE_RATE=16000,  # Sample rate of audio
        NO_CHANNELS=1,  # Number of audio channels (1 for mono, 2 for stereo)
        MIN_CONFIDENCE=0.6,  # Minimum confidence level for voice activity detection
        MIN_DURATION=1,  # Minimum duration of speech segments (in seconds)
        MAX_DURATION=6  # Maximum duration of speech segments (in seconds)
    ):
        """
        Initializes an instance of SERPipeline.

        Args:
            FORMAT (numpy.dtype): Data type of audio samples (default: np.float32).
            SAMPLE_RATE (int): Sample rate of audio (in Hz) (default: 16000).
            NO_CHANNELS (int): Number of audio channels (1 for mono, 2 for stereo) (default: 1).
            MIN_CONFIDENCE (float): Minimum confidence level for voice activity detection (default: 0.6).
            MIN_DURATION (float): Minimum duration of speech segments (in seconds) (default: 1).
            MAX_DURATION (float): Maximum duration of speech segments (in seconds) (default: 6).
        """
        # Set parameters
        self.FORMAT = FORMAT
        self.SAMPLE_RATE = SAMPLE_RATE
        self.NO_CHANNELS = NO_CHANNELS
        self.MIN_CONFIDENCE = MIN_CONFIDENCE
        self.MIN_DURATION = MIN_DURATION
        self.MAX_DURATION = MAX_DURATION
        self.STEP = 1

        # Initialize variables for segmentation
        self.current_y, self.prev_start, self.prev_end = None, None, None
        self.start = 0

        # Load models
        self.vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        self.classifier_model = SERClassifier()

    def process_bytes(self, y):
        """
        Converts raw audio bytes to a numpy array.

        Args:
            y (bytes): Raw audio data in bytes.

        Returns:
            numpy.ndarray: Numpy array of audio data.
        """
        # Convert bytes to numpy array
        y = np.frombuffer(y, self.FORMAT)
        if self.FORMAT == np.int32:
            abs_max = np.abs(y).max()
            y = y.astype('float32')
            if abs_max > 0:
                y *= 1/abs_max
            y = y.squeeze()
        elif self.FORMAT != np.float32:
            y = y.astype('float32')
        return y

    def normalize_audio(self, y):
        """
        Resamples and converts audio to mono if necessary.

        Args:
            y (numpy.ndarray): Numpy array of audio data.

        Returns:
            numpy.ndarray: Resampled and mono audio data.
        """
        # Resample and convert to mono if necessary
        if self.SAMPLE_RATE != 16000:
            y = librosa.resample(y, self.SAMPLE_RATE, 16000)
        if self.NO_CHANNELS != 1:
            y = librosa.to_mono(y)
        return y

    def consume(self, binary_audio):
        """
        Processes a segment of raw audio data and returns the predicted emotion probabilities.

        Args:
            binary_audio (bytes): Raw audio data in bytes.

        Returns:
            dict or None: A dictionary of emotion probabilities, or None if not enough audio data has been processed.
        """
        # Initialize emotion probabilities
        emotion_prob = None

        # Update start and end times
        self.start += self.STEP
        end = self.start + self.STEP

        # Convert binary audio to numpy array
        y = self.process_bytes(binary_audio)

        # Normalize audio
        y = self.normalize_audio(y)

        # Perform voice activity detection
        confidence = self.vad_model(
            torch.from_numpy(y), self.SAMPLE_RATE).item()

        if confidence >= self.MIN_CONFIDENCE:
            # If confident voiced speech detected, add to current segment
            if self.prev_end == None:
                self.current_y = y
                self.prev_start, self.prev_end = self.start, end
            else:
                self.current_y = np.append(self.current_y, y)
                self.prev_end = end
                # If current segment exceeds maximum duration, classify segment
                if (self.prev_end - self.prev_start) >= self.MAX_DURATION:
                    emotion_prob = self.classifier_model.predict_segment(
                        self.current_y,
                        return_proba=True
                    )
                    self.prev_start, self.prev_end, self.current_y = None, None, None
        elif self.prev_end:
            # If voiced speech stops and
            # the previous segment duration exceeds minimum duration,
            # classify the current segment
            if (self.prev_end - self.prev_start) > self.MIN_DURATION:
                emotion_prob = self.classifier_model.predict_segment(
                    self.current_y,
                    return_proba=True
                )
            self.prev_start, self.prev_end, self.current_y = None, None, None

        # Return emotion probabilities
        return emotion_prob
