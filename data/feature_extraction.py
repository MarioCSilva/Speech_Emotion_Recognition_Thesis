import sys
import os
# go to upper diretory
sys.path.append(os.path.abspath('./../../'))
import glob
import librosa
from tqdm import tqdm
import numpy as np
import librosa.display
import moviepy.editor as mp
from Audio_Sentiment_Analysis.utils.Configuration import Configuration
import csv

AUDIO_DIR = f"{os.path.abspath('./../../')}/eNTERFACE05_Dataset/*/*/*/*.avi"
EXTRACTED_FEATURES_FILE = 'extracted_features_ent05.csv'
CONFIG_FILE = f"{os.path.abspath('./../../')}/Audio_Sentiment_Analysis/data/config.json"

def extract_features(audio_file, subject, emotion):
    y, sr = librosa.load(audio_file, res_type='kaiser_fast')

    file = audio_file.split(".")[-2].split("\\")[-1]
    rms = librosa.feature.rms(y)
    chroma_stft = librosa.feature.chroma_stft(y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y, sr=sr)
    spec_cont = librosa.feature.spectral_contrast(y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y, sr=sr)
    mel_spect = librosa.feature.melspectrogram(y, sr=sr, n_mels=config.n_mels)

    features_str = f'{file} {subject} {emotion} {np.mean(mel_spect)} {np.min(mel_spect)} {np.max(mel_spect)} {np.var(mel_spect)} {np.std(mel_spect)}\
        {np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_cont)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'

    for e in mfcc:
        features_str += f' {np.mean(e)}'

    return features_str.split()


def process_data(audio_dir, proc_feat_dataset):
    # Create a CSV for storing all processed features and write the header
    header = 'File Subject Emotion mel_mean mel_min mel_max mel_var mel_std chroma_stft rmse spectral_centroid spectral_contrast spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header = header.split()
    file = open(proc_feat_dataset, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(header)

    print("Processing audio files from all subjects:")
    for file_path in tqdm(glob.glob(audio_dir)):
        labels =  file_path.split('\\')
        # labels =  file_path.split('/')
        subject = labels[-4].split()[1]
        emotion = labels[-3]
        file_type = labels[-1].split('.')[-1]

        if file_type == 'avi':
            audio_file_path = file_path[:-3] + 'wav'
            # convert file type to wav
            if not os.path.isfile(audio_file_path):
                audio_clip = mp.VideoFileClip(file_path)
                audio_clip.audio.write_audiofile(audio_file_path)
            file_path = audio_file_path

        processed_data = extract_features(file_path, subject, emotion)
        writer.writerow(processed_data)


if __name__ == "__main__":
    global config
    config = Configuration.load_json(CONFIG_FILE)

    # extract features from audio files and store them in a dataset
    process_data(AUDIO_DIR, EXTRACTED_FEATURES_FILE)
