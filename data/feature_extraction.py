import sys
import os
# go to upper diretory
sys.path.append(os.path.abspath('./../../'))
from collections import defaultdict
import glob
import librosa
from tqdm import tqdm
import moviepy.editor as mp
from Audio_Sentiment_Analysis.utils.Configuration import Configuration

print(sys.path[-1])
AUDIO_DIR = f"{sys.path[-1]}/eNTERFACE05_Dataset/*/*/*/*.avi"
CONFIG_FILE = f"{sys.path[-1]}/Audio_Sentiment_Analysis/data/config.json"


def get_audio_files(audio_dir) -> defaultdict:
    emotion_files = defaultdict(list)

    for file_path in glob.glob(audio_dir):
        emotion = file_path.split('/')[-3]

        file_type = file_path.split('.')[-1]

        if file_type == 'avi':
            audio_file_path = file_path[:-3] + 'wav'

            # convert file type to wav
            if not os.path.isfile(audio_file_path):
                audio_clip = mp.VideoFileClip(file_path)
                audio_clip.audio.write_audiofile(audio_file_path)

            file_path = audio_file_path

        emotion_files[emotion].append(file_path)

    return emotion_files


def process_data(emotion_files, config):
    for emotion, files in tqdm(emotion_files.items()):
        # taking just one audio file per emotion for testing purposes
        audio_file = files[0]

        print(emotion)
        print(audio_file)

        x, sr = librosa.load(audio_file)

        spec = librosa.feature.melspectrogram(x, sr=sr, n_mels=config.n_mels)



if __name__ == "__main__":
    config = Configuration.load_json(CONFIG_FILE)

    emotion_files = get_audio_files(AUDIO_DIR)

    process_data(emotion_files, config)
