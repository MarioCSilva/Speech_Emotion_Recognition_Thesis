import IPython.display
import numpy as np
import json

def Audio(audio: np.ndarray, sr: int):
    """
    Use instead of IPython.display.Audio as a workaround for VS Code.
    `audio` is an array with shape (channels, samples) or just (samples,) for mono.
    """

    if np.ndim(audio) == 1:
        channels = [audio.tolist()]
    else:
        channels = audio.tolist()

    return IPython.display.HTML("""
        <script>
            function stopAudio() {
                if (window.audioContext)
                    window.audioContext.close();
            }
            function playAudio(audioChannels, sr) {
                if (window.audioContext)
                    stopAudio()
                const buffer = audioContext.createBuffer(audioChannels.length, audioChannels[0].length, sr);
                for (let [channel, data] of audioChannels.entries()) {
                    buffer.copyToChannel(Float32Array.from(data), channel);
                }
                window.audioContext = new AudioContext();
                const source = audioContext.createBufferSource();
                source.buffer = buffer;
                source.connect(audioContext.destination);
                source.start();
            }
        </script>
        <button onclick="playAudio(%s, %s)">Play</button>
        <button onclick="stopAudio()">Stop</button>
    """ % (json.dumps(channels), sr))