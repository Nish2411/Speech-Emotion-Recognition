import numpy as np
import librosa


TARGET_SAMPLE_RATE = 48000
MIN_AUDIO_SAMPLES = 2048


def load_audio(file_name, sample_rate=TARGET_SAMPLE_RATE):
    audio, _ = librosa.load(file_name, sr=sample_rate, mono=True)

    # Remove leading/trailing silence so live recordings look more like training clips.
    audio, _ = librosa.effects.trim(audio, top_db=25)

    if len(audio) < MIN_AUDIO_SAMPLES:
        return None, sample_rate

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    return audio.astype(np.float32), sample_rate


def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    audio, sample_rate = load_audio(file_name)
    if audio is None:
        return None

    result = np.array([], dtype=np.float32)
    stft = np.abs(librosa.stft(audio))

    if mfcc:
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        result = np.hstack((result, np.mean(mfccs.T, axis=0)))

    if chroma:
        chroma_feature = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        result = np.hstack((result, np.mean(chroma_feature.T, axis=0)))

    if mel:
        mel_feature = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        result = np.hstack((result, np.mean(mel_feature.T, axis=0)))

    return result.astype(np.float32)
