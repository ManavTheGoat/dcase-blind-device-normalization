import numpy as np
import librosa


FEATURE_NAMES = [
    'spectral_centroid',
    'spectral_rolloff_85',
    'noise_floor',
    'spectral_flatness',
    'hf_lf_ratio',
]


def extract_device_features(audio: np.ndarray, sr: int = 22050) -> np.ndarray:
    """
    Extract signal-level features that characterize the recording device.
    These features capture device-induced spectral distortions without
    requiring knowledge of which device was used (blind estimation).

    Returns a feature vector of shape (5,).
    """
    features = []

    # 1. Spectral centroid — consumer mics shift centroid downward (less highs)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features.append(float(np.mean(centroid)))

    # 2. Spectral rolloff at 85% — frequency below which 85% of energy sits
    #    Professional mics have higher rolloff (energy spread to higher freqs)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)
    features.append(float(np.mean(rolloff)))

    # 3. Noise floor — mean energy in top 20 frequency bins of STFT
    #    Consumer mics have higher noise floors
    stft = np.abs(librosa.stft(audio))
    features.append(float(np.mean(stft[-20:, :])))

    # 4. Spectral flatness — 1.0 = white noise, 0.0 = pure tone
    #    Consumer mics add noise → higher flatness
    flatness = librosa.feature.spectral_flatness(y=audio)
    features.append(float(np.mean(flatness)))

    # 5. High-frequency to low-frequency energy ratio
    #    Professional mic (device A) preserves more high-freq energy
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    lf_energy = np.mean(mel[:32, :]) + 1e-8   # low 25% of mel bins
    hf_energy = np.mean(mel[96:, :])           # top 25% of mel bins
    features.append(float(hf_energy / lf_energy))

    return np.array(features, dtype=np.float32)


def extract_batch_features(audio_batch: np.ndarray, sr: int = 22050) -> np.ndarray:
    """Extract features for a batch of audio arrays. Returns (batch, 5)."""
    return np.stack([extract_device_features(a, sr) for a in audio_batch])
