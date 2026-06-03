import os

import joblib
import numpy as np
from scipy.fftpack import dct
from scipy.io import wavfile
from scipy.signal import resample_poly, spectrogram
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DATASET_PATH = "dataset"
MODEL_PATH = "model_asr_v2.pkl"
SCALER_PATH = "scaler_v2.pkl"
SAMPLE_RATE = 16000
N_MFCC = 40
EPSILON = 1e-10
AUGMENT_PER_FILE = 3
RANDOM_SEED = 42


def preprocess_audio(audio):
    audio = audio.astype(np.float32)
    peak = np.max(np.abs(audio))

    if peak > 0:
        audio = audio / peak

    threshold = max(0.02, 0.08 * np.max(np.abs(audio)))
    voiced = np.where(np.abs(audio) > threshold)[0]

    if len(voiced) == 0:
        return audio

    start = max(0, voiced[0] - int(0.1 * SAMPLE_RATE))
    end = min(len(audio), voiced[-1] + int(0.1 * SAMPLE_RATE))
    trimmed = audio[start:end]

    if len(trimmed) < int(0.2 * SAMPLE_RATE):
        return audio

    return trimmed


def load_audio(file_path):
    sr, audio = wavfile.read(file_path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    else:
        audio = audio.astype(np.float32)

    if sr != SAMPLE_RATE:
        gcd = np.gcd(sr, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // gcd, sr // gcd).astype(np.float32)
        sr = SAMPLE_RATE

    return audio, sr


def frame_features(audio, sr):
    frequencies, _, power = spectrogram(
        audio,
        fs=sr,
        window="hann",
        nperseg=512,
        noverlap=256,
        mode="magnitude",
    )
    power = power.T.astype(np.float32)
    power = np.maximum(power, EPSILON)
    log_power = np.log(power)

    mfcc = dct(log_power, type=2, axis=1, norm="ortho")[:, :N_MFCC]
    delta = np.gradient(mfcc, axis=0)
    delta2 = np.gradient(delta, axis=0)

    weights = power / np.sum(power, axis=1, keepdims=True)
    centroid = np.sum(weights * frequencies, axis=1)
    bandwidth = np.sqrt(np.sum(weights * (frequencies - centroid[:, None]) ** 2, axis=1))
    cumulative = np.cumsum(weights, axis=1)
    rolloff_idx = np.argmax(cumulative >= 0.85, axis=1)
    rolloff = frequencies[rolloff_idx]
    rms = np.sqrt(np.mean(power**2, axis=1))

    signs = np.signbit(audio)
    zcr = np.mean(signs[1:] != signs[:-1])

    return mfcc, delta, delta2, centroid, bandwidth, rolloff, rms, zcr


def extract_features(file_path):
    audio, sr = load_audio(file_path)
    return extract_features_from_audio(audio, sr)


def extract_features_from_audio(audio, sr):
    audio = preprocess_audio(audio)

    mfcc, delta, delta2, centroid, bandwidth, rolloff, rms, zcr = frame_features(audio, sr)

    features = np.concatenate(
        [
            np.mean(mfcc, axis=0),
            np.std(mfcc, axis=0),
            np.mean(delta, axis=0),
            np.std(delta, axis=0),
            np.mean(delta2, axis=0),
            np.std(delta2, axis=0),
            [np.mean(centroid), np.std(centroid)],
            [np.mean(bandwidth), np.std(bandwidth)],
            [np.mean(rolloff), np.std(rolloff)],
            [np.mean(rms), np.std(rms)],
            [zcr],
        ]
    )

    return features


def augment_audio(audio, rng):
    augmented = audio.copy()

    speed = rng.uniform(0.9, 1.1)
    if abs(speed - 1.0) > 0.01:
        new_length = max(1, int(len(augmented) / speed))
        x_old = np.linspace(0, 1, len(augmented))
        x_new = np.linspace(0, 1, new_length)
        augmented = np.interp(x_new, x_old, augmented).astype(np.float32)

    gain = rng.uniform(0.75, 1.25)
    augmented = augmented * gain

    noise_level = rng.uniform(0.001, 0.008)
    augmented = augmented + rng.normal(0, noise_level, size=len(augmented)).astype(np.float32)

    peak = np.max(np.abs(augmented))
    if peak > 0:
        augmented = augmented / peak

    return augmented.astype(np.float32)


def load_dataset():
    X = []
    y = []
    rng = np.random.default_rng(RANDOM_SEED)

    for label in sorted(os.listdir(DATASET_PATH)):
        folder_path = os.path.join(DATASET_PATH, label)

        if not os.path.isdir(folder_path):
            continue

        for file_name in sorted(os.listdir(folder_path)):
            if not file_name.lower().endswith(".wav"):
                continue

            file_path = os.path.join(folder_path, file_name)

            try:
                audio, sr = load_audio(file_path)
                X.append(extract_features_from_audio(audio, sr))
                y.append(label)

                for _ in range(AUGMENT_PER_FILE):
                    X.append(extract_features_from_audio(augment_audio(audio, rng), sr))
                    y.append(label)

                print(f"Berhasil proses: {file_path}")
            except Exception as e:
                print(f"Error di {file_path}: {e}")

    return np.array(X), np.array(y)


def main():
    X, y = load_dataset()

    print("\nJumlah data:", len(X))
    print("Jumlah label:", len(np.unique(y)))

    if len(X) == 0:
        raise RuntimeError("Dataset kosong. Pastikan file .wav ada di folder dataset/<label>.")

    model = SVC(kernel="rbf", C=10, gamma="scale", probability=True, class_weight="balanced")
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Akurasi test:", round(accuracy * 100, 2), "%")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=pipeline.classes_))
    print("Urutan label:", list(pipeline.classes_))

    joblib.dump(pipeline, MODEL_PATH)

    # Disimpan untuk kompatibilitas dengan file lama, walau predict.py sekarang memakai pipeline.
    joblib.dump(pipeline.named_steps["scaler"], SCALER_PATH)

    print("\nModel berhasil disimpan!")


if __name__ == "__main__":
    main()
