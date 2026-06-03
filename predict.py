import joblib
import numpy as np
import sounddevice as sd
from scipy.fftpack import dct
from scipy.io import wavfile
from scipy.signal import resample_poly, spectrogram


MODEL_PATH = "model_asr_v2.pkl"
TEST_AUDIO_PATH = "test.wav"
SAMPLE_RATE = 16000
RECORD_SAMPLE_RATE = 44100
N_MFCC = 40
DURATION = 3
MIN_VOLUME = 0.01
MIN_PEAK = 0.04
MIN_VOICED_DURATION = 0.15
EPSILON = 1e-10


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


def validate_voice_activity(audio, sr):
    abs_audio = np.abs(audio)
    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.max(abs_audio))

    noise_samples = max(1, int(0.35 * sr))
    noise_floor = float(np.sqrt(np.mean(audio[:noise_samples] ** 2)))
    threshold = max(MIN_VOLUME, noise_floor * 4.0)
    voiced = abs_audio > threshold
    voiced_duration = float(np.sum(voiced) / sr)

    if rms < MIN_VOLUME or peak < MIN_PEAK or voiced_duration < MIN_VOICED_DURATION:
        raise ValueError(
            "Tidak ada ucapan terdeteksi. Audio masuk diabaikan karena hanya silence/noise."
        )


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
    validate_voice_activity(audio, sr)

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


def record_audio():
    print("Silakan ucapkan satu nama hewan sesuai label dataset...")
    print("Label yang bisa dikenali tergantung folder di dataset.")

    audio = sd.rec(
        int(DURATION * RECORD_SAMPLE_RATE),
        samplerate=RECORD_SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )

    sd.wait()
    wavfile.write(TEST_AUDIO_PATH, RECORD_SAMPLE_RATE, audio)


def predict_file(file_path):
    model = joblib.load(MODEL_PATH)
    features = extract_features(file_path)
    prediction = model.predict([features])[0]
    probabilities = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([features])[0]

    return model, prediction, probabilities


def main():
    record_audio()
    print("Memproses...")

    try:
        model, prediction, probabilities = predict_file(TEST_AUDIO_PATH)

        print("\nHasil Prediksi:", prediction)

        if probabilities is not None:
            top_indices = np.argsort(probabilities)[::-1][:3]
            confidence = probabilities[top_indices[0]] * 100

            print("\nKemungkinan teratas:")
            for index in top_indices:
                label = model.classes_[index]
                label_confidence = probabilities[index] * 100
                print(f"- {label}: {label_confidence:.2f}%")

            if confidence < 55:
                print("\nConfidence masih rendah. Tambahkan sampel suara kamu lalu train ulang.")

    except ValueError as e:
        print(f"\n{e}")


if __name__ == "__main__":
    main()
