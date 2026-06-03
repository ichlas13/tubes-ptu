import os
from datetime import datetime

import sounddevice as sd
from scipy.io import wavfile
//coba

DATASET_PATH = "dataset"
SAMPLE_RATE = 44100
DURATION = 2.5
DEFAULT_REPEATS = 5


def record_sample(label, index):
    folder_path = os.path.join(DATASET_PATH, label)
    os.makedirs(folder_path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(folder_path, f"live_{timestamp}_{index}.wav")

    input(f"Tekan Enter, lalu ucapkan '{label}'...")
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    wavfile.write(file_path, SAMPLE_RATE, audio)
    print(f"Tersimpan: {file_path}")


def main():
    labels = [
        label
        for label in sorted(os.listdir(DATASET_PATH))
        if os.path.isdir(os.path.join(DATASET_PATH, label))
    ]

    print("Label tersedia:", ", ".join(labels))
    label = input("Label yang mau direkam: ").strip().lower()

    if label not in labels:
        raise ValueError("Label tidak ada di folder dataset.")

    repeats_input = input(f"Jumlah rekaman untuk '{label}' [{DEFAULT_REPEATS}]: ").strip()
    repeats = int(repeats_input) if repeats_input else DEFAULT_REPEATS

    for index in range(1, repeats + 1):
        record_sample(label, index)

    print("\nSelesai. Jalankan: python train.py")


if __name__ == "__main__":
    main()
