import os
import subprocess
import threading
from datetime import datetime

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from scipy.io import wavfile

from predict import MODEL_PATH, TEST_AUDIO_PATH, predict_file, record_audio
from tts import synthesize_speech


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
UPLOAD_PATH = os.path.join(BASE_DIR, "web_upload.wav")

app = Flask(__name__)
CORS(app)

train_state = {
    "running": False,
    "returncode": None,
    "log": "",
}


def get_labels():
    if not os.path.isdir(DATASET_PATH):
        return []

    return [
        label
        for label in sorted(os.listdir(DATASET_PATH))
        if os.path.isdir(os.path.join(DATASET_PATH, label))
    ]


def label_counts():
    counts = {}

    for label in get_labels():
        folder = os.path.join(DATASET_PATH, label)
        counts[label] = len([
            name for name in os.listdir(folder)
            if name.lower().endswith(".wav")
        ])

    return counts


def save_uploaded_audio(file_storage, path):
    file_storage.save(path)

    # validasi file audio
    wavfile.read(path)


@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "model_exists": os.path.exists(os.path.join(BASE_DIR, MODEL_PATH)),
        "labels": get_labels(),
        "counts": label_counts(),
    })


@app.get("/api/labels")
def labels():
    return jsonify({
        "labels": get_labels(),
        "counts": label_counts()
    })


# =========================
# TTS BARU
# =========================
@app.post("/api/tts")
def text_to_speech():
    data = request.get_json(silent=True) or {}

    text = data.get("text", "").strip()
    speed = data.get("speed", "normal").strip().lower()
    gender = data.get("gender", "cowo").strip().lower()

    if not text:
        return jsonify({"error": "Teks belum diisi."}), 400

    if gender not in {"cowo", "cewe"}:
        return jsonify({"error": "Pilihan gender tidak valid."}), 400

    if speed not in {"slow", "normal", "fast"}:
        return jsonify({"error": "Pilihan speed tidak valid."}), 400

    try:
        audio_path = synthesize_speech(text, gender, speed)

        return jsonify({
            "ok": True,
            "text": text,
            "speed": speed,
            "gender": gender,
            "path": audio_path
        })

    except Exception as e:
        return jsonify({"error": f"Gagal generate suara: {e}"}), 500


@app.get("/api/tts/download")
def download_tts():
    text = request.args.get("text", "").strip()
    speed = request.args.get("speed", "normal").strip().lower()
    gender = request.args.get("gender", "cowo").strip().lower()

    if not text:
        return jsonify({"error": "Teks belum diisi."}), 400

    if gender not in {"cowo", "cewe"}:
        return jsonify({"error": "Pilihan gender tidak valid."}), 400

    if speed not in {"slow", "normal", "fast"}:
        return jsonify({"error": "Pilihan speed tidak valid."}), 400

    try:
        audio_path = synthesize_speech(text, gender, speed)

        return send_file(
            audio_path,
            mimetype="audio/wav",
            as_attachment=True,
            download_name=f"tts_{text}_{speed}_{gender}.wav",
        )

    except Exception as e:
        return jsonify({"error": f"Gagal menyimpan audio: {e}"}), 500


# =========================
# ASR (TIDAK DIUBAH)
# =========================
@app.post("/api/predict")
def predict_audio():
    if "audio" not in request.files:
        return jsonify({"error": "File audio tidak ditemukan."}), 400

    try:
        save_uploaded_audio(request.files["audio"], UPLOAD_PATH)
        model, prediction, probabilities = predict_file(UPLOAD_PATH)

        top = []
        if probabilities is not None:
            top_indices = probabilities.argsort()[::-1][:3]
            top = [
                {
                    "label": str(model.classes_[index]),
                    "confidence": round(float(probabilities[index]) * 100, 2),
                }
                for index in top_indices
            ]

        return jsonify({
            "prediction": str(prediction),
            "top": top
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 422

    except Exception as e:
        return jsonify({"error": f"Gagal memproses audio: {e}"}), 500


@app.post("/api/predict-live")
def predict_live_audio():
    try:
        record_audio()

        model, prediction, probabilities = predict_file(
            os.path.join(BASE_DIR, TEST_AUDIO_PATH)
        )

        top = []
        if probabilities is not None:
            top_indices = probabilities.argsort()[::-1][:3]
            top = [
                {
                    "label": str(model.classes_[index]),
                    "confidence": round(float(probabilities[index]) * 100, 2),
                }
                for index in top_indices
            ]

        return jsonify({
            "prediction": str(prediction),
            "top": top
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 422

    except Exception as e:
        return jsonify({"error": f"Gagal menjalankan predict.py: {e}"}), 500


@app.post("/api/samples")
def save_sample():
    label = request.form.get("label", "").strip().lower()

    if label not in get_labels():
        return jsonify({"error": "Label tidak valid."}), 400

    if "audio" not in request.files:
        return jsonify({"error": "File audio tidak ditemukan."}), 400

    folder = os.path.join(DATASET_PATH, label)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(folder, f"web_{timestamp}.wav")

    try:
        save_uploaded_audio(request.files["audio"], file_path)

        return jsonify({
            "ok": True,
            "path": file_path,
            "counts": label_counts()
        })

    except Exception as e:
        return jsonify({"error": f"Gagal menyimpan sample: {e}"}), 500


def run_training():
    train_state["running"] = True
    train_state["returncode"] = None
    train_state["log"] = ""

    process = subprocess.Popen(
        ["python", "-u", "train.py"],
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    lines = []
    assert process.stdout is not None

    for line in process.stdout:
        lines.append(line)
        train_state["log"] = "".join(lines[-180:])

    process.wait()
    train_state["returncode"] = process.returncode
    train_state["running"] = False


@app.post("/api/train")
def train_model():
    if train_state["running"]:
        return jsonify({
            "ok": True,
            "running": True,
            "message": "Training sedang berjalan."
        })

    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()

    return jsonify({
        "ok": True,
        "running": True,
        "message": "Training dimulai."
    })


@app.get("/api/train/status")
def train_status():
    return jsonify(train_state)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)