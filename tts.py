import asyncio
import time
import pygame
import edge_tts
from pydub import AudioSegment

OUTPUT_MP3 = "tts_output.mp3"
OUTPUT_WAV = "tts_output.wav"

VOICES = {
    "cewe": "id-ID-GadisNeural",
    "cowo": "id-ID-ArdiNeural"
}

SPEEDS = {
    "slow": 0.8,
    "normal": 1.0,
    "fast": 2.0
}


async def generate_tts(text, gender):
    voice = VOICES.get(gender, VOICES["cewe"])

    communicate = edge_tts.Communicate(
        text=text,
        voice=voice
    )

    await communicate.save(OUTPUT_MP3)


def synthesize_speech(text, gender="cewe", speed="normal"):
    if not text.strip():
        raise ValueError("Teks tidak boleh kosong!")

    asyncio.run(generate_tts(text, gender))

    audio = AudioSegment.from_mp3(OUTPUT_MP3)

    speed_factor = SPEEDS.get(speed, 1.0)

    # Fast
    if speed_factor > 1.0:
        audio = audio.speedup(playback_speed=speed_factor)

    # Slow
    elif speed_factor < 1.0:
        new_frame_rate = int(audio.frame_rate * speed_factor)

        audio = audio._spawn(
            audio.raw_data,
            overrides={"frame_rate": new_frame_rate}
        ).set_frame_rate(audio.frame_rate)

    audio.export(OUTPUT_WAV, format="wav")

    return OUTPUT_WAV


def play_audio(audio_path):
    pygame.mixer.init()
    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(0.5)

    pygame.mixer.quit()


def main():
    text = input("Masukkan teks: ")
    gender = input("Pilih gender (cewe / cowo): ").lower()
    speed = input("Pilih speed (slow / normal / fast): ").lower()

    try:
        audio_path = synthesize_speech(text, gender, speed)
        print("Memutar audio...")
        play_audio(audio_path)

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
