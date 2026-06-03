import streamlit as st
import pyttsx3
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

# --- Tampilan Judul Aplikasi ---
st.title("🎙️ Aplikasi Text-to-Speech (TTS)")
st.write("Projek Akhir PTU - Ubah teks menjadi suara dengan fitur visualisasi MFCC.")

# --- Fitur 1 & 2: Input Teks (Bahasa Indonesia) ---
input_teks = st.text_area("Masukkan teks di sini:", "Halo, selamat datang di aplikasi Text to Speech kelompok kami.")

# --- Pengaturan (Fitur 5 & Fitur Tambahan 1) ---
col1, col2 = st.columns(2)
with col1:
    speed_option = st.selectbox("Kecepatan Bicara:", ["Normal", "Slow", "Fast"])
with col2:
    gender_option = st.radio("Pilih Suara:", ["Laki-laki", "Perempuan"])

# --- Tombol Eksekusi ---
if st.button("Generate Suara"):
    if input_teks:
        with st.spinner('Sedang memproses suara...'):
            # Inisialisasi engine pyttsx3
            engine = pyttsx3.init()
            
            # Atur Kecepatan (Rate)
            if speed_option == "Slow":
                engine.setProperty('rate', 100)
            elif speed_option == "Normal":
                engine.setProperty('rate', 150)
            else: # Fast
                engine.setProperty('rate', 220)

            # Atur Gender Suara
            voices = engine.getProperty('voices')
            # Catatan: Indeks [0] & [1] bisa berbeda tiap laptop tergantung bahasa OS.
            if gender_option == "Perempuan" and len(voices) > 1:
                engine.setProperty('voice', voices[1].id) 
            else:
                engine.setProperty('voice', voices[0].id)
                
            # --- Fitur 4: Simpan Audio ---
            file_name = "output_suara.wav"
            engine.save_to_file(input_teks, file_name)
            engine.runAndWait() # Proses penyimpanan
            
            st.success("✅ Suara berhasil dibuat!")
            
            # --- Fitur 6: Output Suara (Putar di Web) ---
            audio_file = open(file_name, 'rb')
            st.audio(audio_file.read(), format='audio/wav')
            
            # --- Fitur Tambahan 2: Visualisasi MFCC ---
            st.subheader("Visualisasi Fitur MFCC")
            # Membaca file audio yang baru disimpan menggunakan librosa
            y, sr = librosa.load(file_name, sr=None)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # Membuat grafik
            fig, ax = plt.subplots(figsize=(10, 4))
            img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
            fig.colorbar(img, ax=ax)
            ax.set(title='Mel-frequency cepstral coefficients (MFCC)')
            
            # Menampilkan grafik di Streamlit
            st.pyplot(fig)
            
    else:
        st.warning("Teks tidak boleh kosong!")