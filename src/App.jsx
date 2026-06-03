import { useEffect, useState } from 'react';
import { Activity, Download, Loader2, Mic, MicOff, Volume2 } from 'lucide-react';

const API_URL = 'http://127.0.0.1:5000/api';

function App() {
  const [status, setStatus] = useState('Siap digunakan');
  const [labels, setLabels] = useState([]);
  const [ttsText, setTtsText] = useState('');
  const [ttsSpeed, setTtsSpeed] = useState('normal');
  const [ttsGender, setTtsGender] = useState('cowo');
  const [isRecording, setIsRecording] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [result, setResult] = useState(null);

  useEffect(() => {
    fetch(`${API_URL}/labels`)
      .then((response) => response.json())
      .then((data) => {
        const nextLabels = data.labels || [];
        setLabels(nextLabels);
        setTtsText(nextLabels[0] || '');
      })
      .catch(() => setStatus('Backend belum aktif'));
  }, []);

  const predictAudio = async () => {
    setIsRecording(true);
    setResult(null);
    setStatus('Merekam suara...');

    try {
      const response = await fetch(`${API_URL}/predict-live`, {
        method: 'POST'
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Prediksi gagal');
      }

      setResult(data);
      setStatus('Prediksi selesai');
    } catch (error) {
      setResult({ error: error.message });
      setStatus('Audio tidak diprediksi');
    } finally {
      setIsRecording(false);
    }
  };

  const playTts = async () => {
    setIsSpeaking(true);
    setStatus('Membuat dan memutar suara...');

    try {
      // Trigger generation (ke server) to validate input
      const postResp = await fetch(`${API_URL}/tts`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          text: ttsText,
          speed: ttsSpeed,
          gender: ttsGender
        }),
      });

      const postData = await postResp.json();
      if (!postResp.ok) throw new Error(postData.error || 'TTS gagal');

      // Download generated WAV and play it
      const params = new URLSearchParams({
        text: ttsText,
        speed: ttsSpeed,
        gender: ttsGender
      });

      const resp = await fetch(`${API_URL}/tts/download?${params.toString()}`);
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ error: 'Gagal mengambil audio' }));
        throw new Error(err.error || 'Gagal mengunduh audio');
      }

      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);

      const audio = new Audio(url);
      audio.play();

      audio.onended = () => {
        URL.revokeObjectURL(url);
      };

      setStatus(`Memutar suara: "${postData.text}"`);
    } catch (error) {
      setStatus(error.message);
    } finally {
      setIsSpeaking(false);
    }
  };

  const downloadTts = async () => {
    setIsDownloading(true);
    setStatus('Menyiapkan file audio...');

    try {
      const params = new URLSearchParams({
        text: ttsText,
        speed: ttsSpeed,
        gender: ttsGender
      });

      const response = await fetch(`${API_URL}/tts/download?${params.toString()}`);

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || 'Gagal menyimpan audio');
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);

      const link = document.createElement('a');
      link.href = url;
      link.download = `tts_${ttsText}_${ttsSpeed}_${ttsGender}.wav`;

      document.body.appendChild(link);
      link.click();
      link.remove();

      URL.revokeObjectURL(url);

      setStatus('Audio berhasil disimpan');
    } catch (error) {
      setStatus(error.message);
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <main className="app-shell">
      <section className="topbar">
        <div>
          <p className="eyebrow">Animal Speech Recognition & Text To Speech</p>
          <h1>ASR & TTS Nama Hewan</h1>
        </div>

        <div className="status-pill">
          <Activity size={18} />
          <span>{status}</span>
        </div>
      </section>

      <section className="dual-workspace">
        <div className="recorder-panel">
          <div className="panel-heading">
            <span>ASR</span>
            <h2>Prediksi Suara</h2>
          </div>

          <div className="meter">
            {isRecording ? <Mic size={54} /> : <MicOff size={54} />}
            <span className={isRecording ? 'pulse active' : 'pulse'} />
          </div>

          <div className="actions single">
            <button
              className="primary"
              disabled={isRecording || isSpeaking}
              onClick={predictAudio}
            >
              {isRecording ? (
                <Loader2 className="spin" size={20} />
              ) : (
                <Mic size={20} />
              )}
              <span>{isRecording ? 'Merekam...' : 'Prediksi Suara'}</span>
            </button>
          </div>

          <div className="result-box">
            {result?.error ? (
              <p className="error-text">{result.error}</p>
            ) : result ? (
              <>
                <p className="result-label">Hasil Prediksi</p>
                <strong>{result.prediction}</strong>

                <div className="bars">
                  {result.top?.map((item) => (
                    <div className="bar-row" key={item.label}>
                      <span>{item.label}</span>

                      <div className="bar-track">
                        <div
                          className="bar-fill"
                          style={{ width: `${item.confidence}%` }}
                        />
                      </div>

                      <b>{item.confidence}%</b>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <p>Klik tombol prediksi, lalu ucapkan satu nama hewan.</p>
            )}
          </div>
        </div>

        <div className="recorder-panel tts-panel">
          <div className="panel-heading">
            <span>TTS</span>
            <h2>Text To Speech</h2>
          </div>

          <div className="tts-visual">
            <Volume2 size={58} />
            <span className={isSpeaking ? 'sound-wave active' : 'sound-wave'} />
          </div>

          <div className="tts-form">
            <label htmlFor="tts-text">Teks</label>
            <input
              id="tts-text"
              value={ttsText}
              onChange={(event) => setTtsText(event.target.value)}
              placeholder="Contoh: halo nama saya firman"
            />

            <label htmlFor="tts-speed">Kecepatan bicara</label>
            <select
              id="tts-speed"
              value={ttsSpeed}
              onChange={(event) => setTtsSpeed(event.target.value)}
            >
              <option value="slow">Lambat</option>
              <option value="normal">Normal</option>
              <option value="fast">Cepat</option>
            </select>

            <label htmlFor="tts-gender">Gender suara</label>
            <select
              id="tts-gender"
              value={ttsGender}
              onChange={(event) => setTtsGender(event.target.value)}
            >
              <option value="cowo">Laki-laki</option>
              <option value="cewe">Perempuan</option>
            </select>
          </div>

          <div className="actions">
            <button
              className="primary secondary-action"
              disabled={isRecording || isSpeaking || isDownloading || !ttsText}
              onClick={playTts}
            >
              {isSpeaking ? (
                <Loader2 className="spin" size={20} />
              ) : (
                <Volume2 size={20} />
              )}

              <span>{isSpeaking ? 'Memutar...' : 'Putar Suara'}</span>
            </button>

            <button
              disabled={isRecording || isSpeaking || isDownloading || !ttsText}
              onClick={downloadTts}
            >
              {isDownloading ? (
                <Loader2 className="spin" size={20} />
              ) : (
                <Download size={20} />
              )}

              <span>{isDownloading ? 'Menyimpan...' : 'Simpan Audio'}</span>
            </button>
          </div>

          <div className="result-box tts-info">
            <p>
              Masukkan teks bebas, pilih kecepatan bicara, lalu pilih gender suara.
            </p>
          </div>
        </div>
      </section>
    </main>
  );
}

export default App;
